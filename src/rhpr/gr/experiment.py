"""Audited Sentence-T5 -> GRID RK-Means -> GRID encoder-decoder baseline.

The finite prefix schedule, complete SentenceTransformer encoder, and item-level
evaluator are explicit adapters around pinned upstream GRID components. This is
a TIGER-style functional baseline, not a reproduction of a published score.
"""
from __future__ import annotations

import importlib.metadata
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.resources import unpack_grid, verify_resources, verify_parent_artifacts


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _event(phase: str, **values) -> None:
    print(json.dumps({"phase": phase, **values}, allow_nan=False), flush=True)


def _encode_items(config: dict, resources: Path, texts: list[str], output: Path, device: str,
                  parent_root: Path | None = None):
    import numpy as np
    from rhpr.gr.sid import validate_item_embeddings

    specification = config["representation"]
    started = time.monotonic()
    if specification["kind"] == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(resources / "sentence-t5-base"), device=device,
                                    local_files_only=True)
        model.max_seq_length = specification["max_length"]
        embeddings = model.encode(texts, batch_size=specification["batch_size"],
                                  show_progress_bar=True, convert_to_numpy=True)
        details = {"kind": specification["kind"], "model": specification["model"],
                   "revision": specification["revision"],
                   "modules": [type(module).__name__ for module in model],
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "max_length": model.max_seq_length}
        del model
    elif specification["kind"] == "external_embeddings":
        # A deliberately narrow plug-in seam: every future encoder must supply
        # the same catalogue row order and a separately recorded objective.
        if "parent_file" in specification:
            name = specification["parent_file"]
            if parent_root is None or Path(name).name != name:
                raise ValueError("parent embeddings require verified parent artifacts")
            source = parent_root / name
        else:
            source = resources / specification["path"]
        if file_sha256(source) != specification["sha256"]:
            raise ValueError("external embedding digest differs from the experiment")
        if specification["catalog_sha256"] != file_sha256(resources / "data/catalog.json"):
            raise ValueError("external embeddings use a different catalogue order")
        embeddings = np.load(source, allow_pickle=False)
        details = dict(specification)
    else:
        raise ValueError("unsupported item representation")
    embeddings = validate_item_embeddings(embeddings, num_items=len(texts))
    np.save(output / "item-embeddings.npy", embeddings, allow_pickle=False)
    details.update(seconds=time.monotonic() - started, shape=list(embeddings.shape),
                   sha256=file_sha256(output / "item-embeddings.npy"))
    _write(output / "representation.json", details)
    _event("item_embeddings_ready", **details)
    return embeddings, details


def _model_batch(contexts, users, targets, sids, device):
    import numpy as np
    import torch
    observed = contexts >= 0
    inputs = sids[np.maximum(contexts, 0)].copy()
    inputs[~observed] = -1
    return {
        "input_ids": torch.as_tensor(inputs.reshape(len(users), -1), dtype=torch.long, device=device),
        "attention_mask_encoder": torch.as_tensor(
            np.repeat(observed, sids.shape[1], axis=1), dtype=torch.long, device=device),
        "user_id": torch.as_tensor(users[:, None], dtype=torch.long, device=device),
        "future_ids": torch.as_tensor(sids[targets], dtype=torch.long, device=device),
    }


def _loss(model, batch):
    import torch
    output = model(**batch)[:, :-1]
    return sum(torch.nn.functional.cross_entropy(head(output[:, level]),
                                                batch["future_ids"][:, level])
               for level, head in enumerate(model.decoder.decoder_mlp))


def _autocast(precision: str, device: str):
    """Make numeric precision explicit; T4 has no native BF16 execution."""
    import torch
    if precision == "fp32":
        return nullcontext()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("GR precision must be fp32, fp16, or bf16")
    kind = torch.device(device).type
    if precision == "fp16" and kind != "cuda":
        raise ValueError("GR fp16 mixed precision requires CUDA")
    if precision == "bf16" and kind == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("this GPU does not provide native BF16 support")
    return torch.autocast(device_type=kind, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def _evaluate(model, examples, sid_index, batch_size: int, device: str, precision: str = "fp32"):
    import numpy as np
    import torch
    from rhpr.gr.sid import evaluate_ranked_sids

    contexts, users, targets = examples
    generated, probabilities = [], []
    started = time.monotonic()
    model.eval()
    with torch.inference_mode(), _autocast(precision, device):
        for start in range(0, len(targets), batch_size):
            end = start + batch_size
            batch = _model_batch(contexts[start:end], users[start:end], targets[start:end],
                                 sid_index.sids, device)
            ids, scores = model.generate(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask_encoder"],
                                         user_id=batch["user_id"])
            if not torch.isfinite(scores).all():
                raise FloatingPointError("nonfinite beam scores")
            generated.append(ids.cpu().numpy())
            probabilities.append(scores.cpu().numpy())
    predictions = np.concatenate(generated)
    values = evaluate_ranked_sids(sid_index, predictions, targets, cutoffs=(5, 10))
    values["seconds"] = time.monotonic() - started
    return values, predictions, np.concatenate(probabilities)


def _train_generator(model, users, sid_index, config: dict, output: Path, device: str):
    import numpy as np
    import torch

    specification = config["training"]
    seed = config["seed"]
    train = build_examples(users, config["data"]["history_items"], "train")
    validation = build_examples(users, config["data"]["history_items"], "validation")
    optimizer = torch.optim.Adam(model.parameters(), lr=specification["learning_rate"],
                                 weight_decay=specification["weight_decay"])
    precision = specification.get("precision", "fp16" if specification.get("amp", False) and device.startswith("cuda") else "fp32")
    # Validate the requested policy before allocating optimizer state or taking
    # an update. GradScaler is only needed for FP16, not BF16 or FP32.
    with _autocast(precision, device):
        pass
    amp = precision == "fp16"
    evaluation_precision = specification.get("evaluation_precision", "fp32")
    with _autocast(evaluation_precision, device):
        pass
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    curves = []
    best_metric, best_epoch, stale = -1.0, 0, 0
    best_path = output / "generator-best.pt"
    contexts, user_ids, targets = train
    batch_size = specification["batch_size"]
    physical = specification["physical_batch_size"]
    updates = 0
    skipped_updates = 0
    started = time.monotonic()
    stop_reason = "epoch_limit"
    mean_epoch_seconds = 0.0
    last_validation = None
    for epoch in range(1, specification["max_epochs"] + 1):
        # Budget exits only at an epoch boundary, preserving an interpretable
        # full pass. Save selected and last checkpoints for later continuation.
        if epoch > 1 and time.monotonic() - started + mean_epoch_seconds > specification["max_seconds"]:
            stop_reason = "time_budget"
            break
        order = np.random.default_rng(seed + epoch).permutation(len(targets))
        model.train()
        epoch_started, total_loss = time.monotonic(), 0.0
        for begin in range(0, len(order), batch_size):
            chosen = order[begin:begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            for position in range(0, len(chosen), physical):
                part = chosen[position:position + physical]
                batch = _model_batch(contexts[part], user_ids[part], targets[part], sid_index.sids, device)
                with _autocast(precision, device):
                    loss = _loss(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite GR training loss")
                total_loss += loss.detach().float().item() * len(part)
                scaler.scale(loss * (len(part) / len(chosen))).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), specification["grad_clip_norm"])
            finite_gradients = bool(torch.isfinite(norm))
            if not finite_gradients and not amp:
                raise FloatingPointError("nonfinite GR training gradient")
            # AMP's scaler skips an overflowing update and reduces the scale.
            # Count that explicitly rather than treating normal scaling as an
            # LR fallback, or claiming that a skipped batch updated the model.
            scaler.step(optimizer)
            scaler.update()
            if finite_gradients:
                updates += 1
            else:
                skipped_updates += 1
                _event("gr_amp_update_skipped", epoch=epoch, scale=scaler.get_scale(),
                       skipped_updates=skipped_updates)
                if skipped_updates > 16:
                    raise FloatingPointError("repeated GR AMP gradient overflows")
        train_seconds = time.monotonic() - epoch_started
        row = {"epoch": epoch, "updates": updates, "amp_skipped_updates": skipped_updates,
               "train_examples": len(targets),
               "train_loss_sum_over_codes": total_loss / len(targets), "train_seconds": train_seconds}
        if epoch == 1 or epoch % specification["validation_interval"] == 0 or epoch == specification["max_epochs"]:
            last_validation, _, _ = _evaluate(model, validation, sid_index,
                                             specification["evaluation_batch_size"], device,
                                             precision=evaluation_precision)
            row["validation"] = last_validation
            score = last_validation["ndcg@10"]
            if score > best_metric:
                best_metric, best_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "updates": updates,
                            "config": config, "validation": last_validation}, best_path)
            else:
                stale += 1
        curves.append(row)
        _write(output / "training-curves.json", curves)
        _event("gr_epoch_complete", **row)
        # Exact resume state at every completed epoch, separate from selection.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "updates": updates,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                    "config": config, "best_epoch": best_epoch, "best_metric": best_metric,
                    "stale_validation_checks": stale, "amp_skipped_updates": skipped_updates},
                   output / "generator-last.pt")
        mean_epoch_seconds = time.monotonic() - epoch_started
        # An explicit epoch-based patience remains interpretable when changing
        # validation cadence. Legacy configurations keep their check-based rule.
        patience_exhausted = (
            epoch - best_epoch >= specification["patience_epochs"]
            if "patience_epochs" in specification else stale >= specification["patience"]
        )
        eligible_check = "validation" in row if "patience_epochs" in specification else True
        if epoch >= specification["min_epochs"] and eligible_check and patience_exhausted:
            stop_reason = "early_stopping"
            break
    if not best_path.is_file():
        raise RuntimeError("training did not produce a validation-selected checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    values = {"training_epochs": len(curves), "optimizer_updates": updates,
              "amp_skipped_updates": skipped_updates,
              "training_precision": precision, "evaluation_precision": evaluation_precision,
              "best_epoch": best_epoch, "training_seconds": time.monotonic() - started,
              "stopping_reason": stop_reason,
              "validation_ndcg10": best_metric,
              "training_examples_per_epoch": len(targets)}
    _write(output / "training-selection.json", {**values, "criterion": "validation_item_ndcg@10",
                                                "selected_validation": checkpoint["validation"]})
    return values


def run_gr_baseline(config: dict, output: Path) -> dict:
    import numpy as np
    import torch
    from rhpr.gr.grid import build_grid_generator, fit_grid_rkmeans, prepare_grid_source
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = config["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("this registered GR experiment requires CUDA")
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    resources = resolve_dataset_directory(config["resources"]["path"])
    input_manifest = verify_resources(resources, config["resources"]["manifest_sha256"])
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    scratch = Path(config["resources"].get("scratch_path", "/kaggle/temp/rhpr-gr"))
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(scratch / "huggingface")
    upstream = unpack_grid(resources / "grid-source.bin", scratch / "grid-original")
    grid_root = prepare_grid_source(upstream, scratch / "grid-runtime")
    texts, users, original_ids, counts = load_beauty(resources / "data", config["data"])
    parent_root = verify_parent_artifacts(config["parent"]) if "parent" in config else None
    _write(output / "input-provenance.json", {"manifest": input_manifest, "counts": counts,
                                              "data": config["data"]})
    embeddings, representation_report = _encode_items(config, resources, texts, output, device,
                                                      parent_root=parent_root)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    codes, centroids, quantizer_report = fit_grid_rkmeans(
        embeddings, config["quantizer"], seed=seed, device=device, source_root=grid_root)
    sid_index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
    np.save(output / "semantic-ids.npy", sid_index.sids, allow_pickle=False)
    np.save(output / "quantizer-centroids.npy", centroids, allow_pickle=False)
    _write(output / "sid-diagnostics.json", sid_index.diagnostics())
    _write(output / "quantizer.json", quantizer_report)
    _write(output / "sid-catalogue.json", {"original_item_ids": original_ids,
                                          "row_order": "internal_item_id_0_based",
                                          "semantic_ids_sha256": file_sha256(output / "semantic-ids.npy")})
    _event("gr_semantic_ids_ready", diagnostics=sid_index.diagnostics(), quantizer=quantizer_report)
    model = build_grid_generator(sid_index.sids, config["generator"], seed=seed,
                                 device=device, source_root=grid_root)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model_report = {"parameters": parameters, "generator": config["generator"],
                    "framework_revision": config["resources"]["grid_revision"],
                    "device": device, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "none",
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ["transformers", "sentence-transformers", "lightning", "numpy"]}}
    # A partial operation count with a declared convention, rather than an
    # estimate presented as total model FLOPs. Encoding/quantization timings
    # and wall-clock training/evaluation costs are reported separately.
    one = build_examples(users[:1], config["data"]["history_items"], "validation")
    batch = _model_batch(one[0], one[1], one[2], sid_index.sids, device)
    model.eval()
    with torch.inference_mode(), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        _loss(model, batch)
    model_report["profiled_forward_flops_batch1"] = sum(event.flops for event in profile.key_averages())
    model_report["flop_convention"] = "torch.profiler with_flops; supported matmul/convolution ops only; teacher-forced forward; not beam search or backward"
    _write(output / "generator.json", model_report)
    _event("gr_generator_ready", **model_report)
    values = _train_generator(model, users, sid_index, config, output, device)
    if config["evaluation"]["test_after_selection"]:
        test_examples = build_examples(users, config["data"]["history_items"], "test")
        metrics, ids, scores = _evaluate(model, test_examples, sid_index,
                                         config["training"]["evaluation_batch_size"], device,
                                         precision=config["training"].get("evaluation_precision", "fp32"))
        np.savez_compressed(output / "test-predictions.npz", semantic_ids=ids, scores=scores,
                            user_ids=test_examples[1], target_item_ids=test_examples[2])
        _write(output / "test-metrics.json", metrics)
        values.update({"test_" + key.replace("@", ""): value for key, value in metrics.items()})
    values.update(experiment=config["name"], seed=seed, generator_parameters=parameters,
                  embedding_seconds=representation_report["seconds"],
                  quantizer_seconds=quantizer_report["elapsed_seconds"],
                  catalogue_items=counts["items"], users=counts["users"],
                  exact_published_reproduction=False)
    return values
