"""One-device native TPU pipeline: Flan XXL, frozen RK-Means, TIGER windows.

Native torch/torch_xla are never replaced. Host input processing and metrics are
explicit; the model, BF16 gradients and beam computation execute on actual TPU.
"""
from __future__ import annotations

import gc
from itertools import islice
import json
from pathlib import Path
import time

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write


def build_model(sids, source, output, *, tiny=False):
    import torch
    from hydra.utils import instantiate
    from rhpr.gr.genpas import compose_beauty
    folder = output / "model-input"
    (folder / "sids").mkdir(parents=True, exist_ok=True)
    torch.save(torch.as_tensor(sids.T.copy()), folder / "sids/flan-t5-xxl_rkmeans_3_256_seed42.pt")
    cfg = compose_beauty(source, folder, output)
    if tiny:
        for name, value in dict(vocab_size=16, d_model=8, d_ff=16, d_kv=4,
                                num_heads=2, num_layers=1, dropout_rate=0.).items():
            cfg.model.huggingface_model.config[name] = value
        cfg.model.decoder.config.num_layers = 1
    return instantiate(cfg.model), cfg


def bootstrap(config, output):
    import os
    import platform
    from importlib import metadata
    import sys
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    native_versions = {}
    for name in ("torch", "torch-xla", "libtpu", "jax", "tensorflow", "numpy", "scikit-learn", "pyarrow"):
        try:
            native_versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            native_versions[name] = None
    _write(output / "native-runtime.json", {"python": platform.python_version(), "packages": native_versions})
    _event("genpas_native_runtime", python=platform.python_version(), packages=native_versions)
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    if xr.device_type() != "TPU":
        raise RuntimeError("This run requires actual TPU hardware; no CPU fallback")
    from rhpr.gr.tpu_probe import install_probe_dependencies
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    resources = resolve_dataset_directory(config["resources"]["path"])
    verify_resources(resources, config["resources"]["manifest_sha256"])
    scratch = Path("/kaggle/temp/rhpr-genpas-tpu")
    scratch.mkdir(parents=True, exist_ok=True)
    installed = install_probe_dependencies(resources, scratch / "site", config["wheels_manifest_sha256"])
    os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", MPLCONFIGDIR=str(scratch / "matplotlib"))
    from rhpr.gr.genpas import activate_source
    root = unpack_grid(resources / "genpas-source.bin", scratch / "original")
    source = activate_source(root)
    device = xm.xla_device()
    xm.set_rng_state(config["seed"], device=device)
    _write(output / "runtime-bootstrap.json", {"python": platform.python_version(),
        "torch": torch.__version__, "torch_xla": torch_xla.__version__, "actual_device": str(device),
        "device_type": xr.device_type(), "addressable_devices": xr.addressable_device_count(),
        "used_devices": 1, "native_torch_xla_retained": True, "auxiliary_wheels": installed})
    return resources, source, device, xm


def probe(source, output, device, xm):
    import numpy as np
    import torch
    from rhpr.gr.genpas_xla import adapt_beams, per_example_loss
    torch.manual_seed(42)
    sids = np.array([[i % 256, i // 256, 0, 0] for i in range(512)], dtype=np.int64)
    model, _ = build_model(sids, source, output)
    model.eval()
    batch = {"input_ids": torch.tensor(np.tile(sids[:25].reshape(1, -1), (2, 1))),
        "attention_mask_encoder": torch.ones(2, 100, dtype=torch.long),
        "user_id": torch.zeros(2, 100, dtype=torch.long), "future_ids": torch.tensor(sids[25:27])}
    with torch.no_grad():
        reference = per_example_loss(model, batch)
        original_ids, original_scores = model.generate(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])
    adapt_beams(model)
    with torch.no_grad():
        ids, scores = model.generate(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])
    torch.testing.assert_close(ids, original_ids, rtol=0, atol=0)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)
    model.to(device)
    native = {name: value.to(device) for name, value in batch.items()}
    with torch.no_grad():
        actual = per_example_loss(model, native)
    xm.mark_step()
    torch.testing.assert_close(actual.cpu(), reference, rtol=3e-4, atol=3e-4)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    before = next(model.parameters()).detach().cpu().clone()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("xla", dtype=torch.bfloat16):
            loss = per_example_loss(model, native).mean()
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        losses.append(float(loss.detach().cpu()))
    if not np.isfinite(losses).all() or torch.equal(before, next(model.parameters()).detach().cpu()):
        raise FloatingPointError("TPU BF16 optimizer did not make finite effective updates")
    # Compare *the same post-update model* on CPU and TPU, not just output shapes.
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        xla_ids, xla_scores = model.generate(input_ids=native["input_ids"],
            attention_mask=native["attention_mask_encoder"], user_id=native["user_id"])
    xm.mark_step()
    xla_ids, xla_scores = xla_ids.cpu(), xla_scores.cpu()
    model.to("cpu").load_state_dict(state)
    with torch.no_grad():
        cpu_ids, cpu_scores = model.generate(input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])
    # BF16 training changes the weights; only post-update FP32 generation is compared.
    torch.testing.assert_close(xla_scores, cpu_scores, rtol=2e-3, atol=2e-6)
    agreement = float((xla_ids == cpu_ids).all(-1).float().mean())
    if agreement < .9:
        raise FloatingPointError(f"TPU/CPU post-update beam agreement is only {agreement}")
    report = {"passed": True, "quality_not_evaluated": True, "bf16_updates": 3,
        "bf16_losses": losses, "fp32_loss_max_difference": float((actual.cpu() - reference).abs().max()),
        "cpu_tpu_beam_agreement": agreement, "source_vs_static_cpu_beams_equal": True,
        "generator_parameters": sum(p.numel() for p in model.parameters())}
    _write(output / "tpu-compatibility.json", report)
    del optimizer, model, state, native, actual
    gc.collect()
    xm.mark_step()
    return report


def encode_catalogue(config, texts, output, device, xm):
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, T5Config, T5EncoderModel
    from rhpr.gr.resources import verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    spec = config["representation"]
    folder = resolve_dataset_directory(spec["path"])
    verify_resources(folder, spec["manifest_sha256"])
    provenance = json.loads((folder / "encoder-provenance.json").read_text())
    if provenance["revision"] != spec["revision"] or provenance["model"] != "google/flan-t5-xxl":
        raise ValueError("Wrong Flan XXL encoder provenance")
    started = time.monotonic()
    with torch.device("meta"):
        model = T5EncoderModel(T5Config.from_pretrained(folder, local_files_only=True))
    with safe_open(folder / "model.safetensors", framework="pt") as source:
        state = {name: source.get_tensor(name) for name in source.keys()}
    if "encoder.embed_tokens.weight" not in state:
        state["encoder.embed_tokens.weight"] = state["shared.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval().to(device)
    del state
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    size = spec["batch_size"]
    results = []
    with torch.no_grad():
        for begin in range(0, len(texts), size):
            chosen = texts[begin:begin + size]
            count = len(chosen)
            chosen += [chosen[-1]] * (size - count)
            tokens = tokenizer(chosen, max_length=spec["max_length"], padding="max_length",
                               truncation=True, return_tensors="pt")
            tokens = {name: value.to(device) for name, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            xm.mark_step()
            results.append(pooled.cpu().numpy()[:count])
            if begin % (size * 100) == 0:
                _event("genpas_xxl_encoding", items=min(begin + size, len(texts)), total=len(texts))
    values = np.concatenate(results)
    if values.shape != (len(texts), 4096) or not np.isfinite(values).all():
        raise FloatingPointError("Invalid Flan XXL catalogue embeddings")
    np.save(output / "item-embeddings.npy", values)
    _write(output / "representation.json", {**spec, "dtype_weights": "bf16", "pooled_dtype": "fp32",
        "shape": list(values.shape), "seconds": time.monotonic() - started,
        "sha256": file_sha256(output / "item-embeddings.npy"), "source": provenance["model_sha256"]})
    del model, tokenizer, hidden, pooled, results, tokens
    gc.collect()
    xm.mark_step()
    return values


def fit_rkmeans(values, spec, output, seed):
    """Declared CPU RK-Means: normalized residuals, frozen integer catalogue IDs.

    GenPAS ships no quantizer training pipeline or SID weights. Use a standard
    reproducible Lloyd implementation, rather than mislabeling old XL SIDs as XXL.
    """
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    from rhpr.gr.sid import SidIndex
    started = time.monotonic()
    residual = np.asarray(values, dtype=np.float32).copy()
    codes, centers, reports = [], [], []
    with threadpool_limits(limits=spec["cpu_threads"]):
        for level in range(spec["levels"]):
            residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
            model = KMeans(n_clusters=spec["codebook_size"], init="k-means++", n_init=spec["n_init"],
                max_iter=spec["max_iter"], random_state=seed + level, algorithm="lloyd")
            assignment = model.fit_predict(residual)
            residual -= model.cluster_centers_[assignment]
            codes.append(assignment)
            centers.append(model.cluster_centers_)
            reports.append({"level": level, "iterations": model.n_iter_, "inertia": model.inertia_})
            _event("genpas_rkmeans_level", **reports[-1])
    index = SidIndex(np.stack(codes, 1), width=spec["codebook_size"])
    np.save(output / "semantic-ids.npy", index.sids)
    np.save(output / "quantizer-centroids.npy", np.stack(centers))
    _write(output / "quantizer.json", {**spec, "backend": "sklearn.cluster.KMeans", "sklearn": sklearn.__version__,
        "normalization": "incoming residual L2 normalization at every level", "seed": seed,
        "seconds": time.monotonic() - started, "levels_report": reports,
        "author_genpas_tokenizer_reproduced": False})
    _write(output / "sid-diagnostics.json", index.diagnostics())
    return index


def make_rows(users, sids):
    import numpy as np
    import torch
    result = {stage: [] for stage in ("train", "validation", "test")}
    for user in users:
        sequences = {"train": user.train, "validation": (*user.train[-20:], user.validation),
                     "test": (*(*user.train, user.validation)[-20:], user.test)}
        for stage, history in sequences.items():
            result[stage].append({"sequence_data": torch.tensor(sids[np.asarray(history)].reshape(-1)),
                                  "user_id": torch.tensor([user.user_id])})
    return result


def evaluate(model, rows, index, batch_size, collate, labels, output, stage, device, xm):
    import numpy as np
    import torch
    from rhpr.gr.genpas_xla import kwargs_from_collation, padded_batch
    from rhpr.gr.sid import evaluate_ranked_sids
    predictions, probabilities, targets = [], [], []
    started = time.monotonic()
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            batch = kwargs_from_collation(collate(rows[begin:begin + batch_size], labels=labels,
                sequence_length=100, masking_token=-1, padding_token=-1, oov_token=None))
            count = len(batch["future_ids"])
            # Complete SIDs are unique; the item metric is equivalent to source SID matching.
            targets.extend(index.sid_to_item[tuple(row.tolist())] for row in batch["future_ids"])
            batch, _ = padded_batch(batch, batch_size)
            batch = {name: value.to(device) for name, value in batch.items()}
            ids, scores = model.generate(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])
            xm.mark_step()
            predictions.append(ids.cpu().numpy()[:count])
            probabilities.append(scores.cpu().numpy()[:count])
    predictions, probabilities, targets = np.concatenate(predictions), np.concatenate(probabilities), np.asarray(targets)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite TPU beam probabilities")
    scores = evaluate_ranked_sids(index, predictions, targets, cutoffs=(5, 10))
    scores["seconds"] = time.monotonic() - started
    if stage == "test":
        np.savez_compressed(output / "test-predictions.npz", semantic_ids=predictions,
                            scores=probabilities, target_item_ids=targets)
    return scores


def train(model, cfg, rows, index, config, output, device, xm):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from types import SimpleNamespace
    from src.data.loading.components.collate_functions import collate_fn_train
    from src.data.loading.components.label_function import NextKTokenMasking
    from rhpr.gr.genpas_xla import adapt_beams, epoch_batches, padded_batch, per_example_loss
    spec = config["training"]
    labels = {"sequence_data": SimpleNamespace(transform=NextKTokenMasking(next_k=4))}
    adapt_beams(model).to(device)
    optimizer = instantiate(cfg.optim.optimizer)(model.parameters())
    scheduler = instantiate(cfg.optim.scheduler)(optimizer=optimizer)
    started, examples, updates, best, stale, curves = time.monotonic(), 0, 0, -1., 0, []
    reason, selected_epoch = "epoch_limit", 0
    for epoch in range(1, spec["max_epochs"] + 1):
        epoch_started, epoch_examples, losses = time.monotonic(), 0, []
        model.train()
        stream = iter(epoch_batches(rows["train"], seed=config["seed"] + epoch - 1,
            source_size=spec["source_user_batch_size"], physical_size=spec["physical_batch_size"],
            collate=collate_fn_train, labels=labels))
        while group := list(islice(stream, spec["accumulate_grad_batches"])):
            total = sum(len(batch["future_ids"]) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = None
            for cpu_batch in group:
                count = len(cpu_batch["future_ids"])
                batch, valid = padded_batch(cpu_batch, spec["physical_batch_size"])
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                weights = (valid / total).to(device)
                with torch.autocast("xla", dtype=torch.bfloat16):
                    loss = (per_example_loss(model, batch) * weights).sum()
                loss.backward()
                loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
                xm.mark_step()
                epoch_examples += count
            xm.optimizer_step(optimizer, barrier=True)
            value = float(loss_sum.cpu())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite TPU training loss")
            losses.append(value)
            updates += 1
        scheduler.step()  # Native GenPAS configure_optimizers interval is epoch.
        examples += epoch_examples
        row = {"epoch": epoch, "optimizer_updates": updates, "epoch_examples": epoch_examples,
            "model_examples_consumed": examples, "training_loss": sum(losses) / len(losses),
            "epoch_training_seconds": time.monotonic() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"]}
        if epoch % spec["validation_interval_epochs"] == 0:
            scores = evaluate(model, rows["validation"], index, spec["evaluation_batch_size"],
                              collate_fn_train, labels, output, "validation", device, xm)
            row["validation"] = scores
            if scores["recall@5"] > best:
                best, stale, selected_epoch = scores["recall@5"], 0, epoch
                xm.save({"model": model.state_dict(), "epoch": epoch, "update": updates,
                    "validation": scores}, output / "generator-best.pt")
            else:
                stale += 1
            xm.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "update": updates,
                "model_examples_consumed": examples, "automatic_resume": False}, output / "generator-last.pt")
        curves.append(row)
        _write(output / "learning-curves.json", curves)
        _event("genpas_tpu_epoch", **row)
        if "validation" in row and (stale >= spec["patience_checks"] or time.monotonic() - started >= spec["max_seconds"]):
            reason = "early_stopping" if stale >= spec["patience_checks"] else "time_budget"
            break
    selected = torch.load(output / "generator-best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    xm.mark_step()
    test = evaluate(model, rows["test"], index, spec["evaluation_batch_size"],
                    collate_fn_train, labels, output, "test", device, xm)
    selection = {"completed_epochs": epoch, "optimizer_updates": updates, "selected_epoch": selected_epoch,
        "model_examples_consumed": examples, "validation_checks": sum("validation" in row for row in curves),
        "stop_reason": reason, "selected_validation": selected["validation"],
        "test_evaluations": 1, "elapsed_seconds": time.monotonic() - started}
    _write(output / "training-selection.json", selection)
    _write(output / "test-metrics.json", test)
    return {**{k: v for k, v in selection.items() if k != "selected_validation"},
            **{"test_" + k.replace("@", ""): v for k, v in test.items()}}


def run_genpas_tpu(config, output):
    report = {"stage": "bootstrap", "quality_not_evaluated": True}
    started = time.monotonic()
    try:
        resources, source, device, xm = bootstrap(config, output)
        report["stage"] = "native_generator_probe"
        _write(output / "pipeline-status.json", report)
        compatibility = probe(source, output, device, xm)
        if config.get("mode") == "probe":
            return {"experiment": config["name"], "actual_tpu_probe_passed": True,
                    "quality_not_evaluated": True, "elapsed_seconds": time.monotonic() - started}
        from rhpr.gr.data import load_beauty
        texts, users, original_ids, counts = load_beauty(resources / "data", config["data"])
        _write(output / "input-provenance.json", {"counts": counts, "original_item_ids": original_ids,
            "resources_manifest": config["resources"]["manifest_sha256"],
            "split": "author full training; original last20-context validation/test"})
        report["stage"] = "flan_xxl_encoding"
        _write(output / "pipeline-status.json", report)
        values = encode_catalogue(config, texts, output, device, xm)
        report["stage"] = "cpu_rkmeans"
        _write(output / "pipeline-status.json", report)
        index = fit_rkmeans(values, config["quantizer"], output, config["seed"])
        del values
        report["stage"] = "generator_training"
        _write(output / "pipeline-status.json", report)
        import torch
        torch.manual_seed(config["seed"])
        xm.set_rng_state(config["seed"], device=device)
        model, cfg = build_model(index.sids, source, output)
        rows = make_rows(users, index.sids)
        result = train(model, cfg, rows, index, config, output, device, xm)
        _write(output / "pipeline-status.json", {"stage": "complete", "quality_not_evaluated": False})
        return {"experiment": config["name"], "seed": config["seed"], "actual_tpu_probe_passed": True,
                "exact_published_reproduction": False, **result}
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        _write(output / "pipeline-status.json", report)
        raise
