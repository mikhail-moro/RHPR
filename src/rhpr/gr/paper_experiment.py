"""One-seed paper-priority GRID quality run with bounded GPU time."""
from __future__ import annotations

import os
from pathlib import Path
import random
import time

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.experiment import _evaluate, _event, _write


def encode_flan_catalogue(config, texts, output, device):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from rhpr.gr.paper_profile import _load_encoder, masked_mean
    from rhpr.gr.profiling import _clear, _synchronize
    from rhpr.gr.resources import verify_resources
    from rhpr.gr.sid import validate_item_embeddings
    from rhpr.offline_runtime import resolve_dataset_directory

    source = config["representation"]
    directory = resolve_dataset_directory(source["path"])
    verify_resources(directory, source["manifest_sha256"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = _load_encoder(directory, device)
    size = source["batch_size"]
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(texts), size):
            batch = tokenizer(texts[begin:begin + size], max_length=source["max_length"],
                padding="max_length", truncation=True, add_special_tokens=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded = masked_mean(model(**batch).last_hidden_state, batch["attention_mask"])
            if not bool(torch.isfinite(encoded).all()):
                raise FloatingPointError("nonfinite FP32 Flan embeddings")
            chunks.append(encoded.cpu().numpy())
            if begin % (100 * size) == 0:
                _event("grid_catalogue_encoding", encoded_items=min(begin + size, len(texts)))
    values = validate_item_embeddings(np.concatenate(chunks), num_items=len(texts))
    np.save(output / "item-embeddings.npy", values, allow_pickle=False)
    _synchronize(device)
    report = {**source, "precision": "fp32", "shape": list(values.shape),
        "seconds": time.perf_counter() - started,
        "sha256": file_sha256(output / "item-embeddings.npy")}
    _write(output / "representation.json", report)
    del model, tokenizer, chunks
    _clear(device)
    return values, report


def train_paper_generator(model, users, sid_index, config, output, device, *,
                          max_seconds=None, evaluator=_evaluate, clock=time.monotonic):
    import torch
    from rhpr.gr.paper_profile import _update
    from rhpr.gr.paper_protocol import PaperBatchStream

    spec = config["training"]
    precision = spec["precision"]
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"],
                                 weight_decay=spec["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024)
    stream = PaperBatchStream(users, sid_index.sids, seed=config["seed"],
        source_batch_size=spec["source_user_batch_size"], effective_batch_size=spec["batch_size"],
        sequence_length=spec["sequence_length"], max_augmented_batch=spec["max_augmented_batch"])
    validation = build_examples(users, config["data"]["history_items"], "validation")
    best_recall, best_ndcg, stale = -1.0, -1.0, 0
    selected_update, updates, skipped = 0, 0, 0
    curves, interval_losses = [], []
    started, train_seconds, validation_seconds = clock(), 0.0, 0.0
    limit = spec["max_seconds"] if max_seconds is None else max_seconds
    reason = "update_limit"

    def validate():
        nonlocal best_recall, best_ndcg, stale, selected_update, validation_seconds
        before = clock()
        scores, _, _ = evaluator(model, validation, sid_index, spec["evaluation_batch_size"],
                                  device, precision=spec["evaluation_precision"])
        validation_seconds += clock() - before
        recall, ndcg = scores["recall@10"], scores["ndcg@10"]
        row = {"update": updates, "validation": scores,
            "train_loss_sum_over_codes": sum(interval_losses) / len(interval_losses),
            "training_seconds": train_seconds, "elapsed_seconds": clock() - started,
            "amp_skipped_updates": skipped, "model_examples_consumed": stream.report()["model_examples_yielded"]}
        if recall > best_recall:
            best_recall, selected_update = recall, updates
            torch.save({"model": model.state_dict(), "update": updates, "validation": scores,
                "criterion": "validation_recall@10"}, output / "generator-best.pt")
        if ndcg > best_ndcg:
            best_ndcg, stale = ndcg, 0
        else:
            stale += 1
        row.update(best_recall=best_recall, best_ndcg=best_ndcg, stale_checks=stale)
        curves.append(row)
        _write(output / "learning-curves.json", curves)
        _write(output / "training-stream.json", stream.report())
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "update": updates, "validation": scores,
            "rng_cpu": torch.get_rng_state(), "stream_report": stream.report(),
            "automatic_resume": False}, output / "generator-last.pt")
        _event("grid_paper_validation", **row)
        interval_losses.clear()

    while updates < spec["max_updates"]:
        if updates and clock() - started >= limit:
            reason = "time_budget"
            break
        model.train()
        before = clock()
        batch = stream.next_batch()
        result = _update(model, batch, spec["physical_batch_size"], optimizer, scaler,
                         precision, device, spec["grad_clip_norm"])
        train_seconds += clock() - before
        if not result["optimizer_update_applied"]:
            skipped += 1
            if skipped > 16:
                raise FloatingPointError("repeated GRID gradient overflows")
            continue
        updates += 1
        interval_losses.append(result["loss"])
        if updates % spec["validation_interval_updates"] == 0:
            validate()
            if stale >= spec["patience_checks"]:
                reason = "early_stopping"
                break
    # A bounded run still selects only by validation; test is never consulted.
    if interval_losses:
        validate()
    checkpoint = torch.load(output / "generator-best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    report = {"updates": updates, "selected_update": selected_update,
        "checkpoint_metric": "validation_recall@10", "early_stopping_metric": "validation_ndcg@10",
        "best_validation_recall10": best_recall, "best_validation_ndcg10": best_ndcg,
        "selected_validation": checkpoint["validation"], "stop_reason": reason,
        "early_stopping_reached": reason == "early_stopping", "validation_checks": len(curves),
        "training_seconds": train_seconds, "validation_seconds": validation_seconds,
        "elapsed_seconds": clock() - started, "amp_skipped_updates": skipped,
        "model_examples_consumed": stream.report()["model_examples_yielded"],
        "training_stream": stream.report(), "precision": precision}
    _write(output / "training-selection.json", report)
    return report


def run_grid_paper_quality(config, output):
    import numpy as np
    import torch
    from rhpr.gr.distances import make_grid_distance
    from rhpr.gr.grid import build_grid_generator, load_grid, prepare_grid_source
    from rhpr.gr.paper_fit import fit_paper_rkmeans
    from rhpr.gr.paper_profile import (PaperProfileBudget, benchmark_paper_training,
                                       runtime_sid_index)
    from rhpr.gr.paper_protocol import PaperBatchStream
    from rhpr.gr.paper_quantizer_profile import benchmark_logical_rank_quantizer
    from rhpr.gr.profiling import _clear
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory

    started = time.monotonic()
    seed, device = config["seed"], config["training"]["device"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("registered GRID quality run requires CUDA")
    resources = resolve_dataset_directory(config["resources"]["path"])
    manifest = verify_resources(resources, config["resources"]["manifest_sha256"])
    scratch = Path(config["resources"]["scratch_path"])
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_HOME=str(scratch / "huggingface"))
    grid = prepare_grid_source(unpack_grid(resources / "grid-source.bin", scratch / "original"),
                               scratch / "runtime")
    texts, users, original_ids, counts = load_beauty(resources / "data", config["data"])
    _write(output / "input-provenance.json", {"manifest": manifest, "counts": counts, "data": config["data"]})
    # Test the larger AMP batch before spending time on catalogue encoding.
    proxy_index = runtime_sid_index(counts["items"], config["generator"]["codebook_size"], seed)
    proxy_model = build_grid_generator(proxy_index.sids, config["generator"], seed=seed,
                                      device=device, source_root=grid)
    spec = config["training"]

    def stream_factory(effective):
        return PaperBatchStream(users, proxy_index.sids, seed=seed,
            source_batch_size=spec["source_user_batch_size"], effective_batch_size=effective,
            sequence_length=spec["sequence_length"], max_augmented_batch=spec["max_augmented_batch"])

    preflight = benchmark_paper_training(proxy_model, stream_factory, spec["physical_batch_size"],
        spec["precision"], "full_pipeline", spec, {"warmup": 2, "iterations": 5, "seed": seed},
        PaperProfileBudget(180), device)
    preflight["quality_not_evaluated"] = True
    _write(output / "generator-runtime-preflight.json", preflight)
    _event("grid_generator_runtime_preflight", **preflight)
    if preflight["measured_updates_applied"] != preflight["measured_updates_attempted"]:
        raise FloatingPointError("GRID runtime preflight skipped optimizer updates")
    del proxy_model, proxy_index
    _clear(device)
    source_grid = load_grid(grid)
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device).index or 0]
                                              if device.startswith("cuda") else []):
        torch.manual_seed(seed)
        points = torch.nn.functional.normalize(
            torch.randn(256, config["quantizer"].get("embedding_dim", 2048), device=device), dim=-1)
        expected = source_grid.SquaredEuclideanDistance().compute(points, points[:128])
        actual = make_grid_distance(source_grid, config["quantizer"]).compute(points, points[:128])
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-6)
        if not torch.equal(actual.argmin(1), expected.argmin(1)):
            raise FloatingPointError("distance runtime preflight changed nearest-centroid assignments")
        distance_check = {"maximum_absolute_difference": float((actual - expected).abs().max()),
                          "assignments_equal": True, "device": device}
    _write(output / "distance-numerical-preflight.json", distance_check)
    del points, expected, actual
    # Same operator proxy as the completed source profile; no quality inference.
    timing = benchmark_logical_rank_quantizer(grid, config, config["distance_profile"],
                                             PaperProfileBudget(180), device)
    timing["upstream_reference_seconds_per_logical_update"] = 2.0870529487
    timing["speedup_vs_prior_T4_proxy"] = 2.0870529487 / timing["seconds_per_logical_update"]
    _write(output / "rk-distance-profile.json", timing)
    _event("grid_distance_profile_complete", **timing)
    embeddings, representation = encode_flan_catalogue(config, texts, output, device)
    fit_started = time.monotonic()

    def quantizer_progress(values):
        _event("grid_rk_fit", **values)
        if time.monotonic() - fit_started > config["quantizer"]["max_seconds"]:
            raise TimeoutError("GRID quantizer time budget exhausted")

    codes, centroids, quantizer = fit_paper_rkmeans(embeddings, config["quantizer"], seed=seed,
        device=device, source_root=grid, progress=quantizer_progress)
    index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
    np.save(output / "semantic-ids.npy", index.sids, allow_pickle=False)
    np.save(output / "quantizer-centroids.npy", centroids, allow_pickle=False)
    _write(output / "quantizer.json", quantizer)
    _write(output / "sid-diagnostics.json", index.diagnostics())
    _write(output / "sid-catalogue.json", {"original_item_ids": original_ids,
        "row_order": "internal_item_id_0_based", "semantic_ids_sha256": file_sha256(output / "semantic-ids.npy")})
    del embeddings
    _clear(device)
    model = build_grid_generator(index.sids, config["generator"], seed=seed, device=device, source_root=grid)
    parameters = sum(p.numel() for p in model.parameters())
    _write(output / "generator.json", {"parameters": parameters, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else "none",
        "provenance": model.rhpr_grid_provenance, "config": config["generator"]})
    remaining = config["max_run_seconds"] - (time.monotonic() - started) - 180
    if remaining <= 0:
        raise TimeoutError("no time remains for GRID training and final evaluation")
    selection = train_paper_generator(model, users, index, config, output, device,
        max_seconds=min(config["training"]["max_seconds"], remaining))
    # Exactly one held-out test pass, using the validation-Recall-selected state.
    test = build_examples(users, config["data"]["history_items"], "test")
    scores, ids, probabilities = _evaluate(model, test, index,
        config["training"]["evaluation_batch_size"], device, precision=config["training"]["evaluation_precision"])
    np.savez_compressed(output / "test-predictions.npz", semantic_ids=ids, scores=probabilities,
        user_ids=test[1], target_item_ids=test[2])
    _write(output / "test-metrics.json", scores)
    return {"experiment": config["name"], "seed": seed, "generator_parameters": parameters,
        "catalogue_items": counts["items"], "users": counts["users"],
        "embedding_seconds": representation["seconds"], "quantizer_seconds": quantizer["elapsed_seconds"],
        "training_seconds": selection["training_seconds"], "validation_seconds": selection["validation_seconds"],
        "updates": selection["updates"], "selected_update": selection["selected_update"],
        "stop_reason": selection["stop_reason"], "early_stopping_reached": selection["early_stopping_reached"],
        "best_validation_recall10": selection["best_validation_recall10"],
        "best_validation_ndcg10": selection["best_validation_ndcg10"],
        "rk_proxy_speedup_vs_prior_T4": timing["speedup_vs_prior_T4_proxy"],
        "test_evaluations": 1, "elapsed_seconds": time.monotonic() - started,
        "exact_published_reproduction": False,
        **{"test_" + key.replace("@", ""): value for key, value in scores.items()}}
