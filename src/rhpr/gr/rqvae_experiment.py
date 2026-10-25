"""Fast, single-seed Sentence-T5 / native GRID RQ-VAE / GRID TIGER run."""
from __future__ import annotations

import os
import json
from pathlib import Path
import random
import time

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.experiment import _encode_items, _evaluate, _event, _write


def _train_rqvae_generator(model, users, index, config, output, device):
    """Select a declared sampler; retain the previous GRID stream for old runs."""
    sampling = config["training"].get("sampling", "grid_contiguous_windows")
    if sampling == "grid_contiguous_windows":
        from rhpr.gr.epoch_training import train_epoch_generator
        return train_epoch_generator(model, users, index, config, output, device)
    if sampling != "all_train_prefixes":
        raise ValueError(f"unsupported RQ-VAE generator sampling: {sampling}")
    from rhpr.gr.experiment import _train_generator

    values = _train_generator(model, users, index, config, output, device)
    saved = json.loads((output / "training-selection.json").read_text())
    definition = "every noninitial training position exactly once per epoch"
    _write(output / "training-stream.json", {
        "sampling": sampling, "epoch_definition": definition,
        "examples_per_epoch": values["training_examples_per_epoch"],
        "history_items": config["data"]["history_items"],
        "order": "numpy.default_rng(seed + epoch).permutation; epoch starts at 1",
        "drop_last": False,
    })
    result = {**saved, "epochs": values["training_epochs"],
        "updates": values["optimizer_updates"], "selected_epoch": values["best_epoch"],
        "training_seconds": values["optimizer_training_seconds"],
        "elapsed_training_loop_seconds": values["training_seconds"],
        "stop_reason": values["stopping_reason"], "checkpoint_metric": "validation_item_ndcg@10",
        "model_examples_consumed": values["training_examples_per_epoch"] * values["training_epochs"],
        "sampling": sampling, "epoch_definition": definition}
    _write(output / "training-selection.json", result)
    return result


def run_rqvae_baseline(config, output):
    import numpy as np
    import torch
    from rhpr.gr.grid import build_grid_generator, prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources, verify_parent_artifacts
    from rhpr.gr.rqvae import fit_grid_rqvae
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory

    started = time.monotonic()
    seed, device = config["seed"], config["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU RQ-VAE experiment requires CUDA; no CPU fallback")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    resources = resolve_dataset_directory(config["resources"]["path"])
    manifest = verify_resources(resources, config["resources"]["manifest_sha256"])
    scratch = Path(config["resources"]["scratch_path"])
    original = unpack_grid(resources / "grid-source.bin", scratch / "original")
    grid = prepare_grid_source(original, scratch / "runtime")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    texts, users, source_ids, counts = load_beauty(resources / "data", config["data"])
    _write(output / "input-provenance.json", {"manifest": manifest, "counts": counts,
        "data": config["data"], "telemetry": "disabled; no study tables, traces, exact panels or probes"})
    parent = verify_parent_artifacts(config["parent"])
    embeddings, representation = _encode_items(config, resources, texts, output, device, parent)
    codes, quantizer = fit_grid_rqvae(embeddings, config["quantizer"], seed, device,
        grid, original / "configs/experiment/rqvae_train_flat.yaml", output)
    index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
    np.save(output / "semantic-ids.npy", index.sids, allow_pickle=False)
    _write(output / "sid-diagnostics.json", index.diagnostics())
    _write(output / "sid-catalogue.json", {"original_item_ids": source_ids,
        "row_order": "internal_item_id_0_based", "semantic_ids_sha256": file_sha256(output / "semantic-ids.npy")})
    del embeddings
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    model = build_grid_generator(index.sids, config["generator"], seed, device, grid)
    parameters = sum(p.numel() for p in model.parameters())
    _write(output / "generator.json", {"parameters": parameters, "config": config["generator"],
        "provenance": model.rhpr_grid_provenance})
    _event("rqvae_generator_start", parameters=parameters, epochs=config["training"]["max_epochs"])
    selection = _train_rqvae_generator(model, users, index, config, output, device)
    test = build_examples(users, config["data"]["history_items"], "test")
    scores, ids, probabilities = _evaluate(model, test, index,
        config["training"]["evaluation_batch_size"], device,
        precision=config["training"]["evaluation_precision"])
    _write(output / "test-metrics.json", scores)
    np.savez_compressed(output / "test-predictions.npz", semantic_ids=ids, scores=probabilities,
        user_ids=test[1], target_item_ids=test[2])
    return {"experiment": config["name"], "seed": seed, "generator_parameters": parameters,
        "catalogue_items": counts["items"], "users": counts["users"],
        "embedding_seconds": representation["seconds"], "quantizer_seconds": quantizer["elapsed_seconds"],
        "quantizer_steps": quantizer["steps"], "epochs": selection["epochs"],
        "selected_epoch": selection["selected_epoch"], "updates": selection["updates"],
        "training_seconds": selection["training_seconds"], "validation_seconds": selection["validation_seconds"],
        "validation_checks": selection["validation_checks"], "stop_reason": selection["stop_reason"],
        "test_evaluations": 1, "telemetry_enabled": False, "exact_published_reproduction": False,
        "elapsed_seconds": time.monotonic() - started,
        **{"test_" + key.replace("@", ""): value for key, value in scores.items()}}
