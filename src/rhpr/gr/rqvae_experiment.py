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


def _encode_titles(specification, resources, texts, output, device):
    import hashlib
    import numpy as np
    from rhpr.gr.study_index import extract_features

    titles = [next(value for name, value, status in extract_features(text) if name == "title") for text in texts]
    observed = np.array([bool(title and title.strip()) for title in titles])
    values = [title if valid else "" for title, valid in zip(titles, observed)]
    directory = output / "title-encoder"
    directory.mkdir(parents=True, exist_ok=True)
    embeddings, report = _encode_items({"representation": specification}, resources, values, directory, device)
    report.update(input_scope="title_only", observed_titles=int(observed.sum()),
        texts_sha256=hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest(), frozen=True)
    _write(directory / "representation.json", report)
    return embeddings, observed, report


def _quantizer_only_metrics(config, statistics, quantizer, representation, counts, elapsed):
    full = statistics["prefixes"][-1]
    result = {"experiment": config["name"], "seed": config["seed"], "stage": "quantizer_only",
        "catalogue_items": counts["items"], "quantizer_steps": quantizer["steps"],
        "quantizer_seconds": quantizer["elapsed_seconds"], "embedding_seconds": representation["seconds"],
        "elapsed_seconds": elapsed, "generator_trained": False, "test_evaluations": 0,
        "semantic_unique_ids": statistics["semantic_unique_ids"],
        "max_collision_group": statistics["max_collision_group"],
        "suffix_representable": statistics["suffix_representable"],
        "collision_groups": full["collision_groups"], "collided_items": full["collided_items"],
        "collision_excess_items": full["collision_excess_items"],
        "collision_excess_fraction": full["collision_excess_items"] / statistics["items"],
        "collided_item_fraction": full["collided_items"] / statistics["items"],
        "overflow_groups": full["groups_exceeding_suffix_width"],
        "items_in_overflow_groups": full["items_in_overflow_groups"]}
    result.update({f"used_codes_level_{level + 1}": count for level, count in enumerate(quantizer["used_codes_per_level"])})
    if "reconstruction_mse_normalized_input" in quantizer:
        result["embedding_reconstruction_mse"] = quantizer["reconstruction_mse_normalized_input"]
    for name, fields in quantizer.get("field_reconstruction", {}).items():
        if fields["loss"] is not None:
            result[f"reconstruction_{name}_loss"] = fields["loss"]
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
    stage = config.get("pipeline", {}).get("stage", "full")
    if stage not in {"full", "quantizer_only", "continuous_autoencoder"}:
        raise ValueError("unsupported pipeline.stage")
    if stage == "continuous_autoencoder" and config["representation"]["kind"] != "field_encoder":
        raise ValueError("continuous control requires field_encoder representation")
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
    if config["representation"]["kind"] == "field_encoder":
        from rhpr.gr.field_rqvae import fit_field_rqvae
        fitter = fit_field_rqvae
        if stage == "continuous_autoencoder":
            from rhpr.gr.field_autoencoder import fit_field_autoencoder
            fitter = fit_field_autoencoder
        title_arguments = {}
        if "title" in config["representation"]["fields"]:
            title_values, title_observed, title_report = _encode_titles(
                config["representation"]["title"], resources, texts, output, device)
            title_arguments = {"title_embeddings": title_values, "title_observed": title_observed}
        codes, quantizer, representation = fitter(texts, config["representation"],
            config["quantizer"], seed, device, grid,
            original / "configs/experiment/rqvae_train_flat.yaml", output, **title_arguments)
        if title_arguments:
            representation["seconds"] += title_report["seconds"]
            representation["title_encoder"] = title_report
            _write(output / "representation.json", representation)
            del title_values, title_arguments
    else:
        parent = verify_parent_artifacts(config["parent"])
        embeddings, representation = _encode_items(config, resources, texts, output, device, parent)
        codes, quantizer = fit_grid_rqvae(embeddings, config["quantizer"], seed, device,
            grid, original / "configs/experiment/rqvae_train_flat.yaml", output)
        del embeddings
    if stage == "continuous_autoencoder":
        _write(output / "autoencoder-catalogue.json", {"original_item_ids": source_ids,
            "row_order": "internal_item_id_0_based", "quantization_enabled": False})
        return {"experiment": config["name"], "seed": seed, "stage": stage,
            "autoencoder_steps": quantizer["steps"], "catalogue_items": counts["items"],
            "autoencoder_seconds": quantizer["elapsed_seconds"], "embedding_seconds": representation["seconds"],
            "quantization_enabled": False, "generator_trained": False, "test_evaluations": 0,
            "elapsed_seconds": time.monotonic() - started,
            **{f"reconstruction_{name}_loss": value["loss"] for name, value in quantizer["field_reconstruction"].items()
               if value["loss"] is not None}}
    if config["representation"]["kind"] == "field_encoder" or stage == "quantizer_only":
        statistics = quantizer["semantic_code_statistics"]
        status = {"stage": "sid_collision_resolution", "semantic_codes_exported": True,
            "status": "representable" if statistics["suffix_representable"] else "collision_overflow",
            "max_collision_group": statistics["max_collision_group"],
            "code_sha": os.environ.get("GITHUB_SHA", "local"), "seed": config["seed"],
            "suffix_width": config["quantizer"]["codebook_size"]}
        _write(output / "tokenizer-status.json", status)
        if stage == "quantizer_only":
            _write(output / "sid-catalogue.json", {"original_item_ids": source_ids,
                "row_order": "internal_item_id_0_based", "semantic_codes_sha256": file_sha256(output / "semantic-codes.npy")})
            return _quantizer_only_metrics(config, statistics, quantizer, representation, counts, time.monotonic() - started)
        if not statistics["suffix_representable"]:
            raise ValueError(f"semantic collision group has {statistics['max_collision_group']} items, "
                f"exceeding suffix width {config['quantizer']['codebook_size']}; "
                "see semantic-code-diagnostics.json and quantizer.pt")
    index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
    np.save(output / "semantic-ids.npy", index.sids, allow_pickle=False)
    _write(output / "sid-diagnostics.json", index.diagnostics())
    _write(output / "sid-catalogue.json", {"original_item_ids": source_ids,
        "row_order": "internal_item_id_0_based", "semantic_ids_sha256": file_sha256(output / "semantic-ids.npy")})
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
