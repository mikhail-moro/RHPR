"""A common-recipe compression pilot; no HPO or test-based selection."""
from __future__ import annotations

import copy
import gc
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def validate_pilot_config(config: dict[str, Any]) -> None:
    base = config["base"]
    variants = config["variants"]
    if len(variants) != 4 or len({v["name"] for v in variants}) != 4:
        raise ValueError("pilot requires four distinctly named variants")
    settings = {(v["compressor_type"], v["use_time_features"]) for v in variants}
    if settings != {(c, t) for c in ("identity", "pair_mlp") for t in (False, True)}:
        raise ValueError("pilot must cover Full/PairMLP x Time OFF/ON")
    if any(type(v["use_time_features"]) is not bool for v in variants):
        raise ValueError("use_time_features must be boolean")
    tail = int(base["model"]["tail_size"])
    if tail < 1 or int(base["loss"]["tail_size"]) != tail:
        raise ValueError("all variants require the same protected tail and loss tail")
    if base["loss"]["name"] != "sampled_softmax" or int(base["loss"]["num_negatives"]) < 1:
        raise ValueError("pilot requires common sampled softmax")
    if base["evaluation"].get("evaluate_test", True):
        raise ValueError("pilot uses validation; reserve test for final frozen decisions")
    training = base["training"]
    if int(training["effective_batch_size"]) % int(training["physical_batch_size"]):
        raise ValueError("effective batch must be divisible by physical batch")
    if float(config.get("training_budget_seconds", 19800)) <= 0:
        raise ValueError("training budget must be positive")


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_compression_pilot(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import polars as pl
    import torch

    from rhpr.metrics import validate_metrics
    from rhpr.sasrec.data import load_yambda_sequences, resolve_yambda_parquet_path
    from rhpr.sasrec.diagnostics import analyze_cohorts, extract_history_features, write_diagnostic_rows
    from rhpr.sasrec.experiment import run_yambda_experiment

    validate_pilot_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = int(config.get("seed", 42))
    base = copy.deepcopy(config["base"])
    base["seed"] = seed
    base["kind"] = "yambda_sasrec"
    data = base["data"]
    dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
                                   timestamp_unit=data.get("timestamp_unit", "seconds"))
    if any(split.timestamps is None for split in dataset.splits):
        raise ValueError("Time ON/OFF pilot requires observed timestamps")
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    reference_path: Path | None = None

    def persist(complete: bool) -> dict[str, Any]:
        metrics: dict[str, Any] = {"experiment": config["name"], "seed": seed,
            "pilot_complete": complete, "completed_variants": len(records),
            "planned_variants": len(config["variants"]), "pilot_seconds": time.time() - started,
            "test_evaluated": False}
        for record in records:
            for key, value in record["metrics"].items():
                metrics[f"{record['name']}_{key}"] = value
        _json(output_dir / "pilot-results.json", records)
        _json(output_dir / "metrics.json", validate_metrics(metrics))
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "pilot_complete": complete})
        (output_dir / "summary.md").write_text(
            f"# Compression pilot\n\nCompleted {len(records)}/4 variants; test reserved.\n", encoding="utf-8")
        return metrics

    persist(False)
    for variant in config["variants"]:
        candidate = copy.deepcopy(base)
        candidate["name"] = variant["name"]
        candidate["model"].update({"compressor_type": variant["compressor_type"],
                                  "use_time_features": variant["use_time_features"]})
        candidate["training"]["keep_checkpoint"] = True
        candidate["training"]["deadline_unix_seconds"] = started + float(config.get("training_budget_seconds", 19800))
        candidate["diagnostics"] = {"enabled": True}
        directory = output_dir / "runs" / variant["name"]
        directory.mkdir(parents=True, exist_ok=True)
        _json(directory / "config.json", candidate)
        print(json.dumps({"phase": "pilot_variant_started", "variant": variant["name"]}), flush=True)
        try:
            metrics = validate_metrics(run_yambda_experiment(candidate, directory, dataset=dataset))
            _json(directory / "metrics.json", metrics)
            records.append({"name": variant["name"], "config": candidate, "metrics": metrics,
                            "checkpoint": str(directory / "best-model.pt")})
            predictions.extend(pl.read_parquet(directory / "predictions.parquet").to_dicts())
            if variant["compressor_type"] == "identity" and not variant["use_time_features"]:
                reference_path = directory / "best-model.pt"
            persist(False)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if reference_path is None:
        raise RuntimeError("missing Full/OFF reference checkpoint")
    print(json.dumps({"phase": "pilot_diagnostics_started"}), flush=True)
    state = torch.load(reference_path, map_location="cpu", weights_only=True)
    embeddings = state["item_embedding.weight"].numpy()
    features = extract_history_features(dataset.splits, embeddings,
        max_seq_len=int(base["model"]["max_seq_len"]), tail_size=int(base["model"]["tail_size"]),
        split_name="validation", num_gap_buckets=int(base["model"].get("num_gap_buckets", 33)) - 1,
        time_unit=float(data.get("gap_time_unit", 1.0)))
    del state, embeddings
    # These answer-dependent fields are offline diagnostics, never router inputs.
    train_frequencies = np.bincount(np.concatenate([s.train for s in dataset.splits]),
                                  minlength=dataset.num_items + 1)
    by_user = {s.user_id: s for s in dataset.splits}
    for row in predictions:
        context = by_user[row["user_id"]].train[-int(base["model"]["max_seq_len"]):]
        tail = min(len(context), int(base["model"]["tail_size"]))
        target = row["target"]
        row["target_seen_in_tail"] = bool(target in context[-tail:])
        row["target_seen_only_in_old"] = bool(target in context[:-tail] and not row["target_seen_in_tail"])
        row["target_train_frequency"] = int(train_frequencies[target])
    write_diagnostic_rows(output_dir, features=features, predictions=predictions)
    reference_record = next(r for r in records if r["checkpoint"] == str(reference_path))
    _json(output_dir / "diagnostic-reference.json", {"checkpoint": str(reference_path.relative_to(output_dir)),
        "git_sha": os.environ.get("GITHUB_SHA", "local"), "seed": seed,
        "config": reference_record["config"], "basis": "item embeddings before PE/time features",
        "target_fields": "offline-only; excluded from history-features"})
    analysis = analyze_cohorts(features, predictions, output_dir,
                    bootstrap_seed=seed, bootstrap_samples=int(config.get("bootstrap_samples", 1000)))
    metrics = persist(True)
    lines = ["# Compression pilot", "", "Common starting recipe; validation only, one seed.", "",
             "| Variant | NDCG@10 | Recall@10 | History p50 ms | Full p50 ms | B32 users/s | Best epoch |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for record in records:
        m = record["metrics"]
        values = [record["name"], f"{m['validation_ndcg_at_10']:.6f}", f"{m['validation_recall_at_10']:.6f}"]
        values.extend(f"{m[key]:.3f}" if key in m else "disabled" for key in (
            "inference_history_b1_p50_ms", "inference_full_b1_p50_ms", "inference_full_b32_users_per_second"))
        values.append(str(m["best_epoch"]))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Paired PairMLP minus Full differences (user bootstrap; fixed checkpoints):", ""])
    for cohort in analysis["cohorts"]:
        if cohort["cohort_type"] == "all":
            lines.append(f"- Time {'ON' if cohort['time_enabled'] else 'OFF'}: "
                f"delta NDCG={cohort['delta_ndcg_at_10']:.6f}; "
                f"95% interval={cohort['delta_ndcg_at_10_ci95']}; n={cohort['user_count']}.")
    if any(r["metrics"]["best_at_epoch_limit"] for r in records):
        lines.extend(["", "Some best checkpoints reached the epoch limit; convergence is unresolved."])
    lines.extend(["", "See cohorts.json/Parquet and per-run benchmarks. Matmul FLOPs are nominal dense counts; "
                  "wall time determines speedup. This pilot does not establish adaptive-depth benefit."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
