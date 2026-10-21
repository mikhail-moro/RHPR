"""Freshly trained recent-tail baseline with the common raw-window protocol."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from rhpr.metrics import validate_metrics
from rhpr.sasrec.data import YambdaSequences, load_yambda_sequences, resolve_yambda_parquet_path, validation_example
from rhpr.sasrec.diagnostics import context_hash, extract_history_features, write_diagnostic_rows
from rhpr.sasrec.experiment import run_yambda_experiment


INITIALIZATION = "fresh_seed42_no_pretrained_weights"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_recent128_config(config: dict[str, Any]) -> None:
    """Require a fresh tail-only model and the common validation-only objective."""
    if config.get("kind") != "yambda_recent128":
        raise ValueError("kind must be yambda_recent128")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    if type(config.get("seed")) is not int or config["seed"] != 42:
        raise ValueError("Recent128 uses the common seed42")
    base = config.get("base")
    if not isinstance(base, dict) or base.get("kind") != "yambda_sasrec" or base.get("seed") != 42:
        raise ValueError("base must be yambda_sasrec with the common seed42")
    model, loss = base.get("model", {}), base.get("loss", {})
    if model.get("compressor_type") != "tail_only":
        raise ValueError("Recent128 requires a tail_only model")
    if model.get("use_time_features", False) is not False:
        raise ValueError("Recent128 uses Time OFF")
    if model.get("max_seq_len") != 1028:
        raise ValueError("preserve the common raw max_seq_len1028 and recency-position convention")
    if model.get("tail_size") != 128 or loss.get("tail_size") != 128:
        raise ValueError("model and loss must preserve the common raw tail128")
    if loss.get("name") != "sampled_softmax" or loss.get("num_negatives") != 512:
        raise ValueError("retain sampled_softmax with 512 catalogue negatives")
    evaluation = base.get("evaluation", {})
    if evaluation.get("evaluate_test") is not False or evaluation.get("k") != 10:
        raise ValueError("use full-catalog validation top10 and reserve test")
    if evaluation.get("max_validation_batches") is not None:
        raise ValueError("Recent128 evaluates all validation users")
    training = base.get("training", {})
    epochs = _positive_integer(training.get("max_epochs"), "max_epochs")
    interval = _positive_integer(training.get("validation_interval", 5), "validation_interval")
    if interval > epochs:
        raise ValueError("validation_interval cannot exceed max_epochs")
    patience = _positive_integer(training.get("early_stopping_patience"), "early_stopping_patience")
    # Stale counts advance only at validation. Even a tie at every scheduled
    # check cannot stop this run before its configured training budget ends.
    checks = (epochs + interval - 1) // interval
    if patience < checks:
        raise ValueError("early_stopping_patience must cover all scheduled validation checks")
    physical = _positive_integer(training.get("physical_batch_size"), "physical_batch_size")
    effective = _positive_integer(training.get("effective_batch_size"), "effective_batch_size")
    if effective % physical:
        raise ValueError("effective batch must be divisible by physical batch")
    if training.get("max_batches_per_epoch") is not None or training.get("mixed_depth") is not None:
        raise ValueError("Recent128 trains all users at one fixed recent-tail budget")
    budget = config.get("training_budget_seconds", 7200)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("training_budget_seconds must be finite and positive")


def run_recent128(config: dict[str, Any], output_dir: Path, *,
                  dataset: YambdaSequences | None = None) -> dict[str, int | float | str | bool]:
    """Train tail-only weights from scratch and retain full-context identities."""
    validate_recent128_config(config)
    import polars as pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = config["seed"]
    candidate = copy.deepcopy(config["base"])
    candidate["name"] = config["name"]
    candidate["source_initialization"] = INITIALIZATION
    candidate["training"].update(keep_checkpoint=True,
        deadline_unix_seconds=started + config.get("training_budget_seconds", 7200))
    candidate["diagnostics"] = {"enabled": True}
    _json(output_dir / "training-config.json", candidate)
    results: dict[str, int | float | str | bool] = {}

    def persist(complete: bool) -> dict[str, int | float | str | bool]:
        metrics = validate_metrics({**results, "experiment": config["name"], "seed": seed,
            "source_initialization": INITIALIZATION, "training_performed": True,
            "recent128_complete": complete, "test_evaluated": False,
            "recent128_seconds": time.time() - started})
        _json(output_dir / "metrics.json", metrics)
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "source_initialization": INITIALIZATION,
            "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "recent128_complete": complete})
        (output_dir / "summary.md").write_text(
            f"# Recent128\n\nComplete: {complete}; fresh training; raw tail128; test reserved.\n",
            encoding="utf-8")
        if not complete:
            (output_dir / "comparison.md").write_text(
                "# Recent128\n\nIncomplete; do not interpret partial results.\n", encoding="utf-8")
        return metrics

    persist(False)
    if dataset is None:
        data = candidate["data"]
        dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
            timestamp_unit=data.get("timestamp_unit", "seconds"))
    results.update(validate_metrics(run_yambda_experiment(candidate, output_dir, dataset=dataset)))
    persist(False)
    if (results.get("epochs_trained") != candidate["training"]["max_epochs"]
            or results.get("test_evaluated") is not False
            or results.get("validation_examples") != len(dataset.splits)):
        raise RuntimeError("Recent128 must complete its training budget and evaluate all validation users only")
    checkpoint = output_dir / "best-model.pt"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if results.get("checkpoint_sha256") != digest.hexdigest():
        raise RuntimeError("Recent128 selected checkpoint checksum mismatch")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"phase": "recent128_diagnostics_started"}), flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(key.startswith("compressor.") for key in state):
        raise RuntimeError("Recent128 must not contain trainable compression summaries")
    features = extract_history_features(dataset.splits, state["item_embedding.weight"].numpy(),
        max_seq_len=candidate["model"]["max_seq_len"], tail_size=128, split_name="validation",
        num_gap_buckets=candidate["model"].get("num_gap_buckets", 33) - 1,
        time_unit=candidate["data"].get("gap_time_unit", 1.0))
    del state
    for row in features:
        row["output_length_recent128"] = row["tail_length"]
    predictions = pl.read_parquet(output_dir / "predictions.parquet").to_dicts()
    expected = {}
    for split in dataset.splits:
        items, target = validation_example(split, candidate["model"]["max_seq_len"])
        if split.user_id in expected:
            raise RuntimeError("Recent128 requires unique validation user IDs")
        expected[split.user_id] = items, target
    seen = set()
    for row in predictions:
        user = row["user_id"]
        if user not in expected or user in seen:
            raise RuntimeError("Recent128 prediction users must match the full validation set exactly")
        seen.add(user)
        items, target = expected[user]
        full_hash = context_hash(items)
        if (row.get("model") != "recent128" or row.get("split") != "validation"
                or row.get("seed") != seed or row.get("time_enabled") is not False
                or row.get("context_hash") != full_hash or row.get("target") != target):
            raise RuntimeError("Recent128 predictions must preserve the common full context and target")
        tail = items[-128:]
        old = items[:-128]
        seen_tail, seen_old = bool(target in tail), bool(target in old)
        row.update({"reference_context_hash": full_hash, "reference_window_length": len(items),
            "input_context_hash": context_hash(tail), "input_length": len(tail),
            "target_seen_tail": seen_tail, "target_seen_old": seen_old,
            "target_seen_only_old": seen_old and not seen_tail,
            "target_not_in_window": not seen_tail and not seen_old})
    if seen != expected.keys():
        raise RuntimeError("Recent128 predictions must include every validation user")
    write_diagnostic_rows(output_dir, features=features, predictions=predictions)
    _json(output_dir / "diagnostic-reference.json", {"checkpoint": "best-model.pt", "seed": seed,
        "checkpoint_sha256": results["checkpoint_sha256"],
        "git_sha": os.environ.get("GITHUB_SHA", "local"), "best_epoch": results["best_epoch"],
        "source_initialization": INITIALIZATION,
        "basis": "own freshly trained Recent128 checkpoint item embedding table before PE",
        "feature_context_basis": "full raw validation context; model uses only the final128 events",
        "feature_output_length_basis": "depth1 diagnostic convention; explicit output_length_recent128 provided",
        "cross_model_embedding_features": "use one common MixedDepth reference table for paired cohort analysis",
        "target_fields": "prediction-only offline diagnostics; excluded from history-features",
        "checkpoint_selection": "own validation NDCG@10 at scheduled epochs, including final epoch"})
    _json(output_dir / "recent128-results.json", {"checkpoint": "best-model.pt",
        "checkpoint_sha256": results["checkpoint_sha256"], "best_epoch": results["best_epoch"],
        "epochs_trained": results["epochs_trained"], "training_performed": True,
        "source_initialization": INITIALIZATION, "complete": True})
    lines = ["# Recent128", "", "Freshly trained SASRec with only the final128 raw events and no summaries.",
        "The raw max_seq_len1028, recency PE, tail128 supervision and full-catalog validation targets match MixedDepth.",
        "Own best checkpoint selected by validation NDCG@10; test reserved; one seed.", "",
        "| NDCG@10 | Recall@10 | History B1 p50 ms | Full B1 p50 ms | B32 users/s |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    values = [f"{results['validation_ndcg_at_10']:.6f}", f"{results['validation_recall_at_10']:.6f}"]
    values.extend(f"{results[key]:.3f}" if key in results else "disabled" for key in
        ("inference_history_b1_p50_ms", "inference_full_b1_p50_ms", "inference_full_b32_users_per_second"))
    lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"Best epoch: {results['best_epoch']}; trained epochs: {results['epochs_trained']}.",
        "Full context hashes identify matched users/targets; input hashes separately record the retained tail.",
        "History features use this model's own embedding table. Compare models using a single common reference table.",
        "Target-dependent prediction flags are offline diagnostics and cannot be used as production router inputs.",
        "No convergence claim, hyperparameter retuning or test-based selection is made."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return persist(True)
