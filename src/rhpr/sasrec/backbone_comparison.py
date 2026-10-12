"""Matched fresh training at two context budgets for one chosen backbone."""
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

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.sasrec.data import YambdaSequences, load_yambda_sequences, resolve_yambda_parquet_path, validation_example
from rhpr.sasrec.depth_diagnostics import _bootstrap_means
from rhpr.sasrec.diagnostics import context_hash, write_diagnostic_rows
from rhpr.sasrec.experiment import run_yambda_experiment


CONTEXTS = (128, 1028)
INITIALIZATION = "fresh_seed42_no_pretrained_weights"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_backbone_context_config(config: dict[str, Any]) -> None:
    """Keep architecture as the only difference between the two pair kernels."""
    if config.get("kind") != "yambda_backbone_context":
        raise ValueError("kind must be yambda_backbone_context")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    if type(config.get("seed")) is not int or config["seed"] != 42:
        raise ValueError("the backbone screen uses the common seed42")
    contexts = config.get("contexts")
    if (not isinstance(contexts, list) or any(type(value) is not int for value in contexts)
            or contexts != list(CONTEXTS)):
        raise ValueError("contexts must be [128,1028]")
    base = config.get("base")
    if not isinstance(base, dict) or base.get("kind") != "yambda_sasrec" or base.get("seed") != 42:
        raise ValueError("base must be yambda_sasrec with seed42")
    model, loss = base.get("model", {}), base.get("loss", {})
    if model.get("architecture") not in {"sasrec", "hstu"}:
        raise ValueError("model.architecture must be sasrec or hstu")
    if model.get("max_seq_len") != 1028 or model.get("tail_size") != 128:
        raise ValueError("retain raw max_seq_len1028 and tail128 recency positions")
    if model.get("use_time_features", False) is not False:
        raise ValueError("the common screen uses Time OFF")
    if model.get("compressor_type") != "identity":
        raise ValueError("base model must be identity; the wrapper selects full or tail-only per run")
    if loss.get("name") != "sampled_softmax" or loss.get("num_negatives") != 512 or loss.get("tail_size") != 128:
        raise ValueError("retain sampled_softmax with512 negatives and the last128 raw supervised positions")
    evaluation = base.get("evaluation", {})
    if evaluation.get("evaluate_test") is not False or evaluation.get("k") != 10:
        raise ValueError("use validation-only full-catalog top10 and reserve test")
    if evaluation.get("max_validation_batches") is not None:
        raise ValueError("evaluate every validation user")
    training = base.get("training", {})
    epochs = _positive_int(training.get("max_epochs"), "max_epochs")
    interval = _positive_int(training.get("validation_interval", 5), "validation_interval")
    if interval > epochs:
        raise ValueError("validation_interval cannot exceed the training budget")
    patience = _positive_int(training.get("early_stopping_patience"), "early_stopping_patience")
    if patience < (epochs + interval - 1) // interval:
        raise ValueError("patience must cover all scheduled validation checks")
    physical = _positive_int(training.get("physical_batch_size"), "physical_batch_size")
    effective = _positive_int(training.get("effective_batch_size"), "effective_batch_size")
    if effective % physical:
        raise ValueError("effective batch must be divisible by physical batch")
    if training.get("mixed_depth") is not None or training.get("max_batches_per_epoch") is not None:
        raise ValueError("train all users with one fixed context budget per model")
    if base.get("benchmark", {}).get("allow_oom_fallback", False) is not False:
        raise ValueError("benchmark OOM must fail rather than silently change its batch size")
    budget = config.get("training_budget_seconds", 14400)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("training_budget_seconds must be finite and positive")
    _positive_int(config.get("bootstrap_samples", 1000), "bootstrap_samples")


def _raw_features(dataset: YambdaSequences) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, int]]]:
    """Use item IDs and observed history lengths; never compare embedding tables."""
    features, observed = [], {}
    for split in sorted(dataset.splits, key=lambda row: row.user_id):
        if split.user_id in observed:
            raise ValueError("validation user IDs must be unique")
        items, target = validation_example(split, 1028)
        observed[split.user_id] = items, target
        tail, old = items[-128:], items[:-128]
        features.append({"user_id": split.user_id, "split": "validation", "context_hash": context_hash(items),
            "observed_length_raw": len(split.train), "window_length": len(items),
            "old_length": len(old), "tail_length": len(tail),
            "old_unique_fraction": float(len(np.unique(old)) / len(old)) if len(old) else None,
            "tail_unique_fraction": float(len(np.unique(tail)) / len(tail)),
            "window_unique_fraction": float(len(np.unique(items)) / len(items))})
    return features, observed


def _check_predictions(rows: list[dict[str, Any]], *, observed: dict[int, tuple[np.ndarray, int]],
                       architecture: str, context: int, checkpoint_sha256: str, best_epoch: int) -> list[dict[str, Any]]:
    indexed = {}
    label = f"{architecture}_{context}"
    for row in rows:
        user = row.get("user_id")
        if user not in observed or user in indexed:
            raise ValueError("context prediction users must exactly match validation")
        items, target = observed[user]
        full_hash = context_hash(items)
        if (row.get("model") != label or row.get("split") != "validation" or row.get("seed") != 42
                or row.get("time_enabled") is not False or row.get("context_hash") != full_hash
                or row.get("target") != target):
            raise ValueError("context modes must use the common full raw context and target identity")
        for name in ("ndcg_at_10", "recall_at_10"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid per-user {name}")
        tail, old = items[-128:], items[:-128]
        seen_tail, seen_old = bool(target in tail), bool(target in old)
        actual = tail if context == 128 else items
        indexed[user] = {**row, "architecture": architecture, "context_budget": context,
            "checkpoint_sha256": checkpoint_sha256, "best_epoch": best_epoch,
            "reference_context_hash": full_hash, "reference_window_length": len(items),
            "input_context_hash": context_hash(actual), "input_length": len(actual),
            "target_seen_tail": seen_tail, "target_seen_old": seen_old,
            "target_seen_only_old": seen_old and not seen_tail,
            "target_not_in_window": not seen_old and not seen_tail}
    if indexed.keys() != observed.keys():
        raise ValueError("every context mode must contain all validation users")
    return [indexed[user] for user in sorted(indexed)]


def _summarize(rows: list[dict[str, Any]], *, bootstrap_seed: int, samples: int,
               cohort: str) -> dict[str, Any]:
    result: dict[str, Any] = {"cohort": cohort, "user_count": len(rows)}
    for metric in ("ndcg_at_10", "recall_at_10"):
        for prefix in ("short", "long", "delta", "relative_delta"):
            result[f"{prefix}_{metric}"] = None
            result[f"{prefix}_{metric}_ci95"] = None
    result.update(short_hit_long_miss=0, long_hit_short_miss=0, both_hit=0, both_miss=0)
    if not rows:
        return result
    values = np.asarray([[row[f"{mode}_{metric}"] for metric in ("ndcg_at_10", "recall_at_10")
                          for mode in ("short", "long")] for row in rows], dtype=np.float64)
    stable = hashlib.sha256(json.dumps([bootstrap_seed, cohort]).encode("utf-8")).digest()
    draws = _bootstrap_means(values, samples, np.random.default_rng(int.from_bytes(stable[:8], "little")))
    for offset, metric in ((0, "ndcg_at_10"), (2, "recall_at_10")):
        short, long = values[:, offset].mean(), values[:, offset + 1].mean()
        short_draws, long_draws = draws[:, offset], draws[:, offset + 1]
        for prefix, mean, distribution in (("short", short, short_draws), ("long", long, long_draws),
                ("delta", long - short, long_draws - short_draws)):
            result[f"{prefix}_{metric}"] = float(mean)
            result[f"{prefix}_{metric}_ci95"] = np.quantile(distribution, [.025, .975]).tolist()
        result[f"relative_delta_{metric}"] = float(100 * (long / short - 1)) if short > 0 else None
        if np.all(short_draws > 0):
            result[f"relative_delta_{metric}_ci95"] = np.quantile(100 * (long_draws / short_draws - 1), [.025, .975]).tolist()
    short_hit, long_hit = values[:, 2] > 0, values[:, 3] > 0
    result.update(short_hit_long_miss=int(np.count_nonzero(short_hit & ~long_hit)),
        long_hit_short_miss=int(np.count_nonzero(long_hit & ~short_hit)),
        both_hit=int(np.count_nonzero(short_hit & long_hit)), both_miss=int(np.count_nonzero(~short_hit & ~long_hit)))
    return result


def _analyze(features: list[dict[str, Any]], predictions: list[dict[str, Any]], *,
             architecture: str, bootstrap_seed: int, samples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_mode = {context: {} for context in CONTEXTS}
    for row in predictions:
        by_mode[row["context_budget"]][row["user_id"]] = row
    paired = []
    for feature in features:
        user = feature["user_id"]
        short, long = by_mode[128][user], by_mode[1028][user]
        if any(short[field] != long[field] for field in ("user_id", "context_hash", "target", "split", "seed", "time_enabled")):
            raise ValueError("paired context modes differ in evaluation identity")
        row = {**feature, "architecture": architecture, "target": short["target"],
               "seed": short["seed"], "time_enabled": short["time_enabled"],
               "target_seen_tail": short["target_seen_tail"], "target_seen_old": short["target_seen_old"],
               "target_seen_only_old": short["target_seen_only_old"], "target_not_in_window": short["target_not_in_window"]}
        for mode, prediction in (("short", short), ("long", long)):
            for field in ("ndcg_at_10", "recall_at_10", "best_epoch", "checkpoint_sha256", "input_length", "input_context_hash"):
                row[f"{mode}_{field}"] = prediction[field]
        paired.append(row)
    memberships = {"all": paired, "old_prefix_present": [row for row in paired if row["old_length"] > 0],
        "history_le128": [row for row in paired if row["window_length"] <= 128],
        "history_129_512": [row for row in paired if 128 < row["window_length"] <= 512],
        "history_gt512": [row for row in paired if row["window_length"] > 512],
        "target_seen_only_old": [row for row in paired if row["target_seen_only_old"]]}
    old_fractions = [row["old_unique_fraction"] for row in paired if row["old_unique_fraction"] is not None]
    edges = np.quantile(old_fractions, [.25, .5, .75]).tolist() if old_fractions else None
    if edges is not None:
        for quantile in range(4):
            memberships[f"old_uniqueness_q{quantile + 1}"] = [row for row in paired
                if row["old_unique_fraction"] is not None
                and int(np.searchsorted(edges, row["old_unique_fraction"], side="left")) == quantile]
    cohorts = [_summarize(rows, bootstrap_seed=bootstrap_seed, samples=samples, cohort=name)
               for name, rows in memberships.items()]
    return paired, {"schema_version": 1, "architecture": architecture, "comparison": "long1028_minus_short128",
        "bootstrap_seed": bootstrap_seed, "bootstrap_samples": samples,
        "bootstrap_unit": "paired_validation_user_within_cohort", "confidence_interval": "percentile95_unadjusted",
        "relative_interval_protocol": "null_if_any_bootstrap_short_mean_is_zero",
        "history_feature_basis": "raw observed item IDs and lengths; no learned embedding geometry",
        "old_uniqueness_quartiles": {"fit_split": "validation", "edges": edges, "assignment": "right_closed"},
        "limitations": ["One seed and own validation-selected checkpoints; bootstrap excludes training and checkpoint-selection uncertainty.",
            "Fixed15epoch common screen, not convergence or per-architecture hyperparameter optimization.",
            "Target-seen-only-old is an offline diagnostic and cannot be used as a production routing feature.",
            "A both-miss outcome does not establish that a user's context budget is safe."], "cohorts": cohorts}


def run_backbone_context(config: dict[str, Any], output_dir: Path, *,
                         dataset: YambdaSequences | None = None) -> dict[str, int | float | str | bool]:
    """Train two independent contexts, each selecting its own validation checkpoint."""
    validate_backbone_context_config(config)
    import polars as pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    architecture = config["base"]["model"]["architecture"]
    results: dict[str, int | float | str | bool] = {}

    def persist(complete: bool) -> dict[str, int | float | str | bool]:
        metrics = validate_metrics({**results, "experiment": config["name"], "architecture": architecture,
            "seed": config["seed"], "planned_contexts": len(CONTEXTS), "training_performed": True,
            "source_initialization": INITIALIZATION, "raw_max_seq_len": 1028, "loss_tail_size": 128,
            "backbone_context_complete": complete, "test_evaluated": False,
            "backbone_context_seconds": time.time() - started})
        _json(output_dir / "metrics.json", metrics)
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "source_initialization": INITIALIZATION, "planned_contexts": list(CONTEXTS),
            "started_at": datetime.fromtimestamp(started, UTC).isoformat(), "updated_at": datetime.now(UTC).isoformat(),
            "backbone_context_complete": complete})
        (output_dir / "summary.md").write_text(
            f"# Backbone context screen\n\nComplete: {complete}; {architecture}; independent fresh128/1028 models; test reserved.\n",
            encoding="utf-8")
        if not complete:
            (output_dir / "comparison.md").write_text("# Backbone context screen\n\nIncomplete; do not interpret partial results.\n", encoding="utf-8")
        return metrics

    persist(False)
    if dataset is None:
        data = config["base"]["data"]
        dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
            timestamp_unit=data.get("timestamp_unit", "seconds"))
    features, observed = _raw_features(dataset)
    predictions = []
    mode_results = {}
    # The dense HSTU full-context case is the memory limit; fail there before
    # spending quota on its short-context training if it cannot fit.
    execution_contexts = tuple(reversed(CONTEXTS)) if architecture == "hstu" else CONTEXTS
    for context in execution_contexts:
        directory = output_dir / "runs" / f"context_{context}"
        directory.mkdir(parents=True, exist_ok=True)
        candidate = copy.deepcopy(config["base"])
        candidate["name"] = f"{config['name']}-{context}"
        candidate["source_initialization"] = INITIALIZATION
        candidate["model"]["compressor_type"] = "tail_only" if context == 128 else "identity"
        candidate["training"].update(keep_checkpoint=True,
            deadline_unix_seconds=started + config.get("training_budget_seconds", 14400))
        candidate["diagnostics"] = {"enabled": True}
        _json(directory / "training-config.json", candidate)
        print(json.dumps({"phase": "backbone_context_training", "architecture": architecture, "context_budget": context}), flush=True)
        metrics = validate_metrics(run_yambda_experiment(candidate, directory, dataset=dataset))
        if (metrics.get("epochs_trained") != candidate["training"]["max_epochs"]
                or metrics.get("test_evaluated") is not False or metrics.get("validation_examples") != len(dataset.splits)
                or metrics.get("architecture") != architecture or metrics.get("seed") != config["seed"]):
            raise RuntimeError("each backbone context must complete its fixed training budget and full validation protocol")
        required = ["training-config.json", "model-config.json", "history.json", "predictions.parquet", "best-model.pt"]
        if candidate.get("benchmark", {}).get("enabled", False):
            required.extend(("benchmark.json", "benchmark_samples.json"))
        if any(not (directory / name).is_file() for name in required):
            raise RuntimeError("backbone context is missing required training/evaluation artifacts")
        checkpoint = directory / "best-model.pt"
        if metrics.get("checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError("backbone context checkpoint checksum mismatch")
        history = json.loads((directory / "history.json").read_text())
        if [row.get("epoch") for row in history] != list(range(1, candidate["training"]["max_epochs"] + 1)):
            raise RuntimeError("backbone context history must contain its entire epoch budget")
        evaluated = [row for row in history if row.get("validation_performed")]
        if metrics.get("best_epoch") not in [row["epoch"] for row in evaluated]:
            raise RuntimeError("backbone context selected an epoch without validation")
        if candidate.get("benchmark", {}).get("enabled", False):
            benchmark = json.loads((directory / "benchmark.json").read_text())
            if benchmark.get("metadata", {}).get("config", {}).get("checkpoint_sha256") != metrics["checkpoint_sha256"]:
                raise RuntimeError("backbone context benchmark must use its selected checkpoint")
        rows = _check_predictions(pl.read_parquet(directory / "predictions.parquet").to_dicts(),
            observed=observed, architecture=architecture, context=context,
            checkpoint_sha256=metrics["checkpoint_sha256"], best_epoch=metrics["best_epoch"])
        predictions.extend(rows)
        write_diagnostic_rows(directory, features=features, predictions=rows)
        _json(directory / "diagnostic-reference.json", {"checkpoint": "best-model.pt", "checkpoint_sha256": metrics["checkpoint_sha256"],
            "git_sha": os.environ.get("GITHUB_SHA", "local"), "best_epoch": metrics["best_epoch"], "seed": config["seed"],
            "architecture": architecture, "context_budget": context, "source_initialization": INITIALIZATION,
            "history_feature_basis": "raw observed item IDs and lengths; no learned embeddings",
            "context_identity_basis": "full raw1028 validation window; input_context_hash separately records the retained input",
            "target_fields": "prediction-only offline diagnostics", "checkpoint_selection": "own validation NDCG@10 at scheduled epochs"})
        metrics.update(context_budget=context, training_performed=True, source_initialization=INITIALIZATION)
        _json(directory / "metrics.json", metrics)
        mode_results[str(context)] = metrics
        results.update({f"context_{context}_{key}": value for key, value in metrics.items()})
        persist(False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    paired, analysis = _analyze(features, predictions, architecture=architecture,
        bootstrap_seed=config["seed"], samples=config.get("bootstrap_samples", 1000))
    write_diagnostic_rows(output_dir, features=features, predictions=predictions)
    _json(output_dir / "backbone-cohorts.json", analysis)
    pl.DataFrame(paired, infer_schema_length=None).write_parquet(output_dir / "paired-predictions.parquet")
    pl.DataFrame(analysis["cohorts"], infer_schema_length=None).write_parquet(output_dir / "cohort-results.parquet")
    _json(output_dir / "backbone-results.json", {"architecture": architecture, "contexts": list(CONTEXTS),
        "checkpoint_selection": "own_validation_ndcg_at_10", "independent_fresh_training": True,
        "results": mode_results, "complete": True})
    overall = analysis["cohorts"][0]
    for metric in ("ndcg_at_10", "recall_at_10"):
        results[f"long_minus_short_{metric}"] = overall[f"delta_{metric}"]
        relative = overall[f"relative_delta_{metric}"]
        if relative is not None:
            results[f"long_minus_short_relative_{metric}_percent"] = relative
    lines = ["# Backbone context comparison", "", f"Backbone: {architecture}. Two independent fresh models; same seed/objective/users; own checkpoints.",
        "Common15epoch screen at raw max_seq_len1028; tail128 supervised positions; full-catalog validation; test reserved.",
        "This screen retains the common recipe and does not tune each architecture or establish convergence.", "",
        "| Context | NDCG@10 | Recall@10 | Best epoch | Full B1 p50 ms | B32 users/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for context in CONTEXTS:
        metrics = mode_results[str(context)]
        values = [str(context), f"{metrics['validation_ndcg_at_10']:.6f}", f"{metrics['validation_recall_at_10']:.6f}", str(metrics["best_epoch"])]
        values.extend(f"{metrics[name]:.3f}" if name in metrics else "disabled" for name in
            ("inference_full_b1_p50_ms", "inference_full_b32_users_per_second"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Paired confidence intervals and raw-history cohorts: backbone-cohorts.json.",
        "All cohorts use identical full raw-window hashes and held-out targets; retained-input hashes are separate.",
        "History features use raw IDs and lengths, so cohorts can also be joined across architectures.",
        "Target-seen-only-old is descriptive and target-dependent; it cannot be a production routing input.",
        "Inference includes the executed model and catalogue scoring; preprocessing/serving exclusions follow benchmark.json.",
        "This is a fixed-window backbone screen, without recursive compression, routing, caches or test-based selection."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return persist(True)
