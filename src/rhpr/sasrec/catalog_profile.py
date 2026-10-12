"""Train-only catalogue popularity and hypothetical event-coverage diagnostics.

This CPU experiment does not alter item mappings, histories, splits or targets.
Popularity is counted over all train events, rather than cropped model windows.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any

import numpy as np

from rhpr.metrics import validate_metrics
from rhpr.sasrec.data import YambdaSequences, load_yambda_sequences, resolve_yambda_parquet_path
from rhpr.sasrec.diagnostics import context_hash


TIE_POLICY = "train_count_descending_then_external_item_id_ascending; partial_cutoff_ties_allowed"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coverage(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 1):
        raise ValueError(f"{name} must be finite and lie in (0, 1]")
    return float(value)


def validate_catalog_profile_config(config: dict[str, Any]) -> None:
    if config.get("kind") != "yambda_catalog_profile":
        raise ValueError("kind must be yambda_catalog_profile")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data = config.get("data", {})
    if not isinstance(data, dict) or not isinstance(data.get("path"), str) or not data["path"].strip():
        raise ValueError("data.path must be a nonempty string")
    if data.get("format", "sequential") != "sequential":
        raise ValueError("data.format must be sequential")
    if data.get("timestamp_unit", "seconds") not in {"seconds", "milliseconds", "microseconds", "nanoseconds"}:
        raise ValueError("unsupported timestamp_unit")
    profile = config.get("profile", {})
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    if _coverage(profile.get("event_coverage", 0.99), "event_coverage") != 0.99:
        raise ValueError("the primary catalogue profile must retain 99% of train events")
    diagnostic = profile.get("diagnostic_coverages", [0.95, 0.99, 0.995, 0.999])
    if not isinstance(diagnostic, list) or not diagnostic:
        raise ValueError("diagnostic_coverages must be a nonempty list")
    values = [_coverage(value, "diagnostic coverage") for value in diagnostic]
    if len(set(values)) != len(values):
        raise ValueError("diagnostic_coverages must not contain duplicates")
    window = _positive_integer(profile.get("max_seq_len", 1028), "max_seq_len")
    tail = _positive_integer(profile.get("tail_size", 128), "tail_size")
    if tail > window:
        raise ValueError("tail_size cannot exceed max_seq_len")
    _positive_integer(profile.get("embedding_dimension", 256), "embedding_dimension")
    if type(profile.get("num_negatives", 512)) is not int or profile.get("num_negatives", 512) != 512:
        raise ValueError("retain the existing 512 sampled negatives for cost interpretation")
    if "model" in config or "training" in config or "base" in config:
        raise ValueError("catalog profiling is data-only; no model or training configuration is accepted")


def _train_counts(dataset: YambdaSequences) -> np.ndarray:
    catalog = np.asarray(dataset.catalog_item_ids)
    if (catalog.ndim != 1 or not catalog.size or not np.issubdtype(catalog.dtype, np.integer)
            or np.unique(catalog).size != catalog.size):
        raise ValueError("external catalogue IDs must be a nonempty unique integer array")
    if not dataset.splits:
        raise ValueError("dataset must contain at least one user")
    counts = np.zeros(dataset.num_items + 1, dtype=np.int64)
    users = set()
    for split in dataset.splits:
        if split.user_id in users:
            raise ValueError("user IDs must be unique")
        users.add(split.user_id)
        items = np.asarray(split.train)
        if (items.ndim != 1 or not items.size or not np.issubdtype(items.dtype, np.integer)
                or items.min() < 1 or items.max() > dataset.num_items):
            raise ValueError("train events must be nonempty mapped item IDs in 1..num_items")
        # One catalogue-sized array for the entire dataset. Bincount per user
        # with minlength=num_items would repeatedly allocate ~877k counters.
        np.add.at(counts, items, 1)
    return counts


def _cutoff(counts: np.ndarray, order: np.ndarray, coverage: float) -> tuple[np.ndarray, dict]:
    cumulative = np.cumsum(counts[order], dtype=np.int64)
    total = int(cumulative[-1])
    required = int((Decimal(str(coverage)) * total).to_integral_value(rounding=ROUND_CEILING))
    number = int(np.searchsorted(cumulative, required, side="left")) + 1
    retained = order[:number]
    mask = np.zeros(counts.size, dtype=np.bool_)
    mask[retained] = True
    kept_events = int(cumulative[number - 1])
    cutoff_count = int(counts[retained[-1]])
    descriptor = {
        "requested_train_event_coverage": coverage,
        "required_train_event_count": required,
        "train_event_count": total,
        "retained_train_event_count": kept_events,
        "retained_train_event_fraction": kept_events / total,
        "retained_item_count": number,
        "retained_item_fraction": number / (counts.size - 1),
        "cutoff_train_count": cutoff_count,
        "events_retained_with_one_fewer_item": int(cumulative[number - 2]) if number > 1 else 0,
        "cutoff_tie_item_count": int(np.count_nonzero(counts[1:] == cutoff_count)),
        "cutoff_tie_retained_item_count": int(np.count_nonzero(counts[retained] == cutoff_count)),
        "tie_policy": TIE_POLICY,
    }
    return mask, descriptor


def _distribution(values: list[float | int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "min": float(array.min()),
            "q25": float(np.quantile(array, 0.25)), "median": float(np.quantile(array, 0.5)),
            "q75": float(np.quantile(array, 0.75)), "max": float(array.max())}


def _label(coverage: float) -> str:
    return "p" + format((Decimal(str(coverage)) * 100).normalize(), "f").replace(".", "_")


def run_catalog_profile(config: dict[str, Any], output_dir: Path, *,
                        dataset: YambdaSequences | None = None) -> dict[str, int | float | str | bool]:
    """Profile minimal popularity prefixes without training or filtering data."""
    validate_catalog_profile_config(config)
    import polars as pl

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = config.get("profile", {})
    results: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "training_performed": False,
        "gpu_used": False, "test_evaluated": False, "test_targets_inspected": False,
        "catalog_filtered": False, "split_changed": False, "popularity_count_source": "all_split_train_events",
        "primary_train_event_coverage": 0.99, "tie_policy": TIE_POLICY,
        "embedding_dimension": profile.get("embedding_dimension", 256),
        "sampled_negative_count": 512, "sampled_negative_count_changed": False,
    }

    def persist(complete: bool) -> dict[str, int | float | str | bool]:
        metrics = validate_metrics({**results, "catalog_profile_complete": complete,
                                    "catalog_profile_seconds": time.time() - started})
        _json(output_dir / "metrics.json", metrics)
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "catalog_profile_complete": complete,
            "popularity_count_source": "all_split_train_events", "test_targets_inspected": False,
            "training_performed": False, "gpu_used": False, "catalog_filtered": False})
        if not complete:
            (output_dir / "summary.md").write_text("# Catalogue profile\n\nIncomplete; no model or GPU run.\n",
                                                   encoding="utf-8")
        return metrics

    persist(False)
    if dataset is None:
        data = config["data"]
        dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
                                       timestamp_unit=data.get("timestamp_unit", "seconds"))
    counts = _train_counts(dataset)
    mapped = np.arange(1, dataset.num_items + 1, dtype=np.int64)
    order = mapped[np.lexsort((dataset.catalog_item_ids, -counts[1:]))]
    targets, context_rows = [], []
    maximum, tail_size = profile.get("max_seq_len", 1028), profile.get("tail_size", 128)
    for split in dataset.splits:
        target = split.validation
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)) or not 1 <= target <= dataset.num_items:
            raise ValueError("validation targets must be mapped IDs in 1..num_items")
        targets.append(int(target))
        items = split.train[-maximum:]
        old, tail = items[:-tail_size], items[-tail_size:]
        seen_tail, seen_old = bool(target in tail), bool(target in old)
        context_rows.append({"user_id": int(split.user_id), "split": "validation",
            "context_hash": context_hash(items), "observed_length_raw": int(split.train.size),
            "window_length": int(items.size), "old_length": int(old.size), "tail_length": int(tail.size),
            "target_seen_tail": seen_tail, "target_seen_only_old": seen_old and not seen_tail,
            "target_not_in_window": not seen_tail and not seen_old})
    targets_array = np.asarray(targets, dtype=np.int64)
    coverages = sorted(set(profile.get("diagnostic_coverages", [0.95, 0.99, 0.995, 0.999])) | {0.99})
    descriptors = []
    primary = None
    dimension = profile.get("embedding_dimension", 256)
    for coverage in coverages:
        keep, descriptor = _cutoff(counts, order, coverage)
        target_kept = keep[targets_array]
        descriptor.update(validation_target_count=len(targets),
            validation_targets_retained=int(target_kept.sum()),
            validation_target_retained_fraction=float(target_kept.mean()))
        for name in ("target_seen_tail", "target_seen_only_old", "target_not_in_window"):
            members = np.asarray([row[name] for row in context_rows], dtype=np.bool_)
            number, retained = int(members.sum()), int(target_kept[members].sum())
            descriptor[f"validation_{name}_count"] = number
            descriptor[f"validation_{name}_targets_retained"] = retained
            descriptor[f"validation_{name}_target_retained_fraction"] = retained / number if number else None
        # PAD is included in the table estimates; only real items are selected.
        full_bytes = (dataset.num_items + 1) * dimension * 4
        kept_bytes = (descriptor["retained_item_count"] + 1) * dimension * 4
        descriptor.update(embedding_fp32_full_bytes=full_bytes, embedding_fp32_retained_bytes=kept_bytes,
            embedding_fp32_saved_bytes=full_bytes - kept_bytes,
            embedding_fp32_with_grad_adam_m_v_full_bytes=4 * full_bytes,
            embedding_fp32_with_grad_adam_m_v_retained_bytes=4 * kept_bytes,
            embedding_fp32_with_grad_adam_m_v_saved_bytes=4 * (full_bytes - kept_bytes))
        descriptors.append(descriptor)
        for key, value in descriptor.items():
            if value is not None:
                results[f"{_label(coverage)}_{key}"] = value
        if coverage == 0.99:
            primary = keep
    assert primary is not None
    for split, row, target in zip(dataset.splits, context_rows, targets, strict=True):
        items = split.train[-maximum:]
        old, tail = items[:-tail_size], items[-tail_size:]
        old_kept, tail_kept = int(primary[old].sum()), int(primary[tail].sum())
        row.update(retained_old_events=old_kept, retained_tail_events=tail_kept,
            retained_window_events=old_kept + tail_kept,
            retained_old_fraction=old_kept / old.size if old.size else None,
            retained_tail_fraction=tail_kept / tail.size,
            retained_window_fraction=(old_kept + tail_kept) / items.size,
            validation_target_retained=bool(primary[target]))
    distributions = {field: _distribution([row[field] for row in context_rows if row[field] is not None])
                     for field in ("retained_old_events", "retained_tail_events", "retained_window_events",
                                   "retained_old_fraction", "retained_tail_fraction", "retained_window_fraction")}
    users_with_old = [row for row in context_rows if row["old_length"]]
    lost_old = sum(row["retained_old_events"] == 0 for row in users_with_old)
    results.update(full_catalog_item_count=dataset.num_items, train_event_count=int(counts.sum()),
        train_observed_item_count=int(np.count_nonzero(counts[1:])),
        catalog_item_count_without_train_events=int(np.count_nonzero(counts[1:] == 0)),
        validation_user_count=len(context_rows), raw_window_max_seq_len=maximum, raw_tail_size=tail_size,
        validation_users_with_old_prefix=len(users_with_old), validation_users_losing_entire_old_prefix=lost_old,
        validation_users_losing_entire_old_prefix_fraction=lost_old / len(users_with_old) if users_with_old else 0.0,
        embedding_fp32_bytes_per_real_item=dimension * 4,
        embedding_fp32_with_grad_adam_m_v_bytes_per_real_item=dimension * 16)
    for field, distribution in distributions.items():
        for statistic, value in distribution.items():
            if value is not None:
                results[f"p99_context_{field}_{statistic}"] = value
    ranks = np.empty(dataset.num_items, dtype=np.int64)
    ranks[order - 1] = np.arange(1, dataset.num_items + 1)
    pl.DataFrame({"mapped_item_id": mapped, "external_item_id": dataset.catalog_item_ids,
                  "train_count": counts[1:], "popularity_rank": ranks, "keep_99_events": primary[1:]}).write_parquet(
                      output_dir / "catalog-frequency.parquet")
    pl.DataFrame(context_rows).write_parquet(output_dir / "context-retention.parquet")
    _json(output_dir / "coverage-thresholds.json", {"schema_version": 1,
        "full_catalog_item_count": dataset.num_items, "count_source": "all_split_train_events",
        "primary_train_event_coverage": 0.99, "tie_policy": TIE_POLICY, "thresholds": descriptors,
        "embedding_estimate": "FP32 weights plus FP32 gradient and Adam m/v; PAD included; no activation, encoder, AMP or temporary-buffer costs",
        "sampled_negative_count": 512, "catalogue_and_splits_modified": False, "test_targets_inspected": False})
    _json(output_dir / "context-retention.json", {"schema_version": 1, "train_event_coverage": 0.99,
        "raw_window_max_seq_len": maximum, "raw_tail_size": tail_size, "validation_user_count": len(context_rows),
        "users_with_old_prefix": len(users_with_old), "users_losing_entire_old_prefix": lost_old,
        "distributions": distributions,
        "target_fields": "validation-only offline diagnostics; excluded from popularity selection and unusable as router inputs",
        "actual_filtering_performed": False, "per_user_artifact": "context-retention.parquet"})
    lines = ["# Train-only catalogue profile", "", "CPU diagnostics; no model training, GPU use, filtering or split changes.",
        "Select the fewest most-popular items covering at least 99% of all train EVENTS.",
        "Popularity excludes validation and test targets. Model windows do not limit frequency counting.",
        f"Ties: {TIE_POLICY}.", "",
        f"Full catalogue: {dataset.num_items:,} real items; train events: {int(counts.sum()):,}.", "",
        "| Train event coverage | Retained items | Catalogue fraction | Actual train events retained | Validation targets retained |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for descriptor in descriptors:
        lines.append(f"| {descriptor['requested_train_event_coverage']:.3%} | {descriptor['retained_item_count']:,} | "
                     f"{descriptor['retained_item_fraction']:.3%} | {descriptor['retained_train_event_fraction']:.3%} | "
                     f"{descriptor['validation_target_retained_fraction']:.3%} |")
    primary_descriptor = next(row for row in descriptors if row["requested_train_event_coverage"] == 0.99)
    lines.extend(["", f"99% cutoff train count: {primary_descriptor['cutoff_train_count']:,}; "
                  f"old-only replay validation targets retained: {primary_descriptor['validation_target_seen_only_old_targets_retained']}/"
                  f"{primary_descriptor['validation_target_seen_only_old_count']}.",
        f"Entire old prefix hypothetically lost: {lost_old}/{len(users_with_old)} validation users with an old prefix.",
        "", f"FP32 embedding weights: {primary_descriptor['embedding_fp32_full_bytes'] / 2**30:.3f} → "
        f"{primary_descriptor['embedding_fp32_retained_bytes'] / 2**30:.3f} GiB.",
        f"FP32 weights + gradient + Adam m/v estimate: {primary_descriptor['embedding_fp32_with_grad_adam_m_v_full_bytes'] / 2**30:.3f} → "
        f"{primary_descriptor['embedding_fp32_with_grad_adam_m_v_retained_bytes'] / 2**30:.3f} GiB.",
        "These estimates exclude backbone weights, activations, temporary buffers and implementation-specific AMP state.",
        "Full-catalog scoring scales with the number of retained items; 512 sampled training negatives remain unchanged.",
        "Validation-target retention describes hypothetical candidate eligibility, not recommendation quality.",
        "Filtering events would change histories and targets; it requires a separately declared experiment protocol.",
        "Test targets were not inspected."])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return persist(True)
