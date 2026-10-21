"""Exploratory, paired depth comparisons from one mixed-depth checkpoint.

This module describes held-out outcomes; it neither trains a depth router nor
labels an individual history safe to compress. The caller must extract history
features using the same frozen joint checkpoint for all evaluated depths.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cohorts import (
    OPTIONAL_QUANTILE_FEATURES,
    QUANTILE_FEATURES,
    _feature_key,
    _finite,
    _integer,
    _quantile_bin,
    _write_table,
)


DEPTHS = (0, 1, 2, 3)
COMPARISONS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3))
LENGTH_BINS = (
    {"label": "le_128", "lower": 1, "upper": 128},
    {"label": "129_512", "lower": 129, "upper": 512},
    {"label": "gt_512", "lower": 513, "upper": None},
)
METRICS = ("ndcg_at_10", "recall_at_10")


def _comparison_protocol(
    depths: Sequence[int] | None, comparisons: Sequence[tuple[int, int]] | None,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    selected = DEPTHS if depths is None else tuple(_integer(depth, "depth") for depth in depths)
    if len(selected) < 2 or len(set(selected)) != len(selected) or any(not 0 <= depth <= 6 for depth in selected):
        raise ValueError("depths must contain at least two distinct integers between zero and six")
    selected = tuple(sorted(selected))
    if comparisons is None:
        if selected == DEPTHS:
            pairs = COMPARISONS
        else:
            # The usual deep experiment uses 0..6: six comparisons to the full
            # mode, three to the previous maximum depth, and two adjacent deep
            # comparisons. Avoid all-pairs expansion of overlapping cohorts.
            pairs = tuple((selected[0], depth) for depth in selected[1:])
            pairs += tuple((3, depth) for depth in selected if depth > 3 and 3 in selected and selected[0] != 3)
            pairs += tuple((depth, depth + 1) for depth in (4, 5) if depth in selected and depth + 1 in selected
                           and (depth, depth + 1) not in pairs)
    else:
        pairs = tuple(comparisons)
    checked = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("comparisons must contain pairs of selected depths")
        reference, depth = (_integer(value, "comparison depth") for value in pair)
        if reference not in selected or depth not in selected or reference == depth:
            raise ValueError("comparisons must contain distinct selected depths")
        checked.append((reference, depth))
    if not checked or len(set(checked)) != len(checked):
        raise ValueError("comparisons must be nonempty and contain no duplicate pairs")
    return selected, tuple(checked)


def _context_hash(row: dict[str, Any]) -> str:
    value = row.get("context_hash")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("context_hash must be a nonempty string")
    return value


def _index_features(features: list[dict]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for row in features:
        key = _feature_key(row)
        if key in indexed:
            raise ValueError(f"duplicate feature key: {key}")
        length = _integer(row.get("window_length"), "window_length")
        old_length = _integer(row.get("old_length"), "old_length")
        if length < 1 or not 0 <= old_length < length:
            raise ValueError("window_length must be positive and 0 <= old_length < window_length")
        tail_length = length - old_length
        if "tail_length" in row and _integer(row["tail_length"], "tail_length") != tail_length:
            raise ValueError("old_length + tail_length must equal window_length")
        normalized = dict(
            row, split=key[0], user_id=key[1], window_length=length,
            old_length=old_length, tail_length=tail_length, context_hash=_context_hash(row),
        )
        if "target" in row:
            normalized["target"] = _integer(row["target"], "feature target")
            if normalized["target"] < 1:
                raise ValueError("feature target must be a positive mapped item ID")
        if "seed" in row:
            normalized["seed"] = _integer(row["seed"], "feature seed")
        if "time_enabled" in row and not isinstance(row["time_enabled"], bool):
            raise ValueError("feature time_enabled must be a boolean")
        for name in QUANTILE_FEATURES:
            if name not in row and name not in OPTIONAL_QUANTILE_FEATURES:
                raise ValueError(f"feature row is missing {name}")
            normalized[name] = _finite(row.get(name), name, nullable=True)
        indexed[key] = normalized
    return indexed


def _fit_bins(features: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    validation = [row for (split, _), row in features.items() if split == "validation"]
    if not validation:
        raise ValueError("validation features or frozen_bins are required; never fit bins on test")
    descriptors = {}
    for name in QUANTILE_FEATURES:
        values = [row[name] for row in validation if row[name] is not None]
        descriptors[name] = {
            "edges": np.quantile(values, [0.25, 0.5, 0.75]).tolist() if values else None,
            "validation_count": len(values),
        }
    return {
        "schema_version": 1,
        "fit_split": "validation",
        "validation_user_count": len(validation),
        "length_bins": list(LENGTH_BINS),
        "quantile_levels": [0.25, 0.5, 0.75],
        "quantile_assignment": "right_closed",
        "quantile_features": descriptors,
    }


def _check_bins(frozen_bins: dict) -> dict[str, Any]:
    if not isinstance(frozen_bins, dict) or frozen_bins.get("fit_split") != "validation":
        raise ValueError("frozen_bins must contain thresholds fitted on validation")
    if frozen_bins.get("length_bins") != list(LENGTH_BINS):
        raise ValueError("frozen_bins must use the predefined input length bins")
    if frozen_bins.get("quantile_levels") != [0.25, 0.5, 0.75]:
        raise ValueError("frozen_bins must contain quartile thresholds")
    if frozen_bins.get("quantile_assignment") != "right_closed":
        raise ValueError("frozen_bins must use right_closed quantile assignment")
    checked = json.loads(json.dumps(frozen_bins, allow_nan=False))
    count = _integer(checked.get("validation_user_count"), "validation_user_count")
    if count < 1:
        raise ValueError("frozen_bins must have a positive validation_user_count")
    descriptors = checked.get("quantile_features")
    if not isinstance(descriptors, dict):
        raise ValueError("frozen_bins is missing quantile_features")
    for name in QUANTILE_FEATURES:
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict) or "edges" not in descriptor:
            raise ValueError(f"frozen_bins is missing thresholds for {name}")
        valid_count = _integer(descriptor.get("validation_count"), "validation_count")
        edges = descriptor["edges"]
        if not 0 <= valid_count <= count:
            raise ValueError(f"invalid validation_count for {name}")
        if edges is None:
            if valid_count:
                raise ValueError(f"missing thresholds with nonzero count for {name}")
            continue
        if not valid_count or not isinstance(edges, list) or len(edges) != 3:
            raise ValueError(f"invalid quartile thresholds for {name}")
        values = [_finite(edge, f"{name} threshold") for edge in edges]
        if values != sorted(values):
            raise ValueError(f"quartile thresholds must be sorted for {name}")
    return checked


def _length_bin(length: int) -> str:
    return next(
        row["label"] for row in LENGTH_BINS
        if row["lower"] <= length and (row["upper"] is None or length <= row["upper"])
    )


def _encoder_length(row: dict[str, Any], depth: int) -> int:
    factor = 2 ** depth
    return row["tail_length"] + (row["old_length"] + factor - 1) // factor


def _join_predictions(
    features: dict[tuple[str, int], dict[str, Any]], predictions: list[dict],
    selected_depths: tuple[int, ...] = DEPTHS,
) -> dict[tuple[str, bool, int], list[dict[str, Any]]]:
    indexed = defaultdict(lambda: {depth: {} for depth in selected_depths})
    for row in predictions:
        split, user_id = _feature_key(row)
        if (split, user_id) not in features:
            raise ValueError(f"prediction has no features: {(split, user_id)}")
        depth = _integer(row.get("compression_depth"), "compression_depth")
        if depth not in selected_depths or row.get("model") != f"depth_{depth}":
            raise ValueError("model must match compression_depth in the selected depth modes")
        time_enabled = row.get("time_enabled")
        if not isinstance(time_enabled, bool):
            raise ValueError("time_enabled must be a boolean")
        seed = _integer(row.get("seed"), "seed")
        target = _integer(row.get("target"), "target")
        if target < 1:
            raise ValueError("target must be a positive mapped item ID")
        group = indexed[(split, time_enabled, seed)][depth]
        if user_id in group:
            raise ValueError(f"duplicate prediction key: {(split, time_enabled, seed, depth, user_id)}")
        normalized = dict(row, target=target, context_hash=_context_hash(row))
        for metric in METRICS:
            value = _finite(row.get(metric), metric)
            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must lie between zero and one")
            if metric == "recall_at_10" and value not in (0.0, 1.0):
                raise ValueError("single-target recall_at_10 must be zero or one")
            normalized[metric] = value
        group[user_id] = normalized

    if not indexed:
        raise ValueError("predictions must contain all selected depth modes for at least one group")
    users_by_split = {
        split: {user_id for feature_split, user_id in features if feature_split == split}
        for split in ("validation", "test")
    }
    joined = {}
    for key, depths in sorted(indexed.items()):
        expected_users = users_by_split[key[0]]
        for depth in selected_depths:
            missing = sorted(expected_users - depths[depth].keys())
            if missing:
                raise ValueError(f"paired user IDs differ for {key}: missing depth_{depth}={missing[:10]}")
        rows = []
        for user_id in sorted(expected_users):
            feature = features[(key[0], user_id)]
            predictions_for_user = [depths[depth][user_id] for depth in selected_depths]
            if any(row["context_hash"] != feature["context_hash"] for row in predictions_for_user):
                raise ValueError(f"context_hash differs for {(key[0], user_id)}")
            targets = {row["target"] for row in predictions_for_user}
            if len(targets) != 1:
                raise ValueError(f"target differs for {(key[0], user_id)}")
            if "target" in feature and feature["target"] != predictions_for_user[0]["target"]:
                raise ValueError(f"feature target differs for {(key[0], user_id)}")
            for field, expected in (("time_enabled", key[1]), ("seed", key[2])):
                if field in feature and feature[field] != expected:
                    raise ValueError(f"feature {field} differs for {(key[0], user_id)}")
            paired = dict(feature, time_enabled=key[1], seed=key[2], target=predictions_for_user[0]["target"])
            for depth, prediction in zip(selected_depths, predictions_for_user, strict=True):
                for metric in METRICS:
                    paired[f"depth_{depth}_{metric}"] = prediction[metric]
                paired[f"depth_{depth}_encoder_length"] = _encoder_length(feature, depth)
            rows.append(paired)
        joined[key] = rows
    return joined


def _bootstrap_means(values: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Use one user draw for every depth and metric in a cohort."""
    bootstrap = np.empty((samples, values.shape[1]), dtype=np.float64)
    unique, counts = np.unique(values, axis=0, return_counts=True)
    if len(unique) < len(values) / 4:
        # Exact ordinary user-bootstrap law, using counts of identical outcomes.
        for start in range(0, samples, 64):
            stop = min(samples, start + 64)
            weights = rng.multinomial(len(values), counts / len(values), size=stop - start)
            bootstrap[start:stop] = weights @ unique / len(values)
    else:
        batch_size = max(1, min(64, 1_000_000 // len(values)))
        for start in range(0, samples, batch_size):
            stop = min(samples, start + batch_size)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            for column in range(values.shape[1]):
                bootstrap[start:stop, column] = values[:, column][indices].mean(axis=1)
    return bootstrap


def _interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def _summaries(
    rows: list[dict[str, Any]], samples: int, rng: np.random.Generator,
    depths: tuple[int, ...] = DEPTHS, comparisons: tuple[tuple[int, int], ...] = COMPARISONS,
) -> list[dict[str, Any]]:
    columns = [f"depth_{depth}_{metric}" for metric in METRICS for depth in depths]
    positions = {depth: index for index, depth in enumerate(depths)}
    values = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    means = values.mean(axis=0) if rows else None
    bootstrap = _bootstrap_means(values, samples, rng) if rows else None
    results = []
    for reference, depth in comparisons:
        result: dict[str, Any] = {
            "comparison": f"depth_{depth}_minus_depth_{reference}",
            "reference_depth": reference,
            "compression_depth": depth,
            "user_count": len(rows),
            "actually_compressed_user_count": sum(row["old_length"] >= 2 for row in rows),
            "changed_length_user_count": sum(
                row[f"depth_{depth}_encoder_length"] < row[f"depth_{reference}_encoder_length"]
                for row in rows
            ),
            "reference_mean_encoder_length": None,
            "depth_mean_encoder_length": None,
            "reference_hit_depth_miss": 0,
            "depth_hit_reference_miss": 0,
            "both_hit": 0,
            "both_miss": 0,
        }
        for metric in METRICS:
            for prefix in ("reference", "depth", "delta"):
                result[f"{prefix}_{metric}"] = None
                result[f"{prefix}_{metric}_ci95"] = None
            result[f"relative_delta_{metric}_percent"] = None
            result[f"relative_delta_{metric}_percent_ci95"] = None
            result[f"relative_delta_{metric}_bootstrap_defined_samples"] = 0
        if rows:
            result["reference_mean_encoder_length"] = float(np.mean([
                row[f"depth_{reference}_encoder_length"] for row in rows
            ]))
            result["depth_mean_encoder_length"] = float(np.mean([
                row[f"depth_{depth}_encoder_length"] for row in rows
            ]))
            for index, metric in enumerate(METRICS):
                ref_index = index * len(depths) + positions[reference]
                depth_index = index * len(depths) + positions[depth]
                ref_mean, depth_mean = float(means[ref_index]), float(means[depth_index])
                ref_bootstrap, depth_bootstrap = bootstrap[:, ref_index], bootstrap[:, depth_index]
                for prefix, mean, draws in (
                    ("reference", ref_mean, ref_bootstrap),
                    ("depth", depth_mean, depth_bootstrap),
                    ("delta", depth_mean - ref_mean, depth_bootstrap - ref_bootstrap),
                ):
                    result[f"{prefix}_{metric}"] = mean
                    result[f"{prefix}_{metric}_ci95"] = _interval(draws)
                if ref_mean > 0:
                    result[f"relative_delta_{metric}_percent"] = 100 * (depth_mean / ref_mean - 1)
                defined = ref_bootstrap > 0
                result[f"relative_delta_{metric}_bootstrap_defined_samples"] = int(defined.sum())
                # Do not silently condition the interval on bootstrap draws
                # with at least one reference hit in a sparse single-target set.
                if defined.all():
                    relative_draws = 100 * (depth_bootstrap / ref_bootstrap - 1)
                    result[f"relative_delta_{metric}_percent_ci95"] = _interval(relative_draws)
            ref_hit = values[:, len(depths) + positions[reference]] > 0
            depth_hit = values[:, len(depths) + positions[depth]] > 0
            result.update(
                reference_hit_depth_miss=int(np.count_nonzero(ref_hit & ~depth_hit)),
                depth_hit_reference_miss=int(np.count_nonzero(depth_hit & ~ref_hit)),
                both_hit=int(np.count_nonzero(ref_hit & depth_hit)),
                both_miss=int(np.count_nonzero(~ref_hit & ~depth_hit)),
            )
        results.append(result)
    return results


def _memberships(rows: list[dict], bins: dict) -> dict[tuple, list[dict]]:
    memberships = defaultdict(list)
    memberships[("all", None, None, None)] = rows
    memberships[("actually_compressed", None, None, None)] = []
    for length in LENGTH_BINS:
        memberships[("length", None, None, length["label"])] = []
    for feature in QUANTILE_FEATURES:
        for quantile in ("q1", "q2", "q3", "q4", "missing", "unavailable"):
            memberships[("feature_quantile", feature, quantile, None)] = []
            if feature == "old_unique_fraction":
                for length in LENGTH_BINS:
                    memberships[("length_feature_quantile", feature, quantile, length["label"])] = []
    for row in rows:
        length = _length_bin(row["window_length"])
        memberships[("length", None, None, length)].append(row)
        if row["old_length"] >= 2:
            memberships[("actually_compressed", None, None, None)].append(row)
        for feature in QUANTILE_FEATURES:
            quantile = _quantile_bin(row[feature], bins["quantile_features"][feature])
            memberships[("feature_quantile", feature, quantile, None)].append(row)
            if feature == "old_unique_fraction":
                memberships[("length_feature_quantile", feature, quantile, length)].append(row)
    return memberships


def analyze_depth_cohorts(
    features: list[dict],
    predictions: list[dict],
    output_dir: Path,
    *,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 1000,
    frozen_bins: dict | None = None,
    depths: Sequence[int] | None = None,
    comparisons: Sequence[tuple[int, int]] | None = None,
) -> dict:
    """Compare forced depths from one checkpoint using paired users.

    Each represented split/seed/time group must cover every supplied feature
    user at all selected depths with the same observed context and next-item target.
    The default preserves the original four-depth comparison protocol. For
    depths 0..6, default contrasts compare every compressed mode to zero, new
    depths to three, and adjacent new depths to each other. Explicit pairs may
    select a different contrast set without changing the joint bootstrap draws.
    Bins use validation history features only; test-only analysis requires the
    saved ``feature_bins.json`` as ``frozen_bins``. Intervals condition on the
    chosen checkpoint and resample users jointly across all depths.
    """
    samples = _integer(bootstrap_samples, "bootstrap_samples")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    if samples < 1 or bootstrap_seed < 0:
        raise ValueError("bootstrap_samples must be positive and bootstrap_seed nonnegative")
    selected_depths, selected_comparisons = _comparison_protocol(depths, comparisons)
    indexed_features = _index_features(features)
    bins = _check_bins(frozen_bins) if frozen_bins is not None else _fit_bins(indexed_features)
    groups = _join_predictions(indexed_features, predictions, selected_depths)
    paired_rows = []
    cohorts = []
    for (split, time_enabled, seed), rows in groups.items():
        paired_rows.extend(rows)
        metadata = {"split": split, "time_enabled": time_enabled, "seed": seed}
        for (kind, feature, quantile, length), members in _memberships(rows, bins).items():
            if not members and quantile in {"missing", "unavailable"}:
                continue
            cohort_metadata = dict(
                metadata, cohort_type=kind, feature=feature, quantile=quantile, length_bin=length,
            )
            # Stable by cohort identity: adding a group or reordering input rows
            # cannot change the bootstrap draws of an existing cohort.
            identity = json.dumps([bootstrap_seed, cohort_metadata], sort_keys=True).encode("utf-8")
            cohort_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
            rng = np.random.default_rng(cohort_seed)
            cohorts.extend(dict(cohort_metadata, **result) for result in _summaries(
                members, samples, rng, selected_depths, selected_comparisons,
            ))
    summary = {
        "schema_version": 1,
        "depths": list(selected_depths),
        "comparisons": [f"depth_{depth}_minus_depth_{reference}" for reference, depth in selected_comparisons],
        "checkpoint_protocol": "all_depths_from_one_frozen_mixed_depth_checkpoint",
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "paired_user_within_split_time_seed_cohort_jointly_across_all_depths",
        "confidence_interval": "percentile_95_unadjusted",
        "relative_interval_protocol": "null_if_any_bootstrap_reference_mean_is_zero",
        "feature_bins": bins,
        "paired_user_rows": len(paired_rows),
        "cohorts": cohorts,
        "limitations": [
            "Exploratory comparisons across multiple overlapping cohorts; intervals are not multiplicity adjusted.",
            "User bootstrap conditions on the selected checkpoint and seed; it excludes training and checkpoint-selection uncertainty.",
            "One next-item outcome per user cannot identify a history as safe to compress; both depths missing the target is not evidence of preserved quality.",
            "History features must come from the same frozen joint checkpoint; no depth policy is fitted or evaluated here.",
        ],
    }
    # Check all preserved descriptors as well as metrics before writing files.
    json.dumps(paired_rows, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_bins.json").write_text(
        json.dumps(bins, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    summary["artifacts"] = {
        "paired_predictions": _write_table(output_dir / "depth-paired_predictions.parquet", paired_rows),
        "cohort_results": _write_table(output_dir / "depth-cohort_results.parquet", cohorts),
    }
    (output_dir / "depth-cohorts.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary
