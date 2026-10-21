"""Paired tail-only ablations of a frozen MixedDepth checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .cohorts import _feature_key, _finite, _integer, _write_table
from .depth_diagnostics import _bootstrap_means, _check_bins, _index_features, _interval, _memberships


MODELS = ("depth_0", "depth_3", "tail_128")
COMPARISONS = (("depth_0", "tail_128"), ("depth_3", "tail_128"), ("depth_0", "depth_3"))
METRICS = ("ndcg_at_10", "recall_at_10")


def _required_hash(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _encoder_length(feature: dict[str, Any], model: str) -> int:
    if model == "depth_0":
        return feature["window_length"]
    if model == "depth_3":
        return feature["tail_length"] + (feature["old_length"] + 7) // 8
    return feature["tail_length"]


def _join_predictions(
    features: dict[tuple[str, int], dict[str, Any]], predictions: list[dict]
) -> dict[tuple[str, bool, int], list[dict[str, Any]]]:
    indexed = defaultdict(lambda: {model: {} for model in MODELS})
    for row in predictions:
        split, user_id = _feature_key(row)
        feature = features.get((split, user_id))
        if feature is None:
            raise ValueError(f"prediction has no features: {(split, user_id)}")
        model = row.get("model")
        if model not in MODELS:
            raise ValueError("model must be depth_0, depth_3, or tail_128")
        time_enabled = row.get("time_enabled")
        if not isinstance(time_enabled, bool):
            raise ValueError("time_enabled must be a boolean")
        seed = _integer(row.get("seed"), "seed")
        target = _integer(row.get("target"), "target")
        if target < 1:
            raise ValueError("target must be a positive mapped item ID")
        context = _required_hash(row, "context_hash")
        reference_context = _required_hash(row, "reference_context_hash")
        if context != feature["context_hash"] or reference_context != feature["context_hash"]:
            raise ValueError(f"full context_hash differs for {(split, user_id, model)}")
        input_context = _required_hash(row, "input_context_hash")
        input_length = _integer(row.get("input_length"), "input_length")
        expected_length = (min(feature["window_length"], feature["tail_length"])
                           if model == "tail_128" else feature["window_length"])
        if input_length != expected_length:
            raise ValueError(f"input_length differs for {(split, user_id, model)}: expected {expected_length}")
        if (model != "tail_128" or feature["old_length"] == 0) and input_context != context:
            raise ValueError(f"untruncated input_context_hash differs for {(split, user_id, model)}")
        if model != "tail_128" and "compression_depth" in row:
            depth = _integer(row["compression_depth"], "compression_depth")
            if depth != (0 if model == "depth_0" else 3):
                raise ValueError("compression_depth must agree with the depth model label")
        group = indexed[(split, time_enabled, seed)][model]
        if user_id in group:
            raise ValueError(f"duplicate prediction key: {(split, time_enabled, seed, model, user_id)}")
        normalized = dict(row, target=target, input_length=input_length, input_context_hash=input_context)
        for metric in METRICS:
            value = _finite(row.get(metric), metric)
            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must lie between zero and one")
            if metric == "recall_at_10" and value not in (0.0, 1.0):
                raise ValueError("single-target recall_at_10 must be zero or one")
            normalized[metric] = value
        group[user_id] = normalized
    if not indexed:
        raise ValueError("predictions must contain all three models for at least one group")
    joined = {}
    for key, models in sorted(indexed.items()):
        expected_users = {user_id for split, user_id in features if split == key[0]}
        for model in MODELS:
            missing = sorted(expected_users - models[model].keys())
            if missing:
                raise ValueError(f"paired user IDs differ for {key}: missing {model}={missing[:10]}")
        rows = []
        for user_id in sorted(expected_users):
            feature = features[(key[0], user_id)]
            targets = {models[model][user_id]["target"] for model in MODELS}
            if len(targets) != 1:
                raise ValueError(f"target differs for {(key[0], user_id)}")
            target = models[MODELS[0]][user_id]["target"]
            for field, expected in (("target", target), ("time_enabled", key[1]), ("seed", key[2])):
                if field in feature and feature[field] != expected:
                    raise ValueError(f"feature {field} differs for {(key[0], user_id)}")
            paired = dict(feature, seed=key[2], time_enabled=key[1], target=target,
                          reference_context_hash=feature["context_hash"])
            for model in MODELS:
                prediction = models[model][user_id]
                for field in (*METRICS, "input_length", "input_context_hash"):
                    paired[f"{model}_{field}"] = prediction[field]
                paired[f"{model}_encoder_length"] = _encoder_length(feature, model)
            rows.append(paired)
        joined[key] = rows
    return joined


def _summaries(rows: list[dict], samples: int, rng: np.random.Generator) -> list[dict]:
    columns = [f"{model}_{metric}" for metric in METRICS for model in MODELS]
    values = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    means = values.mean(axis=0) if rows else None
    bootstrap = _bootstrap_means(values, samples, rng) if rows else None
    summaries = []
    for reference, model in COMPARISONS:
        result: dict[str, Any] = {
            "comparison": f"{model}_minus_{reference}",
            "reference_model": reference,
            "model": model,
            "user_count": len(rows),
            "old_prefix_present_user_count": sum(row["old_length"] > 0 for row in rows),
            "actually_compressed_user_count": sum(row["old_length"] >= 2 for row in rows),
            "changed_length_user_count": sum(
                row[f"{model}_encoder_length"] < row[f"{reference}_encoder_length"] for row in rows
            ),
            "reference_mean_encoder_length": None,
            "model_mean_encoder_length": None,
            "reference_hit_model_miss": 0,
            "model_hit_reference_miss": 0,
            "both_hit": 0,
            "both_miss": 0,
        }
        for metric in METRICS:
            for prefix in ("reference", "model", "delta"):
                result[f"{prefix}_{metric}"] = None
                result[f"{prefix}_{metric}_ci95"] = None
            result[f"relative_delta_{metric}_percent"] = None
            result[f"relative_delta_{metric}_percent_ci95"] = None
            result[f"relative_delta_{metric}_bootstrap_defined_samples"] = 0
        if rows:
            for prefix, label in (("reference", reference), ("model", model)):
                result[f"{prefix}_mean_encoder_length"] = float(np.mean([
                    row[f"{label}_encoder_length"] for row in rows
                ]))
            for index, metric in enumerate(METRICS):
                reference_index = index * len(MODELS) + MODELS.index(reference)
                model_index = index * len(MODELS) + MODELS.index(model)
                reference_mean, model_mean = float(means[reference_index]), float(means[model_index])
                reference_draws, model_draws = bootstrap[:, reference_index], bootstrap[:, model_index]
                for prefix, mean, draws in (
                    ("reference", reference_mean, reference_draws),
                    ("model", model_mean, model_draws),
                    ("delta", model_mean - reference_mean, model_draws - reference_draws),
                ):
                    result[f"{prefix}_{metric}"] = mean
                    result[f"{prefix}_{metric}_ci95"] = _interval(draws)
                if reference_mean > 0:
                    result[f"relative_delta_{metric}_percent"] = 100 * (model_mean / reference_mean - 1)
                defined = reference_draws > 0
                result[f"relative_delta_{metric}_bootstrap_defined_samples"] = int(defined.sum())
                if defined.all():
                    result[f"relative_delta_{metric}_percent_ci95"] = _interval(
                        100 * (model_draws / reference_draws - 1)
                    )
            reference_hit = values[:, len(MODELS) + MODELS.index(reference)] > 0
            model_hit = values[:, len(MODELS) + MODELS.index(model)] > 0
            result.update(
                reference_hit_model_miss=int(np.count_nonzero(reference_hit & ~model_hit)),
                model_hit_reference_miss=int(np.count_nonzero(model_hit & ~reference_hit)),
                both_hit=int(np.count_nonzero(reference_hit & model_hit)),
                both_miss=int(np.count_nonzero(~reference_hit & ~model_hit)),
            )
        summaries.append(result)
    return summaries


def analyze_tail_cohorts(
    features: list[dict],
    predictions: list[dict],
    output_dir: Path,
    *,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 1000,
    frozen_bins: dict,
) -> dict:
    """Describe a tail-only intervention, without fitting bins or a router.

    Both context_hash and reference_context_hash identify the original FULL
    observed window for every mode. input_context_hash identifies the actual
    raw input, which is shorter only for tail_128. The caller computes that hash
    from observed items; features alone cannot reconstruct a truncated history.
    All users must be present in all modes within each split/seed/time group.
    """
    samples = _integer(bootstrap_samples, "bootstrap_samples")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    if samples < 1 or bootstrap_seed < 0:
        raise ValueError("bootstrap_samples must be positive and bootstrap_seed nonnegative")
    indexed_features = _index_features(features)
    if any(row["tail_length"] > 128 for row in indexed_features.values()):
        raise ValueError("feature tail_length must be at most 128 for tail_128 ablation")
    bins = _check_bins(frozen_bins)
    groups = _join_predictions(indexed_features, predictions)
    paired_rows, cohorts = [], []
    for (split, time_enabled, seed), rows in groups.items():
        paired_rows.extend(rows)
        metadata = {"split": split, "time_enabled": time_enabled, "seed": seed}
        memberships = _memberships(rows, bins)
        memberships[("old_prefix_present", None, None, None)] = [row for row in rows if row["old_length"] > 0]
        for (kind, feature, quantile, length), members in memberships.items():
            if not members and quantile in {"missing", "unavailable"}:
                continue
            cohort_metadata = dict(metadata, cohort_type=kind, feature=feature, quantile=quantile, length_bin=length)
            identity = json.dumps([bootstrap_seed, cohort_metadata], sort_keys=True).encode("utf-8")
            cohort_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
            rng = np.random.default_rng(cohort_seed)
            cohorts.extend(dict(cohort_metadata, **row) for row in _summaries(members, samples, rng))
    summary = {
        "schema_version": 1,
        "models": list(MODELS),
        "comparisons": [f"{model}_minus_{reference}" for reference, model in COMPARISONS],
        "checkpoint_protocol": "one_frozen_mixed_depth_checkpoint_no_training",
        "context_hash_protocol": "context_hash_and_reference_context_hash_are_full_raw_window; input_context_hash_is_actual_input",
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "paired_user_within_split_time_seed_cohort_jointly_across_all_models",
        "confidence_interval": "percentile_95_unadjusted",
        "relative_interval_protocol": "null_if_any_bootstrap_reference_mean_is_zero",
        "feature_bins_protocol": "frozen_from_source_mixed_depth_validation_run_without_refitting",
        "feature_bins": bins,
        "paired_user_rows": len(paired_rows),
        "cohorts": cohorts,
        "limitations": [
            "Tail-only deletion is a distribution shift for a frozen checkpoint; it measures prefix dependence of this model, not the optimum of a separately trained Recent128 model.",
            "Exploratory comparisons across overlapping cohorts have no multiplicity adjustment; user bootstrap excludes checkpoint-selection and training-seed uncertainty.",
            "One next-item target per user cannot establish individual safety; both models missing the target is not evidence of preserved information.",
            "All modes must use the same frozen checkpoint and seed; no policy is fitted and no test-based selection is performed here.",
        ],
    }
    json.dumps(paired_rows, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_bins.json").write_text(json.dumps(bins, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary["artifacts"] = {
        "paired_predictions": _write_table(output_dir / "tail-paired_predictions.parquet", paired_rows),
        "cohort_results": _write_table(output_dir / "tail-cohort_results.parquet", cohorts),
    }
    (output_dir / "tail-cohorts.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
