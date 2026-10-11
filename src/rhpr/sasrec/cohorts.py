from __future__ import annotations

import json
import math
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np


QUANTILE_FEATURES = (
    "pair_cosine_distance_mean",
    "old_unique_fraction",
)
OPTIONAL_QUANTILE_FEATURES = {"pair_cosine_distance_mean"}
LENGTH_BINS = (
    {"label": "le_128", "lower": 1, "upper": 128},
    {"label": "129_512", "lower": 129, "upper": 512},
    {"label": "513_1028", "lower": 513, "upper": 1028},
)


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _finite(value: Any, field: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number" + (" or null" if nullable else ""))
    return float(value)


def _feature_key(row: dict[str, Any]) -> tuple[str, int]:
    split = row.get("split")
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    return split, _integer(row.get("user_id"), "user_id")


def _feature_rows(features: list[dict]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for row in features:
        key = _feature_key(row)
        if key in indexed:
            raise ValueError(f"duplicate feature key: {key}")
        window_length = _integer(row.get("window_length"), "window_length")
        if not 1 <= window_length <= 1028:
            raise ValueError("window_length must lie between 1 and 1028")
        normalized = dict(row, split=key[0], user_id=key[1], window_length=window_length)
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
    fitted = {}
    for name in QUANTILE_FEATURES:
        values = [row[name] for row in validation if row[name] is not None]
        fitted[name] = {
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
        "quantile_features": fitted,
    }


def _check_bins(frozen_bins: dict) -> dict[str, Any]:
    if not isinstance(frozen_bins, dict) or frozen_bins.get("fit_split") != "validation":
        raise ValueError("frozen_bins must contain thresholds fitted on validation")
    if frozen_bins.get("quantile_assignment") != "right_closed":
        raise ValueError("frozen_bins must use right_closed quantile assignment")
    if frozen_bins.get("length_bins") != list(LENGTH_BINS):
        raise ValueError("frozen_bins must use the predefined input length bins")
    if frozen_bins.get("quantile_levels") != [0.25, 0.5, 0.75]:
        raise ValueError("frozen_bins must contain quartile thresholds")
    # Copy the artifact so neither fitting nor serialization mutates caller state.
    checked_bins = json.loads(json.dumps(frozen_bins, allow_nan=False))
    descriptors = checked_bins.get("quantile_features")
    if not isinstance(descriptors, dict):
        raise ValueError("frozen_bins is missing quantile_features")
    for name in QUANTILE_FEATURES:
        if name not in descriptors and name in OPTIONAL_QUANTILE_FEATURES:
            descriptors[name] = {"edges": None, "validation_count": 0}
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict) or "edges" not in descriptor:
            raise ValueError(f"frozen_bins is missing thresholds for {name}")
        count = _integer(descriptor.get("validation_count"), "validation_count")
        edges = descriptor["edges"]
        if edges is None:
            if count != 0:
                raise ValueError(f"missing thresholds with nonzero count for {name}")
            continue
        if count <= 0 or not isinstance(edges, list) or len(edges) != 3:
            raise ValueError(f"invalid quartile thresholds for {name}")
        checked = [_finite(edge, f"{name} threshold") for edge in edges]
        if checked != sorted(checked):
            raise ValueError(f"quartile thresholds must be sorted for {name}")
    return checked_bins


def _length_bin(length: int) -> str:
    return next(row["label"] for row in LENGTH_BINS if row["lower"] <= length <= row["upper"])


def _quantile_bin(value: float | None, descriptor: dict[str, Any]) -> str:
    if value is None:
        return "missing"
    if descriptor["edges"] is None:
        return "unavailable"
    # Repeated edges deliberately leave empty bins; identical feature values
    # must not be split by row order or user ID.
    return f"q{int(np.searchsorted(descriptor['edges'], value, side='left')) + 1}"


def _join_predictions(
    features: dict[tuple[str, int], dict[str, Any]], predictions: list[dict]
) -> dict[tuple[str, bool, int], list[dict[str, Any]]]:
    indexed: dict[tuple[str, bool, int], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: {"full": {}, "pairmlp": {}}
    )
    for row in predictions:
        split, user_id = _feature_key(row)
        if (split, user_id) not in features:
            raise ValueError(f"prediction has no features: {(split, user_id)}")
        model = row.get("model")
        if model not in {"full", "pairmlp"}:
            raise ValueError("prediction model must be 'full' or 'pairmlp'")
        time_enabled = row.get("time_enabled")
        if not isinstance(time_enabled, bool):
            raise ValueError("time_enabled must be a boolean")
        seed = _integer(row.get("seed"), "seed")
        group = indexed[(split, time_enabled, seed)][model]
        if user_id in group:
            raise ValueError(f"duplicate prediction key: {(split, time_enabled, seed, model, user_id)}")
        normalized = dict(row)
        for metric in ("ndcg_at_10", "recall_at_10"):
            value = _finite(row.get(metric), metric)
            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must lie between zero and one")
            normalized[metric] = value
        group[user_id] = normalized

    joined = {}
    for key, models in sorted(indexed.items()):
        if models["full"].keys() != models["pairmlp"].keys():
            missing_pair = sorted(models["full"].keys() - models["pairmlp"].keys())
            missing_full = sorted(models["pairmlp"].keys() - models["full"].keys())
            raise ValueError(
                f"paired user IDs differ for {key}: missing pairmlp={missing_pair[:10]}, "
                f"missing full={missing_full[:10]}"
            )
        rows = []
        for user_id in sorted(models["full"]):
            feature = features[(key[0], user_id)]
            full, pair = models["full"][user_id], models["pairmlp"][user_id]
            hashes = {str(row["context_hash"]) for row in (feature, full, pair) if row.get("context_hash") is not None}
            if len(hashes) > 1:
                raise ValueError(f"context_hash differs for {(key[0], user_id)}")
            if "target" in full and "target" in pair and full["target"] != pair["target"]:
                raise ValueError(f"target differs for {(key[0], user_id)}")
            rows.append(
                dict(
                    feature,
                    time_enabled=key[1],
                    seed=key[2],
                    full_ndcg_at_10=full["ndcg_at_10"],
                    pairmlp_ndcg_at_10=pair["ndcg_at_10"],
                    delta_ndcg_at_10=pair["ndcg_at_10"] - full["ndcg_at_10"],
                    full_recall_at_10=full["recall_at_10"],
                    pairmlp_recall_at_10=pair["recall_at_10"],
                    delta_recall_at_10=pair["recall_at_10"] - full["recall_at_10"],
                )
            )
        joined[key] = rows
    if not joined:
        raise ValueError("predictions must contain at least one Full/PairMLP pair")
    return joined


def _summarize(rows: list[dict[str, Any]], samples: int, rng: np.random.Generator) -> dict[str, Any]:
    columns = (
        "full_ndcg_at_10", "pairmlp_ndcg_at_10", "delta_ndcg_at_10",
        "full_recall_at_10", "pairmlp_recall_at_10", "delta_recall_at_10",
    )
    result: dict[str, Any] = {"user_count": len(rows)}
    for name in columns:
        result[name] = None
        result[f"{name}_ci95"] = None
    result.update(full_hit_pair_miss=0, pair_hit_full_miss=0, both_hit=0, both_miss=0)
    if not rows:
        return result
    values = np.asarray([[row[column] for column in columns] for row in rows], dtype=np.float64)
    means = values.mean(axis=0)
    for column, mean in zip(columns, means, strict=True):
        result[column] = float(mean)
    full_hit, pair_hit = values[:, 3] > 0, values[:, 4] > 0
    result.update(
        full_hit_pair_miss=int(np.count_nonzero(full_hit & ~pair_hit)),
        pair_hit_full_miss=int(np.count_nonzero(pair_hit & ~full_hit)),
        both_hit=int(np.count_nonzero(full_hit & pair_hit)),
        both_miss=int(np.count_nonzero(~full_hit & ~pair_hit)),
    )
    bootstrap = np.empty((samples, len(columns)), dtype=np.float64)
    unique, counts = np.unique(values, axis=0, return_counts=True)
    if len(unique) < len(rows) / 4:
        # Single-target rank metrics have very few distinct paired outcomes.
        # Multinomial counts give the exact ordinary user-bootstrap law without
        # repeatedly materializing hundreds of thousands of sampled user IDs.
        for start in range(0, samples, 64):
            stop = min(samples, start + 64)
            weights = rng.multinomial(len(rows), counts / len(rows), size=stop - start)
            bootstrap[start:stop] = weights @ unique / len(rows)
    else:
        # Bound temporary indexing arrays instead of allocating samples * users.
        batch_size = max(1, min(64, 1_000_000 // len(rows)))
        for start in range(0, samples, batch_size):
            stop = min(samples, start + batch_size)
            indices = rng.integers(0, len(rows), size=(stop - start, len(rows)))
            for index in range(len(columns)):
                bootstrap[start:stop, index] = values[:, index][indices].mean(axis=1)
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    for index, column in enumerate(columns):
        result[f"{column}_ci95"] = [float(intervals[0, index]), float(intervals[1, index])]
    return result


def _write_table(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import polars as pl
    except ImportError:
        fallback = path.with_suffix(".json")
        fallback.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return fallback.name
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
    return path.name


def analyze_cohorts(
    features: list[dict],
    predictions: list[dict],
    output_dir: Path,
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    frozen_bins: dict | None = None,
) -> dict:
    """Join user predictions and describe paired compression effects.

    Quartile boundaries depend only on validation histories, never labels or
    test feature values. Test-only analyses require the prior feature_bins.json
    artifact as frozen_bins. Bootstrap draws resample paired users within each
    split/time/seed cohort; intervals are exploratory, unadjusted percentiles.
    Only pair cosine distance and old-item uniqueness receive quartile cohorts;
    other supplied descriptors remain in the paired-user artifact.
    """

    samples = _integer(bootstrap_samples, "bootstrap_samples")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    if samples < 1 or bootstrap_seed < 0:
        raise ValueError("bootstrap_samples must be positive and bootstrap_seed nonnegative")
    indexed_features = _feature_rows(features)
    bins = _check_bins(frozen_bins) if frozen_bins is not None else _fit_bins(indexed_features)
    groups = _join_predictions(indexed_features, predictions)
    rng = np.random.default_rng(bootstrap_seed)
    cohorts = []
    paired_rows = []
    for (split, time_enabled, seed), rows in groups.items():
        paired_rows.extend(rows)
        metadata = {"split": split, "time_enabled": time_enabled, "seed": seed}
        memberships: dict[tuple[str, str | None, str | None, str | None], list[dict]] = defaultdict(list)
        memberships[("all", None, None, None)] = rows
        for length in LENGTH_BINS:
            memberships[("length", None, None, length["label"])] = []
        for feature in QUANTILE_FEATURES:
            for quantile in ("q1", "q2", "q3", "q4", "missing", "unavailable"):
                memberships[("feature_quantile", feature, quantile, None)] = []
                for length in LENGTH_BINS:
                    memberships[("length_feature_quantile", feature, quantile, length["label"])] = []
        for row in rows:
            length = _length_bin(row["window_length"])
            memberships[("length", None, None, length)].append(row)
            for feature in QUANTILE_FEATURES:
                quantile = _quantile_bin(row[feature], bins["quantile_features"][feature])
                memberships[("feature_quantile", feature, quantile, None)].append(row)
                memberships[("length_feature_quantile", feature, quantile, length)].append(row)
        for (kind, feature, quantile, length), cohort_rows in memberships.items():
            # Empty missing/unavailable groups add no information. Keep the four
            # quartiles even when ties leave some empty to expose the binning.
            if not cohort_rows and quantile in {"missing", "unavailable"}:
                continue
            cohorts.append(
                dict(metadata, cohort_type=kind, feature=feature, quantile=quantile,
                     length_bin=length, **_summarize(cohort_rows, samples, rng))
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_bins.json").write_text(
        json.dumps(bins, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    artifacts = {
        "paired_predictions": _write_table(output_dir / "paired_predictions.parquet", paired_rows),
        "cohort_results": _write_table(output_dir / "cohort_results.parquet", cohorts),
    }
    summary = {
        "schema_version": 1,
        "comparison": "pairmlp_minus_full",
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "paired_user_within_split_time_seed_cohort",
        "confidence_interval": "percentile_95_unadjusted",
        "feature_bins": bins,
        "paired_user_rows": len(paired_rows),
        "cohorts": cohorts,
        "artifacts": artifacts,
    }
    (output_dir / "cohorts.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
