"""History-only diagnostics using a single frozen reference embedding table.

These CPU helpers deliberately live outside model inference and latency timing.
Cosines describe the geometry of learned embeddings, not semantic equivalence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rhpr.sasrec.data import SequenceSplit, evaluation_gap_ids, test_example, validation_example


def context_hash(item_ids: Sequence[int] | np.ndarray) -> str:
    """Stable identity of the observed, cropped item window, independent of model."""
    return hashlib.sha256(np.asarray(item_ids, dtype="<i8").tobytes()).hexdigest()


def _summary(values: np.ndarray) -> tuple[float | None, float | None]:
    if not values.size:
        return None, None
    return float(np.mean(values)), float(np.quantile(values, 0.9))


def history_feature_row(
    *,
    user_id: int,
    split_name: str,
    item_ids: Sequence[int] | np.ndarray,
    reference_embeddings: np.ndarray,
    observed_length_raw: int,
    tail_size: int = 128,
    gap_ids: np.ndarray | None = None,
    gap_seconds: np.ndarray | None = None,
    num_gap_buckets: int = 32,
    norm_eps: float = 1e-12,
) -> dict[str, Any]:
    """Describe exactly the observed prefix/tail and pairs used by compression.

    ``old_tail_item_overlap`` is the fraction of distinct recent-tail items also
    present in the old prefix. Missing statistics are null, never artificial zero.
    The table must be the same Full/OFF reference for every compared variant.
    """
    if tail_size < 1 or not np.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("tail_size and norm_eps must be positive")
    items = np.asarray(item_ids, dtype=np.int64)
    table = np.asarray(reference_embeddings)
    if items.ndim != 1 or not items.size:
        raise ValueError("diagnostics require a nonempty one-dimensional context")
    if table.ndim != 2 or table.shape[1] < 1:
        raise ValueError("reference_embeddings must have shape (num_items + 1, d)")
    if np.any(items <= 0) or np.any(items >= table.shape[0]):
        raise ValueError("diagnostic contexts must contain valid non-padding item IDs")
    n = int(items.size)
    if observed_length_raw < n:
        raise ValueError("observed_length_raw cannot be shorter than the cropped window")
    tail_length = min(n, tail_size)
    old_length = n - tail_length
    pair_count = old_length // 2
    vectors = np.asarray(table[items], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    norm_valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > norm_eps)
    units = np.zeros_like(vectors)
    units[norm_valid] = vectors[norm_valid] / norms[norm_valid, None]
    pair_valid = norm_valid[: 2 * pair_count : 2] & norm_valid[1 : 2 * pair_count : 2]
    pair_cosines = np.sum(units[: 2 * pair_count : 2] * units[1 : 2 * pair_count : 2], axis=1)[pair_valid]
    pair_cosines = np.clip(pair_cosines, -1.0, 1.0)
    cosine_mean, cosine_p90 = _summary(pair_cosines)
    distance_mean, distance_p90 = _summary(1.0 - pair_cosines)
    old_units = units[:old_length][norm_valid[:old_length]]
    old_spread = None
    if old_units.size:
        centroid = np.mean(old_units, axis=0)
        old_spread = float(np.clip(1.0 - np.dot(centroid, centroid), 0.0, 1.0))
    old_items = items[:old_length]
    tail_items = items[old_length:]
    old_unique, old_counts = np.unique(old_items, return_counts=True)
    tail_unique = np.unique(tail_items)
    repeats = old_items[: 2 * pair_count : 2] == old_items[1 : 2 * pair_count : 2]
    finite_norms = norms[norm_valid]
    norm_mean, norm_p90 = _summary(finite_norms)
    row: dict[str, Any] = {
        "user_id": int(user_id),
        "split": split_name,
        "context_hash": context_hash(items),
        "observed_length_raw": int(observed_length_raw),
        "window_length": n,
        "old_length": old_length,
        "tail_length": tail_length,
        "output_length": tail_length + (old_length + 1) // 2,
        "pair_count": pair_count,
        "pair_cosine_valid_count": int(pair_valid.sum()),
        "old_norm_valid_count": int(norm_valid[:old_length].sum()),
        "reference_norm_invalid_count": int((~norm_valid).sum()),
        "reference_norm_mean": norm_mean,
        "reference_norm_p90": norm_p90,
        "pair_cosine_mean": cosine_mean,
        "pair_cosine_p90": cosine_p90,
        "pair_cosine_distance_mean": distance_mean,
        "pair_cosine_distance_p90": distance_p90,
        "old_angular_spread": old_spread,
        "window_unique_fraction": float(np.unique(items).size / n),
        "old_unique_fraction": float(old_unique.size / old_length) if old_length else None,
        "tail_unique_fraction": float(tail_unique.size / tail_length),
        "old_top_item_share": float(old_counts.max() / old_length) if old_length else None,
        "exact_pair_repeat_fraction": float(repeats.mean()) if pair_count else None,
        "old_tail_item_overlap": float(np.intersect1d(old_unique, tail_unique).size / tail_unique.size) if old_unique.size else None,
    }
    for region, section in (("window", slice(None)), ("old", slice(0, old_length)), ("tail", slice(old_length, None))):
        bucket_mean = bucket_p90 = gap_mean = gap_p90 = None
        bucket_count = gap_count = 0
        if gap_ids is not None:
            buckets = np.asarray(gap_ids, dtype=np.int64)
            if buckets.shape != items.shape or np.any(buckets < 0) or np.any(buckets > num_gap_buckets):
                raise ValueError("gap_ids must align with context and use valid gap/START IDs")
            values = buckets[section]
            values = values[values != num_gap_buckets]  # START is not a long gap.
            bucket_count = int(values.size)
            bucket_mean, bucket_p90 = _summary(values)
        if gap_seconds is not None:
            seconds = np.asarray(gap_seconds, dtype=np.float64)
            if seconds.shape != items.shape or np.any(np.isinf(seconds)) or np.any(seconds[np.isfinite(seconds)] < 0):
                raise ValueError("gap_seconds must align with context and contain nonnegative observed gaps")
            values = seconds[section]
            values = values[np.isfinite(values)]
            gap_count = int(values.size)
            gap_mean, gap_p90 = _summary(values)
        row.update({
            f"{region}_gap_bucket_count": bucket_count,
            f"{region}_gap_bucket_mean": bucket_mean,
            f"{region}_gap_bucket_p90": bucket_p90,
            f"{region}_gap_seconds_count": gap_count,
            f"{region}_gap_seconds_mean": gap_mean,
            f"{region}_gap_seconds_p90": gap_p90,
        })
    return row


def extract_history_features(
    splits: Sequence[SequenceSplit],
    reference_embeddings: np.ndarray,
    *,
    max_seq_len: int = 1028,
    tail_size: int = 128,
    split_name: str = "validation",
    num_gap_buckets: int = 32,
    time_unit: float = 1.0,
    norm_eps: float = 1e-12,
) -> list[dict[str, Any]]:
    """Extract common reference features without observing a held-out target."""
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be validation or test")
    rows = []
    for split in splits:
        example = validation_example if split_name == "validation" else test_example
        items, _ = example(split, max_seq_len)
        raw_length = split.train.size + int(split_name == "test")
        gaps = seconds = None
        if split.timestamps is not None:
            gaps = evaluation_gap_ids(split, max_seq_len, split_name, num_buckets=num_gap_buckets, time_unit=time_unit)
            times = split.timestamps[:raw_length]
            all_seconds = np.concatenate((np.asarray([np.nan]), np.diff(times)))
            seconds = all_seconds[-max_seq_len:]
        rows.append(history_feature_row(
            user_id=split.user_id,
            split_name=split_name,
            item_ids=items,
            reference_embeddings=reference_embeddings,
            observed_length_raw=int(raw_length),
            tail_size=tail_size,
            gap_ids=gaps,
            gap_seconds=seconds,
            num_gap_buckets=num_gap_buckets,
            norm_eps=norm_eps,
        ))
    return rows


def write_diagnostic_rows(
    output_dir: Path,
    *,
    features: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """Write inspectable JSON and Parquet, independently of timed inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    try:
        import polars as pl
    except ImportError:
        pl = None
    for name, rows in (("history-features", features), ("predictions", predictions)):
        json_path = output_dir / f"{name}.json"
        json_path.write_text(json.dumps(list(rows), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        paths[f"{name}_json"] = str(json_path)
        if pl is not None:
            parquet_path = output_dir / f"{name}.parquet"
            columns = set().union(*(row.keys() for row in rows)) if rows else set()
            if name == "predictions":
                stable_types = {
                    "user_id": pl.Int64, "seed": pl.Int64, "target": pl.Int64,
                    "top10": pl.List(pl.Int64), "rank_in_top10": pl.Int64,
                    "ndcg_at_10": pl.Float64, "recall_at_10": pl.Float64,
                    "time_enabled": pl.Boolean, "context_hash": pl.String,
                }
                overrides = {key: dtype for key, dtype in stable_types.items() if key in columns}
            else:
                # Preserve nullable numeric columns even if every row is missing.
                overrides = {
                    key: pl.Float64 for key in columns
                    if key.endswith(("_mean", "_p90", "_fraction", "_share", "_spread", "_overlap"))
                }
            pl.DataFrame(list(rows), infer_schema_length=None, schema_overrides=overrides).write_parquet(parquet_path)
            paths[f"{name}_parquet"] = str(parquet_path)
    return paths


# Keep analysis callable from one public module while separating its implementation.
from rhpr.sasrec.cohorts import analyze_cohorts  # noqa: E402,F401
