"""Past-only, model-independent descriptors of an observed item history.

Item zero denotes an observed item outside the training catalogue. Its slot and
timestamp remain part of the chronology, but it never participates in item
identity counts or overlap denominators. The diversity fields measure item
identities, not semantic diversity. All durations use observed timestamps only;
the next-item target and its timestamp are deliberately absent from the API.
"""
from __future__ import annotations

from typing import Any

import numpy as np


HISTORY_FEATURE_FIELDS = (
    "context_oov_fraction",
    "old_known_fraction",
    "recent_known_fraction",
    "old_unique_fraction",
    "old_entropy_normalized",
    "old_top_item_share",
    "old_hhi",
    "recent_unique_fraction",
    "recent_old_item_overlap_fraction",
    "old_recent_item_overlap_fraction",
    "recent_old_last_quarter_item_overlap_fraction",
    "old_first_last_quarter_item_jaccard",
    "old_region_mean_unique_fraction_16",
    "context_duration_days",
    "old_duration_days",
    "recent_duration_hours",
    "context_gap_burstiness",
    "log_train_event_count",
)

CATALOGUE_FREQUENCY_FEATURE_FIELDS = (
    "old_mean_log_train_frequency",
    "recent_mean_log_train_frequency",
)


def _integer_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if vector.size and np.any(vector < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return vector


def _identity_statistics(known: np.ndarray) -> tuple[float, float, float, float]:
    if not known.size:
        return 0.0, 0.0, 0.0, 0.0
    _, counts = np.unique(known, return_counts=True)
    probabilities = counts.astype(np.float64) / known.size
    entropy = (-np.sum(probabilities * np.log(probabilities)) / np.log(counts.size)
               if counts.size > 1 else 0.0)
    return (float(counts.size / known.size), float(entropy),
            float(probabilities.max()), float(np.sum(probabilities ** 2)))


def _event_overlap(events: np.ndarray, reference: np.ndarray) -> float:
    if not events.size or not reference.size:
        return 0.0
    return float(np.mean(np.isin(events, reference)))


def _duration(timestamps: np.ndarray, unit_seconds: float) -> float:
    if timestamps.size < 2:
        return 0.0
    return float((int(timestamps[-1]) - int(timestamps[0])) / unit_seconds)


def history_features(
    context: np.ndarray,
    timestamps: np.ndarray,
    *,
    train_event_count: int,
    catalogue_counts: np.ndarray | None = None,
) -> dict[str, float]:
    """Describe one chronological, past-only context using a fixed recent-128 tail.

    Inputs are the already cropped observed context, normally 16,384 slots.
    ``train_event_count`` is the user's complete pre-cutoff training-history
    length, an activity proxy; it is not the model's visible window length.

    Identity fractions divide by known events, except the explicit known/OOV
    diagnostics. Entropy divides by ``log(number of unique known old items)``;
    a singleton has entropy zero. Quarter boundaries and the 16 chronological
    regions include unknown observed slots. Region uniqueness is averaged over
    regions containing at least one known event. Empty identities and empty
    histories return finite zeros. Gap burstiness is ``(std - mean)/(std +
    mean)`` for consecutive observed gaps, with all-zero gaps assigned zero.

    Optional ``catalogue_counts`` is a one-based vector of counts from training
    events only: entry ``item_id`` gives that item's frequency, and entry zero
    is ignored. Frequency descriptors average ``log1p(count)`` over known
    events, retaining the weighting of repeated items. This function never
    computes frequencies from evaluation events.
    """
    context = _integer_vector(context, "context")
    timestamps = _integer_vector(timestamps, "timestamps")
    if context.shape != timestamps.shape:
        raise ValueError("context and timestamps must have matching shapes")
    if timestamps.size and np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("timestamps must be chronological")
    if (isinstance(train_event_count, (bool, np.bool_))
            or not isinstance(train_event_count, (int, np.integer))
            or train_event_count < 0):
        raise ValueError("train_event_count must be a nonnegative integer")

    old, recent = context[:-128], context[-128:]
    old_timestamps, recent_timestamps = timestamps[:-128], timestamps[-128:]
    old_known, recent_known = old[old > 0], recent[recent > 0]
    unique_fraction, entropy, top_share, hhi = _identity_statistics(old_known)
    recent_unique_fraction = _identity_statistics(recent_known)[0]

    # Split by occupied chronological slots, never by known-item filtering.
    if old.size:
        quarter_length = max(1, old.size // 4)
        first_quarter = old[:quarter_length]
        last_quarter = old[-quarter_length:]
        first_ids = np.unique(first_quarter[first_quarter > 0])
        last_ids = np.unique(last_quarter[last_quarter > 0])
        quarter_union = np.union1d(first_ids, last_ids)
        quarter_jaccard = (np.intersect1d(first_ids, last_ids).size / quarter_union.size
                           if quarter_union.size else 0.0)
        region_fractions = []
        for region in np.array_split(old, min(16, old.size)):
            known = region[region > 0]
            if known.size:
                region_fractions.append(_identity_statistics(known)[0])
        region_unique_fraction = float(np.mean(region_fractions)) if region_fractions else 0.0
    else:
        last_ids = np.empty(0, dtype=context.dtype)
        quarter_jaccard = region_unique_fraction = 0.0

    burstiness = 0.0
    if timestamps.size > 1:
        # Conversion before subtraction avoids unsigned timestamp underflow.
        gaps = np.diff(timestamps.astype(np.float64))
        mean_gap, std_gap = float(gaps.mean()), float(gaps.std())
        if mean_gap + std_gap:
            burstiness = (std_gap - mean_gap) / (std_gap + mean_gap)

    result = {
        "context_oov_fraction": float(np.mean(context == 0)) if context.size else 0.0,
        "old_known_fraction": float(old_known.size / old.size) if old.size else 0.0,
        "recent_known_fraction": float(recent_known.size / recent.size) if recent.size else 0.0,
        "old_unique_fraction": unique_fraction,
        "old_entropy_normalized": entropy,
        "old_top_item_share": top_share,
        "old_hhi": hhi,
        "recent_unique_fraction": recent_unique_fraction,
        "recent_old_item_overlap_fraction": _event_overlap(recent_known, old_known),
        "old_recent_item_overlap_fraction": _event_overlap(old_known, recent_known),
        "recent_old_last_quarter_item_overlap_fraction": _event_overlap(recent_known, last_ids),
        "old_first_last_quarter_item_jaccard": float(quarter_jaccard),
        "old_region_mean_unique_fraction_16": region_unique_fraction,
        "context_duration_days": _duration(timestamps, 86400.0),
        "old_duration_days": _duration(old_timestamps, 86400.0),
        "recent_duration_hours": _duration(recent_timestamps, 3600.0),
        "context_gap_burstiness": float(burstiness),
        "log_train_event_count": float(np.log1p(train_event_count)),
    }
    if catalogue_counts is not None:
        frequencies = np.asarray(catalogue_counts)
        if (frequencies.ndim != 1 or not np.issubdtype(frequencies.dtype, np.number)
                or np.issubdtype(frequencies.dtype, np.complexfloating)):
            raise ValueError("catalogue_counts must be a one-dimensional real numeric array")
        if context.size and int(context.max()) >= frequencies.size:
            raise ValueError("catalogue_counts must be indexed by mapped item ID, including zero")
        for label, known in (("old", old_known), ("recent", recent_known)):
            selected_counts = frequencies[known]
            if np.any(~np.isfinite(selected_counts)) or np.any(selected_counts < 0):
                raise ValueError("observed catalogue_counts must be finite and nonnegative")
            result[f"{label}_mean_log_train_frequency"] = (
                float(np.log1p(selected_counts.astype(np.float64)).mean()) if known.size else 0.0
            )
    return result
