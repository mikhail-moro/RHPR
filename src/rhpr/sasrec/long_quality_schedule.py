"""Model-independent windows and matched supervision for the long-history pilot.

An epoch is one randomly ended full window per user, not an exhaustive traversal
of all train events.  The last G targets share a raw tail of 127 + G events when
the deployed recent context is 128.  Their raw contexts are therefore 128 through
127 + G for every variant.  Older summaries end before that entire raw tail.

Only a small window descriptor per user is retained.  Event arrays and uniform
negative sets are materialized for the current physical batch, using separate
identity-bound random streams that never consume a model or dropout generator.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

from rhpr.sasrec.long_history_data import LongHistoryCohort


SCHEDULE_VERSION = "long-quality-uid-window-v1"


@dataclass(frozen=True)
class QualityWindow:
    user_index: int
    user_id: int
    end: int  # Exclusive input boundary; the final target is train[end].
    epoch: int
    negative_seed: int


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _digest(domain: str, *identifiers: int) -> bytes:
    payload = json.dumps([SCHEDULE_VERSION, domain, *identifiers], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).digest()


def _training_shape(raw_context_length: int, supervised_targets: int,
                    raw_tail_size: int) -> tuple[int, int, int, int]:
    context = _integer(raw_context_length, "raw_context_length", minimum=1)
    targets = _integer(supervised_targets, "supervised_targets", minimum=1)
    deployed_tail = _integer(raw_tail_size, "raw_tail_size", minimum=1)
    training_tail = deployed_tail - 1 + targets
    if training_tail >= context:
        raise ValueError("the matched training tail must leave a nonempty older prefix")
    return context, targets, deployed_tail, training_tail


def quality_window_schedule(cohort: LongHistoryCohort, *, seed: int, epoch: int,
                            raw_context_length: int = 16_384) -> tuple[QualityWindow, ...]:
    """One window/user; endpoints and shuffle order are bound to seed/epoch/UID.

    Users with no full shifted window cause an error instead of disappearing
    from a variant's training cohort.  Reordering ``cohort.users`` changes only
    descriptor indexes, not UID order, sampled endings or negative draws.
    """
    seed = _integer(seed, "seed")
    epoch = _integer(epoch, "epoch")
    context = _integer(raw_context_length, "raw_context_length", minimum=1)
    if not cohort.users or cohort.num_items < 1:
        raise ValueError("a nonempty cohort and catalogue are required")
    seen: set[int] = set()
    ordered: list[tuple[bytes, QualityWindow]] = []
    for index, user in enumerate(cohort.users):
        uid = _integer(user.user_id, "user_id")
        if uid in seen:
            raise ValueError("training user IDs must be unique")
        seen.add(uid)
        if user.train.ndim != 1 or len(user.train) <= context:
            raise ValueError(f"user {uid} has no full shifted training window")
        endpoint_seed = int.from_bytes(_digest("endpoint", seed, epoch, uid, context)[:16], "little")
        end = int(np.random.default_rng(endpoint_seed).integers(context, len(user.train)))
        negative_seed = int.from_bytes(_digest("negatives", seed, epoch, uid, end)[:8], "little")
        window = QualityWindow(index, uid, end, epoch, negative_seed)
        ordered.append((_digest("order", seed, epoch, uid), window))
    ordered.sort(key=lambda pair: (pair[0], pair[1].user_id))
    return tuple(window for _, window in ordered)


def quality_negative_ids(window: QualityWindow, *, num_items: int,
                         num_negatives: int = 512) -> np.ndarray:
    """Uniform catalogue IDs shared by all targets of a window, sampled lazily.

    Positive collisions and repeated negative IDs are deliberately retained;
    sampled-softmax loss masks positive collisions under the shared protocol.
    """
    catalogue = _integer(num_items, "num_items", minimum=1)
    count = _integer(num_negatives, "num_negatives", minimum=1)
    seed = _integer(window.negative_seed, "negative_seed")
    return np.random.default_rng(seed).integers(1, catalogue + 1, count, dtype=np.int64)


def _checked_user(cohort: LongHistoryCohort, window: QualityWindow, context: int) -> Any:
    index = _integer(window.user_index, "user_index")
    if index >= len(cohort.users):
        raise ValueError("window user_index is outside the cohort")
    user = cohort.users[index]
    if user.user_id != window.user_id:
        raise ValueError("window UID disagrees with its cohort index")
    end = _integer(window.end, "end")
    if not context <= end < len(user.train):
        raise ValueError("window endpoint is outside its train-only full context")
    return user


def quality_batch_arrays(cohort: LongHistoryCohort, windows: Sequence[QualityWindow], *,
                         raw_context_length: int = 16_384, supervised_targets: int = 32,
                         raw_tail_size: int = 128, num_negatives: int = 512,
                         recent_only: bool = False) -> dict[str, np.ndarray]:
    """Materialize only this batch; targets align with the returned input axis.

    Select the last ``supervised_targets`` encoder states before FP32 conversion
    and classification.  With ``recent_only=True`` crop to the shared training
    raw tail before embedding lookup; its last G targets and negative IDs remain
    exactly equal to those of the complete 16K batch.
    """
    context, _, _, training_tail = _training_shape(
        raw_context_length, supervised_targets, raw_tail_size)
    _integer(num_negatives, "num_negatives", minimum=1)
    if not isinstance(recent_only, bool):
        raise ValueError("recent_only must be a boolean")
    if not windows:
        raise ValueError("a nonempty physical batch is required")
    width = training_tail if recent_only else context
    inputs, targets, negatives = [], [], []
    for window in windows:
        user = _checked_user(cohort, window, context)
        inputs.append(user.train[window.end - width:window.end])
        targets.append(user.train[window.end - width + 1:window.end + 1])
        negatives.append(quality_negative_ids(window, num_items=cohort.num_items,
                                              num_negatives=num_negatives))
    item_ids = np.asarray(inputs, dtype=np.int64)
    return {"item_ids": item_ids, "valid_mask": np.ones(item_ids.shape, dtype=np.bool_),
            "targets": np.asarray(targets, dtype=np.int64),
            "negative_ids": np.asarray(negatives, dtype=np.int64)}


def _schedule_rows(windows: Sequence[QualityWindow]) -> list[list[int]]:
    return [[_integer(w.epoch, "epoch"), _integer(w.user_id, "user_id"),
             _integer(w.end, "end"), _integer(w.negative_seed, "negative_seed")]
            for w in windows]


def _rows_sha256(rows: list[list[int]]) -> str:
    payload = json.dumps([SCHEDULE_VERSION, rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def quality_schedule_fingerprint(windows: Sequence[QualityWindow]) -> str:
    """Canonical identity hash, invariant to physical batches and cohort order."""
    rows = _schedule_rows(windows)
    if not rows:
        raise ValueError("cannot fingerprint an empty schedule")
    return _rows_sha256(sorted(rows))


def quality_schedule_summary(cohort: LongHistoryCohort, windows: Sequence[QualityWindow], *,
                            raw_context_length: int = 16_384, supervised_targets: int = 32,
                            raw_tail_size: int = 128) -> dict[str, Any]:
    """Small reproducibility/count metadata; never expands all candidate labels."""
    context, targets, deployed_tail, training_tail = _training_shape(
        raw_context_length, supervised_targets, raw_tail_size)
    if not windows:
        raise ValueError("a nonempty schedule is required")
    first_boundary = last_boundary = 0
    unique_users, unique_endpoints, epochs = set(), set(), set()
    available_endings: dict[int, int] = {}
    for window in windows:
        user = _checked_user(cohort, window, context)
        unique_users.add(window.user_id)
        unique_endpoints.add((window.user_id, window.end))
        epochs.add(window.epoch)
        available_endings[window.user_id] = len(user.train) - context
        first_boundary += int(window.end == context)
        last_boundary += int(window.end == len(user.train) - 1)
    return {
        "schedule_version": SCHEDULE_VERSION,
        "schedule_sha256": quality_schedule_fingerprint(windows),
        "ordered_schedule_sha256": _rows_sha256(_schedule_rows(windows)),
        "windows": len(windows), "unique_users": len(unique_users),
        "epochs": sorted(epochs), "unique_endpoints": len(unique_endpoints),
        "repeated_endpoint_presentations": len(windows) - len(unique_endpoints),
        "available_full_window_endings": sum(available_endings.values()),
        "sampled_end_min": min(w.end for w in windows),
        "sampled_end_max": max(w.end for w in windows),
        "sampled_first_possible_ending_count": first_boundary,
        "sampled_last_possible_ending_count": last_boundary,
        "raw_context_length": context, "supervised_targets_per_window": targets,
        "target_presentations": len(windows) * targets,
        "deployment_raw_tail_size": deployed_tail,
        "training_raw_tail_size": training_tail,
        "training_old_prefix_size": context - training_tail,
        "training_query_raw_context_min": deployed_tail,
        "training_query_raw_context_max": training_tail,
        "train_is_exhaustive_event_epoch": False,
    }


def choose_uniform_pass_budget(preflight_costs: Sequence[Mapping[str, Any]], *,
                               budget_seconds: float, preflight_seconds: float = 0.,
                               fixed_overhead_seconds: float = 0., min_passes: int = 5,
                               max_passes: int = 20, safety_factor: float = 1.25) -> dict[str, Any]:
    """Use a shared pass count for every model, chosen before quality training.

    Each cost row supplies positive ``training_pass_seconds``, nonnegative
    ``validation_pass_seconds`` (amortized if validation is less frequent), and
    optional fixed ``evaluation_seconds`` for additional/final evaluation.  The
    safety factor applies to future work; already elapsed preflight is charged
    once.  Insufficient budget raises rather than giving later models fewer
    passes or starting a suite that cannot meet its minimum shared supervision.
    """
    budget = _number(budget_seconds, "budget_seconds", positive=True)
    preflight = _number(preflight_seconds, "preflight_seconds")
    overhead = _number(fixed_overhead_seconds, "fixed_overhead_seconds")
    minimum = _integer(min_passes, "min_passes", minimum=1)
    maximum = _integer(max_passes, "max_passes", minimum=1)
    safety = _number(safety_factor, "safety_factor", positive=True)
    if minimum > maximum or safety < 1:
        raise ValueError("min_passes must not exceed max_passes; safety_factor must be >= 1")
    if not preflight_costs:
        raise ValueError("preflight must measure every configured model")
    train = validation = fixed_eval = 0.
    for index, cost in enumerate(preflight_costs):
        if not isinstance(cost, Mapping):
            raise ValueError("each preflight cost must be a mapping")
        train += _number(cost.get("training_pass_seconds"),
                         f"preflight_costs[{index}].training_pass_seconds", positive=True)
        validation += _number(cost.get("validation_pass_seconds", 0.),
                              f"preflight_costs[{index}].validation_pass_seconds")
        fixed_eval += _number(cost.get("evaluation_seconds", 0.),
                              f"preflight_costs[{index}].evaluation_seconds")
    per_pass = train + validation
    available = budget - preflight - safety * (overhead + fixed_eval)
    feasible = math.floor(available / (safety * per_pass))
    if feasible < minimum:
        required = preflight + safety * (overhead + fixed_eval + minimum * per_pass)
        raise ValueError(f"budget cannot support {minimum} common passes: "
                         f"need {required:.1f}s, have {budget:.1f}s")
    passes = min(maximum, feasible)
    estimate = preflight + overhead + fixed_eval + passes * per_pass
    conservative = preflight + safety * (overhead + fixed_eval + passes * per_pass)
    return {
        "passes": passes, "min_passes": minimum, "max_passes": maximum,
        "models": len(preflight_costs), "budget_seconds": budget,
        "preflight_seconds": preflight, "fixed_overhead_seconds": overhead,
        "safety_factor": safety, "training_seconds_per_pass": train,
        "validation_seconds_per_pass": validation,
        "model_fixed_evaluation_seconds": fixed_eval,
        "estimated_suite_seconds": estimate,
        "conservative_suite_seconds": conservative,
        "remaining_budget_seconds": budget - conservative,
        "selection_uses_quality_metrics": False,
    }
