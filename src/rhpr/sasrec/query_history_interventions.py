"""Past-only input interventions for an already trained old-suffix compressor.

These controls change inputs, never architecture or weights. In particular,
``query_only_history`` still runs the original Cross Attention or QSSM over the
retained query history; it is not a separately trained QueryOnly architecture.

For X=[remote, Q, R], Q and R remain byte-for-byte unchanged in every control.
The old-suffix query and both compressors use recency relative to the end of H,
so cropping ``remote`` also preserves position features for all retained events.
SASRec's summary and raw-tail position features are preserved for the same reason.

``shuffle_remote`` permutes only item IDs in remote chronological slots. Their
timestamps stay in place, and time features MUST be disabled by the caller. This
keeps the daily evaluator's past-only chronological metadata valid without
claiming that the shuffled item/time pairs are real observed events. The original
reference hash identifies common evaluation support, not the intervened input.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
import hashlib
import json
from typing import Any

import numpy as np

from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample


INTERVENTION_VERSION = "old-suffix-frozen-history-v1"
INTERVENTION_CONDITIONS = ("full", "query_only_history", "shuffle_remote")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def intervention_spec(condition: str) -> dict[str, Any]:
    """Return a JSON-safe declaration of an input control and its limitations."""
    if condition not in INTERVENTION_CONDITIONS:
        raise ValueError(f"condition must be one of {INTERVENTION_CONDITIONS}")
    return {
        "intervention_version": INTERVENTION_VERSION,
        "condition": condition,
        "input_partition": "X=[remote, Q=H[-query_context_size:], R=last_tail_size]",
        "history_policy": {
            "full": "all observed history in original order",
            "query_only_history": "retain only Q and R, using the same pretrained compressor",
            "shuffle_remote": "permute item IDs only in remote, preserving its multiset and length",
        }[condition],
        "query_and_raw_tail_preserved": True,
        "model_weights_and_architecture_unchanged": True,
        "timestamps_policy": "retain original chronological timestamp slots; crop aligned slots only",
        "requires_time_features_off": True,
        "reference_context_hash_policy": "original common full-context identity, not intervened-input identity",
        "permutation_identity": "SHA256(version, seed, user_id, anchor_id); NumPy PCG64 permutation",
        "target_values_or_timestamps_used_for_permutation": False,
    }


def intervene_example(example: DailyEvaluationExample, *, condition: str, seed: int,
                      tail_size: int = 128, query_context_size: int = 128,
                      ) -> DailyEvaluationExample:
    """Transform one complete past context without consulting its target.

    The caller must use an ``old_suffix`` query compressor with the matching Q/R
    sizes and time features disabled. An occupied OOV ID zero is an ordinary
    context slot here, never padding. User/anchor identities alone bind a remote
    permutation, making it independent of model, physical batching, and traversal
    order. Input arrays are never modified.
    """
    intervention_spec(condition)
    seed = _integer(seed, "seed")
    tail_size = _integer(tail_size, "tail_size", minimum=1)
    query_context_size = _integer(query_context_size, "query_context_size", minimum=1)
    uid = _integer(example.user_id, "user_id")
    if not isinstance(example.anchor_id, str) or not example.anchor_id:
        raise ValueError("anchor_id must be a nonempty stable string")
    context, timestamps = np.asarray(example.context), np.asarray(example.timestamps)
    if (context.ndim != 1 or not np.issubdtype(context.dtype, np.integer)
            or np.any(context < 0)):
        raise ValueError("context must be a one-dimensional array of nonnegative integer item IDs")
    if timestamps.shape != context.shape or not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("timestamps must be integer slots aligned with the context")
    retained = tail_size + query_context_size
    if context.size < retained:
        raise ValueError("context must contain the complete old query suffix and raw tail")
    if condition == "full":
        return example
    if condition == "query_only_history":
        return replace(example, context=context[-retained:].copy(),
                       timestamps=timestamps[-retained:].copy())
    remote_length = int(context.size) - retained
    if remote_length <= 1:
        return example
    identity = json.dumps([INTERVENTION_VERSION, seed, uid, example.anchor_id],
                          ensure_ascii=True, separators=(",", ":")).encode("ascii")
    permutation_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
    generator = np.random.Generator(np.random.PCG64(permutation_seed))
    transformed = context.copy()
    transformed[:remote_length] = context[:remote_length][generator.permutation(remote_length)]
    return replace(example, context=transformed)


def iter_intervened_examples(examples: Iterable[DailyEvaluationExample], *, condition: str,
                             seed: int, tail_size: int = 128,
                             query_context_size: int = 128,
                             ) -> Iterator[DailyEvaluationExample]:
    """Apply the same control lazily, preserving anchor identity and order."""
    intervention_spec(condition)
    for example in examples:
        yield intervene_example(example, condition=condition, seed=seed,
                                tail_size=tail_size, query_context_size=query_context_size)
