"""Full-catalogue daily-anchor evaluation with user-level aggregation.

Several chronological anchors belong to one user. Primary metrics give each
user equal weight, and paired intervals resample users rather than anchors.
Unsupported targets retain zero credit in the primary denominator.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality


_CUTOFFS = (10, 100)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _anchor_metadata(example: Any) -> dict[str, Any]:
    user_id = _integer(example.user_id, "user_id")
    day = _integer(example.day, "day")
    anchor_id = example.anchor_id
    if not isinstance(anchor_id, str) or anchor_id != f"{user_id}:{day}":
        raise ValueError("anchor_id must be the stable user_id:day identity")
    signature = example.reference_context_hash
    if (not isinstance(signature, str) or len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)):
        raise ValueError("reference_context_hash must be a lowercase SHA256 hex digest")
    timestamps = np.asarray(example.timestamps)
    context = np.asarray(example.context)
    target_timestamp = _integer(example.target_timestamp, "target_timestamp")
    if (timestamps.ndim != 1 or timestamps.shape != context.shape or not timestamps.size
            or not np.issubdtype(timestamps.dtype, np.integer)
            or timestamps.min() < 0 or np.any(timestamps[1:] < timestamps[:-1])):
        raise ValueError("occupied context timestamps must be aligned chronological integers")
    # The first event of a day has only strictly earlier observed context.
    # Later same-timestamp events become context only for a later day's anchor.
    if int(timestamps[-1]) >= target_timestamp:
        raise ValueError("context must contain only events strictly before its target")
    return {"user_id": user_id, "anchor_id": anchor_id, "day": day,
            "reference_context_hash": signature}


def _aggregate(records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["user_id"], []).append(record)
    per_user = []
    first_anchors = []
    for user_id, anchors in sorted(groups.items()):
        warm = [record for record in anchors if record["target_known"]]
        first_anchors.append(min(anchors, key=lambda record: record["day"]))
        user = {"user_id": user_id, "anchor_count": len(anchors),
                "warm_anchor_count": len(warm), "oov_target_count": len(anchors) - len(warm)}
        for cutoff in _CUTOFFS:
            for metric in ("hr", "ndcg"):
                key = f"{metric}{cutoff}"
                user[key] = float(np.mean([record[key] for record in anchors]))
                user[f"warm_{key}"] = float(np.mean([record[key] for record in warm])) if warm else 0.0
        per_user.append(user)
    warm_users = [record for record in per_user if record["warm_anchor_count"]]
    metrics = {
        "user_count": len(per_user), "anchor_count": len(records),
        "warm_user_count": len(warm_users),
        "warm_anchor_count": sum(record["warm_anchor_count"] for record in per_user),
        "oov_target_count": sum(record["oov_target_count"] for record in per_user),
        "first_anchor_count": len(first_anchors),
    }
    for cutoff in _CUTOFFS:
        for metric in ("hr", "ndcg"):
            key = f"{metric}{cutoff}"
            metrics[key] = float(np.mean([record[key] for record in per_user]))
            metrics[f"micro_anchor_{key}"] = float(np.mean([record[key] for record in records]))
            metrics[f"warm_{key}"] = float(np.mean([record[f"warm_{key}"] for record in warm_users])) if warm_users else 0.0
            metrics[f"first_anchor_{key}"] = float(np.mean([record[key] for record in first_anchors]))
    return metrics, per_user


def evaluate_daily_quality(model: nn.Module, examples: Iterable[Any], *,
                           device: torch.device, amp: bool = False, batch_size: int = 32,
                           item_chunk_size: int = 65_536) -> dict[str, Any]:
    """Evaluate one final-state prediction per day, reporting @10 and @100.

    Only ``context`` and ``target`` reach the existing full-catalogue evaluator.
    A transient unique row ID lets that evaluator process repeated users. The
    metadata retained alongside it contains no context arrays or item IDs.
    ``reference_context_hash`` identifies the common *full* context even when
    a Recent128 model receives only its final 128 observed slots.
    """
    metadata: list[dict[str, Any]] = []
    anchors: set[str] = set()

    def proxies() -> Iterable[SimpleNamespace]:
        for row, example in enumerate(examples):
            record = _anchor_metadata(example)
            if record["anchor_id"] in anchors:
                raise ValueError("daily evaluation requires unique user-day anchors")
            anchors.add(record["anchor_id"])
            metadata.append(record)
            yield SimpleNamespace(user_id=row, context=example.context, target=example.target)

    result = evaluate_long_quality(model, proxies(), device=device, amp=amp, batch_size=batch_size,
                                   k=10, additional_cutoffs=(100,), item_chunk_size=item_chunk_size)
    rows = {record["user_id"]: record for record in result["per_user"]}
    if len(rows) != len(metadata) or set(rows) != set(range(len(metadata))):
        raise RuntimeError("catalogue evaluator changed the proxy anchor identities")
    per_anchor = [{**rows[row], **record} for row, record in enumerate(metadata)]
    metrics, per_user = _aggregate(per_anchor)
    metrics["num_items"] = result["metrics"]["num_items"]
    metrics["evaluation_wall_seconds"] = result["metrics"]["evaluation_wall_seconds"]
    return {"metrics": metrics, "per_user": per_user, "per_anchor": per_anchor,
            "metadata": {
                **result["metadata"],
                "daily_quality_format_version": 1,
                "primary_denominator": "mean across users of each user's mean across all daily anchors, including OOV zero credit",
                "micro_denominator": "all daily anchors with equal anchor weight; secondary",
                "warm_denominator": "mean across users with warm anchors of each user's mean across warm anchors",
                "zero_warm_count_policy": "warm metrics set to zero; warm_user_count and warm_anchor_count record absent support",
                "first_anchor_diagnostic": "earliest daily anchor per user; secondary compatibility check with the previous first-period-anchor protocol",
                "anchor_identity": "user_id:day",
                "reference_context": "common full chronological context hash, independent of model-specific cropping",
                "forward_inputs": ["observed item IDs", "observed-slot validity mask"],
                "temporal_boundary": "every occupied context event is strictly earlier than its first-of-day target",
            }}


def paired_daily_ndcg_bootstrap(reference_records: Sequence[dict[str, Any]],
                                candidate_records: Sequence[dict[str, Any]], *, k: int = 10,
                                seed: int = 42, draws: int = 5000,
                                confidence: float = 0.95) -> dict[str, int | float]:
    """Paired percentile interval after averaging matched anchors per user.

    The interval measures user-sampling uncertainty for frozen models. It does
    not measure training-seed variation or correct selection among budgets.
    """
    draws = _integer(draws, "draws", minimum=1)
    seed = _integer(seed, "seed")
    k = _integer(k, "k", minimum=1)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")

    def indexed(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {}
        for record in records:
            user_id = _integer(record["user_id"], "user_id")
            day = _integer(record["day"], "day")
            anchor_id = record["anchor_id"]
            if not isinstance(anchor_id, str) or anchor_id != f"{user_id}:{day}" or anchor_id in result:
                raise ValueError("paired records require unique stable user-day anchors")
            signature = record["reference_context_hash"]
            if (not isinstance(signature, str) or len(signature) != 64
                    or any(character not in "0123456789abcdef" for character in signature)):
                raise ValueError("paired records require reference_context_hash SHA256 signatures")
            if type(record["target_known"]) is not bool:
                raise ValueError("paired target catalogue support must be boolean")
            value = record[f"ndcg{k}"]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("paired NDCG values must be finite and lie in [0,1]")
            if not record["target_known"] and value != 0:
                raise ValueError("unsupported targets must retain zero NDCG credit")
            if _integer(record["rank_cutoff"], "rank_cutoff", minimum=1) < k:
                raise ValueError("paired records must retain ranks through the requested cutoff")
            result[anchor_id] = record
        if not result:
            raise ValueError("paired records must be nonempty")
        return result

    reference, candidate = indexed(reference_records), indexed(candidate_records)
    if reference.keys() != candidate.keys():
        raise ValueError("paired records must contain exactly the same anchors")
    groups: dict[int, list[str]] = {}
    for anchor_id, record in reference.items():
        other = candidate[anchor_id]
        for key in ("user_id", "day", "target_known", "reference_context_hash"):
            if record[key] != other[key]:
                raise ValueError(f"paired anchor identity or support disagrees: {key}")
        groups.setdefault(int(record["user_id"]), []).append(anchor_id)
    users = sorted(groups)
    baseline = np.asarray([np.mean([reference[anchor][f"ndcg{k}"] for anchor in groups[user]])
                           for user in users], dtype=np.float64)
    comparison = np.asarray([np.mean([candidate[anchor][f"ndcg{k}"] for anchor in groups[user]])
                             for user in users], dtype=np.float64)
    differences = comparison - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for first in range(0, draws, 64):
        last = min(first + 64, draws)
        indexes = rng.integers(0, len(users), size=(last - first, len(users)))
        samples[first:last] = differences[indexes].mean(axis=1)
    alpha = (1 - confidence) / 2
    tied = np.abs(differences) <= 1e-12
    result: dict[str, int | float] = {
        "matched_users": len(users), "matched_anchors": len(reference),
        "draws": draws, "seed": seed, "confidence": confidence,
        "reference_ndcg": float(baseline.mean()), "candidate_ndcg": float(comparison.mean()),
        "delta_ndcg": float(differences.mean()), "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1 - alpha)),
        "bootstrap_fraction_delta_le_zero": float(np.mean(samples <= 0)),
        "user_win_count": int(np.count_nonzero((differences > 0) & ~tied)),
        "user_loss_count": int(np.count_nonzero((differences < 0) & ~tied)),
        "user_tie_count": int(np.count_nonzero(tied)),
        "user_win_fraction": float(np.mean((differences > 0) & ~tied)),
        "user_loss_fraction": float(np.mean((differences < 0) & ~tied)),
        "user_tie_fraction": float(np.mean(tied)),
    }
    if baseline.mean() > 0:
        result["relative_delta_percent"] = float(differences.mean() / baseline.mean() * 100)
    return result
