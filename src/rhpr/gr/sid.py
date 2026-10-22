"""Collision-safe Semantic IDs and item-level retrieval accounting.

The semantic part of an ID is not an item identity.  Every catalogue row receives
one extra token, its zero-based ordinal among rows with the same semantic part.
An ordinal of zero is also appended to collision-free rows, so all items have
one fixed SID length.  Catalogue row order is therefore part of the tokenizer
artifact and must remain fixed between training and evaluation.
"""
from __future__ import annotations

from collections import Counter
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np


def _integer_matrix(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError(f"{name} must be a nonempty two-dimensional integer matrix")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers, not coerced floats or booleans")
    # Check unsigned values before conversion so large uint64 values cannot wrap.
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains integers outside int64 range")
    return array.astype(np.int64, copy=True)


def validate_item_embeddings(
    embeddings: object, num_items: int | None = None
) -> np.ndarray:
    """Validate an external representation artifact without changing its geometry.

    Returns a float32 matrix.  Neither normalization nor centering is implicit;
    those operations belong to the representation/quantizer configuration.
    """
    array = np.asarray(embeddings)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("item embeddings must have shape [items, dimensions] with both positive")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("item embeddings must contain finite real numbers")
    if num_items is not None:
        if isinstance(num_items, bool) or not isinstance(num_items, (int, np.integer)) or num_items < 1:
            raise ValueError("num_items must be a positive integer")
        if array.shape[0] != num_items:
            raise ValueError("item embedding rows do not match the catalogue")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("item embeddings overflow float32")
    return result


class PrefixTrie:
    """A fixed-depth catalogue trie with O(1) prefix lookup.

    Invalid paths have no allowed next codes.  A complete SID has no next code;
    ``contains`` distinguishes a terminal item from an invalid path.
    """

    def __init__(self, sids: object) -> None:
        matrix = _integer_matrix(sids, "sids")
        if np.any(matrix < 0):
            raise ValueError("sids must contain nonnegative codes")
        self.sid_length = int(matrix.shape[1])
        children: dict[tuple[int, ...], set[int]] = {}
        terminals: set[tuple[int, ...]] = set()
        for row in matrix:
            sid = tuple(map(int, row))
            terminals.add(sid)
            for depth, code in enumerate(sid):
                children.setdefault(sid[:depth], set()).add(code)
        self._children = {prefix: tuple(sorted(codes)) for prefix, codes in children.items()}
        self._terminals = frozenset(terminals)

    @staticmethod
    def _key(prefix: Sequence[int]) -> tuple[int, ...]:
        values = tuple(prefix)
        if any(isinstance(code, (bool, np.bool_)) or not isinstance(code, (int, np.integer)) for code in values):
            raise ValueError("a SID prefix must contain integer codes")
        return tuple(map(int, values))

    def allowed_next(self, prefix: Sequence[int] = ()) -> tuple[int, ...]:
        return self._children.get(self._key(prefix), ())

    def contains(self, sid: Sequence[int]) -> bool:
        return self._key(sid) in self._terminals


class SidIndex:
    """A collision-resolved bijection between catalogue items and fixed-length SIDs.

    Item IDs are exactly ``0 .. N-1``.  An explicit ``item_ids`` argument can
    record a permutation of these IDs if the embedding artifact has a different
    row order; collision ordinals always follow that supplied row order.
    """

    def __init__(
        self,
        semantic_codes: object,
        width: int = 256,
        item_ids: object | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width = int(width)
        self.semantic_codes = _integer_matrix(semantic_codes, "semantic_codes")
        if np.any(self.semantic_codes < 0) or np.any(self.semantic_codes >= self.width):
            raise ValueError("semantic codes must be in [0, width)")
        self.num_items, self.semantic_levels = map(int, self.semantic_codes.shape)
        self.sid_length = self.semantic_levels + 1
        if item_ids is None:
            self.item_ids = np.arange(self.num_items, dtype=np.int64)
        else:
            raw_ids = np.asarray(item_ids)
            if raw_ids.ndim != 1 or raw_ids.shape[0] != self.num_items or raw_ids.dtype.kind not in "iu":
                raise ValueError("item_ids must be a one-dimensional integer catalogue permutation")
            if np.any(raw_ids < 0) or np.any(raw_ids >= self.num_items):
                raise ValueError("item_ids must be in [0, number of catalogue rows)")
            self.item_ids = raw_ids.astype(np.int64, copy=True)
            if len(np.unique(self.item_ids)) != self.num_items:
                raise ValueError("item_ids must not contain duplicates")
        group_counts: Counter[tuple[int, ...]] = Counter()
        suffix = np.empty((self.num_items, 1), dtype=np.int64)
        for row_index, row in enumerate(self.semantic_codes):
            key = tuple(map(int, row))
            ordinal = group_counts[key]
            if ordinal >= self.width:
                raise ValueError("a semantic collision group exceeds the collision-token codebook width")
            suffix[row_index, 0] = ordinal
            group_counts[key] += 1
        self.sids = np.concatenate((self.semantic_codes, suffix), axis=1)
        self._group_counts = group_counts
        self.sid_to_item: Mapping[tuple[int, ...], int] = MappingProxyType({
            tuple(map(int, sid)): int(item_id)
            for sid, item_id in zip(self.sids, self.item_ids, strict=True)
        })
        if len(self.sid_to_item) != self.num_items:
            raise AssertionError("resolved SIDs must be unique")
        self.trie = PrefixTrie(self.sids)
        for array in (self.semantic_codes, self.sids, self.item_ids):
            array.setflags(write=False)

    def diagnostics(self) -> dict[str, object]:
        """Return JSON-safe tokenizer diagnostics; no user interactions are read."""
        levels = []
        for level in range(self.sid_length):
            _, counts = np.unique(self.sids[:, level], return_counts=True)
            probabilities = counts.astype(np.float64) / self.num_items
            entropy = float(-(probabilities * np.log2(probabilities)).sum())
            levels.append({
                "level": level,
                "is_collision_suffix": level == self.semantic_levels,
                "used_codes": int(len(counts)),
                "width": self.width,
                "coverage": len(counts) / self.width,
                "entropy_bits": entropy,
                "entropy_normalized": entropy / math.log2(self.width) if self.width > 1 else 0.0,
            })
        collided_groups = [count for count in self._group_counts.values() if count > 1]
        return {
            "num_items": self.num_items,
            "semantic_levels": self.semantic_levels,
            "sid_length": self.sid_length,
            "width": self.width,
            "semantic_unique_ids": len(self._group_counts),
            "collision_groups": len(collided_groups),
            "collided_items": sum(collided_groups),
            "collision_excess_items": self.num_items - len(self._group_counts),
            "max_collision_group": max(self._group_counts.values()),
            "resolved_unique_ids": len(self.sid_to_item),
            "collision_suffix_order": "catalogue_row_order",
            "level_statistics": levels,
        }


def evaluate_ranked_sids(
    index: SidIndex,
    ranked_sids: object,
    target_item_ids: object,
    cutoffs: Iterable[int] = (5, 10),
) -> dict[str, int | float]:
    """Evaluate exact item identity, keeping invalid/duplicate paths in rank slots.

    Each row has one ground-truth item.  Matching only the semantic prefix is
    never sufficient.  In particular, a different item in the same collision
    group is a miss.  Invalid generated paths occupy their original rank and
    receive no credit; they are not filtered out to improve later ranks.
    """
    predictions = np.asarray(ranked_sids)
    if (predictions.ndim != 3 or predictions.shape[0] < 1
            or predictions.shape[2] != index.sid_length or predictions.dtype.kind not in "iu"):
        raise ValueError("ranked_sids must be an integer matrix [examples, candidates, full SID length]")
    if np.any(predictions > np.iinfo(np.int64).max):
        raise ValueError("ranked_sids contains integers outside int64 range")
    targets = np.asarray(target_item_ids)
    if (targets.ndim != 1 or len(targets) != predictions.shape[0] or targets.dtype.kind not in "iu"
            or np.any(targets < 0) or np.any(targets >= index.num_items)):
        raise ValueError("target_item_ids must identify one valid catalogue item per example")
    requested_cutoffs = tuple(cutoffs)
    if (not requested_cutoffs or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1
        for k in requested_cutoffs
    ) or len(set(requested_cutoffs)) != len(requested_cutoffs)):
        raise ValueError("cutoffs must be distinct positive integers")
    first_hits: list[int | None] = []
    invalid = 0
    duplicates = 0
    for candidates, target in zip(predictions, targets, strict=True):
        seen_items: set[int] = set()
        first_hit = None
        for rank, candidate in enumerate(candidates, start=1):
            item_id = index.sid_to_item.get(tuple(map(int, candidate)))
            if item_id is None:
                invalid += 1
                continue
            if item_id in seen_items:
                duplicates += 1
                continue
            seen_items.add(item_id)
            if item_id == int(target) and first_hit is None:
                first_hit = rank
        first_hits.append(first_hit)
    examples = len(first_hits)
    prediction_count = int(predictions.shape[0] * predictions.shape[1])
    result: dict[str, int | float] = {
        "evaluation_examples": examples,
        "prediction_count": prediction_count,
        "invalid_predictions": invalid,
        "duplicate_predictions": duplicates,
        "invalid_prediction_fraction": invalid / prediction_count if prediction_count else 0.0,
        "duplicate_prediction_fraction": duplicates / prediction_count if prediction_count else 0.0,
    }
    for k in requested_cutoffs:
        hits = [rank for rank in first_hits if rank is not None and rank <= k]
        result[f"recall@{k}"] = len(hits) / examples
        result[f"ndcg@{k}"] = sum(1.0 / math.log2(rank + 1) for rank in hits) / examples
    return result
