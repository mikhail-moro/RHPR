from __future__ import annotations

import math
from collections.abc import Sequence


def ndcg_at_k_single_target(
    ranked_item_ids: Sequence[Sequence[int]], target_item_ids: Sequence[int], k: int = 10
) -> float:
    """Compute standard mean NDCG@k when each user has exactly one relevant item."""

    if k < 1:
        raise ValueError("k must be positive")
    if len(ranked_item_ids) != len(target_item_ids) or not target_item_ids:
        raise ValueError("rankings and non-empty targets must have the same length")
    total = 0.0
    for ranking, target in zip(ranked_item_ids, target_item_ids, strict=True):
        for index, item_id in enumerate(ranking[:k]):
            if int(item_id) == int(target):
                total += 1.0 / math.log2(index + 2.0)
                break
    return total / len(target_item_ids)
