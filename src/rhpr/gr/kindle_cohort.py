"""Deterministic long-reader sampling and a maximal user/item core.

The final conditions describe a unique maximal subgraph of a fixed candidate
sample. Pruning is an implementation of that definition, not a fitted model.
"""
from __future__ import annotations

import numpy as np


def candidate_users(users, *, event_budget=800_000, seed=42):
    """Nearest whole-user prefix of a shuffled, lexically numbered user list."""
    users = np.asarray(users)
    sizes = np.bincount(users)
    if not len(sizes) or np.any(sizes < 66):
        raise ValueError("candidate pool must contain only users with >=66 events")
    order = np.random.default_rng(seed).permutation(len(sizes))
    cumulative = np.cumsum(sizes[order])
    take = min(len(order), int(np.searchsorted(cumulative, event_budget)) + 1)
    if take > 1 and abs(int(cumulative[take-2])-event_budget) < abs(int(cumulative[take-1])-event_budget):
        take -= 1
    selected = np.zeros(len(sizes), dtype=bool)
    selected[order[:take]] = True
    return selected[users]


def core_mask(users, items, *, min_events=66, min_item_users=5):
    """Largest induced subgraph with both degree constraints; count distinct users."""
    users, items = np.asarray(users), np.asarray(items)
    if users.ndim != 1 or users.shape != items.shape or not len(users):
        raise ValueError("nonempty aligned user/item vectors required")
    if users.dtype.kind not in 'iu' or items.dtype.kind not in 'iu' or min_events < 1 or min_item_users < 1:
        raise ValueError("integer IDs and positive degree thresholds required")
    if np.any(users < 0) or np.any(items < 0):
        raise ValueError("negative IDs")
    width = int(items.max()) + 1
    mask = np.ones(len(users), dtype=bool)
    iterations = 0
    while mask.any():
        iterations += 1
        pairs = np.unique(users[mask].astype(np.int64)*width + items[mask])
        supported = np.bincount(pairs % width, minlength=width) >= min_item_users
        after = mask & supported[items]
        active = np.bincount(users[after], minlength=int(users.max())+1) >= min_events
        after &= active[users]
        if np.array_equal(mask, after):
            return mask, iterations
        mask = after
    return mask, iterations


def prefix_cases(histories, width=64):
    """All train prefix targets; one val/test target, test sees the val event."""
    cases = {s: [[], [], []] for s in ('train', 'validation', 'test')}
    for uid, seq in enumerate(histories, 1):
        if len(seq) < 66:
            raise ValueError("each user needs 64 pre-validation events plus val/test")
        for split, positions in (('train', range(1, len(seq)-2)),
                                 ('validation', [len(seq)-2]), ('test', [len(seq)-1])):
            for at in positions:
                row = np.full(width, -1, dtype=np.int32)
                past = seq[max(0, at-width):at]
                row[-len(past):] = past
                for column, value in zip(cases[split], (row, uid, seq[at])):
                    column.append(value)
    return {s: tuple(np.asarray(x, dtype=np.int32) for x in a) for s, a in cases.items()}
