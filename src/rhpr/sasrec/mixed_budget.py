"""One shared hierarchical compressor trained under several summary budgets.

Budget changes alter the chronological regions and hierarchy depth, while every
budget uses the same merge projection and downstream HSTU parameters.  This is
random-budget training, not a learned halting policy or an adaptive controller.
The schedule assigns one budget to each effective optimizer update, so gradient
accumulation and physical microbatch sizes cannot change its random choices.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence

import torch
from torch import nn

from rhpr.sasrec.long_compression import HierarchicalPairMLPCompressor


DEFAULT_SUMMARY_BUDGETS = (1, 2, 4, 8, 16, 32)


def _summary_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    choices = tuple(budgets)
    if (not choices or any(isinstance(value, bool) or not isinstance(value, int)
                           or value < 1 for value in choices)):
        raise ValueError("budgets must be a nonempty sequence of positive integers")
    if len(set(choices)) != len(choices):
        raise ValueError("budgets must contain distinct summary counts")
    return tuple(sorted(choices))


def make_mixed_budget_compressor(
    d_model: int, *, budgets: Sequence[int] = DEFAULT_SUMMARY_BUDGETS,
    tail_size: int = 159, group_arity: int = 4,
) -> HierarchicalPairMLPCompressor:
    """Initialize a single merge projection at the maximum declared budget.

    Preserve the caller's CPU RNG exactly, as the fixed-budget compressor
    factory does, so adding this arm cannot change common HSTU initialization.
    ``mixed_budget_choices`` is immutable tuple metadata, not model state.
    """
    choices = _summary_budgets(budgets)
    with torch.random.fork_rng(devices=[]):
        compressor = HierarchicalPairMLPCompressor(
            d_model=d_model, summary_tokens=max(choices), tail_size=tail_size,
            group_arity=group_arity,
        )
    compressor.mixed_budget_choices = choices
    return compressor


def select_summary_budget(model: nn.Module, budget: int) -> None:
    """Change a shared HSTU model's budget without replacing any parameters.

    The normalization divisor and relative-bias table must accommodate the
    largest declared budget under the current raw-tail policy.  They remain
    fixed across choices, as does the raw-recency position embedding table.
    Tail-policy changes between training and evaluation belong to the caller.
    """
    compressor = getattr(model, "compressor", None)
    if (not isinstance(compressor, HierarchicalPairMLPCompressor)
            or not hasattr(compressor, "mixed_budget_choices")):
        raise ValueError("model must use a declared mixed-budget hierarchical compressor")
    choices = _summary_budgets(compressor.mixed_budget_choices)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget not in choices:
        raise ValueError("budget must be one of the declared mixed-budget summary counts")
    tail_size = compressor.tail_size
    if isinstance(tail_size, bool) or not isinstance(tail_size, int) or tail_size < 1:
        raise ValueError("compressor tail_size must be a positive integer")
    normalization = getattr(model, "hstu_normalization_length", None)
    if (isinstance(normalization, bool) or not isinstance(normalization, int)
            or normalization < tail_size + max(choices)):
        raise ValueError("common HSTU normalization_length must fit the largest declared budget and tail")
    max_seq_len = getattr(model, "max_seq_len", None)
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError("model must declare a positive raw max_seq_len")
    compressor.summary_tokens = budget
    model.hstu_context_len = min(max_seq_len, tail_size + budget)


def balanced_budget_schedule(
    *, seed: int, epoch: int, effective_updates: int,
    budgets: Sequence[int] = DEFAULT_SUMMARY_BUDGETS,
) -> tuple[int, ...]:
    """Return reproducible shuffled choices with counts differing by at most one.

    The local RNG is separate from training-window, negative-item, dropout, and
    initialization RNGs.  A shuffled remainder prevents incomplete cycles from
    systematically favoring the smallest budgets across epochs.  Balance is
    by effective updates; the caller records actual windows and labels for a
    potentially smaller final update instead of claiming equal label counts.
    """
    choices = _summary_budgets(budgets)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for name, value in (("epoch", epoch), ("effective_updates", effective_updates)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    descriptor = json.dumps(
        {"namespace": "mixed_summary_budget_effective_update_v1", "seed": seed,
         "epoch": epoch, "effective_updates": effective_updates, "budgets": choices},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(descriptor).digest(), "big"))
    cycles, remainder = divmod(effective_updates, len(choices))
    extras = list(choices)
    rng.shuffle(extras)
    schedule = list(choices) * cycles + extras[:remainder]
    rng.shuffle(schedule)
    return tuple(schedule)
