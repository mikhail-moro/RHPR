"""Target-aware ceilings for repairing decisions of a FROZEN AR model.

This is an optimistic diagnostic, never a usable recommendation method.
Only the conditional distribution at the TRUE parent prefix is repaired;
other parent distributions and all model weights remain unchanged.
"""
import numpy as np


def repair_true_decisions(scores, sids, target_item, true_logprobs, levels):
    scores = np.asarray(scores, dtype=np.float64)
    sids = np.asarray(sids)
    logp = np.asarray(true_logprobs, dtype=np.float64)
    if scores.shape != (len(sids),) or logp.shape != (sids.shape[1],):
        raise ValueError("Scores, IDs and per-head log probabilities must align")
    if not 0 <= target_item < len(scores) or not np.isfinite(scores).all() or not np.isfinite(logp).all() or (logp > 2e-5).any():
        raise ValueError("Expected finite original log scores/probabilities and a valid target")
    if len(set(levels)) != len(levels) or any(type(l) is not int or not 0 <= l < sids.shape[1] for l in levels):
        raise ValueError("Repair levels must be distinct valid integers")
    result = scores.copy()
    true = sids[target_item]
    for level in levels:
        parent = np.all(sids[:, :level] == true[:level], axis=1)
        correct_child = parent & (sids[:, level] == true[level])
        result[parent & ~correct_child] = -np.inf
        result[correct_child] -= logp[level]
    return result


def target_rank(scores, target_item):
    scores = np.asarray(scores)
    value = scores[target_item]
    if not np.isfinite(value):
        raise ValueError("Target must retain a finite score")
    return int(1 + (scores > value).sum() + ((scores == value) & (np.arange(len(scores)) < target_item)).sum())
