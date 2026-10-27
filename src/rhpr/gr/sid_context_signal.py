"""Frozen ALS preceding-context prototypes for TRAIN-only SID placement."""
from dataclasses import dataclass

import numpy as np

from rhpr.gr.sid_graph_swap import NeighborGraph


@dataclass(frozen=True)
class ContextSignal:
    vectors: np.ndarray
    contexts: np.ndarray
    users: np.ndarray
    length_sum: np.ndarray
    embedded_sum: np.ndarray
    report: dict


def context_prototypes(users, item_vectors, window, event_eligible=None):
    """Mean within each causal H, then equal-weight mean over H ending in i.

    ALS factors are frozen, unnormalized inputs. Only final prototype cosine is
    normalized. Earlier occurrences of i are legitimate observed context events.
    First events have no context and are not targets. Missing factor rows are
    excluded from the mean denominator, never replaced with artificial PADs.
    """
    z = np.asarray(item_vectors, dtype=np.float64)
    if (z.ndim != 2 or not np.isfinite(z).all() or type(window) is not int
            or window < 1):
        raise ValueError('finite item matrix and positive context window required')
    n = len(z)
    valid = np.linalg.norm(z, axis=1) > 0
    if event_eligible is not None:
        mask = np.asarray(event_eligible)
        if mask.shape != (n,) or mask.dtype.kind != 'b':
            raise ValueError('one boolean event eligibility per item required')
        valid &= mask
    z = z.copy()
    z[~valid] = 0
    total = np.zeros_like(z)
    counts = np.zeros(n, dtype=np.int64)
    distinct = counts.copy()
    lengths = counts.copy()
    embedded = counts.copy()
    raw = skipped = partial = 0
    length_hist = np.zeros(window + 1, dtype=np.int64)
    for user in users:
        ids = np.asarray(user.train)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any((ids < 0) | (ids >= n)):
            raise ValueError('TRAIN item outside catalogue')
        if len(ids) < 2:
            continue
        # Prefix sums avoid rescanning the full H at every TRAIN position.
        sums = np.vstack([np.zeros((1, z.shape[1])), np.cumsum(z[ids], axis=0)])
        mass = np.r_[0, np.cumsum(valid[ids])]
        position = np.arange(1, len(ids))
        start = np.maximum(0, position - window)
        denominator = mass[position] - mass[start]
        keep = denominator > 0
        raw += len(position)
        skipped += int((~keep).sum())
        target = ids[1:][keep]
        size = (position - start)[keep]
        denom = denominator[keep]
        context = (sums[position[keep]] - sums[start[keep]]) / denom[:, None]
        np.add.at(total, target, context)
        np.add.at(counts, target, 1)
        distinct[np.unique(target)] += 1
        np.add.at(lengths, target, size)
        np.add.at(embedded, target, denom)
        np.add.at(length_hist, size, 1)
        partial += int((size != denom).sum())
    mean = np.divide(total, counts[:, None], out=np.zeros_like(total), where=counts[:, None] > 0)
    return ContextSignal(mean.astype(np.float32), counts, distinct, lengths, embedded,
                         dict(window=window, train_target_positions=raw,
                              usable_contexts=int(counts.sum()), skipped_empty_factor_contexts=skipped,
                              partially_embedded_contexts=partial, eligible_event_items=int(valid.sum()),
                              items_without_context=int((counts == 0).sum()),
                              context_length_histogram=length_hist.tolist(),
                              aggregation='raw ALS mean within H; equal context weight across H',
                              validation_and_test_used=False))


def context_confidence(signal, minimum_users=2, shrinkage=10.):
    if (type(minimum_users) is not int or minimum_users < 1
            or not np.isfinite(shrinkage) or shrinkage < 0):
        raise ValueError('positive minimum users and nonnegative shrinkage required')
    support = signal.users
    return np.where(support >= minimum_users, support / np.maximum(support + shrinkage, 1), 0.)


def context_neighbors(signal, k, minimum_users=2, shrinkage=10., block=256,
                      ranking='cosine'):
    """Positive cosine neighbors, with confidence-weighted edge importance.

    Default top-k follows similarity among eligible endpoints, then confidence
    weights the selected edges. Optional weighted_cosine ranks by edge weight;
    these are different graphs and must not be conflated. The latter favors
    popular targets and is retained as an audited sensitivity, not the default.
    """
    z = np.asarray(signal.vectors, dtype=np.float64)
    if type(k) is not int or not 0 < k < len(z) or type(block) is not int or block < 1:
        raise ValueError('valid k and positive block required')
    if ranking not in {'cosine', 'weighted_cosine'}:
        raise ValueError('unknown neighbor ranking policy')
    confidence = context_confidence(signal, minimum_users, shrinkage)
    norm = np.linalg.norm(z, axis=1)
    z = z / np.maximum(norm[:, None], 1e-30)
    z[confidence == 0] = 0
    source, target, weight = [], [], []
    for begin in range(0, len(z), block):
        score = z[begin:begin + block] @ z.T
        if ranking == 'weighted_cosine':
            score *= confidence[begin:begin + len(score), None] * confidence[None, :]
        score[np.arange(len(score)), begin + np.arange(len(score))] = -np.inf
        for offset, row in enumerate(score):
            # Stable tie handling also at the kth boundary.
            ids = np.flatnonzero(row > 0)
            ids = ids[np.lexsort((ids, -row[ids]))[:k]]
            source.extend([begin + offset] * len(ids))
            target.extend(ids)
            values = row[ids]
            if ranking == 'cosine':
                values = values * confidence[begin + offset] * confidence[ids]
            weight.extend(values)
    return NeighborGraph(np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64),
                         np.asarray(weight, dtype=np.float64), len(z))


def prototype_reliability(fit, heldout):
    """User-disjoint TRAIN prototype agreement, stratified by FIT user support."""
    a, b = fit.vectors, heldout.vectors
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(np.einsum('ij,ij->i', a, b), denominator,
                       out=np.zeros(len(a)), where=denominator > 0)
    rows = []
    for lo, hi in [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, None)]:
        cohort = (fit.users >= lo) & (fit.users <= hi if hi is not None else True)
        comparable = cohort & (denominator > 0)
        rows.append(dict(fit_users_min=lo, fit_users_max=hi, items=int(cohort.sum()),
                         comparable_items=int(comparable.sum()),
                         mean_cosine=float(cosine[comparable].mean()) if comparable.any() else None,
                         median_cosine=float(np.median(cosine[comparable])) if comparable.any() else None))
    return rows
