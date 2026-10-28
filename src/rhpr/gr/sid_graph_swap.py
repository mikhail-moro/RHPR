"""CPU-only item placement on a fixed SID trie; no generator or encoder fit.

Graphs and contrastive samples must be frozen before search. A move exchanges
complete item paths, including the uniqueness suffix. Centroid decoding is no
longer the assignment rule after a move; the exported item->SID table is.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import logsumexp


def lcp_similarity(codes, left, right, depth=3):
    """Normalized *prefix* length, not Hamming similarity or code-number distance."""
    equal = codes[np.asarray(left), :depth] == codes[np.asarray(right), :depth]
    return np.cumprod(equal, axis=-1).sum(axis=-1) / depth


@dataclass(frozen=True)
class NeighborGraph:
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    items: int

    def __post_init__(self):
        a, b, w = self.source, self.target, self.weight
        if (a.ndim != 1 or b.shape != a.shape or w.shape != a.shape
                or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
                or self.items < 2 or not len(a) or np.any(a == b)
                or np.any(a < 0) or np.any(b < 0)
                or np.any(a >= self.items) or np.any(b >= self.items)
                or not np.isfinite(w).all() or np.any(w <= 0)):
            raise ValueError('graph requires finite positive, nonself catalogue edges')

    def prefix_scores(self, codes, depth=3):
        eq = np.cumprod(codes[self.source, :depth] == codes[self.target, :depth], axis=1)
        return (self.weight @ eq / self.weight.sum()).tolist()

    def score(self, codes, depth=3):
        return float(np.mean(self.prefix_scores(codes, depth)))

    def macro_score(self, codes, depth=3):
        mass = np.bincount(self.source, weights=self.weight, minlength=self.items)
        value = np.bincount(self.source, weights=self.weight *
                           lcp_similarity(codes, self.source, self.target, depth),
                           minlength=self.items)
        return float(np.mean(value[mass > 0] / mass[mass > 0]))

    def top(self, k):
        if type(k) is not int or k < 1:
            raise ValueError('positive k required')
        # Builders emit each source in descending weight order, tie by target ID.
        order = np.lexsort((self.target, -self.weight, self.source))
        a = self.source[order]
        start = np.maximum.accumulate(np.where(np.r_[True, a[1:] != a[:-1]],
                                              np.arange(len(a)), 0))
        keep = order[np.arange(len(a)) - start < k]
        return NeighborGraph(self.source[keep], self.target[keep], self.weight[keep], self.items)


def cosine_neighbors(embeddings, k, block=256):
    """Exact blocked top-k, positive cosine only; never allocates N x N."""
    z = np.asarray(embeddings, dtype=np.float32)
    if z.ndim != 2 or not np.isfinite(z).all() or not 0 < k < len(z):
        raise ValueError('finite embeddings and 0 < k < catalogue size required')
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-30)
    aa, bb, ww = [], [], []
    for begin in range(0, len(z), block):
        end = min(begin + block, len(z))
        score = np.clip(z[begin:end] @ z.T, -1, 1)
        score[np.arange(end-begin), np.arange(begin, end)] = -np.inf
        selected = np.argpartition(score, -k, axis=1)[:, -k:]
        for offset, ids in enumerate(selected):
            ids = ids[np.lexsort((ids, -score[offset, ids]))]
            ids = ids[score[offset, ids] > 0]
            aa.extend([begin+offset]*len(ids)); bb.extend(ids); ww.extend(score[offset, ids])
    return NeighborGraph(np.array(aa, dtype=np.int64), np.array(bb, dtype=np.int64),
                         np.array(ww, dtype=np.float64), len(z))


def couser_neighbors(users, items, k, minimum_users=2, block=256):
    """Binary TRAIN incidence cosine; no validation/test access, sparse products."""
    row, col = [], []
    for u, user in enumerate(users):
        ids = np.unique(user.train)
        if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= items):
            raise ValueError('TRAIN item outside catalogue')
        row.extend(ids); col.extend([u]*len(ids))
    x = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(items, len(users)))
    frequency = np.asarray(x.sum(1)).ravel()
    eligible = frequency >= minimum_users
    x = sparse.diags(eligible / np.sqrt(np.maximum(frequency, 1))) @ x
    aa, bb, ww = [], [], []
    for begin in range(0, items, block):
        score = (x[begin:begin+block] @ x.T).tocsr()
        for offset in range(score.shape[0]):
            lo, hi = score.indptr[offset:offset+2]
            ids, weights = score.indices[lo:hi], score.data[lo:hi]
            keep = (ids != begin+offset) & (weights > 0)
            ids, weights = ids[keep], weights[keep]
            order = np.lexsort((ids, -weights))[:k]
            aa.extend([begin+offset]*len(order)); bb.extend(ids[order]); ww.extend(weights[order])
    return NeighborGraph(np.array(aa, dtype=np.int64), np.array(bb, dtype=np.int64),
                         np.array(ww, dtype=np.float64), items), frequency


def _incidence(vertices, items):
    """Item -> all affected rows; includes incoming and negative incidences."""
    v = np.asarray(vertices)
    rows = np.repeat(np.arange(len(v)), v.shape[1])
    matrix = sparse.csr_matrix((np.ones(v.size, dtype=np.int8), (v.ravel(), rows)),
                               shape=(items, len(v)))
    matrix.sum_duplicates()
    return matrix


class PairTerm:
    def __init__(self, graph, depth=3):
        self.graph, self.depth = graph, depth
        self.mass = graph.weight / graph.weight.sum()
        self.incidence = _incidence(np.stack([graph.source, graph.target], axis=1), graph.items)

    def values(self, codes, rows=None):
        r = slice(None) if rows is None else rows
        g = self.graph
        return self.mass[r] * lcp_similarity(codes, g.source[r], g.target[r], self.depth)


class ContrastiveTerm:
    """Fixed sampled InfoNCE with normalized LCP logits, not a PLUM reproduction.

    One or two weight-sampled positive neighbours per anchor; unique uniform
    catalogue negatives exclude anchor and *all* graph positives. Non-neighbours
    are unlabeled candidates, not asserted true negatives. Each anchor has equal
    mass. Return (log(M)-CE)/log(M); this skill is NOT bounded below by zero.
    """
    def __init__(self, graph, seed, negatives=16, positives=2, temperature=.2, depth=3):
        if negatives < 1 or positives < 1 or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive sampling counts and temperature required')
        self.temperature, self.depth = temperature, depth
        rng = np.random.default_rng(seed)
        order = np.argsort(graph.source, kind='stable')
        sources = graph.source[order]
        starts = np.r_[0, np.flatnonzero(sources[1:] != sources[:-1])+1, len(sources)]
        anchors, candidates, mass = [], [], []
        for lo, hi in zip(starts[:-1], starts[1:]):
            ids = order[lo:hi]; anchor = int(sources[lo]); neighbours = graph.target[ids]
            count = min(positives, len(ids))
            chosen = rng.choice(len(ids), size=count, replace=False,
                                p=graph.weight[ids]/graph.weight[ids].sum())
            excluded = set(map(int, neighbours)); excluded.add(anchor)
            if graph.items-len(excluded) < negatives:
                raise ValueError('not enough distinct negative candidates')
            for chosen_id in chosen:
                negative, seen = [], set(excluded)
                while len(negative) < negatives:
                    candidate = int(rng.integers(graph.items))
                    if candidate not in seen:
                        negative.append(candidate); seen.add(candidate)
                anchors.append(anchor)
                candidates.append([int(neighbours[chosen_id]), *negative]); mass.append(1/count)
        self.anchor = np.asarray(anchors, dtype=np.int64)
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.mass = np.asarray(mass)/len(starts[:-1])
        self.uniform_loss = float(np.log(negatives+1))
        self.incidence = _incidence(np.c_[self.anchor, self.candidates], graph.items)

    def values(self, codes, rows=None):
        r = slice(None) if rows is None else rows
        # Broadcasting item IDs requires codes[:, :depth] before advanced indexing.
        anchor = codes[self.anchor[r], :self.depth][:, None, :]
        candidates = codes[self.candidates[r], :self.depth]
        similarity = np.cumprod(anchor == candidates, axis=-1).sum(-1)/self.depth
        logits = similarity/self.temperature
        loss = logsumexp(logits, axis=1)-logits[:, 0]
        return self.mass[r]*(self.uniform_loss-loss)/self.uniform_loss

    def loss(self, codes):
        return float(self.uniform_loss*(1-self.values(codes).sum()))


class SwapSearch:
    """Exact cached local deltas. Callers can audit against full objective scans."""
    def __init__(self, sids, terms, coefficients):
        self.codes = np.asarray(sids).copy()
        if (self.codes.ndim != 2 or self.codes.dtype.kind not in 'iu'
                or len(np.unique(self.codes, axis=0)) != len(self.codes)):
            raise ValueError('complete unique integer SID paths required')
        self.terms = list(terms)
        self.coefficients = np.asarray(coefficients, dtype=float)
        if (len(terms) != len(coefficients) or not np.isfinite(self.coefficients).all()
                or np.any(self.coefficients < 0) or self.coefficients.sum() <= 0):
            raise ValueError('nonnegative finite coefficients with positive total required')
        self.cache = [term.values(self.codes) for term in self.terms]
        self.totals = np.array([v.sum() for v in self.cache])
        self.owner = np.arange(len(self.codes))
        self.accepted = 0

    @property
    def score(self):
        return float(self.coefficients @ self.totals)

    def try_swap(self, a, b, force=False):
        if not 0 <= a < len(self.codes) or not 0 <= b < len(self.codes):
            raise ValueError('swap item outside catalogue')
        if a == b:
            return False, 0.
        rows = []
        for t in self.terms:
            m = t.incidence
            rows.append(np.union1d(m.indices[m.indptr[a]:m.indptr[a+1]],
                                  m.indices[m.indptr[b]:m.indptr[b+1]]))
        self.codes[[a, b]] = self.codes[[b, a]]
        new = [t.values(self.codes, r) for t, r in zip(self.terms, rows)]
        delta = np.array([float((v-c[r]).sum()) for v, c, r in zip(new, self.cache, rows)])
        improvement = float(self.coefficients @ delta)
        accept = bool(force or improvement > 1e-14)
        if accept:
            for c, r, v in zip(self.cache, rows, new):
                c[r] = v
            self.totals += delta
            self.owner[[a, b]] = self.owner[[b, a]]
            self.accepted += 1
        else:
            self.codes[[a, b]] = self.codes[[b, a]]
        return accept, improvement

    def audit(self, initial):
        np.testing.assert_array_equal(self.codes, initial[self.owner])
        np.testing.assert_array_equal(np.sort(self.owner), np.arange(len(self.owner)))
        actual = [term.values(self.codes).sum() for term in self.terms]
        np.testing.assert_allclose(self.totals, actual, atol=2e-12, rtol=1e-10)
        return float(self.coefficients @ actual)


def heldout_gain(codes, fit_proxy, evaluation_proxy):
    """Train the count predictor on fit users, score only unseen TRAIN users."""
    _, labels = np.unique(codes, return_inverse=True)
    k = len(np.unique(labels)); fit_y = labels[fit_proxy.dst]; y = labels[evaluation_proxy.dst]
    prior = (np.bincount(fit_y, minlength=k)+.5)/(len(fit_y)+.5*k)
    keys, mass = np.unique(fit_proxy.src*k+fit_y, return_counts=True)
    query = evaluation_proxy.src*k+y
    pos = np.searchsorted(keys, query); valid = pos < len(keys)
    valid[valid] = keys[pos[valid]] == query[valid]
    matched = np.zeros(len(query)); matched[valid] = mass[pos[valid]]
    total = np.bincount(fit_proxy.src, minlength=len(codes))
    prob = (matched+10*prior[y])/(total[evaluation_proxy.src]+10)
    unigram, conditional = float(-np.log2(prior[y]).mean()), float(-np.log2(prob).mean())
    return dict(heldout_gain_bits=unigram-conditional, heldout_unigram_nll_bits=unigram,
                heldout_conditional_nll_bits=conditional,
                heldout_relative_nll_reduction=1-conditional/unigram)
