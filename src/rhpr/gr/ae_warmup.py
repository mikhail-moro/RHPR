"""Paper-inspired spectral stopping for continuous AE warm-up.

2605.06870, section 4 / appendix C.1: Gaussian reverse water-filling
on the centred catalogue PCA spectrum. For residual books, using the sum
of nominal book rates is an experimental adaptation, not a proved bound.
"""
import math

import numpy as np


def water_filling(eigenvalues, rate_bits):
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("eigenvalues must be finite nonnegative vector")
    if not math.isfinite(rate_bits) or rate_bits < 0:
        raise ValueError("rate must be finite nonnegative")
    positive = values[values > 0]
    if not len(positive):
        return {"rate_bits": rate_bits, "water_level": 0., "active_modes": 0}
    # Solve in log space: sum(max(log2(lambda)-log2(D),0))/2 = R.
    logs = np.log2(positive)
    high, low = float(logs.max()), float(logs.min() - 2 * rate_bits - 1)
    for _ in range(100):
        middle = (low + high) / 2
        if np.maximum(logs - middle, 0).sum() / 2 > rate_bits:
            low = middle
        else:
            high = middle
    level = (low + high) / 2
    return {"rate_bits": float(rate_bits), "water_level": float(2. ** level),
        "active_modes": int((logs > level + 1e-10).sum())}


def latent_spectrum(latents, codebook_size, levels):
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("latent PCA requires at least two finite catalogue rows")
    x = x - x.mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(x.T @ x / (len(x)-1))[::-1], 0)
    total = float(eigenvalues.sum())
    d99 = int(np.searchsorted(np.cumsum(eigenvalues), .99 * total) + 1) if total > 0 else 0
    return {"eigenvalues": eigenvalues.tolist(), "effective_dimension_99pct": d99,
        "total_variance": total,
        "single_book": water_filling(eigenvalues, math.log2(codebook_size)),
        "aggregate_nominal_rate": water_filling(eigenvalues, levels * math.log2(codebook_size)),
        "scope": "full catalogue, centred eval latents; residual aggregate rate is heuristic"}


class SpectralPlateau:
    def __init__(self, spec, max_steps):
        self.spec = dict(spec)
        if spec.get("kind") != "water_filling_plateau":
            raise ValueError("unsupported AE warm-up stopping criterion")
        for key in ("min_steps", "patience_probes"):
            if type(spec.get(key)) is not int or spec[key] < 1:
                raise ValueError(f"warm-up {key} must be positive integer")
        if spec["min_steps"] > max_steps or spec["patience_probes"] < 2:
            raise ValueError("warm-up minimum/patience exceeds sensible budget")
        self.rows = []

    def observe(self, step, spectrum):
        self.rows.append({"step": step, **spectrum})
        eligible = [r for r in self.rows if r["step"] >= self.spec["min_steps"]]
        tail = eligible[-self.spec["patience_probes"]:]
        # Exact stability is explicit: neither a declining count nor a one-probe
        # maximum qualifies. The bounded cap is reported separately from plateau.
        return len(tail) == self.spec["patience_probes"] and len({
            r["aggregate_nominal_rate"]["active_modes"] for r in tail}) == 1
