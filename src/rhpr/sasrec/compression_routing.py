"""Small, reproducible policies for exploratory compression-budget routing.

Fit policies on development users and evaluate their decisions on different
users. The utilities supplied to ``fit_stump_router`` are training labels;
``predict_router`` consumes only history features. Separate pretrained arms
remain separate models, so this analysis does not isolate compression depth.
"""

from __future__ import annotations

import hashlib
import operator
from typing import Any, Sequence

import numpy as np


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _user_hashes(user_ids: Sequence[int] | np.ndarray, seed: int) -> list[int]:
    identifiers = np.asarray(user_ids)
    if identifiers.ndim != 1:
        raise ValueError("user_ids must be one-dimensional")
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    result = []
    for identifier in identifiers:
        user_id = _integer(identifier, "user_id")
        payload = f"{seed}:{user_id}".encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=8, person=b"rhpr-routing-v1").digest()
        result.append(int.from_bytes(digest, "big"))
    return result


def user_hash_split(
    user_ids: Sequence[int] | np.ndarray,
    seed: int = 42,
    train_fraction: float = 0.5,
) -> np.ndarray:
    """Return a development mask determined by UID, independent of row order.

    The fraction is probabilistic rather than an exact sample-size guarantee.
    Repeated occurrences of a user always receive the same assignment.
    """
    if not np.isfinite(train_fraction) or not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be finite and between zero and one")
    threshold = int(train_fraction * (1 << 64))
    return np.asarray([value < threshold for value in _user_hashes(user_ids, seed)], dtype=bool)


def _features(X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("X must be a finite two-dimensional array")
    return features


def _scores(scores: np.ndarray, *, n_users: int | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("utilities must be a nonempty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("utilities must contain only finite values")
    if n_users is not None and values.shape[0] != n_users:
        raise ValueError("X and utilities must have the same number of users")
    return values


def fit_stump_router(
    X: np.ndarray,
    utilities: np.ndarray,
    feature_names: Sequence[str],
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    min_leaf: int = 200,
    arm_labels: Sequence[int] = (4, 8, 16),
) -> dict[str, Any]:
    """Maximize development utility over a constant policy and shallow stumps.

    Each admissible threshold chooses its best arm separately in its two
    leaves. Arm ties prefer the first column (four summaries by default).
    Equivalent split policies never replace the constant policy. Thresholds
    and utility means are learned exclusively from the supplied users.
    """
    features = _features(X)
    values = _scores(utilities, n_users=features.shape[0])
    names = list(feature_names)
    labels = [_integer(label, "arm_label") for label in arm_labels]
    if len(names) != features.shape[1] or not all(isinstance(name, str) for name in names):
        raise ValueError("feature_names must name every column of X")
    if len(labels) != values.shape[1] or len(set(labels)) != len(labels):
        raise ValueError("arm_labels must uniquely label every utility column")
    min_leaf = _integer(min_leaf, "min_leaf")
    if min_leaf < 1:
        raise ValueError("min_leaf must be positive")
    cuts = np.asarray(tuple(quantiles), dtype=np.float64)
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or ((cuts <= 0) | (cuts >= 1)).any():
        raise ValueError("quantiles must be finite and strictly between zero and one")

    means = values.mean(axis=0)
    arm = int(np.argmax(means))
    best = float(means[arm])
    policy: dict[str, Any] = {
        "kind": "constant", "arm_index": arm, "arm_labels": labels,
        "feature_names": names, "n_features": int(features.shape[1]),
        "n_train_users": int(features.shape[0]), "min_leaf": min_leaf,
        "quantiles": cuts.tolist(), "train_utility": best,
    }
    valid_splits = 0
    for feature_index in range(features.shape[1]):
        thresholds = np.unique(np.quantile(features[:, feature_index], cuts)) if cuts.size else []
        for threshold in thresholds:
            left = features[:, feature_index] <= threshold
            n_left = int(left.sum())
            n_right = features.shape[0] - n_left
            if min(n_left, n_right) < min_leaf:
                continue
            valid_splits += 1
            left_means = values[left].mean(axis=0)
            right_means = values[~left].mean(axis=0)
            left_arm = int(np.argmax(left_means))
            right_arm = int(np.argmax(right_means))
            utility = float((n_left * left_means[left_arm] + n_right * right_means[right_arm]) / features.shape[0])
            # Floating-point summation order alone must not create a split.
            if utility > best + 1e-12:
                best = utility
                policy = {
                    **{key: value for key, value in policy.items() if key != "arm_index"},
                    "kind": "stump", "feature_index": feature_index,
                    "feature_name": names[feature_index], "threshold": float(threshold),
                    "left_arm_index": left_arm, "right_arm_index": right_arm,
                    "left_train_users": n_left, "right_train_users": n_right,
                    "train_utility": best,
                }
    policy["n_candidate_splits"] = valid_splits
    policy["n_candidates"] = 1 + valid_splits
    return policy


def fit_interval_router(
    X: np.ndarray,
    utilities: np.ndarray,
    feature_names: Sequence[str],
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    min_leaf: int = 200,
    arm_labels: Sequence[int] = (4, 8, 16),
) -> dict[str, Any]:
    """Extend the fixed stump bank with two cuts on a single history feature.

    The three leaves can capture a nonmonotonic conditional utility pattern.
    All thresholds and leaf arms are fitted on development users only; the
    bank includes constants and stumps, which win equal-utility ties.
    """
    policy = fit_stump_router(
        X, utilities, feature_names, quantiles=quantiles,
        min_leaf=min_leaf, arm_labels=arm_labels,
    )
    features, values = _features(X), _scores(utilities)
    best = policy["train_utility"]
    cuts = np.asarray(policy["quantiles"], dtype=np.float64)
    metadata_keys = (
        "arm_labels", "feature_names", "n_features", "n_train_users",
        "min_leaf", "quantiles", "n_candidate_splits",
    )
    metadata = {key: policy[key] for key in metadata_keys}
    valid_intervals = 0
    for feature_index in range(features.shape[1]):
        thresholds = np.unique(np.quantile(features[:, feature_index], cuts)) if cuts.size else []
        column = features[:, feature_index]
        for lower_index in range(len(thresholds)):
            for upper_index in range(lower_index + 1, len(thresholds)):
                low, high = thresholds[lower_index], thresholds[upper_index]
                leaves = (column <= low, (column > low) & (column <= high), column > high)
                counts = [int(leaf.sum()) for leaf in leaves]
                if min(counts) < min_leaf:
                    continue
                valid_intervals += 1
                leaf_means = [values[leaf].mean(axis=0) for leaf in leaves]
                arms = [int(np.argmax(means)) for means in leaf_means]
                utility = float(sum(count * means[arm] for count, means, arm in zip(counts, leaf_means, arms)) / len(features))
                if utility > best + 1e-12:
                    best = utility
                    policy = {
                        **metadata, "kind": "interval", "feature_index": feature_index,
                        "feature_name": metadata["feature_names"][feature_index],
                        "thresholds": [float(low), float(high)], "arm_indices": arms,
                        "leaf_train_users": counts, "train_utility": best,
                    }
    policy["n_candidate_intervals"] = valid_intervals
    policy["n_candidates"] = 1 + metadata["n_candidate_splits"] + valid_intervals
    return policy


def predict_router(policy: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Choose arm indices using history features, without outcome labels."""
    features = _features(X)
    if features.shape[1] != policy["n_features"]:
        raise ValueError("X feature count differs from the fitted router")
    if policy["kind"] == "constant":
        return np.full(features.shape[0], policy["arm_index"], dtype=np.int64)
    if policy["kind"] == "interval":
        leaves = np.searchsorted(policy["thresholds"], features[:, policy["feature_index"]], side="left")
        return np.asarray(policy["arm_indices"], dtype=np.int64)[leaves]
    if policy["kind"] != "stump":
        raise ValueError("unknown router policy kind")
    left = features[:, policy["feature_index"]] <= policy["threshold"]
    return np.where(left, policy["left_arm_index"], policy["right_arm_index"]).astype(np.int64)


def crossfit_stump_router(
    user_ids: Sequence[int] | np.ndarray,
    X: np.ndarray,
    utilities: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 42,
    quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    min_leaf: int = 200,
    arm_labels: Sequence[int] = (4, 8, 16),
    policy_class: str = "stump",
) -> dict[str, Any]:
    """Secondary stability check; every user's prediction excludes its labels.

    Returns NumPy ``predicted_arms``/``fold_ids`` and JSON-compatible policies.
    This does not replace a fixed held-out evaluation. Repeated UIDs share a
    fold, preventing the same user's labels from crossing the fold boundary.
    """
    features = _features(X)
    values = _scores(utilities, n_users=features.shape[0])
    if policy_class not in ("stump", "interval"):
        raise ValueError("policy_class must be stump or interval")
    fitter = fit_stump_router if policy_class == "stump" else fit_interval_router
    n_folds = _integer(n_folds, "n_folds")
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    hashes = _user_hashes(user_ids, seed)
    if len(hashes) != features.shape[0]:
        raise ValueError("user_ids and X must have the same number of users")
    fold_ids = np.asarray([value % n_folds for value in hashes], dtype=np.int64)
    predictions = np.empty(features.shape[0], dtype=np.int64)
    policies = []
    for fold in range(n_folds):
        heldout = fold_ids == fold
        if not heldout.any():
            continue
        if heldout.all():
            raise ValueError("hash folds leave no development users")
        policy = fitter(
            features[~heldout], values[~heldout], feature_names,
            quantiles=quantiles, min_leaf=min_leaf, arm_labels=arm_labels,
        )
        predictions[heldout] = predict_router(policy, features[heldout])
        policies.append({"fold_index": fold, "n_heldout_users": int(heldout.sum()), "policy": policy})
    return {"predicted_arms": predictions, "fold_ids": fold_ids, "policies": policies,
            "n_folds": n_folds, "seed": int(seed), "policy_class": policy_class}


def paired_mean_bootstrap(
    delta: np.ndarray,
    seed: int = 42,
    draws: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, int | float]:
    """Percentile CI for a paired user-level mean, conditional on a fixed policy.

    For learned policies, pass only held-out differences. This bootstrap does
    not incorporate policy-training uncertainty or repeated-seed variation.
    """
    differences = np.asarray(delta, dtype=np.float64)
    if differences.ndim != 1 or not differences.size or not np.isfinite(differences).all():
        raise ValueError("delta must be a nonempty finite one-dimensional array")
    draws = _integer(draws, "draws")
    seed = _integer(seed, "seed")
    if draws < 1 or seed < 0:
        raise ValueError("draws must be positive and seed nonnegative")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    chunk_size = max(1, min(256, 1_000_000 // differences.size))
    for start in range(0, draws, chunk_size):
        size = min(chunk_size, draws - start)
        indices = rng.integers(0, differences.size, size=(size, differences.size))
        bootstrap_means[start:start + size] = differences[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(bootstrap_means, [alpha, 1 - alpha])
    return {"mean": float(differences.mean()), "ci_low": float(low), "ci_high": float(high),
            "n_users": int(differences.size), "draws": draws,
            "confidence_level": float(confidence_level), "seed": seed}


def routing_evaluation(
    scores: np.ndarray,
    predicted_arms: np.ndarray,
    reference_arm: int = 1,
    batch_latency_ms: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate decisions and an outcome-independent mixture with equal shares.

    Mixture utility uses the expectation over independent arm assignments,
    not a noisy sampled assignment. Latencies must use comparable batch sizes
    and measurement conditions. Their mixture is a cost estimate; it does not
    measure batch fragmentation, routing overhead, or loading separate models.
    """
    values = _scores(scores)
    predictions = np.asarray(predicted_arms)
    if predictions.ndim != 1 or predictions.shape[0] != values.shape[0] or predictions.dtype.kind not in "iu":
        raise ValueError("predicted_arms must contain one integer arm index per user")
    if ((predictions < 0) | (predictions >= values.shape[1])).any():
        raise ValueError("predicted arm index is out of range")
    reference_arm = _integer(reference_arm, "reference_arm")
    if not 0 <= reference_arm < values.shape[1]:
        raise ValueError("reference arm index is out of range")
    selected = values[np.arange(values.shape[0]), predictions]
    reference = values[:, reference_arm]
    shares = np.bincount(predictions, minlength=values.shape[1]) / values.shape[0]
    mixture = values @ shares
    result: dict[str, Any] = {
        "n_users": int(values.shape[0]), "reference_arm_index": reference_arm,
        "mean_selected_score": float(selected.mean()), "mean_reference_score": float(reference.mean()),
        "mean_matched_mixture_score": float(mixture.mean()),
        "selected_minus_reference": float((selected - reference).mean()),
        "selected_minus_matched_mixture": float((selected - mixture).mean()),
        "arm_shares": shares.tolist(),
    }
    if batch_latency_ms is not None:
        latencies = np.asarray(batch_latency_ms, dtype=np.float64)
        if latencies.shape != (values.shape[1],) or not np.isfinite(latencies).all() or (latencies < 0).any():
            raise ValueError("batch_latency_ms must give finite nonnegative latency for every arm")
        result["expected_batch_latency_ms"] = float(shares @ latencies)
        result["reference_batch_latency_ms"] = float(latencies[reference_arm])
    return result


def matched_mixture_bootstrap(
    scores: np.ndarray,
    predicted_arms: np.ndarray,
    seed: int = 42,
    draws: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, int | float | str]:
    """Bootstrap router-minus-mixture utility, recomputing each sample's shares.

    Resample complete user rows jointly across the pretrained arms. Unlike
    bootstrapping a difference vector with fixed shares, this includes sampling
    variation of the router's assignment fractions. The fitted router remains
    fixed; development fitting and model-training uncertainty are not included.
    """
    values = _scores(scores)
    predictions = np.asarray(predicted_arms)
    evaluation = routing_evaluation(values, predictions, reference_arm=0)
    draws, seed = _integer(draws, "draws"), _integer(seed, "seed")
    if draws < 1 or seed < 0:
        raise ValueError("draws must be positive and seed nonnegative")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    rng = np.random.default_rng(seed)
    selected = values[np.arange(len(values)), predictions]
    bootstrap_deltas = np.empty(draws, dtype=np.float64)
    chunk_size = max(1, min(256, 1_000_000 // len(values)))
    for start in range(0, draws, chunk_size):
        size = min(chunk_size, draws - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        sampled_arms = predictions[indices]
        shares = np.column_stack([(sampled_arms == arm).mean(axis=1) for arm in range(values.shape[1])])
        arm_means = np.column_stack([values[:, arm][indices].mean(axis=1) for arm in range(values.shape[1])])
        bootstrap_deltas[start:start + size] = selected[indices].mean(axis=1) - (shares * arm_means).sum(axis=1)
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(bootstrap_deltas, [alpha, 1 - alpha])
    return {"mean": evaluation["selected_minus_matched_mixture"], "ci_low": float(low),
            "ci_high": float(high), "n_users": len(values), "draws": draws,
            "confidence_level": float(confidence_level), "seed": seed,
            "resampling": "user_rows_with_assignment_shares_recomputed"}
