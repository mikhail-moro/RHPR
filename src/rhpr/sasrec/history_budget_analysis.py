"""Exploratory user-level analysis of fixed pretrained compression models.

Only past-context features enter routing. Different models' future-target
utilities are outcomes, never inputs. A test-anchor oracle is an optimistic
upper bound, not an estimate of which compression a user actually needs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CORE_FEATURES = (
    "old_unique_fraction", "old_entropy_normalized", "recent_old_item_overlap_fraction",
    "old_duration_days", "log_train_event_count",
)
ARM_LABELS = (4, 8, 16)
COMPRESSION_CONTRASTS = ("pair16_minus_pair4", "pair16_minus_pair8", "pair8_minus_pair4")


def load_paired_scores(directory: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    names = ["recent128"] + [f"{kind}_m{budget}" for kind in ("mean", "uniform", "pair_mlp", "qformer")
                              for budget in (16, 8, 4)]
    reference_ids, reference_known, scores = None, None, {}
    for name in names:
        rows = json.loads((Path(directory) / f"{name}-test-users.json").read_text())
        if not rows or any(type(row.get("user_id")) is not int or row["user_id"] < 0 for row in rows):
            raise ValueError("paired records require nonempty nonnegative integer user IDs")
        rows.sort(key=lambda r: r["user_id"])
        ids = np.array([r["user_id"] for r in rows], dtype=np.int64)
        known = np.array([r["target_known"] for r in rows], dtype=np.bool_)
        if len(np.unique(ids)) != len(ids) or any(type(r["target_known"]) is not bool for r in rows):
            raise ValueError("duplicate user IDs or invalid target support flags")
        values = np.array([r["ndcg10"] for r in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite user utilities")
        for row in rows:
            rank = row.get("rank")
            if type(rank) is not int or not 0 <= rank <= 10:
                raise ValueError("invalid top10 rank")
            expected = 1 / math.log2(rank + 1) if rank else 0.
            if (not math.isclose(row["ndcg10"], expected, abs_tol=1e-12)
                    or row["hr10"] != float(rank > 0) or (not row["target_known"] and rank)):
                raise ValueError("rank, support and metric records disagree")
        if reference_ids is None:
            reference_ids, reference_known = ids, known
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(known, reference_known):
            raise ValueError("models must contain exactly the same users and target support")
        scores[name] = values
    return reference_ids, reference_known, scores


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("ranks require a nonempty finite vector")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts + 1) / 2)[inverse]


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p values must lie in [0,1]")
    if not values.size:
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.)
    return result.tolist()


def permutation_correlations(features: np.ndarray, outcomes: np.ndarray, *, feature_names: Sequence[str],
                             outcome_names: Sequence[str], seed: int = 42, draws: int = 5000) -> list[dict[str, Any]]:
    """Spearman associations; jointly shuffle outcomes to preserve their covariance.

    FDR is adjusted across the entire supplied feature×outcome family, including
    constant columns with p=1. The permutation p-value is two-sided and includes
    the observed configuration via the usual +1 correction.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(outcomes, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or x.shape[1] != len(feature_names) or y.shape[1] != len(outcome_names)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or type(draws) is not int or draws < 1 or type(seed) is not int or seed < 0):
        raise ValueError("invalid correlation matrix or permutation parameters")
    xr = np.column_stack([average_ranks(x[:, i]) for i in range(x.shape[1])])
    yr = np.column_stack([average_ranks(y[:, i]) for i in range(y.shape[1])])
    xr -= xr.mean(axis=0)
    yr -= yr.mean(axis=0)
    xnorm, ynorm = np.linalg.norm(xr, axis=0), np.linalg.norm(yr, axis=0)
    xr = np.divide(xr, xnorm, out=np.zeros_like(xr), where=xnorm > 0)
    yr = np.divide(yr, ynorm, out=np.zeros_like(yr), where=ynorm > 0)
    observed = xr.T @ yr
    counts = np.zeros(observed.shape, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        permuted = xr.T @ yr[rng.permutation(len(y))]
        counts += np.abs(permuted) >= np.abs(observed) - 1e-12
    p_values = (counts + 1) / (draws + 1)
    p_values[(xnorm == 0), :] = 1.
    p_values[:, (ynorm == 0)] = 1.
    adjusted = np.asarray(bh_adjust(p_values.ravel())).reshape(observed.shape)
    rows = []
    for i, feature in enumerate(feature_names):
        for j, outcome in enumerate(outcome_names):
            constant = bool(xnorm[i] == 0 or ynorm[j] == 0)
            rows.append({"feature": feature, "outcome": outcome, "users": len(y),
                         "spearman_rho": None if constant else float(observed[i, j]),
                         "constant_column": constant, "permutation_p": float(p_values[i, j]),
                         "bh_q_within_family": float(adjusted[i, j]), "permutations": draws,
                         "outcome_nonzero_users": int(np.count_nonzero(y[:, j])), "seed": seed})
    return rows


def feature_quartile_effects(features: np.ndarray, utilities: np.ndarray, *, feature_names: Sequence[str],
                            warm_mask: np.ndarray, seed: int = 42, draws: int = 2000) -> list[dict[str, Any]]:
    from rhpr.sasrec.compression_routing import paired_mean_bootstrap

    x, y = np.asarray(features), np.asarray(utilities)
    contrasts = np.column_stack((y[:, 2] - y[:, 0], y[:, 2] - y[:, 1], y[:, 1] - y[:, 0]))
    rows = []
    for i, feature in enumerate(feature_names):
        edges = np.quantile(x[:, i], (.25, .5, .75))
        labels = np.searchsorted(edges, x[:, i], side="right")
        for label in range(4):
            mask = labels == label
            if not mask.any():
                continue
            effect = {name: paired_mean_bootstrap(contrasts[mask, j], seed=seed + 100 * i + 10 * label + j, draws=draws)
                      for j, name in enumerate(COMPRESSION_CONTRASTS)}
            rows.append({"feature": feature, "quartile": label + 1, "edges": edges.tolist(),
                         "users": int(mask.sum()), "warm_users": int(np.count_nonzero(warm_mask[mask])),
                         "feature_min": float(x[mask, i].min()), "feature_max": float(x[mask, i].max()),
                         "pair_ndcg": {str(arm): float(y[mask, j].mean()) for j, arm in enumerate(ARM_LABELS)},
                         "contrasts": effect})
    return rows


def quartile_dependency_tests(features: np.ndarray, outcomes: np.ndarray, *, feature_names: Sequence[str],
                             outcome_names: Sequence[str], seed: int = 42, draws: int = 5000) -> list[dict[str, Any]]:
    """Secondary omnibus tests of differences across fixed feature quartiles.

    Between-group sum of squares detects nonmonotone conditional mean changes.
    Group boundaries depend only on past features. The complete feature/outcome
    family receives BH correction; pointwise subgroup intervals are not tests
    of treatment heterogeneity. This follow-up is exploratory.
    """
    x, y = np.asarray(features, dtype=np.float64), np.asarray(outcomes, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or x.shape[1] != len(feature_names) or y.shape[1] != len(outcome_names)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or type(draws) is not int or draws < 1 or type(seed) is not int or seed < 0):
        raise ValueError("invalid quartile-test matrices")
    basis, edges_by_feature, counts_by_feature = [], [], []
    for i in range(x.shape[1]):
        edges = np.quantile(x[:, i], (.25, .5, .75))
        labels = np.searchsorted(edges, x[:, i], side="right")
        counts = np.bincount(labels, minlength=4)
        basis.append(np.stack([(labels == group) / np.sqrt(max(1, counts[group])) for group in range(4)]))
        edges_by_feature.append(edges.tolist())
        counts_by_feature.append(counts.tolist())
    weights = np.concatenate(basis)
    centered = y - y.mean(axis=0)
    total_ss = (centered ** 2).sum(axis=0)
    def statistic(values: np.ndarray) -> np.ndarray:
        return ((weights @ values).reshape(x.shape[1], 4, y.shape[1]) ** 2).sum(axis=1)
    observed = statistic(centered)
    rng, counts = np.random.default_rng(seed), np.zeros(observed.shape, dtype=np.int64)
    for _ in range(draws):
        counts += statistic(centered[rng.permutation(len(y))]) >= observed - 1e-14
    p = (counts + 1) / (draws + 1)
    q = np.asarray(bh_adjust(p.ravel())).reshape(p.shape)
    rows = []
    for i, feature in enumerate(feature_names):
        for j, outcome in enumerate(outcome_names):
            rows.append({"feature": feature, "outcome": outcome, "users": len(y),
                         "quartile_edges": edges_by_feature[i], "quartile_counts": counts_by_feature[i],
                         "eta_squared": float(observed[i, j] / total_ss[j]) if total_ss[j] > 0 else 0.,
                         "permutation_p": float(p[i, j]), "bh_q_within_family": float(q[i, j]),
                         "permutations": draws, "seed": seed})
    return rows


def oracle_summary(utilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(utilities, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 3 or not len(y) or not np.isfinite(y).all():
        raise ValueError("oracle requires finite three-arm utilities")
    maxima = y.max(axis=1)
    ties = (y == maxima[:, None]).sum(axis=1)
    return {"users": len(y), "fixed_arm_mean_ndcg": dict(zip(map(str, ARM_LABELS), y.mean(axis=0).tolist())),
            "oracle_ndcg": float(maxima.mean()), "best_fixed_ndcg": float(y.mean(axis=0).max()),
            "oracle_lift_vs_best_fixed_percent": float((maxima.mean() / y.mean(axis=0).max() - 1) * 100) if y.mean(axis=0).max() else None,
            "all_three_zero_users": int(np.count_nonzero(maxima == 0)),
            "unique_best_arm_users": int(np.count_nonzero(ties == 1)),
            "tied_best_arm_users": int(np.count_nonzero(ties > 1)),
            "scope": "future-target oracle upper bound across separately trained models; not a deployable policy"}


def run_history_budget_analysis(run_directory: Path, feature_path: Path, output_directory: Path, *,
                                seed: int = 42, permutations: int = 5000, bootstrap_draws: int = 2000) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from rhpr.sasrec.history_features import HISTORY_FEATURE_FIELDS, CATALOGUE_FREQUENCY_FEATURE_FIELDS
    from rhpr.sasrec.compression_routing import (
        user_hash_split, fit_stump_router, predict_router, routing_evaluation, paired_mean_bootstrap,
        crossfit_stump_router, matched_mixture_bootstrap, fit_interval_router,
    )

    ids, warm, scores = load_paired_scores(run_directory)
    rows = pq.read_table(feature_path).to_pylist()
    features_by_user = {int(row["user_id"]): row for row in rows}
    if len(features_by_user) != len(rows) or set(features_by_user) != set(ids.tolist()):
        raise ValueError("feature users must exactly match paired prediction users")
    all_fields = list(HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS)
    x_all = np.array([[features_by_user[int(uid)][key] for key in all_fields] for uid in ids], dtype=np.float64)
    x = x_all[:, [all_fields.index(name) for name in CORE_FEATURES]]
    y = np.column_stack([scores[f"pair_mlp_m{arm}"] for arm in ARM_LABELS])
    contrasts = np.column_stack((y[:, 2] - y[:, 0], y[:, 2] - y[:, 1], y[:, 1] - y[:, 0]))
    history_benefit = y - scores["recent128"][:, None]
    run = json.loads((Path(run_directory) / "long-quality.json").read_text())
    if not run["complete"] or any(m["test"]["anchor_count"] != len(ids) for m in run["models"].values()):
        raise ValueError("incomplete or mismatched source run")
    validation_scores = {arm: max(row["validation"]["ndcg10"] for row in run["models"][f"pair_mlp_m{arm}"]["passes"]
                                  if "validation" in row) for arm in ARM_LABELS}
    reference_arm = int(np.argmax([validation_scores[arm] for arm in ARM_LABELS]))
    dev = user_hash_split(ids, seed=seed, train_fraction=.5)
    policy = fit_stump_router(x[dev], y[dev], CORE_FEATURES, min_leaf=200, arm_labels=ARM_LABELS)
    selected = predict_router(policy, x[~dev])
    routing = routing_evaluation(y[~dev], selected, reference_arm=reference_arm)
    held_scores = y[~dev]
    chosen_scores = held_scores[np.arange(len(selected)), selected]
    shares = np.bincount(selected, minlength=3) / len(selected)
    mixed_scores = held_scores @ shares
    routing["vs_validation_selected_fixed_ci"] = paired_mean_bootstrap(
        chosen_scores - held_scores[:, reference_arm], seed=seed, draws=bootstrap_draws)
    routing["vs_assignment_matched_random_mixture_ci"] = matched_mixture_bootstrap(
        held_scores, selected, seed=seed + 1, draws=bootstrap_draws)
    routing["policy"] = policy
    routing["development_users"], routing["heldout_users"] = int(dev.sum()), int((~dev).sum())
    routing["reference_summary_tokens"] = ARM_LABELS[reference_arm]
    crossfit = crossfit_stump_router(ids, x, y, CORE_FEATURES, seed=seed, n_folds=5, min_leaf=200, arm_labels=ARM_LABELS)
    predictions = np.asarray(crossfit.pop("predicted_arms"), dtype=np.int64)
    fold_ids = crossfit.pop("fold_ids")
    crossfit["fold_user_counts"] = np.bincount(fold_ids, minlength=5).tolist()
    crossfit["evaluation"] = routing_evaluation(y, predictions, reference_arm=reference_arm)
    crossfit["conditional_fixed_prediction_ci"] = paired_mean_bootstrap(
        y[np.arange(len(y)), predictions] - y[:, reference_arm], seed=seed + 2, draws=bootstrap_draws)
    interval_policy = fit_interval_router(x[dev], y[dev], CORE_FEATURES, min_leaf=200, arm_labels=ARM_LABELS)
    interval_selected = predict_router(interval_policy, x[~dev])
    interval_routing = routing_evaluation(held_scores, interval_selected, reference_arm=reference_arm)
    interval_routing["policy"] = interval_policy
    interval_routing["vs_validation_selected_fixed_ci"] = paired_mean_bootstrap(
        held_scores[np.arange(len(interval_selected)), interval_selected] - held_scores[:, reference_arm],
        seed=seed + 6, draws=bootstrap_draws)
    interval_routing["vs_assignment_matched_random_mixture_ci"] = matched_mixture_bootstrap(
        held_scores, interval_selected, seed=seed + 7, draws=bootstrap_draws)
    report = {
        "analysis_kind": "exploratory_existing_test_users_not_new_confirmatory_test",
        "seed": seed, "source_code_sha": run["code_sha"], "users": len(ids), "warm_users": int(warm.sum()),
        "core_features_fixed_before_joining_outcomes": list(CORE_FEATURES),
        "feature_provenance": "past-only item identities and observed timestamps; no target features, no learned embedding features",
        "primary_fdr_family_size": len(CORE_FEATURES) * len(COMPRESSION_CONTRASTS),
        "primary_compression_correlations": permutation_correlations(x, contrasts, feature_names=CORE_FEATURES,
             outcome_names=COMPRESSION_CONTRASTS, seed=seed, draws=permutations),
        "warm_sensitivity_correlations": permutation_correlations(x[warm], contrasts[warm], feature_names=CORE_FEATURES,
             outcome_names=COMPRESSION_CONTRASTS, seed=seed + 1, draws=permutations),
        "secondary_all_feature_compression_correlations": permutation_correlations(x_all, contrasts, feature_names=all_fields,
             outcome_names=COMPRESSION_CONTRASTS, seed=seed + 2, draws=permutations),
        "secondary_history_benefit_correlations": permutation_correlations(x, history_benefit, feature_names=CORE_FEATURES,
             outcome_names=[f"pair{arm}_minus_recent" for arm in ARM_LABELS], seed=seed + 3, draws=permutations),
        "secondary_quartile_dependency_tests": quartile_dependency_tests(x, contrasts, feature_names=CORE_FEATURES,
             outcome_names=COMPRESSION_CONTRASTS, seed=seed + 4, draws=permutations),
        "secondary_warm_quartile_dependency_tests": quartile_dependency_tests(x[warm], contrasts[warm], feature_names=CORE_FEATURES,
             outcome_names=COMPRESSION_CONTRASTS, seed=seed + 5, draws=permutations),
        "quartile_effects": feature_quartile_effects(x, y, feature_names=CORE_FEATURES, warm_mask=warm,
             seed=seed, draws=bootstrap_draws),
        "oracle": oracle_summary(y), "heldout_router": routing, "secondary_five_fold_router": crossfit,
        "secondary_nonlinear_heldout_router": interval_routing,
        "limitations": [
            "only test per-user predictions were saved; validation contains aggregate metrics only",
            "one future target per user; zero ties dominate and oracle is optimistic",
            "three budgets are separately trained models, not counterfactual compression of one model",
            "one training seed; user bootstrap cannot estimate variation across training seeds",
            "crossfit bootstrap is conditional on fixed out-of-fold predictions and omits policy-refitting uncertainty",
            "quartile CIs are descriptive pointwise intervals, not multiple-comparison adjusted",
            "static level latency is nearly equal; quality routing alone does not establish compute savings",
        ],
        "source_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                [Path(run_directory) / "long-quality.json", Path(feature_path),
                                 *sorted(Path(run_directory).glob("*-test-users.json"))]},
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "history-budget-analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
