from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.compression_routing import (
    crossfit_stump_router, fit_interval_router, fit_stump_router, matched_mixture_bootstrap, paired_mean_bootstrap,
    predict_router, routing_evaluation, user_hash_split,
)


class CompressionRoutingTest(unittest.TestCase):
    def test_uid_partition_is_reorder_and_duplicate_stable(self) -> None:
        users = np.arange(1000, dtype=np.int64)
        mask = user_hash_split(users)
        reordered = users[::-1]
        np.testing.assert_array_equal(user_hash_split(reordered)[::-1], mask)
        np.testing.assert_array_equal(user_hash_split([19, 19, 27]), [user_hash_split([19])[0]] * 2 + [user_hash_split([27])[0]])
        self.assertGreater(int(mask.sum()), 400)
        self.assertLess(int(mask.sum()), 600)
        self.assertFalse(np.array_equal(mask, user_hash_split(users, seed=43)))
        self.assertFalse(user_hash_split(users, train_fraction=0).any())
        self.assertTrue(user_hash_split(users, train_fraction=1).all())

    def test_hand_calculated_stump_and_arm_indices(self) -> None:
        X = np.arange(4, dtype=float)[:, None]
        scores = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
        policy = fit_stump_router(X, scores, ["unique_fraction"], quantiles=(0.5,), min_leaf=2)
        self.assertEqual(policy["kind"], "stump")
        self.assertEqual(policy["threshold"], 1.5)
        self.assertEqual(policy["left_arm_index"], 0)
        self.assertEqual(policy["right_arm_index"], 2)
        self.assertEqual(policy["train_utility"], 1.0)
        self.assertEqual(policy["n_candidates"], 2)
        np.testing.assert_array_equal(predict_router(policy, X), [0, 0, 2, 2])
        np.testing.assert_array_equal(predict_router(policy, np.array([[1.5], [1.6]])), [0, 2])
        json.dumps(policy, allow_nan=False)

    def test_zero_outcomes_ties_and_equal_leaf_means_prefer_constant_four(self) -> None:
        X = np.arange(6, dtype=float)[:, None]
        for scores in (np.zeros((6, 3)), np.tile([0.5, 0.5, 0.5], (6, 1))):
            policy = fit_stump_router(X, scores, ["feature"], min_leaf=1)
            self.assertEqual(policy["kind"], "constant")
            self.assertEqual(policy["arm_index"], 0)
            np.testing.assert_array_equal(predict_router(policy, X), np.zeros(6, dtype=int))
        constant = fit_stump_router(X, np.tile([0, 1, 0], (6, 1)), ["feature"], min_leaf=1)
        self.assertEqual(constant["kind"], "constant")
        self.assertEqual(constant["arm_index"], 1)

    def test_minimum_leaf_and_duplicate_quantiles(self) -> None:
        X = np.array([[0], [0], [0], [1]], dtype=float)
        scores = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        policy = fit_stump_router(X, scores, ["feature"], quantiles=(0.2, 0.4, 0.6), min_leaf=2)
        self.assertEqual(policy["kind"], "constant")
        self.assertEqual(policy["n_candidate_splits"], 0)
        policy = fit_stump_router(X, scores, ["feature"], quantiles=(0.2, 0.4, 0.6), min_leaf=1)
        self.assertEqual(policy["n_candidate_splits"], 1)
        self.assertEqual(policy["kind"], "stump")

    def test_majority_zero_ndcg_still_preserves_conditional_gain(self) -> None:
        X = np.arange(100, dtype=float)[:, None]
        scores = np.zeros((100, 3))
        scores[4, 0] = 1
        scores[94, 2] = 0.5
        policy = fit_stump_router(X, scores, ["feature"], quantiles=(0.5,), min_leaf=20)
        self.assertEqual(policy["kind"], "stump")
        self.assertAlmostEqual(policy["train_utility"], 0.015)
        result = routing_evaluation(scores, predict_router(policy, X), reference_arm=0)
        self.assertAlmostEqual(result["selected_minus_reference"], 0.005)
        self.assertEqual(result["arm_shares"], [0.5, 0.0, 0.5])

    def test_prediction_and_thresholds_do_not_consume_holdout_labels(self) -> None:
        development_X = np.arange(8, dtype=float)[:, None]
        development_scores = np.array([[1, 0, 0]] * 4 + [[0, 0, 1]] * 4, dtype=float)
        policy = fit_stump_router(development_X, development_scores, ["feature"], min_leaf=2)
        heldout_X = np.array([[-10.0], [2.5], [4.5], [100.0]])
        before = predict_router(policy, heldout_X)
        heldout_scores = np.zeros((4, 3))
        heldout_scores[:, 1] = 1
        routing_evaluation(heldout_scores, before)
        heldout_scores[:] = np.array([1, 0, 0])
        np.testing.assert_array_equal(before, predict_router(policy, heldout_X))
        repeated = fit_stump_router(development_X, development_scores, ["feature"], min_leaf=2)
        self.assertEqual(policy, repeated)

    def test_crossfit_predictions_exclude_entire_heldout_user_fold(self) -> None:
        users = np.arange(100, dtype=np.int64)
        X = users.astype(float)[:, None]
        scores = np.zeros((100, 3))
        scores[:, 0] = 1
        first = crossfit_stump_router(users, X, scores, ["feature"], min_leaf=5)
        selected_fold = first["fold_ids"][0]
        heldout = first["fold_ids"] == selected_fold
        changed = scores.copy()
        changed[heldout] = [0, 0, 100]
        second = crossfit_stump_router(users, X, changed, ["feature"], min_leaf=5)
        np.testing.assert_array_equal(first["predicted_arms"][heldout], second["predicted_arms"][heldout])
        np.testing.assert_array_equal(first["fold_ids"], second["fold_ids"])
        self.assertEqual(sum(policy["n_heldout_users"] for policy in first["policies"]), 100)

    def test_fixed_interval_bank_captures_nonmonotonic_utility(self) -> None:
        X = np.arange(10, dtype=float)[:, None]
        scores = np.zeros((10, 3))
        scores[:2, 0] = 1
        scores[2:6, 2] = 1
        scores[6:, 0] = 1
        stump = fit_stump_router(X, scores, ["duration"], min_leaf=2)
        policy = fit_interval_router(X, scores, ["duration"], min_leaf=2)
        self.assertLess(stump["train_utility"], 1.0)
        self.assertEqual(policy["kind"], "interval")
        self.assertEqual(policy["train_utility"], 1.0)
        np.testing.assert_array_equal(policy["leaf_train_users"], [2, 4, 4])
        self.assertEqual(policy["arm_indices"], [0, 2, 0])
        self.assertEqual(policy["n_candidate_splits"], 4)
        self.assertEqual(policy["n_candidate_intervals"], 6)
        self.assertEqual(policy["n_candidates"], 11)
        np.testing.assert_array_equal(predict_router(policy, X), [0, 0, 2, 2, 2, 2, 0, 0, 0, 0])
        low, high = policy["thresholds"]
        np.testing.assert_array_equal(predict_router(policy, np.array([[low], [low + 0.001], [high], [high + 0.001]])), [0, 2, 2, 0])
        self.assertNotIn("left_arm_index", policy)
        json.dumps(policy, allow_nan=False)

    def test_interval_bank_preserves_constant_and_stump_ties(self) -> None:
        X = np.arange(10, dtype=float)[:, None]
        zero = fit_interval_router(X, np.zeros((10, 3)), ["feature"], min_leaf=2)
        self.assertEqual(zero["kind"], "constant")
        self.assertEqual(zero["arm_index"], 0)
        scores = np.array([[1, 0, 0]] * 4 + [[0, 0, 1]] * 6, dtype=float)
        tied = fit_interval_router(X, scores, ["feature"], min_leaf=2)
        self.assertEqual(tied["kind"], "stump")
        self.assertEqual(tied["train_utility"], 1.0)
        too_small = fit_interval_router(X, scores, ["feature"], min_leaf=6)
        self.assertEqual(too_small["kind"], "constant")
        self.assertEqual(too_small["n_candidate_intervals"], 0)

    def test_interval_crossfit_uses_no_heldout_fold_labels(self) -> None:
        users = np.arange(100, dtype=np.int64)
        X = users.astype(float)[:, None]
        scores = np.zeros((100, 3))
        scores[:, 0] = 1
        first = crossfit_stump_router(users, X, scores, ["feature"], min_leaf=5, policy_class="interval")
        heldout = first["fold_ids"] == first["fold_ids"][0]
        scores[heldout] = [0, 100, 0]
        second = crossfit_stump_router(users, X, scores, ["feature"], min_leaf=5, policy_class="interval")
        np.testing.assert_array_equal(first["predicted_arms"][heldout], second["predicted_arms"][heldout])
        self.assertEqual(first["policy_class"], "interval")
        with self.assertRaises(ValueError):
            crossfit_stump_router(users, X, scores, ["feature"], policy_class="unknown")

    def test_matched_mixture_is_expected_assignment_independent_of_user(self) -> None:
        scores = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
        result = routing_evaluation(scores, np.array([0, 2]), reference_arm=0, batch_latency_ms=[2, 3, 4])
        self.assertEqual(result["mean_selected_score"], 1.0)
        self.assertEqual(result["mean_matched_mixture_score"], 0.5)
        self.assertEqual(result["selected_minus_matched_mixture"], 0.5)
        self.assertEqual(result["expected_batch_latency_ms"], 3.0)
        self.assertEqual(result["reference_batch_latency_ms"], 2.0)
        constant = routing_evaluation(scores, np.array([2, 2]), reference_arm=0)
        self.assertEqual(constant["selected_minus_matched_mixture"], 0.0)
        self.assertEqual(constant["arm_shares"], [0.0, 0.0, 1.0])
        json.dumps(result, allow_nan=False)

    def test_paired_bootstrap_reproducible_and_exact_constant_interval(self) -> None:
        result = paired_mean_bootstrap(np.full(100, 0.125), draws=100)
        self.assertEqual(result["mean"], 0.125)
        self.assertEqual(result["ci_low"], 0.125)
        self.assertEqual(result["ci_high"], 0.125)
        delta = np.array([-1.0, -0.1, 0.0, 0.1, 1.0])
        first = paired_mean_bootstrap(delta, seed=9, draws=200)
        self.assertEqual(first, paired_mean_bootstrap(delta, seed=9, draws=200))
        self.assertLess(first["ci_low"], 0)
        self.assertGreater(first["ci_high"], 0)

    def test_mixture_bootstrap_recomputes_each_samples_assignment_shares(self) -> None:
        scores = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
        predictions = np.array([0, 2])
        result = matched_mixture_bootstrap(scores, predictions, seed=9, draws=1000)
        self.assertEqual(result, matched_mixture_bootstrap(scores, predictions, seed=9, draws=1000))
        self.assertEqual(result["mean"], 0.5)
        # Repeated rows assign every user to one arm; their matched mixture
        # then equals routing. A fixed-share bootstrap would wrongly stay .5.
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.5)
        constant = matched_mixture_bootstrap(scores, np.array([0, 0]), draws=100)
        self.assertEqual(constant["mean"], 0.0)
        self.assertEqual(constant["ci_low"], 0.0)
        self.assertEqual(constant["ci_high"], 0.0)
        json.dumps(result, allow_nan=False)

    def test_invalid_or_leaking_shapes_fail_explicitly(self) -> None:
        X = np.zeros((4, 1))
        scores = np.zeros((4, 3))
        for features, utilities in ((X, scores[:2]), (np.full((4, 1), np.nan), scores), (X, np.full((4, 3), np.inf))):
            with self.assertRaises(ValueError):
                fit_stump_router(features, utilities, ["feature"], min_leaf=1)
        with self.assertRaises(ValueError):
            fit_stump_router(X, scores, ["feature"], quantiles=(1.0,), min_leaf=1)
        with self.assertRaises(ValueError):
            routing_evaluation(scores, np.array([0, 1, 2, 3]))
        with self.assertRaises(ValueError):
            routing_evaluation(scores, np.zeros(4, dtype=int), batch_latency_ms=[1, 2])
        with self.assertRaises(ValueError):
            paired_mean_bootstrap(np.array([]))
        with self.assertRaises(ValueError):
            user_hash_split([1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
