"""Causal averaging, support accounting and frozen-path context search."""
import unittest

import numpy as np

from rhpr.gr.sid_context_signal import (
    context_confidence, context_neighbors, context_prototypes, prototype_reliability,
)
from rhpr.gr.sid_graph_swap import PairTerm, SwapSearch


class TrainOnlyUser:
    def __init__(self, train):
        self.train = train

    @property
    def validation(self):
        raise AssertionError('validation accessed')

    @property
    def test(self):
        raise AssertionError('test accessed')


class ContextTests(unittest.TestCase):
    def test_equal_context_weight_not_pooled_event_weight_and_no_target_leak(self):
        z = np.array([[0., 0.], [2., 0.], [0., 6.], [90., 80.]])
        result = context_prototypes([TrainOnlyUser([1, 3]), TrainOnlyUser([1, 2, 3])], z, 20)
        # For target3: ([2,0] + mean([2,0],[0,6])) / 2, not mean(all events).
        np.testing.assert_allclose(result.vectors[3], [1.5, 1.5])
        np.testing.assert_array_equal(result.contexts, [0, 0, 1, 2])
        np.testing.assert_array_equal(result.users, [0, 0, 1, 2])
        self.assertEqual(result.length_sum[3], 3)
        self.assertFalse(result.report['validation_and_test_used'])

    def test_window_and_short_context_mean_exclude_future(self):
        z = np.array([[1., 0.], [0., 2.], [3., 0.], [50., 60.], [900., 900.]])
        a = context_prototypes([TrainOnlyUser([0, 1, 2, 3])], z, 2)
        b = context_prototypes([TrainOnlyUser([0, 1, 2, 3, 4])], z, 2)
        np.testing.assert_allclose(a.vectors[1], z[0])
        np.testing.assert_allclose(a.vectors[3], [1.5, 1.])
        np.testing.assert_array_equal(a.vectors, b.vectors * np.array([1, 1, 1, 1, 0])[:, None])
        self.assertEqual(a.report['context_length_histogram'], [0, 1, 2])

    def test_masked_factor_rows_are_not_pads_in_denominator(self):
        z = np.array([[2., 0.], [0., 9.], [4., 4.]])
        a = context_prototypes([TrainOnlyUser([1, 0, 2])], z, 2, np.array([True, False, True]))
        self.assertEqual(a.report['skipped_empty_factor_contexts'], 1)
        self.assertEqual(a.report['partially_embedded_contexts'], 1)
        np.testing.assert_allclose(a.vectors[2], [2., 0.])
        self.assertEqual(a.contexts[0], 0)
        self.assertEqual(a.length_sum[2], 2)
        self.assertEqual(a.embedded_sum[2], 1)

    def test_repeated_target_observations_are_not_distinct_users(self):
        z = np.array([[2., 0.], [0., 4.], [1., 1.]])
        a = context_prototypes([TrainOnlyUser([0, 1, 1, 1])], z, 20)
        self.assertEqual(a.contexts[1], 3)
        self.assertEqual(a.users[1], 1)
        # Earlier target occurrences legitimately remain in H.
        np.testing.assert_allclose(a.vectors[1], [11/9, 14/9])
        self.assertEqual(context_confidence(a, 2, 10.)[1], 0.)

    def test_topk_accounts_for_confidence_before_selection(self):
        z = np.array([[1., 0.], [1., 0.], [.8, .6], [-1., 0.]])
        # Generate one observed context for each target per user, then substitute
        # controlled prototype directions/support for an exact ranking assertion.
        a = context_prototypes([TrainOnlyUser([0, 1, 2, 3])], z, 2)
        a.vectors[:] = z
        a.users[:] = [100, 2, 100, 0]
        graph = context_neighbors(a, 1, 2, 10., block=2, ranking='weighted_cosine')
        np.testing.assert_array_equal(graph.target[graph.source == 0], [2])
        self.assertFalse(np.any(graph.source == 3) or np.any(graph.target == 3))
        expected = .8 * (100/110)**2
        self.assertAlmostEqual(float(graph.weight[graph.source == 0][0]), expected, places=6)
        nearest = context_neighbors(a, 1, 2, 10., block=2)
        np.testing.assert_array_equal(nearest.target[nearest.source == 0], [1])
        self.assertAlmostEqual(float(nearest.weight[nearest.source == 0][0]), (100/110)*(2/12))

    def test_split_reliability_reports_missing_heldout_support(self):
        z = np.eye(4)
        fit = context_prototypes([TrainOnlyUser([0, 1, 2])], z, 2)
        held = context_prototypes([TrainOnlyUser([0, 1])], z, 2)
        rows = prototype_reliability(fit, held)
        self.assertEqual(rows[1]['items'], 2)
        self.assertEqual(rows[1]['comparable_items'], 1)
        self.assertAlmostEqual(rows[1]['mean_cosine'], 1.)

    def test_invalid_inputs_fail_and_factors_remain_frozen(self):
        z = np.eye(4)
        original = z.copy()
        context_prototypes([TrainOnlyUser([0, 1, 2])], z, 2)
        np.testing.assert_array_equal(z, original)
        for ids in [[-1, 0], [0, 4], [0., 1.]]:
            with self.assertRaises(ValueError):
                context_prototypes([TrainOnlyUser(ids)], z, 2)
        with self.assertRaises(ValueError):
            context_prototypes([TrainOnlyUser([0, 1])], z, 0)

    def test_search_preserves_complete_paths_and_exact_objective(self):
        z = np.array([[1., 0.], [0., 1.], [1., .1], [.1, 1.]])
        users = [TrainOnlyUser([0, 1, 2, 3]), TrainOnlyUser([2, 1, 0, 3])]
        signal = context_prototypes(users, z, 2)
        graph = context_neighbors(signal, 2, minimum_users=1)
        codes = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
        search = SwapSearch(codes, [PairTerm(graph)], [1.])
        for a, b in [(0, 3), (1, 2), (0, 1), (2, 3)]:
            search.try_swap(a, b)
            search.audit(codes)
            self.assertAlmostEqual(search.score, graph.score(search.codes))
            np.testing.assert_array_equal(search.codes, codes[search.owner])


if __name__ == '__main__':
    unittest.main()
