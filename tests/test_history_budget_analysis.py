from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'numpy required')
class HistoryBudgetAnalysisTest(unittest.TestCase):
    def test_average_tied_ranks_and_fdr_adjustment(self):
        from rhpr.sasrec.history_budget_analysis import average_ranks, bh_adjust
        np.testing.assert_equal(average_ranks(np.array([3., 1., 2., 2.])), [4., 1., 2.5, 2.5])
        np.testing.assert_allclose(bh_adjust([.01, .04, .03, .2]), [.04, .0533333333, .0533333333, .2])
        with self.assertRaises(ValueError):
            bh_adjust([float('nan')])

    def test_joint_permutations_detect_association_and_handle_constants(self):
        from rhpr.sasrec.history_budget_analysis import permutation_correlations
        x = np.column_stack((np.arange(30), np.ones(30)))
        y = np.column_stack((np.arange(30), -np.arange(30)))
        args = dict(feature_names=['varying', 'constant'], outcome_names=['positive', 'negative'], seed=4, draws=300)
        result = permutation_correlations(x, y, **args)
        self.assertEqual(result, permutation_correlations(x, y, **args))
        self.assertAlmostEqual(result[0]['spearman_rho'], 1.)
        self.assertAlmostEqual(result[1]['spearman_rho'], -1.)
        self.assertLess(result[0]['bh_q_within_family'], .02)
        self.assertTrue(result[2]['constant_column'])
        self.assertIsNone(result[2]['spearman_rho'])
        self.assertEqual(result[2]['permutation_p'], 1.)

    def test_paired_loader_reorders_users_and_rejects_missing_support_mismatches(self):
        from rhpr.sasrec.history_budget_analysis import load_paired_scores
        names = ['recent128'] + [f'{kind}_m{budget}' for kind in ('mean', 'uniform', 'pair_mlp', 'qformer') for budget in (16, 8, 4)]
        records = [dict(user_id=4, target_known=False, rank=0, hr10=0., ndcg10=0.),
                   dict(user_id=2, target_known=True, rank=1, hr10=1., ndcg10=1.)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for i, name in enumerate(names):
                (directory / f'{name}-test-users.json').write_text(json.dumps(records if i % 2 else list(reversed(records))))
            ids, known, scores = load_paired_scores(directory)
            np.testing.assert_equal(ids, [2, 4])
            np.testing.assert_equal(known, [True, False])
            np.testing.assert_equal(scores['pair_mlp_m4'], [1., 0.])
            changed = [dict(r) for r in records]
            changed[0]['target_known'] = True
            (directory / 'pair_mlp_m4-test-users.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'same users and target support'):
                load_paired_scores(directory)
            (directory / 'pair_mlp_m4-test-users.json').write_text(json.dumps(records[:1]))
            with self.assertRaises(ValueError):
                load_paired_scores(directory)

    def test_oracle_does_not_assign_tied_zero_users_a_required_budget(self):
        from rhpr.sasrec.history_budget_analysis import oracle_summary
        y = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 1.]])
        result = oracle_summary(y)
        self.assertEqual(result['all_three_zero_users'], 1)
        self.assertEqual(result['unique_best_arm_users'], 1)
        self.assertEqual(result['tied_best_arm_users'], 2)
        self.assertAlmostEqual(result['oracle_ndcg'], 2 / 3)
        self.assertAlmostEqual(result['best_fixed_ndcg'], 1 / 3)

    def test_quartile_omnibus_detects_nonmonotonic_changes_and_handles_constants(self):
        from rhpr.sasrec.history_budget_analysis import quartile_dependency_tests
        x = np.column_stack((np.arange(40), np.ones(40)))
        outcome = np.repeat([0., 1., 1., 0.], 10)[:, None]
        result = quartile_dependency_tests(x, outcome, feature_names=['nonlinear', 'constant'],
                                          outcome_names=['utility_difference'], seed=4, draws=500)
        self.assertAlmostEqual(result[0]['eta_squared'], 1.)
        self.assertLess(result[0]['bh_q_within_family'], .01)
        self.assertEqual(result[1]['quartile_counts'], [0, 0, 0, 40])
        self.assertEqual(result[1]['permutation_p'], 1.)


if __name__ == '__main__':
    unittest.main()
