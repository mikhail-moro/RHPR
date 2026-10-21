from __future__ import annotations

import json
import math
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

    def test_larger_cutoff_requires_uncensored_saved_ranks_and_metrics(self):
        from rhpr.sasrec.history_budget_analysis import load_paired_scores
        names = ['recent128'] + [f'{kind}_m{budget}' for kind in ('mean', 'uniform', 'pair_mlp', 'qformer')
                                for budget in (16, 8, 4)]
        legacy = [dict(user_id=1, target_known=True, rank=0, hr10=0., ndcg10=0.)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in names:
                (directory / f'{name}-test-users.json').write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError, 'saved ndcg100/hr100'):
                load_paired_scores(directory, k=100)
            # A fabricated zero at the larger cutoff still cannot establish
            # where a legacy top-10 miss actually ranked.
            fabricated = [dict(legacy[0], hr100=0., ndcg100=0.)]
            (directory / 'recent128-test-users.json').write_text(json.dumps(fabricated))
            with self.assertRaisesRegex(ValueError, 'rank_cutoff >= 100'):
                load_paired_scores(directory, k=100)

    def test_top100_ranks_preserve_legacy_scores_and_exclude_rank101(self):
        from rhpr.sasrec.history_budget_analysis import load_paired_scores
        names = ['recent128'] + [f'{kind}_m{budget}' for kind in ('mean', 'uniform', 'pair_mlp', 'qformer')
                                for budget in (16, 8, 4)]
        records = [dict(user_id=uid, target_known=rank != 0, rank=0, hr10=0., ndcg10=0.,
                        rank_cutoff=200, rank_at_max_k=rank, hr100=float(0 < rank <= 100),
                        ndcg100=1 / math.log2(rank + 1) if 0 < rank <= 100 else 0.)
                   for uid, rank in enumerate((11, 100, 101, 0), 1)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in names:
                (directory / f'{name}-test-users.json').write_text(json.dumps(records))
            _, _, old_scores = load_paired_scores(directory)
            np.testing.assert_equal(old_scores['pair_mlp_m8'], np.zeros(4))
            ids, known, scores = load_paired_scores(directory, k=100)
            np.testing.assert_equal(ids, [1, 2, 3, 4])
            np.testing.assert_equal(known, [True, True, True, False])
            np.testing.assert_allclose(scores['pair_mlp_m8'], [1 / math.log2(12), 1 / math.log2(101), 0., 0.])
            for field, value in [('rank_cutoff', 10), ('rank_at_max_k', 201),
                                 ('ndcg100', 0.), ('target_known', False)]:
                changed = [dict(row) for row in records]
                changed[0][field] = value
                (directory / 'recent128-test-users.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_paired_scores(directory, k=100)
            (directory / 'recent128-test-users.json').write_text(json.dumps(records))
            for cutoff in (0, -1, True, 100.):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    load_paired_scores(directory, k=cutoff)

    def test_analysis_routes_using_selected_cutoff_at_original_checkpoint(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow required')
        from rhpr.sasrec.history_budget_analysis import run_history_budget_analysis
        from rhpr.sasrec.history_features import HISTORY_FEATURE_FIELDS, CATALOGUE_FREQUENCY_FEATURE_FIELDS
        names = ['recent128'] + [f'{kind}_m{budget}' for kind in ('mean', 'uniform', 'pair_mlp', 'qformer')
                                for budget in (16, 8, 4)]
        records = [dict(user_id=uid, target_known=True, rank=0, hr10=0., ndcg10=0.,
                        rank_cutoff=100, rank_at_max_k=11, hr100=1., ndcg100=1 / math.log2(12))
                   for uid in range(30)]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            run = dict(complete=True, code_sha='test-source', models={})
            for name in names:
                (directory / f'{name}-test-users.json').write_text(json.dumps(records))
                run['models'][name] = dict(test=dict(anchor_count=len(records)), best_epoch=2, passes=[
                    dict(epoch=1, validation=dict(ndcg10=.1, ndcg100=.99)),
                    dict(epoch=2, validation=dict(ndcg10=.2, ndcg100=.3 if name == 'pair_mlp_m8' else .2))])
            (directory / 'long-quality.json').write_text(json.dumps(run))
            feature_path = directory / 'features.parquet'
            pq.write_table(pa.Table.from_pylist([
                dict(user_id=row['user_id'], **{field: float(row['user_id'])
                     for field in HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS})
                for row in records]), feature_path)
            report = run_history_budget_analysis(directory, feature_path, directory / 'analysis',
                                                metric_k=100, permutations=2, bootstrap_draws=2)
            self.assertEqual(report['metric_k'], 100)
            self.assertEqual(report['utility_metric'], 'NDCG@100')
            self.assertIn('secondary_exploratory', report['metric_cutoff_status'])
            self.assertEqual(report['checkpoint_selection_metric'], 'NDCG@10')
            self.assertEqual(report['heldout_router']['reference_summary_tokens'], 8)
            self.assertAlmostEqual(report['oracle']['best_fixed_ndcg'], 1 / math.log2(12))
            # Saved test utilities alone do not substitute for validation at
            # the checkpoint whose weights produced those utilities.
            del run['models']['pair_mlp_m8']['passes'][1]['validation']['ndcg100']
            (directory / 'long-quality.json').write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'NDCG@10-selected checkpoint'):
                run_history_budget_analysis(directory, feature_path, directory / 'analysis-missing',
                                            metric_k=100, permutations=2, bootstrap_draws=2)

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
