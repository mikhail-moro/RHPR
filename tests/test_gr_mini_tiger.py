import copy
import json
import unittest
from pathlib import Path

from rhpr.gr.mini_tiger import ARCHITECTURES, correlate, rank_correlation, selection_stats, validate


class MiniCalibrationTests(unittest.TestCase):
    def test_ranks_use_ties_and_return_undefined_for_constant(self):
        self.assertAlmostEqual(rank_correlation([1, 1, 3], [2, 2, 5]), 1.)
        self.assertAlmostEqual(rank_correlation([1, 2, 3], [6, 5, 4]), -1.)
        self.assertIsNone(rank_correlation([1, 1], [1, 2]))
        with self.assertRaises(ValueError): rank_correlation([1, float('nan')], [1, 2])

    def test_regret_measures_target_quality_not_proxy_score(self):
        row = selection_stats([3, 2, 1], [.1, .2, .5])
        self.assertAlmostEqual(row['selected_target_regret'], .4)
        self.assertEqual(row['top2_overlap'], .5)

    def test_independent_seed_aggregation_and_fixed_epoch_target(self):
        ids = ['a', 'b', 'c', 'd']; refs = {}; records = {}
        c = dict(index_ids=ids, generator_seeds=[31415, 27182])
        for j, i in enumerate(ids):
            refs[i] = dict(full_seed=42 if j < 2 else 101,
                           epochs={str(e): (j + 1) * .01 for e in [5, 10, 15, 20, 25, 30]})
            for arch in ARCHITECTURES:
                for seed in c['generator_seeds']:
                    records[f'{arch}_{i}_{seed}'] = dict(architecture=arch, index_id=i, seed=seed,
                        epochs={str(e): (j + 1) * .02 for e in [5, 10, 15, 20]})
        result = correlate(c, records, refs)
        self.assertEqual(result['groups']['all']['mini']['narrow_epoch20']['mean']['spearman'], 1.)
        self.assertFalse(result['test_used'])
        del records['narrow_a_31415']
        with self.assertRaises(ValueError): correlate(c, records, refs)

    def config(self):
        return json.loads((Path(__file__).resolve().parents[1] / 'experiments/ml1m_mini_tiger_2t4.json').read_text())

    def test_complete_crossed_matrix_and_balanced_initialization_groups(self):
        from rhpr.gr.ae_suite import assignments
        c = self.config(); validate(c)
        jobs = assignments(c, 2)
        self.assertEqual([len(j['trials']) for j in jobs], [24, 24])
        self.assertTrue(c['compare_initialization_by_shape'])
        self.assertEqual(set(c['generator_seeds']) & {42, 101, 202}, set())

    def test_reject_duplicate_and_test_evaluation(self):
        for change in ['duplicate', 'test', 'shorter']:
            c = copy.deepcopy(self.config())
            if change == 'duplicate': c['trials'][-1] = c['trials'][0]
            if change == 'test': c['evaluation']['test_after_selection'] = True
            if change == 'shorter': c['training']['max_epochs'] = 10
            with self.assertRaises(ValueError): validate(c)
