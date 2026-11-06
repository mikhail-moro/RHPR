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

    def transfer(self):
        return json.loads((Path(__file__).resolve().parents[1] / 'experiments/ml1m_mini_tiger_transfer_2t4.json').read_text())

    def test_transfer_is_eight_narrow_fits_with_balanced_workers(self):
        from rhpr.gr.ae_suite import assignments
        c = self.transfer(); validate(c)
        self.assertEqual([len(j['trials']) for j in assignments(c, 2)], [4, 4])
        self.assertEqual({t['architecture'] for t in c['trials']}, {'narrow'})
        self.assertTrue(all(i.endswith('seed202') for i in c['index_ids']))

    def test_transfer_reuses_pilot_settings_and_narrow_initialization(self):
        old = self.config(); c = self.transfer()
        for k in ['generator', 'data', 'movie_input', 'modern_runtime', 'runtime', 'evaluation', 'execution']:
            self.assertEqual(c[k], old[k])
        self.assertEqual(c['initialization_pins']['narrow'], old['initialization_pins']['narrow'])
        a = dict(c['training']); b = dict(old['training'])
        for k in ['max_epochs', 'min_epochs', 'patience_epochs', 'max_seconds']:
            a.pop(k); b.pop(k)
        self.assertEqual(a, b)
        self.assertEqual(c['parents']['202']['files'], old['parents']['101']['files'])
        self.assertEqual(len(c['kaggle']['kernel_sources']), 1)
        self.assertNotIn('index_input', c)

    def test_transfer_rejects_architecture_epoch_or_candidate_tuning(self):
        for key in ['architecture', 'epoch', 'candidate', 'phase']:
            c = self.transfer()
            if key == 'architecture': c['trials'][0]['architecture'] = 'shallow'
            if key == 'epoch': c['primary_proxy_epoch'] = 5
            if key == 'candidate': c['index_ids'][0] = 'sum_seed42'
            if key == 'phase': c['calibration_phase'] = 'other'
            with self.assertRaises(ValueError): validate(c)

    def test_transfer_primary_uses_epoch10_not_selected_best_or_epoch5(self):
        c = self.transfer(); refs = {}; records = {}
        for j, i in enumerate(c['index_ids']):
            refs[i] = dict(full_seed=202, epochs={str(e): (j + 1) * .01 for e in [5, 10, 15, 20, 25, 30]})
            for s in c['generator_seeds']:
                records[f'{i}_{s}'] = dict(architecture='narrow', index_id=i, seed=s,
                    epochs={'5': .5 - j * .01, '10': (j + 1) * .01})
        result = correlate(c, records, refs)
        self.assertEqual(result['primary_proxy_epoch'], 10)
        self.assertEqual(result['primary']['mean']['spearman'], 1.)
        self.assertEqual(set(result['groups']['all']['mini']), {'narrow_epoch5', 'narrow_epoch10'})


class WidthReductionTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1] / 'experiments/ml1m_mini_tiger_width_2t4.json').read_text())

    def test_exact_matrix_and_seeds_balanced(self):
        from rhpr.gr.ae_suite import assignments
        c = self.config(); validate(c)
        self.assertEqual(len(c['index_ids']), 12)
        jobs = assignments(c, 2)
        self.assertEqual([len(j['trials']) for j in jobs], [24, 24])
        self.assertEqual([{t['seed'] for t in j['trials']} for j in jobs], [{31415}, {27182}])
        self.assertEqual({t['architecture'] for t in c['trials']}, {'ffn_small', 'attention_small'})

    def test_no_extra_budget_or_candidates(self):
        for change in ['epoch', 'deadline', 'matrix', 'test']:
            c = self.config()
            if change == 'epoch': c['primary_proxy_epoch'] = 5
            if change == 'deadline': c['suite_deadline_seconds'] = 15000
            if change == 'matrix': c['trials'].pop()
            if change == 'test': c['evaluation']['test_after_selection'] = True
            with self.assertRaises(ValueError): validate(c)

    def test_methods_and_fixed_epoch_are_reported(self):
        c = self.config(); refs = {}; records = {}
        for j, i in enumerate(c['index_ids']):
            method, seed = i.split('_seed')
            refs[i] = dict(full_seed=int(seed), method=method,
                epochs={str(e): (j + 1)*.001 for e in [5,10,15,20,25,30]})
        for t in c['trials']:
            j = c['index_ids'].index(t['index_id'])
            records[t['id']] = {**t, 'epochs': {'5': .5-j*.001, '10': (j+1)*.001}}
        r = correlate(c, records, refs)
        self.assertEqual(r['primary_proxy_epoch'], 10)
        self.assertEqual(r['primary']['ffn_small']['mean']['spearman'], 1.)
        self.assertEqual(len(r['groups']['method_sum']['index_ids']), 3)
        self.assertEqual(r['groups']['method_sum']['mini']['attention_small_epoch10']['mean']['spearman'], 1.)
