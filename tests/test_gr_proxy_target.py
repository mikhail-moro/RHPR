import copy
import json
import unittest
from pathlib import Path

from rhpr.gr.ae_suite import assignments
from rhpr.gr.mini_tiger import protocol_shape, validate
from rhpr.gr.proxy_target import INDEX_IDS, TARGET_SEED, correlate_target


class IndependentTargetTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1] /
            'experiments/ml1m_proxy_target_replication_2t4.json').read_text())

    def test_only_six_full_fits_share_one_new_seed_and_initialization(self):
        c = self.config(); validate(c)
        self.assertEqual(protocol_shape(c), (['full'], [42, 101, 202], 30))
        self.assertEqual([len(j['trials']) for j in assignments(c, 2)], [3, 3])
        self.assertEqual(c['index_ids'], INDEX_IDS)
        self.assertNotIn(TARGET_SEED, [42, 101, 202, 31415, 27182])
        self.assertEqual({t['architecture'] for t in c['trials']}, {'full'})

    def test_reject_compute_expansion_test_and_changed_proxy(self):
        for field in ['seed', 'epoch', 'candidate', 'test', 'proxy', 'shape', 'budget']:
            with self.subTest(field=field):
                c = self.config()
                if field == 'seed': c['generator_seeds'] = [31415]
                if field == 'epoch': c['training']['max_epochs'] = 60
                if field == 'candidate': c['trials'].append(copy.deepcopy(c['trials'][0]))
                if field == 'test': c['evaluation']['test_after_selection'] = True
                if field == 'proxy': c['proxy_reference']['indexes'][INDEX_IDS[0]]['scores_by_seed']['31415'] += .001
                if field == 'shape': c['generator']['encoder_layers'] = 2
                if field == 'budget': c['suite_deadline_seconds'] = 10000
                with self.assertRaises(ValueError): validate(c)

    def fixtures(self):
        c = self.config(); records = {}; refs = {}
        for j, t in enumerate(c['trials']):
            i = t['index_id']
            # Reverse an early peak so best-checkpoint selection would give the wrong ranking.
            records[t['id']] = {**t, 'epochs': {str(e): .9-j*.01 if e==5 else (j+1)*.01
                                               for e in [5,10,15,20,25,30]}}
            refs[i] = {'epochs': {'30': (j+1)*.01}}
            c['proxy_reference']['indexes'][i]['scores_by_seed'] = {'31415': (j+1)*.01, '27182': (j+1)*.02}
        return c, records, refs

    def test_primary_target_is_fixed30_and_mini_seeds_are_averaged(self):
        c, records, refs = self.fixtures()
        r = correlate_target(c, records, refs)
        group = r['groups']['all']['estimators']
        self.assertAlmostEqual(group['mini_mean10']['spearman'], 1.)
        self.assertAlmostEqual(group['early_new_full5']['spearman'], -1.)
        self.assertAlmostEqual(r['estimator_scores']['mini_mean10'][0], .015)
        self.assertEqual(group['mini_mean10']['selected_index'], INDEX_IDS[-1])
        self.assertEqual(group['mini_mean10']['shortlist3_target_regret'], 0.)
        self.assertFalse(r['test_evaluated'])

    def test_incomplete_wrong_seed_or_missing_epoch_is_not_a_complete_calibration(self):
        for change in ['missing', 'seed', 'epoch']:
            c, records, refs = self.fixtures()
            k = next(iter(records))
            if change == 'missing': records.pop(k)
            if change == 'seed': records[k]['seed'] = 31415
            if change == 'epoch': records[k]['epochs'].pop('30')
            with self.assertRaises(ValueError): correlate_target(c, records, refs)


if __name__ == '__main__':
    unittest.main()
