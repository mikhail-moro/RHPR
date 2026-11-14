import copy,json,unittest
from pathlib import Path
from rhpr.gr.kindle_capacity import validate,check_parent


class KindleCapacityTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.c=json.loads((root/'experiments/kindle_capacity_2t4.json').read_text())
        self.old=json.loads((root/'experiments/kindle_long_windows_2t4.json').read_text())

    def test_only_two_new_fits_and_same_frozen_index_protocol(self):
        validate(self.c);check_parent(self.c,self.old)
        self.assertEqual([t['history_items']for t in self.c['trials']],[20,32])
        self.assertEqual({t['seed']for t in self.c['trials']},{42})
        self.assertEqual(self.c['training'],self.old['training'])
        self.assertIn('frozen-indexes/items-42.npy',self.c['parent']['files'])
        for w in [20,32]:self.assertIn(f'trials/recent{w}_42/training-curves.json',self.c['parent']['files'])

    def test_reject_changed_training_targets_and_unrequested_architecture(self):
        for group,key,value in [('training','learning_rate',.001),('training','patience_epochs',30),('data','train_examples',100000),('generator','encoder_layers',6),('generator','dropout',.2),('evaluation','test_after_selection',True)]:
            changed=copy.deepcopy(self.c);changed[group][key]=value
            with self.assertRaises(ValueError):check_parent(changed,self.old)
        changed=copy.deepcopy(self.c);changed['generator']['d_kv']=32
        with self.assertRaises(ValueError):validate(changed)
        changed=copy.deepcopy(self.c);changed['trials'][1]['history_items']=64
        with self.assertRaises(ValueError):validate(changed)

    def test_baseline_has_complete_pinned_prediction_trajectories(self):
        for w,epochs in [(20,range(5,46,5)),(32,range(5,56,5))]:
            for epoch in epochs:
                record=self.c['parent']['files'][f'trials/recent{w}_42/validation-epoch-{epoch:03d}.npz']
                self.assertEqual(len(record['sha256']),64);self.assertGreater(record['size_bytes'],0)
