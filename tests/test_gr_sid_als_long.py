"""A longer run compares the byte-pinned original and selected swapped paths."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.full_sid_index import prepare_full_sid_indices, load_generator_index
from rhpr.gr.generator_suite import assignments
from rhpr.gr.sid import SidIndex

ROOT = Path(__file__).resolve().parents[1]


class LongRunTests(unittest.TestCase):
    def test_two_matched_fits_have_sixty_complete_epochs_and_six_validations(self):
        c = json.loads((ROOT/'experiments/amazon_beauty_sid_als_long_2t4.json').read_text())
        screen = json.loads((ROOT/'experiments/amazon_beauty_sid_als_screen_2t4.json').read_text())
        for key in ['seed', 'data', 'generator', 'quantizer', 'execution', 'frozen_indices',
                    'expected_initial_state_sha256', 'expected_train_cases_sha256']:
            self.assertEqual(c[key], screen[key])
        for key, value in screen['training'].items():
            if key not in ['max_epochs', 'min_epochs', 'validation_interval', 'patience_epochs', 'max_seconds']:
                self.assertEqual(c['training'][key], value)
        self.assertEqual(c['training']['max_epochs'], 60)
        self.assertEqual(c['training']['min_epochs'], 60)
        self.assertEqual(c['training']['validation_interval'], 10)
        self.assertFalse(c['evaluation']['test_after_selection'])
        self.assertTrue(c['evaluation']['save_validation_predictions'])
        self.assertNotIn('controls', c)
        jobs = assignments(c, 2)
        self.assertEqual([[t['id'] for t in j['trials']] for j in jobs],
                         [['t5_rqkmeans_original'], ['t5_rqkmeans_uniform']])
        self.assertTrue(c['trials'][0]['use_original_paths'])
        self.assertEqual(c['trials'][1], next(t for t in screen['trials'] if t['id']=='t5_rqkmeans_uniform'))

    def fixture(self, root):
        assets=root/'assets';assets.mkdir()
        original=SidIndex(np.array([[0,1,0],[0,1,0],[1,0,0],[1,0,0]]), width=4).sids
        owner=np.array([1,2,3,0]);changed=original[owner]
        for name, a in [('original', original), ('changed', changed), ('owner', owner)]:
            np.save(assets/(name+'.npy'), a)
        (assets/'catalogue.json').write_text(json.dumps({'original_item_ids':list(range(4))}))
        candidate=dict(family='fixture',role='swapped',variant='als_uniform',semantic_lambda=.3,
                       sids='changed.npy',original_sids='original.npy',owner='owner.npy')
        manifest=dict(kind='frozen_sid_graph_placement',data_sha256={},indices={'swap':candidate},
                      files=[dict(path=p.name,size_bytes=p.stat().st_size,sha256=file_sha256(p)) for p in assets.iterdir()])
        (assets/'resources-manifest.json').write_text(json.dumps(manifest))
        swap=dict(id='swap',asset_id='swap',family='fixture',role='swapped',expected_variant='als_uniform',semantic_lambda=.3)
        control={**swap,'id':'original','role':'original','expected_variant':'original','semantic_lambda':0,'use_original_paths':True}
        c=dict(frozen_indices=dict(path=str(assets),manifest_sha256=file_sha256(assets/'resources-manifest.json')),
               data=dict(sha256={}),quantizer=dict(num_hierarchies=3,codebook_size=4),trials=[control,swap])
        return c, original, changed

    def test_original_control_uses_pinned_original_bytes_and_identity_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c, original, changed=self.fixture(root)
            records=prepare_full_sid_indices(c,list(range(4)),root/'out')
            for name, expected in [('original', original), ('swap', changed)]:
                p=root/'out/indices'/name
                np.testing.assert_array_equal(load_generator_index(p,4).sids,expected)
            np.testing.assert_array_equal(np.load(root/'out/indices/original/owner.npy'),np.arange(4))
            self.assertEqual(file_sha256(root/'assets/original.npy'),file_sha256(root/'out/indices/original/semantic-ids.npy'))
            self.assertEqual(records['original']['source']['role'],'original')
            self.assertNotIn('selected_snapshot',records['original']['source'])
            # The saved within-tuple suffix ownership must survive this import.
            self.assertFalse(np.array_equal(changed,SidIndex(changed[:,:3],width=4).sids))

    def test_control_opt_in_does_not_allow_relabeling_or_wrong_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,_,_=self.fixture(root)
            for i, delta in enumerate([{'role':'swapped'}, {'semantic_lambda':.3},
                    {'expected_variant':'als_uniform'}, {'family':'other'}, {'use_original_paths':1}]):
                bad=copy.deepcopy(c);bad['trials']=[{**bad['trials'][0],**delta}]
                with self.assertRaisesRegex(ValueError,'identity'):
                    prepare_full_sid_indices(bad,list(range(4)),root/('bad'+str(i)))
            bad=copy.deepcopy(c);bad['trials']=[{**bad['trials'][1],'role':'original'}]
            with self.assertRaisesRegex(ValueError,'identity'):
                prepare_full_sid_indices(bad,list(range(4)),root/'unrequested')


if __name__=='__main__':
    unittest.main()
