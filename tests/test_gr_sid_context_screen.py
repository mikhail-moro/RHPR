"""Fixed-budget launch contract and optional real frozen-asset roundtrip."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhpr.gr.full_sid_index import prepare_full_sid_indices, load_generator_index
from rhpr.gr.generator_suite import assignments

ROOT = Path(__file__).resolve().parents[1]
ASSETS = os.environ.get('RHPR_CONTEXT_SCREEN_ASSETS')


class ContextScreenTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / 'experiments/amazon_beauty_sid_context_screen_2t4.json').read_text())

    def test_four_matched_twenty_epoch_fits_and_exact_controls(self):
        c = self.config()
        previous = json.loads((ROOT / 'experiments/amazon_beauty_sid_als_long_2t4.json').read_text())
        for key in ['seed', 'data', 'generator', 'quantizer', 'execution',
                    'expected_initial_state_sha256', 'expected_train_cases_sha256']:
            self.assertEqual(c[key], previous[key])
        for key, value in previous['training'].items():
            if key not in {'max_epochs', 'min_epochs', 'patience_epochs', 'validation_interval', 'max_seconds'}:
                self.assertEqual(c['training'][key], value)
        self.assertEqual(c['training']['max_epochs'], 20)
        self.assertEqual(c['training']['min_epochs'], 20)
        self.assertEqual(c['training']['patience_epochs'], 20)
        self.assertEqual(c['training']['validation_interval'], 5)
        self.assertFalse(c['training']['validate_first_epoch'])
        self.assertFalse(c['evaluation']['test_after_selection'])
        self.assertTrue(c['evaluation']['save_validation_predictions'])
        self.assertEqual([[t['id'] for t in a['trials']] for a in assignments(c, 2)],
                         [['rqkmeans_original', 'context_als_20'], ['item_als', 'context_als_5']])
        self.assertTrue(c['trials'][0]['use_original_paths'])
        self.assertEqual(c['trials'][1]['expected_variant'], 'als_uniform')
        self.assertEqual({t['asset_id'] for t in c['trials'][2:]}, {'context_20', 'context_5'})
        self.assertEqual(c['kaggle']['machine_shape'], 'NvidiaTeslaT4')
        self.assertFalse(c['kaggle']['kernel_sources'])

    @unittest.skipUnless(ASSETS, 'Set RHPR_CONTEXT_SCREEN_ASSETS for the real private inputs')
    def test_real_packaged_indices_roundtrip_without_suffix_reassignment(self):
        c = copy.deepcopy(self.config())
        c['frozen_indices']['path'] = ASSETS
        assets = Path(ASSETS)
        source_ids = json.loads((assets / 'catalogue.json').read_text())['original_item_ids']
        original = np.load(assets / 'original.npy')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = prepare_full_sid_indices(c, source_ids, output)
            for trial in c['trials']:
                root = output / 'indices' / trial['id']
                index = load_generator_index(root, 256)
                owner = np.load(root / 'owner.npy')
                np.testing.assert_array_equal(index.sids, original[owner])
                self.assertEqual(records[trial['id']]['source']['variant'], trial['expected_variant'])
                for depth in range(1, 5):
                    a, n = np.unique(original[:, :depth], axis=0, return_counts=True)
                    b, m = np.unique(index.sids[:, :depth], axis=0, return_counts=True)
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(n, m)


if __name__ == '__main__':
    unittest.main()
