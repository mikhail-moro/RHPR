import copy
import json
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_alignment import (POSITIONS, VARIANTS, encode_layout,
    extend_sids, pair_keys, segments, validate)
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_windows import encode_window


class KindleAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.h = np.arange(2*64, dtype=np.int32).reshape(2, 64)
        self.sids = np.column_stack([np.arange(128)]*4).astype(np.int32)
        self.keys = np.unique(np.concatenate([pair_keys(self.h, v, 128) for v in VARIANTS]))
        self.pairs = np.column_stack([np.arange(len(self.keys))]*4).astype(np.int32)

    def test_every_layout_preserves_every_original_event_in_order(self):
        for v in VARIANTS:
            width = int(v[:2])
            for h in self.h:
                parts = segments(h, v)
                self.assertEqual([i for part in parts for i in part[1:]], list(h[-width:]))
                self.assertEqual(4*len(parts)+1, POSITIONS[v])
                if v != '32_single':
                    self.assertEqual(parts[-width//2:], [('i', int(i)) for i in h[-(width//2):]])
            with self.assertRaises(ValueError): segments(np.full(64, -1), v)

    def test_reference_and_no_pairs_exactly_equal_original_encoders(self):
        np.testing.assert_array_equal(encode_layout(self.h, '32_reference', self.sids, self.keys, self.pairs),
            encode(self.h, self.sids, self.keys, self.pairs, 32))
        np.testing.assert_array_equal(encode_layout(self.h, '32_single', self.sids, self.keys, self.pairs),
            encode_window(self.h, self.sids, 32))
        for v in VARIANTS:
            x = encode_layout(self.h, v, self.sids, self.keys, self.pairs)
            self.assertEqual(x.shape, (2, POSITIONS[v]-1))
            self.assertTrue((x >= 0).all())
        missing = int(np.searchsorted(self.keys, pair_keys(self.h, '32_reference', 128)[0]))
        with self.assertRaises(ValueError):
            encode_layout(self.h, '32_reference', self.sids, np.delete(self.keys, missing), np.delete(self.pairs, missing, axis=0))

    def test_cross_split_shift_variants_preserve_expected_old_pairs(self):
        val = np.arange(64); test = np.arange(1, 65)
        def pairs(h, v): return {p[1:] for p in segments(h, v) if p[0] == 'p'}
        self.assertFalse(pairs(val, '32_reference') & pairs(test, '32_reference'))
        self.assertEqual(len(pairs(val, '32_reference') & pairs(test, '32_edges')), 7)
        self.assertEqual(len(pairs(val, '32_edges') & pairs(test, '32_reference')), 7)
        self.assertEqual(len(pairs(val, '32_reference') & pairs(test, '34_first')), 8)
        self.assertEqual(len(pairs(val, '32_reference') & pairs(test, '30_first')), 6)
        self.assertFalse(pairs(val, '32_reference') & pairs(test, '30_last'))

    def test_extension_retains_original_sids_and_appends_unique_suffix(self):
        old = np.array([[1, 2, 3, 0], [1, 2, 3, 3], [2, 3, 4, 0]])
        keys, values = extend_sids(np.array([10, 30, 50]), old, np.array([20, 40]), np.array([[1, 2, 3], [1, 2, 3]]))
        np.testing.assert_array_equal(values[np.searchsorted(keys, [10, 30, 50])], old)
        np.testing.assert_array_equal(values[np.searchsorted(keys, [20, 40]), 3], [4, 5])
        self.assertEqual(len(np.unique(values, axis=0)), 5)
        empty_k, empty_v = extend_sids(np.array([10, 30, 50]), old, np.array([], dtype=int), np.empty((0, 3), dtype=int))
        np.testing.assert_array_equal(empty_k, [10, 30, 50]); np.testing.assert_array_equal(empty_v, old)
        with self.assertRaises(ValueError):
            extend_sids(np.array([10]), np.array([[1, 2, 3, 255]]), np.array([20]), np.array([[1, 2, 3]]))

    def test_frozen_contract_no_training_or_posthoc_test_selection(self):
        c = json.loads((Path(__file__).resolve().parents[1]/'experiments/kindle_alignment_2t4.json').read_text())
        validate(c)
        self.assertNotIn('learning_rate', c['training']); self.assertNotIn('max_epochs', c['training'])
        for field, value in [('selected_epoch', 60), ('training_forbidden', False), ('seen_items_filtered', True)]:
            changed = copy.deepcopy(c); changed['evaluation'][field] = value
            with self.assertRaises(ValueError): validate(changed)


if __name__ == '__main__': unittest.main()
