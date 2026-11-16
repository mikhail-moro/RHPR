import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.beauty_pairs import TailTokenization, compact, encode, pair_keys, prepare_cases, validate


class BeautyPairsTests(unittest.TestCase):
    def setUp(self):
        self.h = np.full((21, 20), -1, dtype=np.int32)
        for n in range(1, 21): self.h[n, -n:] = np.arange(n)
        self.sids = np.column_stack([np.arange(32), np.zeros((32, 3), dtype=int)]).astype(np.int32)
        self.keys = np.arange(31, dtype=np.int64)*32+np.arange(1, 32)
        self.pairs = np.column_stack([np.arange(31), np.ones((31, 3), dtype=int)]).astype(np.int32)

    def unpack(self, row):
        tokens = row[row >= 0].reshape(-1, 4); events = []; merged = []
        for tok in tokens:
            if tok[0] >= 1024:
                k = int(tok[0]-1024); events.extend([k, k+1]); merged.append(k)
            else: events.append(int(tok[0]))
        return events, merged

    def test_all_short_lengths_order_tail_and_suffix_endpoints(self):
        for tail in (3, 5):
            views = TailTokenization(self.h, self.sids, self.keys, self.pairs, tail=tail)
            for count in range(views.capacity+1):
                out, _, _ = views.encode_batch(np.arange(21), count, 1, 0)
                self.assertEqual(out.shape, (21, 4*(20-count)))
                for n, row in enumerate(out):
                    events, merged = self.unpack(row)
                    self.assertEqual(events, list(range(n)))
                    self.assertTrue(all(j+1 < max(n-tail, 0) for j in merged))
                    if count == 0: self.assertFalse(merged)
                if count == views.capacity:
                    np.testing.assert_array_equal(out, encode(self.h, self.sids, self.keys, self.pairs, tail=tail))
            for n in range(tail+2):
                self.assertFalse(self.unpack(out[n])[1])
            self.assertEqual(len(self.unpack(out[tail+2])[1]), 1)

    def test_cache_enumerates_train_adjacencies_not_targets_and_fixed_eval(self):
        for tail in (3, 5):
            random_keys = pair_keys(self.h, 32, tail, random_tiling=True)
            fixed = pair_keys(self.h, 32, tail, random_tiling=False)
            np.testing.assert_array_equal(random_keys, self.keys[:20-tail-1])
            np.testing.assert_array_equal(fixed, self.keys[:20-tail-1:2])
            self.assertTrue(np.isin(fixed, random_keys).all())
        with self.assertRaises(ValueError): TailTokenization(self.h, self.sids, self.keys[:1], self.pairs[:1], tail=3)

    def test_short_counts_are_binomial_not_clipped(self):
        # n=5,tail3 has capacity1; clipping a global Binomial(8,.5) would merge almost always.
        h = np.repeat(self.h[5:6], 2048, axis=0)
        views = TailTokenization(h, self.sids, self.keys, self.pairs, tail=3, batch_size=1)
        choices = views.schedule(1)
        merges = [views.encode_batch(np.array([i]), int(c), 1, i)[1] for i, c in enumerate(choices)]
        self.assertGreater(np.mean(merges), .45); self.assertLess(np.mean(merges), .55)

    def test_epoch_replay_microbatches_and_counts(self):
        h = np.tile(self.h, (4, 1)); n = len(h)
        a = TailTokenization(h, self.sids, self.keys, self.pairs, tail=3, batch_size=16)
        b = TailTokenization(h, self.sids, self.keys, self.pairs, tail=3, batch_size=16)
        order = np.random.default_rng(43).permutation(n)
        for begin in range(0, n, 16):
            chosen = order[begin:begin+16]; full = a(1, begin//16, chosen)
            parts = [b(1, begin//16, chosen[k:k+3]) for k in range(0, len(chosen), 3)]
            np.testing.assert_array_equal(full, np.concatenate(parts))
        self.assertEqual(a.epoch_summary(1), b.epoch_summary(1))
        self.assertEqual(a.epoch_summary(1)['protected_tail_items'], 3)
        with self.assertRaises(RuntimeError): a.epoch_summary(2)

    def test_invalid_padding_ids_and_boundaries_fail(self):
        bad = self.h.copy(); bad[10, 0] = 0
        with self.assertRaises(ValueError): compact(bad, 32)
        for tail in (0, 4, 20):
            with self.assertRaises(ValueError): TailTokenization(self.h, self.sids, self.keys, self.pairs, tail=tail)
        bad = self.h.copy(); bad[-1, -1] = 32
        with self.assertRaises(ValueError): encode(bad, self.sids, recent=20)

    def test_original_all_prefix_split_is_preserved_including_one_item_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'resources/data').mkdir(parents=True); (root/'out').mkdir()
            (root/'resources/data/catalog.json').write_text(json.dumps([dict(item_id=i, text=str(i)) for i in range(5)]))
            (root/'resources/data/sequences.json').write_text(json.dumps([dict(user_id=1, train=[0, 1, 2], validation=3, test=4)]))
            _, data, report = prepare_cases(dict(data={}), root/'resources', root/'out')
            self.assertEqual(report['train_examples'], 2)
            for split, contexts, targets in [('train', [[0], [0, 1]], [1, 2]), ('validation', [[0, 1, 2]], [3]), ('test', [[0, 1, 2, 3]], [4])]:
                with np.load(data/(split+'-cases.npz')) as z:
                    self.assertEqual([r[r >= 0].tolist() for r in z['contexts']], contexts)
                    self.assertEqual(z['targets'].tolist(), targets)

    def test_only_declared_four_fits_and_no_test_selection(self):
        c = json.loads(Path('experiments/beauty_pairs_2t4.json').read_text()); validate(c)
        self.assertEqual(len(c['trials']), 4)
        self.assertEqual(c['expected_pair_counts'], dict(train=51400, validation=28944, test=36736))
        for key, value in [('max_epochs', 30), ('evaluation_precision', 'fp16')]:
            bad = copy.deepcopy(c); bad['training'][key] = value
            with self.assertRaises(ValueError): validate(bad)
        bad = copy.deepcopy(c); bad['kaggle']['kernel_sources'] = ['unrelated/kernel']
        with self.assertRaises(ValueError): validate(bad)

    def test_recent12_truncates_raw_items_with_all_four_codes(self):
        out = encode(self.h, self.sids, recent=12)
        self.assertEqual(out.shape, (21, 48))
        for n, row in enumerate(out):
            actual = row[row >= 0].reshape(-1, 4)-np.arange(4)*256
            np.testing.assert_array_equal(actual, self.sids[max(0, n-12):n])
        np.testing.assert_array_equal(out, encode(self.h, self.sids, recent=20)[:, -48:])

    def test_recent12_control_has_one_fit_and_original_training_settings(self):
        c = json.loads(Path('experiments/beauty_recent12_control.json').read_text()); validate(c)
        old = json.loads(Path('experiments/beauty_pairs_2t4.json').read_text())
        for key in ('training', 'generator', 'quantizer', 'data', 'kaggle', 'expected_initial_state_sha256'):
            self.assertEqual(c[key], old[key])
        for change in ('extra_fit', 'wrong_window', 'wrong_index'):
            bad = copy.deepcopy(c)
            if change == 'extra_fit': bad['trials'].append(old['trials'][0])
            elif change == 'wrong_window': bad['trials'][0]['recent'] = 13
            else: bad['expected_sid_file_sha256'] = 'wrong'
            with self.assertRaises(ValueError): validate(bad)
