import copy
import itertools
import importlib.util
import json
import math
from pathlib import Path
import unittest

import numpy as np

from rhpr.gr.kindle_random_tokenization import RandomTokenization, matching_starts, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_windows import encode_window


def fixture(n=1031):
    sids = np.column_stack([np.arange(128), np.zeros((128, 3), dtype=np.int32)]).astype(np.int32)
    keys = np.arange(127, dtype=np.int64)*128+np.arange(1, 128)
    pairs = sids[:127].copy()
    h = np.full((n, 64), -1, dtype=np.int32)
    for i in range(n):
        length = i % 33
        if length: h[i, -length:] = np.arange(length)+(i % 3)*32
    return h, sids, keys, pairs


class RandomTokenizationTest(unittest.TestCase):
    def test_every_layout_preserves_all_events_order_recent_singles_and_suffixes(self):
        h, sid, keys, pairs = fixture(100)
        v = RandomTokenization(h, sid, keys, pairs)
        odd = 0
        for count in range(9):
            out, _, odd_starts = v.encode_batch(np.arange(len(h)), count, 1, 0); odd += odd_starts
            self.assertEqual(out.shape, (100, 4*(32-count)))
            for row, raw in zip(out, h):
                original = raw[raw >= 0].tolist(); parts = row[row >= 0].reshape(-1, 4); recovered = []
                for part in parts:
                    if part[0] >= 1024:
                        code = part-(4+np.arange(4))*256
                        np.testing.assert_array_equal(code[1:], [0, 0, 0])
                        recovered += [int(code[0]), int(code[0])+1]
                        self.assertLessEqual(len(recovered), len(original)//2)
                    else:
                        code = part-np.arange(4)*256
                        np.testing.assert_array_equal(code[1:], [0, 0, 0]); recovered.append(int(code[0]))
                self.assertEqual(recovered, original)
                recent = len(original)-len(original)//2
                if recent:
                    np.testing.assert_array_equal(parts[-recent:], sid[original[-recent:]]+np.arange(4)*256)
        self.assertGreater(odd, 0)

    def test_endpoints_exactly_reproduce_full_and_original_pair_including_short_histories(self):
        h, sid, keys, pairs = fixture(100)
        v = RandomTokenization(h, sid, keys, pairs)
        np.testing.assert_array_equal(v.encode_batch(np.arange(100), 0, 2, 0)[0], encode_window(h, sid, 32))
        np.testing.assert_array_equal(v.encode_batch(np.arange(100), 8, 2, 0)[0], encode(h, sid, keys, pairs, 32))

    def test_uniform_matching_bijection_spans_both_phases_without_overlap(self):
        rng = np.random.default_rng(42)
        seen = {tuple(matching_starts(7, 2, rng)) for _ in range(1000)}
        expected = {v for v in itertools.combinations(range(6), 2) if v[1]-v[0] >= 2}
        self.assertEqual(len(expected), math.comb(7-2, 2)); self.assertEqual(seen, expected)
        for starts in seen: self.assertGreaterEqual(starts[1]-starts[0], 2)
        with self.assertRaises(ValueError): matching_starts(7, 4, rng)

    def test_short_histories_do_not_get_forced_to_maximum_compression(self):
        h, sid, keys, pairs = fixture(33); v = RandomTokenization(h, sid, keys, pairs)
        # Eight original events allow only2pairs. A full-window count4must not
        # always force those2pairs:its conditional expected count is1.
        counts = [v.encode_batch(np.array([8]), 4, 1, b)[1] for b in range(256)]
        self.assertEqual(set(counts), {0, 1, 2})
        self.assertLess(abs(float(np.mean(counts))-1), .2)

    def test_rng_microbatching_and_epoch_boundary_replay_do_not_change_tokenization(self):
        h, sid, keys, pairs = fixture()
        whole = RandomTokenization(h, sid, keys, pairs)
        micro = RandomTokenization(h, sid, keys, pairs)
        np.random.seed(819); state = np.random.get_state()
        for epoch in (1, 2):
            order = np.random.default_rng(42+epoch).permutation(len(h))
            fresh = RandomTokenization(h, sid, keys, pairs)
            for b, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]
                expected = whole(epoch, b, part)
                chunks = np.concatenate([micro(epoch, b, part[j:j+73]) for j in range(0, len(part), 73)])
                np.testing.assert_array_equal(expected, chunks)
                np.testing.assert_array_equal(expected, fresh(epoch, b, part))
            self.assertEqual(whole.epoch_summary(epoch), micro.epoch_summary(epoch))
            self.assertEqual(whole.epoch_summary(epoch), fresh.epoch_summary(epoch))
        after = np.random.get_state(); np.testing.assert_array_equal(after[1], state[1]); self.assertEqual(after[2:], state[2:])
        self.assertFalse(np.array_equal(whole.schedule(1), whole.schedule(2)))

    def test_missing_cached_pairs_duplicate_coverage_and_wrong_shuffle_fail_closed(self):
        h, sid, keys, pairs = fixture(100)
        with self.assertRaisesRegex(ValueError, 'missing'):
            RandomTokenization(h, sid, keys[1:], pairs[1:])
        v = RandomTokenization(h, sid, keys, pairs, batch_size=2)
        part = np.random.default_rng(43).permutation(100)[:2]
        v(1, 0, part); v(1, 0, part)
        with self.assertRaises(RuntimeError): v.epoch_summary(1)
        with self.assertRaises(RuntimeError): v(2, 0, part)
        v = RandomTokenization(h, sid, keys, pairs, batch_size=2)
        with self.assertRaisesRegex(ValueError, 'permutation'): v(1, 1, part)

    def test_contract_has_one_constant_probability_fit_and_unchanged_training(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_random_tokenization.json').read_text()); validate(c)
        for key, value in [('merge_fraction', .9), ('schedule', 'linear'), ('inference', 'full32')]:
            changed = copy.deepcopy(c); changed['random_tokenization'][key] = value
            with self.assertRaises(ValueError): validate(changed)
        old = json.loads((root/'experiments/kindle_mixed_views.json').read_text())
        for key in ('training', 'generator', 'data', 'movie_input', 'parent'):
            self.assertEqual(c[key], old[key])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_training_loop_keeps_targets_users_pair_validation_and_once_per_epoch(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        import torch
        from rhpr.gr.experiment import _train_generator
        from rhpr.gr.pair_unique4 import InputIndex4
        from rhpr.gr.sid import SidIndex
        h, sid, keys, pairs = fixture(16); users = np.arange(16)+1; targets = np.arange(16)
        pair = encode(h, sid, keys, pairs, 32); v = RandomTokenization(h, sid, keys, pairs, batch_size=2)
        seen = []; evaluated = []
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.ones(()))
            def rhpr_loss(self, batch):
                seen.append((batch['user_id'].clone(), batch['future_ids'].clone())); return self.w.square()
        def evaluate(model, examples, *args, **kwargs):
            np.testing.assert_array_equal(examples[0], pair); evaluated.append(True)
            return {'ndcg@10': .1}, np.zeros((16, 20, 4)), np.zeros((16, 20))
        c = dict(seed=42, data=dict(history_items=32), training=dict(batch_size=2, physical_batch_size=2,
            learning_rate=.001, weight_decay=.0001, precision='fp32', evaluation_precision='fp32',
            max_epochs=2, min_epochs=2, patience=5, max_seconds=60, grad_clip_norm=1.,
            validation_interval=1, evaluation_batch_size=2))
        with tempfile.TemporaryDirectory() as tmp, patch('rhpr.gr.experiment._evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = _train_generator(Model(), None, InputIndex4(SidIndex.from_full_sids(sid, width=256)), c,
                Path(tmp), 'cpu', example_builder=lambda *args: (pair, users, targets), training_context_selector=v)
            curves = json.loads((Path(tmp)/'training-curves.json').read_text())
        self.assertEqual(result['optimizer_updates'], 16); self.assertEqual(len(evaluated), 2)
        for u, t in seen: np.testing.assert_array_equal(t.numpy(), sid[u[:, 0].numpy()-1])
        self.assertTrue(all(x['training_views']['examples_consumed_once'] for x in curves))


if __name__ == '__main__': unittest.main()
