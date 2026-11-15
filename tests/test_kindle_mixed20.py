import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from rhpr.gr.kindle_mixed20 import TRIALS, extend_by_split, needed_pairs, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_random_tokenization import RandomTokenization
from rhpr.gr.kindle_windows import encode_window


def fixture(n=1031):
    ids = np.arange(64, dtype=np.int32)
    sid = np.column_stack([ids, ids % 3, ids % 5, ids % 7])
    keys = ids[:-1].astype(np.int64)*64+ids[1:]
    pairs = sid[:-1].copy(); h = np.full((n, 64), -1, dtype=np.int32)
    for i in range(n):
        length = i % 21
        if length: h[i, -length:] = np.arange(length)
    return h, sid, keys, pairs


class Mixed20Test(unittest.TestCase):
    def setUp(self):
        # The legacy ML100K case-loader changes these module globals on import.
        # Importing it inside a patched integration test must not alter later
        # ML1M fixtures in the same unittest process.
        from rhpr.gr import pair_sid, pair_unique4
        for module in (pair_sid, pair_unique4):
            self.addCleanup(setattr, module, 'NITEMS', module.NITEMS)

    def test_all_six_layouts_preserve_events_recent_half_suffixes_and_native_lengths(self):
        h, sid, keys, pairs = fixture(21)
        view = RandomTokenization(h, sid, keys, pairs, history_items=20)
        for count in range(6):
            x, _, _ = view.encode_batch(np.arange(21), count, 1, 0)
            self.assertEqual(x.shape, (21, 4*(20-count)))
            for original, encoded in zip(h, x):
                original = original[original >= 0]; recovered = []
                parts = encoded[encoded >= 0].reshape(-1, 4)
                for part in parts:
                    if part[0] >= 1024:
                        value = part-(4+np.arange(4))*256
                        item = int(value[0]); np.testing.assert_array_equal(value, pairs[item])
                        recovered.extend([item, item+1]); self.assertLessEqual(len(recovered), len(original)//2)
                    else:
                        value = part-np.arange(4)*256; item = int(value[0])
                        np.testing.assert_array_equal(value, sid[item]); recovered.append(item)
                np.testing.assert_array_equal(recovered, original)
                recent = len(original)-len(original)//2
                if recent: np.testing.assert_array_equal(parts[-recent:], sid[original[-recent:]]+np.arange(4)*256)
        np.testing.assert_array_equal(view.encode_batch(np.arange(21), 0, 1, 0)[0], encode_window(h, sid, 20))
        np.testing.assert_array_equal(view.encode_batch(np.arange(21), 5, 1, 0)[0], encode(h, sid, keys, pairs, 20))
        full = encode_window(h, sid, 20)
        np.testing.assert_array_equal(encode_window(h, sid, 10), full[:, -40:])

    def test_augmentation_seed_changes_tiling_without_changing_training_order_or_targets(self):
        h, sid, keys, pairs = fixture()
        a = RandomTokenization(h, sid, keys, pairs, history_items=20, augmentation_seed=42)
        b = RandomTokenization(h, sid, keys, pairs, history_items=20, augmentation_seed=101)
        micro = RandomTokenization(h, sid, keys, pairs, history_items=20, augmentation_seed=42)
        different = False
        for epoch in (1, 2):
            order = np.random.default_rng(42+epoch).permutation(len(h))
            for batch, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]
                x = a(epoch, batch, part); y = b(epoch, batch, part)
                different |= x.shape != y.shape or not np.array_equal(x, y)
                chunks = np.concatenate([micro(epoch, batch, part[j:j+73]) for j in range(0, len(part), 73)])
                np.testing.assert_array_equal(chunks, x)
                np.testing.assert_array_equal(a.order, b.order)
            self.assertEqual(a.epoch_summary(epoch), micro.epoch_summary(epoch))
            self.assertTrue(b.epoch_summary(epoch)['examples_consumed_once'])
        self.assertTrue(different)

    def test_short_history_merge_counts_and_cache_need_only_use_old_half(self):
        h, sid, keys, pairs = fixture(21)
        v = RandomTokenization(h, sid, keys, pairs, history_items=20)
        counts = [v.encode_batch(np.array([8]), 3, 1, b)[1] for b in range(300)]
        self.assertEqual(set(counts), {0, 1, 2})
        self.assertLess(abs(np.mean(counts)-1.2), .15)
        full = h[-1:]
        np.testing.assert_array_equal(needed_pairs(full, 64, True), keys[:9])
        np.testing.assert_array_equal(needed_pairs(full, 64, False), keys[:9:2])
        with self.assertRaises(ValueError): needed_pairs(np.zeros((1, 20)), 64, True)

    def test_extension_preserves_old_sids_and_allocates_train_before_heldout(self):
        oldkeys = np.array([100, 200]); oldsids = np.array([[1, 2, 3, 0], [1, 2, 3, 1]])
        missing = np.array([5, 7, 9]); codes = np.tile([1, 2, 3], (3, 1))
        sets = dict(train=np.array([9]), validation=np.array([7]), test=np.array([5]))
        keys, sids = extend_by_split(oldkeys, oldsids, sets, missing, codes)
        np.testing.assert_array_equal(sids[np.searchsorted(keys, oldkeys)], oldsids)
        self.assertEqual(sids[np.searchsorted(keys, 9), 3], 2)
        self.assertEqual(sids[np.searchsorted(keys, 7), 3], 3)
        self.assertEqual(sids[np.searchsorted(keys, 5), 3], 4)

    def test_contract_declares_only_requested_two_fits_and_unchanged_optimization(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_mixed20_2t4.json').read_text()); validate(c)
        old = json.loads((root/'experiments/kindle_random_tokenization.json').read_text())
        for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
            self.assertEqual(c[key], old[key])
        for mutation in ('third_trial', 'augmentation_seed', 'inference', 'probability', 'epochs'):
            changed = copy.deepcopy(c)
            if mutation == 'third_trial': changed['trials'].append(TRIALS[0])
            elif mutation == 'augmentation_seed': changed['seed_roles']['augmentation_seed'] = 101
            elif mutation == 'inference': changed['random_tokenization']['inference'] = 'full20'
            elif mutation == 'probability': changed['random_tokenization']['merge_fraction'] = .9
            else: changed['training']['max_epochs'] = 100
            with self.assertRaises(ValueError): validate(changed)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_both_trials_use_same_targets_fixed_validation_and_block_training_test_access(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        import torch
        from rhpr.gr.data import file_sha256
        from rhpr.gr.kindle_mixed20 import run_trial
        from rhpr.gr.movie_data import case_digest
        from rhpr.gr.sid_study import state_digest

        h, sid, keys, pairs = fixture(21); u = np.arange(21)+1; t = np.arange(21)
        sid[:, 3] = 0  # Each unique three-code prefix has ordinal suffix0.
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'experiments/kindle_mixed20_2t4.json').read_text())
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.ones(()))
            def rhpr_loss(self, batch): return self.w.square()
            def to(self, *args, **kwargs): return self
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            p = Path(temp); indexes = p/'indexes'; indexes.mkdir()
            for name, x in [('items-42', sid), ('pair-keys', keys), ('pairs-42', pairs)]: np.save(indexes/(name+'.npy'), x)
            baseline = dict(initial_state_sha256=state_digest(dict(Model().named_parameters())),
                sid_file_sha256=file_sha256(indexes/'items-42.npy'), forward=dict(scope='test'),
                raw_case_sha256={s: case_digest((h, u, t)) for s in ('train', 'validation')},
                case_targets_sha256={s: case_digest((u, t)) for s in ('train', 'validation')},
                test_case_sha256=case_digest((h, u, t)))
            dest = p/'parent/trials/pair32_wide_42'; dest.mkdir(parents=True)
            (dest/'trial-result.json').write_text(json.dumps(baseline))
            selection = dict(training_epochs=40, validation_checks=8, stopping_reason='early_stopping',
                optimizer_updates=40, amp_skipped_updates=0, best_epoch=35)
            for trial in TRIALS:
                def train(model, data, wrapped, c, out, device, *, example_builder, training_context_selector):
                    x, users, targets = example_builder(None, None, 'validation')
                    expected = encode(h, sid, keys, pairs, 20) if trial['mode'] == 'random' else encode_window(h, sid, 10)
                    np.testing.assert_array_equal(x, expected); np.testing.assert_array_equal(targets, t)
                    np.testing.assert_array_equal(users, u)
                    with self.assertRaises(ValueError): example_builder(None, None, 'test')
                    self.assertEqual(training_context_selector is not None, trial['mode'] == 'random')
                    if training_context_selector:
                        order = np.random.default_rng(43).permutation(len(h))
                        training_context_selector(1, 0, order)
                        self.assertTrue(training_context_selector.epoch_summary(1)['examples_consumed_once'])
                    (out/'training-curves.json').write_text(json.dumps([dict(train_seconds=.1)]*40))
                    return selection
                def test(model, examples, *args):
                    self.assertEqual(examples[0].shape[1], 60 if trial['mode'] == 'random' else 40)
                    np.testing.assert_array_equal(examples[2], t)
                    return dict(epoch=35, metrics={'ndcg@10': .01})
                plan = dict(config=config, indexes=str(indexes), parent=str(p/'parent'), output=str(p/trial['id']),
                    data='data', source='source')
                with patch('rhpr.gr.grid.build_grid_generator', side_effect=lambda *a: Model()), \
                     patch('rhpr.gr.pair_unique4.attach4', side_effect=lambda m: m), \
                     patch('rhpr.gr.ml100k_pair_raw.cases', return_value=(h, u, t)), \
                     patch('rhpr.gr.kindle_mixed20._train_generator', side_effect=train), \
                     patch('rhpr.gr.kindle_pairs.test_selected', side_effect=test), \
                     patch('torch.cuda.reset_peak_memory_stats'), \
                     patch('torch.cuda.max_memory_allocated', return_value=0), \
                     patch('torch.cuda.max_memory_reserved', return_value=0):
                    row = run_trial(trial, plan)
                self.assertEqual(row['forward']['encoder_positions'], 61 if trial['mode'] == 'random' else 41)
                self.assertEqual(row['initial_state_sha256'], baseline['initial_state_sha256'])


if __name__ == '__main__': unittest.main()
