import copy
import itertools
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhpr.gr.kindle_frequency import (FrequencyTokenization, matching_tables,
    sample_matching, train_support, validate, TRIALS)


def fixture(n=1031):
    ids = np.arange(64, dtype=np.int32)
    sid = np.column_stack([ids, ids % 3, ids % 5, ids % 7])
    keys = ids[:-1].astype(np.int64)*64+ids[1:]
    pairs = sid[:-1].copy(); h = np.full((n, 64), -1, dtype=np.int32)
    for i in range(n):
        length = i % 21
        if length: h[i, -length:] = np.arange(length)
    support = np.where(ids % 3 == 1, 3, 12)
    return h, sid, keys, pairs, support


class FrequencyTest(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration_and_uniform_ties(self):
        rng = np.random.default_rng(5)
        for length in range(1, 11):
            costs = rng.integers(0, 3, (1, length-1))
            dp, ways = matching_tables(costs, np.array([length]))
            for k in range(min(5, length//2)+1):
                subsets = [s for s in itertools.combinations(range(length-1), k)
                           if all(b-a > 1 for a, b in zip(s, s[1:]))]
                values = [sum(costs[0, j] for j in s) for s in subsets]
                best = min(values)
                self.assertEqual(int(dp[0, 0, k]), best)
                self.assertEqual(int(ways[0, 0, k]), values.count(best))
                for _ in range(10):
                    x = sample_matching(dp, ways, costs, [k], rng)[0]
                    self.assertEqual(int(x.sum()), k)
                    self.assertEqual(int((x*costs[0]).sum()), best)
        costs = np.zeros((9000, 3), dtype=np.int16)
        dp, ways = matching_tables(costs, np.full(9000, 4))
        chosen = sample_matching(dp, ways, costs, np.ones(9000), rng)
        self.assertTrue(np.all(np.abs(chosen.mean(0)-1/3) < .025))

    def test_every_native_width_preserves_all_events_suffixes_and_latest_four(self):
        h, sid, keys, pairs, support = fixture(21)
        for mode in ('frequency', 'matched_random'):
            view = FrequencyTokenization(h, sid, keys, pairs, support, mode)
            for count in range(6):
                x, _, _ = view.encode_batch(np.arange(21), count, 1, 0)
                self.assertEqual(x.shape, (21, 4*(20-count)))
                for index, row in enumerate(x):
                    tokens = row[row >= 0].reshape(-1, 4); recovered = []
                    for token in tokens:
                        is_pair = token[0] >= 1024
                        raw = token-(np.arange(4)+(4 if is_pair else 0))*256
                        item = int(raw[0]); np.testing.assert_array_equal(raw, sid[item])
                        recovered.extend([item, item+1] if is_pair else [item])
                    np.testing.assert_array_equal(recovered, h[index][h[index] >= 0])
                    recent = index-index*4//5
                    if recent: np.testing.assert_array_equal(tokens[-recent:], sid[np.arange(index-recent, index)]+np.arange(4)*256)
                if mode == 'frequency':
                    k = view.last_starts.sum(1)
                    np.testing.assert_array_equal(view.last_rare_per_row, view.cost[np.arange(21), 0, k])
            np.testing.assert_array_equal(view.evaluation(), view.evaluation())

    def test_counts_padding_shuffle_microbatches_and_rare_priority_are_matched(self):
        h, sid, keys, pairs, support = fixture()
        a = FrequencyTokenization(h, sid, keys, pairs, support, 'frequency')
        b = FrequencyTokenization(h, sid, keys, pairs, support, 'matched_random')
        micro = FrequencyTokenization(h, sid, keys, pairs, support, 'frequency')
        improved = False
        for epoch in (1, 2):
            order = np.random.default_rng(42+epoch).permutation(len(h))
            for batch, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]; x = a(epoch, batch, part); y = b(epoch, batch, part)
                self.assertEqual(x.shape, y.shape)
                np.testing.assert_array_equal((x >= 0).sum(1), (y >= 0).sum(1))
                np.testing.assert_array_equal(x, np.concatenate([micro(epoch, batch, part[j:j+73]) for j in range(0, len(part), 73)]))
                self.assertLessEqual(a.last_rare_merged, b.last_rare_merged)
                improved |= a.last_rare_merged < b.last_rare_merged
            aa = a.epoch_summary(epoch); bb = b.epoch_summary(epoch)
            self.assertEqual(aa, micro.epoch_summary(epoch))
            for key in ('pair_summary_occurrences', 'padding_sid_positions', 'merge_count_batches', 'merge_count_examples', 'schedule_sha256'):
                self.assertEqual(aa[key], bb[key])
        self.assertTrue(improved)

    def test_train_support_excludes_heldout_and_deduplicates_users(self):
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp)/'histories.npz', items=np.array([0, 0, 1, 2, 3, 0, 1, 3, 2]), offsets=[0, 5, 9])
            np.testing.assert_array_equal(train_support(temp, 4), [2, 2, 0, 0])

    def test_aggressive_evaluation_uses_eight_pairs_and_four_recent_singles(self):
        h, sid, keys, pairs, support = fixture(21)
        v = FrequencyTokenization(h, sid, keys, pairs, support, 'matched_random', max_pairs=8)
        x = v.evaluation()
        self.assertEqual(x.shape, (21, 48))
        np.testing.assert_array_equal(np.flatnonzero(v.last_starts[20]), np.arange(0, 16, 2))
        np.testing.assert_array_equal(x[20, -16:], (sid[16:20]+np.arange(4)*256).reshape(-1))
        for m in range(9):
            encoded, count, _ = v.encode_batch(np.arange(21), m, 1, 0)
            self.assertEqual(encoded.shape, (21, 4*(20-m)))
            for n, row in enumerate(encoded):
                recovered = []
                for token in row[row >= 0].reshape(-1, 4):
                    is_pair = token[0] >= 1024; item = int(token[0]-(1024 if is_pair else 0))
                    recovered.extend([item, item+1] if is_pair else [item])
                np.testing.assert_array_equal(recovered, np.arange(n))

    def test_contract_rejects_extra_fits_threshold_or_optimizer_change(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_frequency_2t4.json').read_text()); validate(c)
        for change in ('threshold', 'epochs', 'trial', 'cache_count'):
            d = copy.deepcopy(c)
            if change == 'threshold': d['frequency_policy']['rare_train_users_at_most'] = 8
            elif change == 'epochs': d['training']['max_epochs'] = 100
            elif change == 'trial': d['trials'].append(d['trials'][0])
            else: d['expected_new_pairs'] = 0
            with self.assertRaises(ValueError): validate(d)


    def setUp(self):
        # The legacy ML100K case-loader changes these module globals on import.
        # Importing it inside a patched integration test must not alter later
        # ML1M fixtures in the same unittest process.
        from rhpr.gr import pair_sid, pair_unique4
        for module in (pair_sid, pair_unique4):
            self.addCleanup(setattr, module, 'NITEMS', module.NITEMS)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_both_trials_use_same_targets_fixed_validation_and_block_training_test_access(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        import torch
        from rhpr.gr.data import file_sha256
        from rhpr.gr.kindle_frequency import run_trial
        from rhpr.gr.movie_data import case_digest
        from rhpr.gr.sid_study import state_digest

        h, sid, keys, pairs, support = fixture(21); u = np.arange(21)+1; t = np.arange(21)
        sid[:, 3] = 0  # Each unique three-code prefix has ordinal suffix0.
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'experiments/kindle_frequency_2t4.json').read_text())
        class Model(torch.nn.Module):
            def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.ones(()))
            def rhpr_loss(self, batch): return self.w.square()
            def to(self, *args, **kwargs): return self
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            p = Path(temp); indexes = p/'indexes'; indexes.mkdir()
            for name, x in [('items-42', sid), ('pair-keys', keys), ('pairs-42', pairs)]: np.save(indexes/(name+'.npy'), x)
            np.save(indexes/'train-user-support.npy', support)
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
                    expected = FrequencyTokenization(h, sid, keys, pairs, support, trial['mode'], max_pairs=trial['max_pairs']).evaluation()
                    np.testing.assert_array_equal(x, expected); np.testing.assert_array_equal(targets, t)
                    np.testing.assert_array_equal(users, u)
                    with self.assertRaises(ValueError): example_builder(None, None, 'test')
                    self.assertIsNotNone(training_context_selector)
                    if training_context_selector:
                        order = np.random.default_rng(43).permutation(len(h))
                        training_context_selector(1, 0, order)
                        self.assertTrue(training_context_selector.epoch_summary(1)['examples_consumed_once'])
                    (out/'training-curves.json').write_text(json.dumps([dict(train_seconds=.1)]*40))
                    return selection
                def test(model, examples, *args):
                    self.assertEqual(examples[0].shape[1], 4*(20-trial['max_pairs']))
                    np.testing.assert_array_equal(examples[2], t)
                    return dict(epoch=35, metrics={'ndcg@10': .01})
                plan = dict(config=config, indexes=str(indexes), parent=str(p/'parent'), output=str(p/trial['id']),
                    data='data', source='source')
                with patch('rhpr.gr.grid.build_grid_generator', side_effect=lambda *a: Model()), \
                     patch('rhpr.gr.pair_unique4.attach4', side_effect=lambda m: m), \
                     patch('rhpr.gr.ml100k_pair_raw.cases', return_value=(h, u, t)), \
                     patch('rhpr.gr.kindle_frequency._train_generator', side_effect=train), \
                     patch('rhpr.gr.kindle_pairs.test_selected', side_effect=test), \
                     patch('torch.cuda.reset_peak_memory_stats'), \
                     patch('torch.cuda.max_memory_allocated', return_value=0), \
                     patch('torch.cuda.max_memory_reserved', return_value=0):
                    row = run_trial(trial, plan)
                self.assertEqual(row['forward']['encoder_positions'], 4*(20-trial['max_pairs'])+1)
                self.assertEqual(row['initial_state_sha256'], baseline['initial_state_sha256'])


if __name__ == '__main__': unittest.main()
