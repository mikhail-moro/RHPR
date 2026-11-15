import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from rhpr.gr.kindle_sampling import (CONSTANT25, CONSTANT75, POLICY, RAMP, TRIALS,
    ScheduledTokenization, probability_at, validate)
from rhpr.gr.kindle_random_tokenization import RandomTokenization
from rhpr.gr.kindle_pairs import encode
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


class SamplingTest(unittest.TestCase):
    def setUp(self):
        from rhpr.gr import pair_sid, pair_unique4
        for module in (pair_sid, pair_unique4):
            self.addCleanup(setattr, module, 'NITEMS', module.NITEMS)

    def test_probability_endpoints_mean_and_invalid_schedules(self):
        self.assertEqual([probability_at(CONSTANT25, e) for e in (1, 60)], [.25, .25])
        self.assertEqual([probability_at(CONSTANT75, e) for e in (1, 60)], [.75, .75])
        p = [probability_at(RAMP, e) for e in range(1, 61)]
        self.assertEqual(p[0], 0); self.assertEqual(p[-1], 1)
        self.assertAlmostEqual(float(np.mean(p)), .5)
        self.assertTrue(np.all(np.diff(p) > 0))
        for bad in (dict(kind='constant', probability=-.1), dict(kind='constant', probability=True),
                    dict(kind='constant', probability=float('nan')), {**RAMP, 'end_epoch': 120}):
            with self.assertRaises(ValueError): probability_at(bad, 1)
        for epoch in (0, 61, 2.5, True):
            with self.assertRaises(ValueError): probability_at(RAMP, epoch)

    def test_p50_exactly_replays_original_tensors_schedules_shuffle_and_exposure(self):
        h, sid, keys, pairs = fixture()
        old = RandomTokenization(h, sid, keys, pairs, history_items=20)
        new = ScheduledTokenization(h, sid, keys, pairs, sampling=dict(kind='constant', probability=.5))
        for epoch in (1, 2):
            order = np.random.default_rng(42+epoch).permutation(len(h))
            for b, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]
                np.testing.assert_array_equal(old(epoch, b, part), new(epoch, b, part))
            summary = new.epoch_summary(epoch); summary.pop('probability_schedule')
            self.assertEqual(summary, old.epoch_summary(epoch))

    def test_ramp_endpoints_are_exact_full_and_original_pair_even_for_short_histories(self):
        h, sid, keys, pairs = fixture(21)
        v = ScheduledTokenization(h, sid, keys, pairs, sampling=RAMP)
        for epoch, expected in [(1, encode_window(h, sid, 20)), (60, encode(h, sid, keys, pairs, 20))]:
            order = np.random.default_rng(42+epoch).permutation(len(h))
            np.testing.assert_array_equal(v(epoch, 0, order), expected[order])
            self.assertEqual(v.epoch_summary(epoch)['merge_fraction'], probability_at(RAMP, epoch))

    def test_same_random_draws_give_nested_count_schedules_and_do_not_use_global_rng(self):
        h, sid, keys, pairs = fixture()
        views = [ScheduledTokenization(h, sid, keys, pairs, sampling=dict(kind='constant', probability=p)) for p in (.25, .5, .75)]
        np.random.seed(113); before = np.random.get_state()
        for epoch in range(1, 61):
            a, b, c = [v.schedule(epoch) for v in views]
            self.assertTrue(np.all(a <= b)); self.assertTrue(np.all(b <= c))
        after = np.random.get_state(); np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2:], after[2:])

    def test_microbatch_replay_and_summary_probability_do_not_depend_on_future_schedule_reads(self):
        h, sid, keys, pairs = fixture()
        a = ScheduledTokenization(h, sid, keys, pairs, sampling=RAMP)
        b = ScheduledTokenization(h, sid, keys, pairs, sampling=RAMP)
        for epoch in (25, 26):
            order = np.random.default_rng(42+epoch).permutation(len(h))
            for batch, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]; x = a(epoch, batch, part)
                np.testing.assert_array_equal(x, np.concatenate([b(epoch, batch, part[j:j+73]) for j in range(0, len(part), 73)]))
            a.schedule(60)
            self.assertEqual(a.epoch_summary(epoch), b.epoch_summary(epoch))
            self.assertEqual(a.epoch_summary(epoch)['merge_fraction'], probability_at(RAMP, epoch))

    def test_contract_has_exactly_three_requested_train_schedules_and_fixed_inference(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_sampling_2t4.json').read_text()); validate(c)
        old = json.loads((root/'experiments/kindle_mixed20_2t4.json').read_text())
        for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
            self.assertEqual(c[key], old[key])
        for change in ('epochs', 'fourth_trial', 'inference', 'seed', 'ramp_end'):
            x = copy.deepcopy(c)
            if change == 'epochs': x['training']['max_epochs'] = 100
            elif change == 'fourth_trial': x['trials'].append(x['trials'][0])
            elif change == 'inference': x['sampling_policy']['inference'] = 'random5_first16'
            elif change == 'seed': x['seed_roles']['augmentation_seed'] = 101
            else: x['trials'][1]['sampling']['end_epoch'] = 40
            with self.assertRaises(ValueError): validate(x)


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
        config = json.loads((root/'experiments/kindle_sampling_2t4.json').read_text())
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
                    expected = encode(h, sid, keys, pairs, 20)
                    np.testing.assert_array_equal(x, expected); np.testing.assert_array_equal(targets, t)
                    np.testing.assert_array_equal(users, u)
                    with self.assertRaises(ValueError): example_builder(None, None, 'test')
                    self.assertIsInstance(training_context_selector, ScheduledTokenization)
                    self.assertEqual(c['active_training_view_policy']['probability_schedule'], trial['sampling'])
                    if training_context_selector:
                        order = np.random.default_rng(43).permutation(len(h))
                        training_context_selector(1, 0, order)
                        self.assertTrue(training_context_selector.epoch_summary(1)['examples_consumed_once'])
                    (out/'training-curves.json').write_text(json.dumps([dict(train_seconds=.1)]*40))
                    return selection
                def test(model, examples, *args):
                    self.assertEqual(examples[0].shape[1], 60)
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
                    row = run_trial(trial, plan,
                        view_factory=lambda *args, **kwargs: ScheduledTokenization(*args, **kwargs, sampling=trial['sampling']),
                        policy={**POLICY, 'probability_schedule': trial['sampling']})
                self.assertEqual(row['forward']['encoder_positions'], 61)
                self.assertEqual(row['initial_state_sha256'], baseline['initial_state_sha256'])


if __name__ == '__main__': unittest.main()
