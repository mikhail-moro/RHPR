import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.history_windows import encode_window, prepare_cases, validate


class HistoryWindowsTests(unittest.TestCase):
    def test_window_changes_only_visible_history_and_preserves_suffix(self):
        sid = np.column_stack([np.arange(64), np.ones((64, 2), dtype=int), np.arange(64) % 7])
        h = np.full((31, 30), -1, dtype=np.int32)
        for n in range(1, 31): h[n, -n:] = np.arange(n)
        for w in (10, 20, 30):
            x = encode_window(h, sid, w)
            self.assertEqual(x.shape, (31, 4*w))
            for n, row in enumerate(x):
                values = row[row >= 0].reshape(-1, 4)-np.arange(4)*256
                np.testing.assert_array_equal(values, sid[max(0, n-w):n])
        for bad_width in (0, 15, 32, 64):
            with self.assertRaises(ValueError): encode_window(h, sid, bad_width)
        bad = h.copy(); bad[10, 0] = 0
        with self.assertRaises(ValueError): encode_window(bad, sid, 10)

    def test_all_targets_and_temporal_holdouts_are_shared_across_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); source = p/'source'; source.mkdir()
            (source/'catalog.json').write_text(json.dumps([dict(item_id=i, text=str(i)) for i in range(64)]))
            (source/'sequences.json').write_text(json.dumps([dict(user_id=1, train=list(range(30)), validation=30, test=31)]))
            _, report = prepare_cases(source, dict(users=1, items=64, train_examples=29), p/'cases')
            self.assertEqual(report['train_examples'], 29)
            sid = np.column_stack([np.arange(64), np.zeros((64, 3), dtype=int)])
            for split, expected_targets in [('train', list(range(1, 30))), ('validation', [30]), ('test', [31])]:
                with np.load(p/'cases'/(split+'-cases.npz')) as z:
                    self.assertEqual(z['targets'].tolist(), expected_targets)
                    for width in (10, 20, 30):
                        x = encode_window(z['contexts'], sid, width)
                        for row, target in zip(x, expected_targets):
                            actual = (row[row >= 0].reshape(-1, 4)-np.arange(4)*256)[:, 0].tolist()
                            self.assertEqual(actual, list(range(max(0, target-width), target)))
            seq = [dict(user_id=1, train=list(range(29)), validation=29, test=30)]
            (source/'sequences.json').write_text(json.dumps(seq))
            with self.assertRaises(ValueError): prepare_cases(source, {}, p/'short')

    def test_only_three_authorized_fits_and_previous_training_policy(self):
        c = json.loads(Path('experiments/steam_history_windows_2t4.json').read_text()); validate(c)
        self.assertEqual({v['history_items'] for v in c['trials']}, {10, 20, 30})
        self.assertEqual(len(c['trials']), 3)
        for field, value in [('max_epochs', 30), ('evaluation_precision', 'fp16')]:
            bad = copy.deepcopy(c); bad['training'][field] = value
            with self.assertRaises(ValueError): validate(bad)
        bad = copy.deepcopy(c); bad['cohorts']['kindle2014'] = {}
        with self.assertRaises(ValueError): validate(bad)

    def test_replication_is_disjoint_and_allows_only_two_original_window_fits(self):
        spec = importlib.util.spec_from_file_location('steam_cohort_preparation', 'scripts/prepare_steam_history_cohort.py')
        preparation = importlib.util.module_from_spec(spec); spec.loader.exec_module(preparation)
        eligible = np.arange(15239)
        original, _ = preparation.select_users(eligible)
        selected, excluded = preparation.select_users(eligible, replication=True)
        np.testing.assert_array_equal(excluded, original)
        self.assertEqual(len(set(selected)), 2000)
        self.assertFalse(set(selected) & set(original))
        np.testing.assert_array_equal(selected, preparation.select_users(eligible, replication=True)[0])
        with self.assertRaises(ValueError): preparation.select_users(np.arange(3999), replication=True)
        c = json.loads(Path('experiments/steam_history_replication_2t4.json').read_text()); validate(c)
        self.assertEqual([t['history_items'] for t in c['trials']], [30, 20])
        baseline = json.loads(Path('experiments/steam_history_windows_2t4.json').read_text())
        for key in ('training', 'generator', 'quantizer', 'expected_sid_file_sha256', 'expected_initial_state_sha256'):
            self.assertEqual(c[key], baseline[key])
        for change in ('extra_trial', 'wrong_seed', 'overlap'):
            bad = copy.deepcopy(c)
            if change == 'extra_trial': bad['trials'].append(baseline['trials'][-1])
            elif change == 'wrong_seed': bad['replication']['data_seed'] = 4242
            else: bad['replication']['overlapping_users'] = 1
            with self.assertRaises(ValueError): validate(bad)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'requires Torch/HF for CPU worker protocol test')
    def test_actual_worker_events_survive_parent_collection_and_preserve_partial_results(self):
        from unittest.mock import patch
        from rhpr.gr.ae_suite import assignments, launch_workers
        fixture = '''
import argparse
from unittest.mock import patch
import torch
import rhpr.gr.history_windows as h
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--slot',type=int);a=p.parse_args()
def trial(t,plan):
    if plan.get('injected_failure')==t['id']:raise RuntimeError('injected protocol test failure')
    return dict(trial=t['id'],seed=t['seed'],initial_state_sha256='matched')
with patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_capability',return_value=(7,5)),patch.object(torch.cuda,'empty_cache'),patch.object(h,'run_trial',side_effect=trial):
    h.worker(a.plan,a.slot)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'steam_protocol_fixture.py').write_text(fixture)
            source = Path(__file__).resolve().parents[1]/'src'
            paths = os.pathsep.join([str(root), str(source), os.environ.get('PYTHONPATH', '')])
            for config_name, fail in [('steam_history_windows_2t4', None), ('steam_history_windows_2t4', 'steam_recent20_42'),
                                      ('steam_history_replication_2t4', None), ('steam_history_replication_2t4', 'steam_recent20_42')]:
                c = json.loads(Path('experiments', config_name+'.json').read_text()); jobs = assignments(c, 2)
                out = root/(config_name+('_success' if fail is None else '_failure')); (out/'workers').mkdir(parents=True)
                plan = out/'plan.json'; plan.write_text(json.dumps(dict(config=c, output=str(out), assignments=jobs, injected_failure=fail)))
                # Only backend discovery and ML work are stubbed; the real worker,
                # OS pipes, event parser and status/result collector are exercised.
                with patch.dict(os.environ, {'PYTHONPATH': paths}), patch.object(sys, 'path', [str(root), str(source), *sys.path]):
                    if fail is None:
                        summary = launch_workers(plan, worker_module='steam_protocol_fixture', event_prefix='history_windows')
                        self.assertEqual(len(summary['results']), len(c['trials']))
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                            launch_workers(plan, worker_module='steam_protocol_fixture', event_prefix='history_windows')
                        summary = json.loads((out/'suite-results.json').read_text())
                        self.assertEqual(len(summary['results']), len(c['trials'])-1)
                        self.assertEqual(summary['failures'][0]['trial'], fail)
                for job in jobs:
                    events = [json.loads(line) for line in (out/'workers'/f"worker-{job['slot']}.log").read_text().splitlines() if line.startswith('{')]
                    starts = [e['trial'] for e in events if e['phase'] == 'history_windows_trial_start']
                    self.assertEqual(starts, [t['id'] for t in job['trials']])
