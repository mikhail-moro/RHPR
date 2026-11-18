import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from rhpr.gr.ml1m_last8 import (
    CONTROL_TRIALS, EXTENDED_CONTROL_TRIALS, REMAINING_TRIALS, TRIALS, WINDOWS, build_last8_cases,
    prepare_raw, trial_config, validate, verify_results,
)


class ML1MLast8Tests(unittest.TestCase):
    def test_last8_positions_and_causal_contexts_are_shared_by_all_windows(self):
        histories = {7: np.arange(20), 3: np.arange(45)}
        result = build_last8_cases(histories)
        for split, (h, u, t, pos) in result.items():
            self.assertEqual(len(t), 16 if split == 'train' else 2)
            for row, uid, target, p in zip(h, u, t, pos):
                seq = histories[int(uid)]
                self.assertEqual(target, seq[p])
                np.testing.assert_array_equal(row[row >= 0], seq[max(0, p-256):p])
                for width in WINDOWS:
                    tail = row[-width:]
                    np.testing.assert_array_equal(tail[tail >= 0], seq[max(0, p-width):p])
            for uid, seq in histories.items():
                positions = pos[u == uid]
                expected = (np.arange(len(seq)-10, len(seq)-2) if split == 'train' else
                            np.array([len(seq)-(2 if split == 'validation' else 1)]))
                np.testing.assert_array_equal(positions, expected)
        # VAL is observed at TEST, but neither heldout label enters TRAIN.
        self.assertEqual(result['test'][0][0, -1], histories[3][-2])
        for uid, seq in histories.items():
            self.assertFalse(np.isin(seq[-2:], result['train'][2][result['train'][1] == uid]).any())

    def test_short_inputs_rejected_instead_of_repeating_targets(self):
        for histories in ({1:np.arange(10)}, {1:np.array([-1]*20)}, {1:np.arange(20, dtype=float)}):
            with self.assertRaises(ValueError): build_last8_cases(histories)

    def test_native_metadata_all_ratings_and_users_without_external_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'ml-1m.zip'
            movies = '\n'.join(f'{i}::Film {i} (2000)::Drama|Comedy' for i in range(1, 23))
            # Reverse source order and equal timestamps: sorting is timestamp then movie ID.
            ratings = '\n'.join(f'1::{i}::{1 if i%2 else 5}::{100+i//2}' for i in range(20, 0, -1))
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('ml-1m/movies.dat', movies); z.writestr('ml-1m/ratings.dat', ratings)
            texts, report = prepare_raw(archive, root/'prepared')
            self.assertEqual((report['users'], report['items'], report['events'], report['train_examples']), (1, 20, 20, 8))
            self.assertEqual(report['train_events'], 18)
            self.assertEqual(len(texts), 20)
            self.assertEqual(texts[0], 'Title: Film 1 (2000). Genres: Drama, Comedy.')
            for split, positions in [('train', np.arange(10, 18)), ('validation', [18]), ('test', [19])]:
                with np.load(root/'prepared'/f'{split}-cases.npz') as d:
                    np.testing.assert_array_equal(d['targets'], positions)
                    for row, target in zip(d['contexts'], d['targets']):
                        np.testing.assert_array_equal(row[row >= 0], np.arange(target))
                with np.load(root/'prepared'/f'{split}-anchors.npz') as d:
                    np.testing.assert_array_equal(d['target_positions'], positions)
            with self.assertRaises(ValueError): prepare_raw(archive, root/'bad', {'items':22})

    def test_only_six_fits_and_microbatching_keeps_effective_batch(self):
        c = json.loads(Path('experiments/ml1m_last8_windows_2t4.json').read_text()); validate(c)
        before = copy.deepcopy(c)
        # reduce-overhead enables CUDA Graphs, which failed with native GRID's
        # shared embeddings on all six windows before completing an epoch.
        self.assertEqual(c['training']['compile_training'], dict(
            options={'triton.cudagraphs': False}, fullgraph=False, dynamic=False))
        bad = copy.deepcopy(c)
        bad['training']['compile_training'] = dict(mode='reduce-overhead', fullgraph=False, dynamic=False)
        with self.assertRaises(ValueError): validate(bad)
        for trial in TRIALS:
            active = trial_config(c, trial)
            self.assertEqual(active['data']['history_items'], trial['history_items'])
            self.assertEqual(active['training']['batch_size'], 512)
            self.assertEqual(512 % active['training']['physical_batch_size'], 0)
            self.assertEqual(active['generator'], c['generator'])
        self.assertEqual(c, before)
        for change in ['extra_fit', 'other_seed', 'mini', 'changed_sampling', 'changed_precision']:
            bad = copy.deepcopy(c)
            if change == 'extra_fit': bad['trials'].append(dict(id='extra', seed=42, history_items=20))
            elif change == 'other_seed': bad['seed'] = 101
            elif change == 'mini': bad['generator']['d_model'] = 64
            elif change == 'changed_sampling': bad['sampling']['targets_per_user'] = 1
            else: bad['training']['evaluation_precision'] = 'fp16'
            with self.assertRaises(ValueError): validate(bad)

    def test_matching_verification_catches_changed_targets_or_index(self):
        c = json.loads(Path('experiments/ml1m_last8_windows_2t4.json').read_text())
        row = dict(initial_state_sha256='same', sid_file_sha256='same', raw_case_sha256={'train':'same'},
            case_targets_sha256={'train':'same'}, test_case_sha256='same')
        results = {t['id']:copy.deepcopy(row) for t in TRIALS}
        verify_results(c, results)
        for key in row:
            bad = copy.deepcopy(results); bad[TRIALS[0]['id']][key] = 'different'
            with self.assertRaises(ValueError): verify_results(c, bad)

    def test_remaining_run_trains_only_missing_windows_with_same_model_targets_and_index(self):
        c = json.loads(Path('experiments/ml1m_last8_remaining_2t4.json').read_text())
        old = json.loads(Path('experiments/ml1m_last8_windows_2t4.json').read_text())
        validate(c)
        self.assertEqual(c['windows'], [16, 64])
        self.assertEqual(c['trials'], REMAINING_TRIALS)
        for key in ('generator', 'training', 'quantizer', 'data', 'sampling',
                    'expected_raw_case_sha256', 'expected_initial_state_sha256', 'kaggle'):
            self.assertEqual(c[key], old[key], key)
        for trial in c['trials']:
            self.assertNotIn(trial['history_items'], [8, 32, 128, 256])
            self.assertEqual(trial_config(c, trial)['training']['batch_size'], 512)
        for field, value in [('windows', WINDOWS), ('trials', TRIALS),
                             ('expected_sid_file_sha256', '0'*64), ('remaining_windows', None)]:
            bad=copy.deepcopy(c);bad[field]=value
            with self.assertRaises(ValueError): validate(bad)

    def test_controls_use_each_mixed_seed_with_matching_base_and_original_microbatches(self):
        cfg = json.loads(Path('experiments/ml1m_last8_mixed120_controls_2t4.json').read_text())
        mixed = json.loads(Path('experiments/ml1m_last8_mixed120_seeds_2t4.json').read_text())
        validate(cfg)
        self.assertEqual(cfg['trials'], CONTROL_TRIALS)
        for field in ('data', 'training', 'generator', 'frozen_parent'):
            self.assertEqual(cfg[field], mixed[field])
        results = {}
        for trial in CONTROL_TRIALS:
            active = trial_config(cfg, trial)
            seed, width = trial['seed'], trial['history_items']
            self.assertEqual(active['seed'], seed)
            self.assertEqual(active['expected_initial_state_sha256'],
                             mixed['replication_initializers'][str(seed)]['base'])
            self.assertEqual(active['training']['physical_batch_size'], 4096 // width)
            results[trial['id']] = dict(seed=seed, initial_state_sha256=active['expected_initial_state_sha256'],
                sid_file_sha256='shared', raw_case_sha256={}, case_targets_sha256={}, test_case_sha256='shared')
        verify_results(cfg, results)
        results['recent128_101']['initial_state_sha256'] = results['recent128_202']['initial_state_sha256']
        with self.assertRaisesRegex(ValueError, 'initialization'):
            verify_results(cfg, results)

    def test_extended_budget_matches_recent96_and_resumes_only_recent128(self):
        cfg = json.loads(Path('experiments/ml1m_last8_mixed64_controls_2t4.json').read_text())
        previous = json.loads(Path('experiments/ml1m_last8_mixed120_controls_2t4.json').read_text())
        validate(cfg)
        self.assertEqual(cfg['trials'], EXTENDED_CONTROL_TRIALS)
        self.assertEqual(cfg['training']['scheduler'], previous['training']['scheduler'])
        for trial in cfg['trials']:
            active = trial_config(cfg, trial)
            if trial['history_items'] == 128:
                old = trial_config(previous, trial)
                self.assertEqual(active['data'], old['data'])
                limits = dict(min_epochs=40,max_epochs=40,max_seconds=7200)
                self.assertEqual({**active['training'], **limits}, old['training'])
            else:
                self.assertEqual(active['training']['physical_batch_size'], 64)
                self.assertEqual(active['training']['evaluation_batch_size'], 16)
        broken = copy.deepcopy(cfg)
        del broken['resume_parent']['files']['trials/recent128_101/generator-last.pt']
        with self.assertRaises(ValueError):
            validate(broken)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'requires Torch/HF')
    def test_actual_two_worker_event_collection_and_partial_failure(self):
        from unittest.mock import patch
        from rhpr.gr.ae_suite import assignments, launch_workers
        fixture = '''
import argparse
from unittest.mock import patch
import torch
import rhpr.gr.ml1m_last8 as m
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--slot',type=int);a=p.parse_args()
def trial(t,plan):
    if plan.get('injected_failure')==t['id']:raise RuntimeError('injected protocol failure')
    return dict(trial=t['id'],seed=42,initial_state_sha256='matched')
with patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_capability',return_value=(7,5)),patch.object(torch.cuda,'empty_cache'),patch.object(m,'run_trial',side_effect=trial):
    m.worker(a.plan,a.slot)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'last8_protocol_fixture.py').write_text(fixture)
            source=Path(__file__).resolve().parents[1]/'src'
            for fail in (None, 'recent128_42'):
                c=json.loads(Path('experiments/ml1m_last8_windows_2t4.json').read_text())
                jobs=assignments(c,2); out=root/('success' if fail is None else 'failure'); (out/'workers').mkdir(parents=True)
                plan=out/'plan.json'; plan.write_text(json.dumps(dict(config=c,output=str(out),assignments=jobs,injected_failure=fail)))
                with patch.object(sys,'path',[str(root),str(source),*sys.path]):
                    if fail is None:
                        result=launch_workers(plan,worker_module='last8_protocol_fixture',event_prefix='ml1m_last8')
                        self.assertEqual(len(result['results']),6)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'incomplete'):
                            launch_workers(plan,worker_module='last8_protocol_fixture',event_prefix='ml1m_last8')
                        result=json.loads((out/'suite-results.json').read_text())
                        self.assertEqual(len(result['results']),5)
                        self.assertEqual(result['failures'][0]['trial'],fail)
                for job in jobs:
                    events=[json.loads(line) for line in (out/'workers'/f"worker-{job['slot']}.log").read_text().splitlines() if line.startswith('{')]
                    self.assertEqual([e['trial'] for e in events if e['phase']=='ml1m_last8_trial_start'],[t['id'] for t in job['trials']])


if __name__ == '__main__': unittest.main()
