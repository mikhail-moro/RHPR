"""Change only TRAIN merge probabilities; preserve original Pair20 inference."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr import kindle_mixed20 as base
from rhpr.gr.experiment import _event, _write
from rhpr.gr.kindle_random_tokenization import RandomTokenization

CONSTANT25 = dict(kind='constant', probability=.25)
CONSTANT75 = dict(kind='constant', probability=.75)
RAMP = dict(kind='linear', start_probability=0., end_probability=1., start_epoch=1, end_epoch=60)
TRIALS = [dict(id='random20_p25_42', seed=42, history_items=20, mode='random', sampling=CONSTANT25),
          dict(id='random20_ramp01_42', seed=42, history_items=20, mode='random', sampling=RAMP),
          dict(id='random20_p75_42', seed=42, history_items=20, mode='random', sampling=CONSTANT75)]
POLICY = {**base.POLICY, 'merge_fraction': 'per_trial_schedule',
    'schedule': 'epoch_probability_only', 'inference': 'original_pair20',
    'recent_single_items_at_full_window': 10, 'test_not_used_for_schedule_or_checkpoint_selection': True}
SEEDS = base.SEEDS


def probability_at(schedule, epoch):
    if type(epoch) is not int or not 1 <= epoch <= 60:
        raise ValueError('schedule requires original epochs1..60')
    if schedule.get('kind') == 'constant' and set(schedule) == {'kind', 'probability'}:
        p = schedule['probability']
        if isinstance(p, bool) or not isinstance(p, (float, int)) or not 0 <= p <= 1:
            raise ValueError('invalid constant merge probability')
        return float(p)
    if schedule == RAMP:
        return (epoch-1)/59
    raise ValueError('only a constant probability or declared linear0->1 schedule is supported')


class ScheduledTokenization(RandomTokenization):
    def __init__(self, *args, sampling, **kwargs):
        probability_at(sampling, 1); probability_at(sampling, 60)
        if kwargs.get('history_items', 20) != 20:
            raise ValueError('sampling pilot keeps the original20event window')
        kwargs.setdefault('history_items', 20)
        super().__init__(*args, **kwargs)
        self.sampling = copy.deepcopy(sampling)

    def schedule(self, epoch):
        p = probability_at(self.sampling, epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.augmentation_seed, epoch, POLICY['count_rng_stream']]))
        return (rng.random(((self.n+self.batch_size-1)//self.batch_size, self.capacity)) < p).sum(1).astype(np.int8)

    def epoch_summary(self, epoch):
        return {**super().epoch_summary(epoch), 'merge_fraction': probability_at(self.sampling, epoch),
                'probability_schedule': self.sampling}


def validate(c):
    if (c['kind'] != 'gr_kindle_sampling' or c['trials'] != TRIALS or c['windows'] != [20]
        or c['sampling_policy'] != POLICY or c['seed_roles'] != SEEDS or c['expected_new_pairs'] != 0
        or c['cache_parent']['kernel'] != 'carasik/rhpr-kindle-mixed20'
        or c['cache_parent']['git_sha'] != '4026db0e479881586cad0cdaeae264e0ec4d06ef'
        or c['kaggle']['kernel_sources'] != [c['parent']['kernel'], c['cache_parent']['kernel']]):
        raise ValueError('expected only p25,p75,linear0->1 TRAIN fits with original Pair20 inference')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_mixed20', trials=base.TRIALS, windows=[20, 10],
        random_tokenization=base.POLICY, expected_new_pairs=5011)
    normalized['kaggle']['kernel_sources'] = [c['parent']['kernel']]
    base.validate(normalized)
    for rel in ('trials/mixed20_wide_42/trial-result.json', 'trials/mixed20_wide_42/training-curves.json',
                'trials/mixed20_wide_42/test-best-predictions.npz'):
        if rel not in c['cache_parent']['files']:
            raise ValueError('missing historical p50 control provenance')


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    torch._dynamo.config.recompile_limit = POLICY['compiler_recompile_limit']
    state = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('sampling_trial_start', trial=trial['id'], sampling=trial['sampling'])
        try:
            def factory(*args, **kwargs):
                return ScheduledTokenization(*args, **kwargs, sampling=trial['sampling'])
            policy = {**POLICY, 'probability_schedule': trial['sampling']}
            row = base.run_trial(trial, p, view_factory=factory, policy=policy)
            state['results'][trial['id']] = row; _event('sampling_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('sampling_trial_failed', **state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', state)
        gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output/'workers').mkdir(); torch.set_num_threads(1)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no hardware fallback')
    parent = locate_parent(c['parent']); cache_parent = locate_parent(c['cache_parent'])
    old = json.loads((parent/'run-manifest.json').read_text())['config']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        if c[key] != old[key]: raise ValueError('comparison changes '+key)
    data = resolve_dataset_directory(c['movie_input']['path'])
    for name, sha in c['movie_input']['sha256'].items():
        if file_sha256(data/name) != sha: raise ValueError('input hash differs: '+name)
    indexes = output/'frozen-indexes'; indexes.mkdir()
    for rel in ('items-42.npy', 'pair-keys.npy', 'pairs-42.npy'):
        shutil.copy2(cache_parent/'frozen-indexes'/rel, indexes/rel)
    old_keys, sets, missing = base.cache_plan(data, cache_parent)
    if len(missing): raise ValueError('all original old10adjacencies must already be cached')
    _write(output/'cache-report.json', dict(new_pairs=0, existing_pairs=len(old_keys), tokenizer_refitted=False,
        needed_by_split={s: len(k) for s, k in sets.items()}, original_sids_unchanged=True))
    for label, source in (('original', parent), ('p50', cache_parent)):
        for rel in c['parent' if label == 'original' else 'cache_parent']['files']:
            if rel.startswith(('trials/', 'baselines/', 'control-tests/')):
                dest = output/'baselines'/label/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source/rel, dest)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes),
        parent=str(parent), cache_parent=str(cache_parent), resources=str(resources), assignments=assignments(c, 2))
    path = output/'plan.json'; _write(path, p)
    overlay = scratch/'modern-site'; overlay.mkdir()
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    summary = launch_workers(path, worker_module='rhpr.gr.kindle_sampling', event_prefix='sampling')
    if set(summary['results']) != {t['id'] for t in TRIALS}: raise RuntimeError('missing completed fit')
    rows = list(summary['results'].values())
    for key in ('initial_state_sha256', 'sid_file_sha256', 'pair_sid_file_sha256', 'case_targets_sha256', 'raw_case_sha256', 'test_case_sha256'):
        if any(row[key] != rows[0][key] for row in rows[1:]): raise ValueError('unmatched controls: '+key)
    control = json.loads((cache_parent/'trials/mixed20_wide_42/trial-result.json').read_text())
    for key in ('initial_state_sha256', 'sid_file_sha256', 'pair_sid_file_sha256', 'case_targets_sha256', 'raw_case_sha256', 'test_case_sha256'):
        if rows[0][key] != control[key]: raise ValueError('historical p50 comparison changes '+key)
    if any(row['forward']['supported_forward_flops_batch1'] != 319748392 for row in rows):
        raise ValueError('all inference budgets must match original Random20')
    _write(output/'study-summary.json', dict(results=summary['results'], new_generator_fits=3, tokenizer_refitted=False,
        seed_roles=SEEDS, training_policy=POLICY, inference_format='original_pair20',
        reused_baselines=['random20_p50', 'recent10_wide_42', 'recent20_wide_42']))
    metrics = dict(experiment=c['name'], models=3, seed=42, test_evaluated=True)
    for name, row in summary['results'].items():
        metrics[name+'_validation_ndcg10'] = row['selection']['validation_ndcg10']
        metrics[name+'_test_ndcg10'] = row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops'] = row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); worker(args.plan, args.slot)
