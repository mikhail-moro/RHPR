"""Matched stochastic Pair20 and Recent10 fits on the frozen Kindle cohort."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _event, _loss, _train_generator, _write
from rhpr.gr.kindle_random_tokenization import POLICY as ORIGINAL_POLICY, RandomTokenization

TRIALS = [dict(id='mixed20_wide_42', seed=42, history_items=20, mode='random'),
          dict(id='recent10_wide_42', seed=42, history_items=10, mode='recent')]
POLICY = {**ORIGINAL_POLICY, 'inference': 'pair20'}
SEEDS = dict(index_seed=42, training_seed=42, augmentation_seed=42)


def validate(c):
    from rhpr.gr.kindle_random_tokenization import TRIALS as OLD_TRIALS, validate as old_validate
    if (c['kind'] != 'gr_kindle_mixed20' or c['trials'] != TRIALS or c['windows'] != [20, 10]
        or c['random_tokenization'] != POLICY or c['seed_roles'] != SEEDS
        or c['execution'] != dict(worker_devices=['cuda:0', 'cuda:1'], cpu_threads_per_worker=1)
        or c['expected_new_pairs'] != 5011):
        raise ValueError('expected exactly Mixed20 and Recent10, fixed seeds and constant half probability')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_random_tokenization', trials=OLD_TRIALS, windows=[32],
        random_tokenization=ORIGINAL_POLICY,
        execution=dict(worker_devices=['cuda:0'], cpu_threads_per_worker=1))
    old_validate(normalized)
    for key in ('frozen-indexes/pair-centroids-42.npy',
                'baselines/recent20_wide_42/trial-result.json',
                'control-tests/recent20_wide_42/test-best-predictions.npz'):
        if key not in c['parent']['files']:
            raise ValueError('missing frozen centroid or Recent20 provenance pin')


def needed_pairs(contexts, nitems, random_tiling):
    """TRAIN admits any old-half adjacency; evaluation uses fixed original pairs."""
    h = np.asarray(contexts)
    if h.ndim != 2 or h.shape[1] != 64 or np.any(h < -1) or np.any(h >= nitems):
        raise ValueError('expected original 64-position contexts')
    h = h[:, -20:]; length = (h >= 0).sum(1)
    if np.any((np.arange(20)[None, :] < (20-length)[:, None]) != (h < 0)):
        raise ValueError('expected contiguous left padding')
    compact = np.take_along_axis(h, np.minimum((20-length)[:, None]+np.arange(20), 19), axis=1)
    starts = np.arange(9) if random_tiling else np.arange(0, 9, 2)
    valid = starts[None, :]+1 < (length//2)[:, None]
    keys = compact[:, starts].astype(np.int64)*nitems+compact[:, starts+1]
    return np.unique(keys[valid])


def cache_plan(data, parent):
    from rhpr.gr.ml100k_pair_raw import cases
    idx = Path(parent)/'frozen-indexes'
    old = np.load(idx/'pair-keys.npy'); sid = np.load(idx/'items-42.npy')
    sets = {s: needed_pairs(cases(data, s)[0], len(sid), s == 'train') for s in ('train', 'validation', 'test')}
    missing = np.setdiff1d(np.unique(np.concatenate(list(sets.values()))), old)
    return old, sets, missing


def extend_by_split(old_keys, old_sids, sets, missing, codes):
    """Allocate TRAIN additions first; held-out additions never renumber them."""
    from rhpr.gr.kindle_alignment import extend_sids
    keys, sids = old_keys, old_sids
    for split in ('train', 'validation', 'test'):
        new = np.setdiff1d(sets[split], keys)
        if len(new):
            at = np.searchsorted(missing, new)
            if np.any(at >= len(missing)) or not np.array_equal(missing[at], new):
                raise ValueError('missing pair code in extension')
            keys, sids = extend_sids(keys, sids, new, codes[at])
    np.testing.assert_array_equal(sids[np.searchsorted(keys, old_keys)], old_sids)
    return keys, sids


def prepare_cache(plan_path):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.pair_sid import predict_rq
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError('cache preparation requires isolated T4')
    parent = Path(p['parent'])/'frozen-indexes'; out = Path(p['indexes'])
    old, sets, missing = cache_plan(p['data'], p['parent'])
    if len(missing) != p['config']['expected_new_pairs']:
        raise ValueError('new pair count differs from CPU preflight')
    _event('mixed20_cache_start', new_pairs=len(missing))
    cache = Path(p['output'])/'pair-cache'; cache.mkdir()
    texts = json.loads((Path(p['data'])/'texts.json').read_text())
    encoder = SentenceTransformer(str(Path(p['resources'])/'sentence-t5-base'), device='cuda:0', local_files_only=True)
    encoder.max_seq_length = 2048; encoder.eval()
    encode_pair_cache(encoder, texts, missing, cache/'new-pair-embeddings.npy', ['First book', 'Second book'])
    del encoder; gc.collect(); torch.cuda.empty_cache()
    codes = predict_rq(np.load(cache/'new-pair-embeddings.npy', mmap_mode='r'),
        np.load(parent/'pair-centroids-42.npy'), 'cuda:0')
    np.save(cache/'new-pair-keys.npy', missing); np.save(cache/'new-pair-codes.npy', codes)
    keys, pairs = extend_by_split(old, np.load(parent/'pairs-42.npy'), sets, missing, codes)
    np.save(out/'pair-keys.npy', keys); np.save(out/'pairs-42.npy', pairs)
    report = dict(new_pairs=len(missing), original_pairs=len(old), total_pairs=len(keys),
        needed_by_split={s: len(k) for s, k in sets.items()},
        original_sids_unchanged=True, tokenizer_refitted=False, suffix_allocation='train_then_validation_then_test',
        max_suffix=int(pairs[:, 3].max()))
    _write(cache/'cache-report.json', report); _event('mixed20_cache_complete', **report)


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.kindle_pairs import encode, test_selected
    from rhpr.gr.kindle_windows import encode_window
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import InputIndex4, attach4
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    c = copy.deepcopy(p['config']); width = trial['history_items']; c['data']['history_items'] = width
    seed = c['seed_roles']['training_seed']; c['seed'] = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    root = Path(p['indexes']); sid = np.load(root/'items-42.npy')
    keys = np.load(root/'pair-keys.npy'); pairs = np.load(root/'pairs-42.npy')
    idx = SidIndex.from_full_sids(sid, width=256); wrapped = InputIndex4(idx)
    model = attach4(build_grid_generator(sid, c['generator'], seed, 'cpu', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    baseline = json.loads((Path(p['parent'])/'trials/pair32_wide_42/trial-result.json').read_text())
    if initial != baseline['initial_state_sha256']:
        raise ValueError('initializer differs from completed controls')
    random_mode = trial['mode'] == 'random'
    def encoding(h): return encode(h, sid, keys, pairs, 20) if random_mode else encode_window(h, sid, 10)
    raw = {s: cases(p['data'], s) for s in ('train', 'validation')}
    encoded = {s: (encoding(h), u, t) for s, (h, u, t) in raw.items()}
    views = RandomTokenization(raw['train'][0], sid, keys, pairs, history_items=20,
        seed=seed, augmentation_seed=c['seed_roles']['augmentation_seed']) if random_mode else None
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    model.eval(); probe = tuple(a[:1] for a in encoded['validation'])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss = _loss(model, wrapped.model_batch(*probe, 'cpu'))
    if not torch.isfinite(loss) or state_digest(dict(model.named_parameters())) != initial:
        raise RuntimeError('profiling changed weights or produced nonfinite loss')
    forward = dict(parameters=sum(v.numel() for v in model.parameters()), encoder_positions=probe[0].shape[1]+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()), scope=baseline['forward']['scope'])
    _write(out/'forward-flops.json', forward)
    _write(out/'training-view-provenance.json', dict(seed_roles=SEEDS, policy=POLICY if random_mode else 'recent10',
        evaluation_case_sha256={s: case_digest(a) for s, a in encoded.items()},
        original_train_target_count=len(raw['train'][2]), examples_doubled=False))
    model.to('cuda:0'); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    def builder(root, ignored, split):
        if split not in encoded: raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection = _train_generator(model, p['data'], wrapped, c, out, 'cuda:0',
        example_builder=builder, training_context_selector=views)
    epochs = selection['training_epochs']
    if (not 40 <= epochs <= 60 or selection['validation_checks'] != epochs//5
        or selection['stopping_reason'] not in ('epoch_limit', 'early_stopping')
        or selection['optimizer_updates']+selection['amp_skipped_updates'] != ((len(raw['train'][2])+511)//512)*epochs):
        raise RuntimeError('incomplete declared fit')
    curves = json.loads((out/'training-curves.json').read_text())
    timing = dict(wall_seconds=time.monotonic()-started, first_epoch_training_seconds=curves[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([r['train_seconds'] for r in curves[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    h, u, t = cases(p['data'], 'test')
    test = test_selected(model, (encoding(h), u, t), wrapped, out, out/'generator-best.pt', selection['best_epoch'])
    row = dict(trial=trial['id'], seed=seed, seed_roles=SEEDS, history_items=width, mode=trial['mode'],
        initial_state_sha256=initial, sid_file_sha256=file_sha256(root/'items-42.npy'),
        pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'), selection=selection, forward=forward,
        raw_case_sha256={s: case_digest(a) for s, a in raw.items()},
        case_targets_sha256={s: case_digest(a[1:]) for s, a in raw.items()},
        test_case_sha256=case_digest((h, u, t)), timing=timing, test=test, test_evaluated=True)
    if random_mode: row['random_tokenization'] = POLICY
    for key in ('sid_file_sha256', 'raw_case_sha256', 'case_targets_sha256', 'test_case_sha256'):
        if row[key] != baseline[key]: raise ValueError('comparison changes '+key)
    _write(out/'trial-result.json', row)
    return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    torch._dynamo.config.recompile_limit = POLICY['compiler_recompile_limit']
    state = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('mixed20_trial_start', trial=trial['id'])
        try:
            row = run_trial(trial, p); state['results'][trial['id']] = row
            _event('mixed20_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('mixed20_trial_failed', **state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', state)
        gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output/'workers').mkdir(); torch.set_num_threads(1)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no hardware fallback')
    parent = locate_parent(c['parent']); old = json.loads((parent/'run-manifest.json').read_text())['config']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        if c[key] != old[key]: raise ValueError('comparison changes '+key)
    data = resolve_dataset_directory(c['movie_input']['path'])
    for name, sha in c['movie_input']['sha256'].items():
        if file_sha256(data/name) != sha: raise ValueError('input hash differs: '+name)
    indexes = output/'frozen-indexes'; indexes.mkdir()
    shutil.copy2(parent/'frozen-indexes/items-42.npy', indexes/'items-42.npy')
    for rel in c['parent']['files']:
        if rel.startswith(('trials/', 'baselines/', 'control-tests/')):
            dest = output/'baselines'/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(parent/rel, dest)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes),
        parent=str(parent), resources=str(resources), assignments=assignments(c, 2))
    path = output/'plan.json'; _write(path, p)
    subprocess.run([sys.executable, '-u', '-m', 'rhpr.gr.kindle_mixed20', '--prepare-cache', str(path)],
        env=worker_environment(c, p['assignments'][0]), check=True, timeout=900)
    overlay = scratch/'modern-site'; overlay.mkdir()
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    summary = launch_workers(path, worker_module='rhpr.gr.kindle_mixed20', event_prefix='mixed20')
    if set(summary['results']) != {t['id'] for t in TRIALS}: raise RuntimeError('missing completed fit')
    rows = list(summary['results'].values())
    for key in ('initial_state_sha256', 'sid_file_sha256', 'case_targets_sha256', 'raw_case_sha256', 'test_case_sha256'):
        if rows[0][key] != rows[1][key]: raise ValueError('unmatched new controls: '+key)
    _write(output/'study-summary.json', dict(results=summary['results'], new_generator_fits=2, tokenizer_refitted=False,
        seed_roles=SEEDS, training_policy=POLICY, inference_formats=['pair20', 'recent10'],
        reused_baselines=['recent20_wide_42', 'recent32_wide_42', 'pair32_wide_42']))
    metrics = dict(experiment=c['name'], models=2, seed=42, test_evaluated=True)
    for name, row in summary['results'].items():
        metrics[name+'_validation_ndcg10'] = row['selection']['validation_ndcg10']
        metrics[name+'_test_ndcg10'] = row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops'] = row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan'); parser.add_argument('--slot', type=int)
    parser.add_argument('--prepare-cache'); args = parser.parse_args()
    if args.prepare_cache: prepare_cache(args.prepare_cache)
    else: worker(args.plan, args.slot)
