"""Frequency-prioritized pair placement at an exactly matched token budget."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rhpr.gr.kindle_random_tokenization import RandomTokenization

TRIALS = [dict(id='frequency20_budget5_42', seed=42, history_items=20, mode='frequency', max_pairs=5),
          dict(id='random20_budget5_42', seed=42, history_items=20, mode='matched_random', max_pairs=5),
          dict(id='random20_budget8_42', seed=42, history_items=20, mode='matched_random', max_pairs=8)]
POLICY = dict(history_items=20, eligible_numerator=4, eligible_denominator=5,
    allowed_pair_budgets=[5, 8], rare_train_users_at_most=5, merge_fraction=.5,
    training='batch_binomial_at_trial_budget_then_short_history_slot_restriction',
    frequency_cost='number_of_rare_items_inside_pairs',
    tie_breaking='uniform_among_minimum_cost_matchings',
    inference='trial_pair_budget_or_short_history_capacity', evaluation_seed=8141,
    count_rng_stream=7401, tiling_rng_stream=7402, compiler_recompile_limit=32,
    unseen_items='rare', frequency_source='unique_train_users_before_window_expansion')
SEEDS = dict(index_seed=42, training_seed=42, augmentation_seed=42)


def train_support(data, nitems):
    """Only original TRAIN events; include the initial event, exclude val/test."""
    z = np.load(Path(data)/'histories.npz'); counts = np.zeros(nitems, dtype=np.int32)
    for begin, end in zip(z['offsets'][:-1], z['offsets'][1:]):
        events = z['items'][begin:end-2]
        if np.any(events < 0) or np.any(events >= nitems):
            raise ValueError('TRAIN item outside catalog')
        counts[np.unique(events)] += 1
    return counts


def compact20(contexts):
    h = np.asarray(contexts)
    if h.ndim != 2 or h.shape[1] != 64 or h.dtype.kind not in 'iu' or np.any(h < -1):
        raise ValueError('expected original 64-position integer histories')
    h = h[:, -20:]; lengths = (h >= 0).sum(1)
    if np.any((np.arange(20)[None, :] < 20-lengths[:, None]) != (h < 0)):
        raise ValueError('expected contiguous left padding')
    compact = np.take_along_axis(h, np.minimum(20-lengths[:, None]+np.arange(20), 19), 1)
    return compact, lengths


def matching_tables(edge_costs, old_lengths, max_pairs=5):
    """Suffix DP: exact minimum cost and number of minimizing size-k matchings.

    Costs count rare *items*, not pairs. Each matching has the same probability
    when its edge costs are tied.
    """
    edge_costs = np.asarray(edge_costs); old_lengths = np.asarray(old_lengths)
    n, edges = edge_costs.shape
    cost = np.full((n, edges+3, max_pairs+1), 100, dtype=np.int16)
    ways = np.zeros_like(cost, dtype=np.int32)
    cost[:, :, 0] = 0; ways[:, :, 0] = 1
    for j in range(edges-1, -1, -1):
        for k in range(1, max_pairs+1):
            skip = cost[:, j+1, k]
            take = np.where(j+1 < old_lengths, edge_costs[:, j]+cost[:, j+2, k-1], 100)
            best = np.minimum(skip, take); cost[:, j, k] = best
            ways[:, j, k] = ((skip == best)*ways[:, j+1, k]
                + ((take == best) & (take < 100))*ways[:, j+2, k-1])
    return cost, ways


def sample_matching(cost, ways, edge_costs, counts, rng):
    n, edges = edge_costs.shape; rows = np.arange(n)
    counts = np.asarray(counts, dtype=int).copy(); j = np.zeros(n, dtype=int)
    selected = np.zeros((n, edges), dtype=bool)
    if np.any(ways[rows, 0, counts] == 0): raise ValueError('infeasible matching')
    for _ in range(edges):
        active = (counts > 0) & (j < edges)
        if not active.any(): break
        r = rows[active]; p = j[r]; k = counts[r]
        best = cost[r, p, k]
        take_ways = np.where(edge_costs[r, p]+cost[r, p+2, k-1] == best,
                             ways[r, p+2, k-1], 0)
        take = rng.random(len(r))*ways[r, p, k] < take_ways
        selected[r[take], p[take]] = True
        j[r] += 1+take; counts[r] -= take
    if np.any(counts): raise RuntimeError('matching sampler failed to fill exact budget')
    return selected


class FrequencyTokenization(RandomTokenization):
    """Same batch counts/shuffle/native shapes for both placement policies."""
    def __init__(self, contexts, item_sids, keys, pair_sids, support, mode,
                 batch_size=512, seed=42, augmentation_seed=42, max_pairs=5):
        # Preserve inherited schedule/coverage/hash auditing; replace geometry.
        if mode not in ('frequency', 'matched_random'): raise ValueError('unknown mode')
        if max_pairs not in (5, 8) or (mode == 'frequency' and max_pairs != 5):
            raise ValueError('only matched budget5 and aggressive random budget8 are declared')
        if np.asarray(support).shape != (len(item_sids),) or np.any(np.asarray(support) < 0):
            raise ValueError('invalid TRAIN user counts')
        h, self.lengths = compact20(contexts)
        if np.any(h >= len(item_sids)): raise ValueError('item outside catalog')
        if (item_sids.shape != (len(item_sids), 4) or pair_sids.shape != (len(keys), 4)
            or np.any(keys[1:] <= keys[:-1])): raise ValueError('invalid 3+1 index')
        self.width = 20; self.capacity = max_pairs; self.n = len(h); self.mode = mode
        self.batch_size = batch_size; self.seed = seed; self.augmentation_seed = augmentation_seed
        self.probability = .5; self.epoch = None
        self.old_lengths = self.lengths*4//5
        self.singles = (item_sids[np.maximum(h, 0)]+np.arange(4)*256).astype(np.int32)
        self.pairs = np.zeros((self.n, 15, 4), dtype=np.int32)
        valid = np.arange(15)[None, :]+1 < self.old_lengths[:, None]
        query = h[:, :15].astype(np.int64)*len(item_sids)+h[:, 1:16]
        needed = query[valid]; at = np.searchsorted(keys, needed)
        if np.any(at >= len(keys)) or not np.array_equal(keys[at], needed):
            raise ValueError('eligible pair absent from frozen cache')
        self.pairs[valid] = pair_sids[at]+(4+np.arange(4))*256
        self.unique_adjacent_candidates = len(np.unique(needed))
        rare = np.asarray(support)[np.maximum(h, 0)] <= 5
        self.rare_cost = rare[:, :15].astype(np.int16)+rare[:, 1:16]
        self.edge_cost = self.rare_cost if mode == 'frequency' else np.zeros_like(self.rare_cost)
        self.cost, self.ways = matching_tables(self.edge_cost, self.old_lengths, self.capacity)
        self.last_rare_merged = 0

    def encode_batch(self, part, count, epoch, batch):
        part = np.asarray(part); old = self.old_lengths[part]
        slots = np.r_[0, self.merge_slots(count, epoch, batch).cumsum()]
        k = slots[np.minimum(self.capacity, old//2)].astype(int)
        result = self._encode(part, k, 4*(20-count), np.random.default_rng(
            np.random.SeedSequence([self.augmentation_seed, epoch, batch, POLICY['tiling_rng_stream']])))
        if getattr(self, 'rare_epoch', None) != epoch:
            self.rare_epoch = epoch; self.rare_merged = 0
        self.rare_merged += self.last_rare_merged
        return result

    def _encode(self, part, k, width, rng):
        starts = sample_matching(self.cost[part], self.ways[part], self.edge_cost[part], k, rng)
        n = len(part); skip = np.zeros((n, 20), dtype=bool); skip[:, 1:16] = starts
        first = np.zeros_like(skip); first[:, :15] = starts
        keep = (np.arange(20)[None, :] < self.lengths[part, None]) & ~skip
        tokens = self.singles[part].copy(); r, j = np.nonzero(first)
        tokens[r, j] = self.pairs[part[r], j]
        at = keep.cumsum(1)-1; lengths = keep.sum(1)
        if np.any(4*lengths > width): raise RuntimeError('layout exceeds native count width')
        out = np.full((n, width//4, 4), -1, dtype=np.int32)
        r, j = np.nonzero(keep); out[r, width//4-lengths[r]+at[r, j]] = tokens[r, j]
        self.last_rare_merged = int((starts*self.rare_cost[part]).sum())
        self.last_starts = starts
        self.last_rare_per_row = (starts*self.rare_cost[part]).sum(1)
        return out.reshape(n, width), int(k.sum()), int(starts[:, 1::2].sum())

    def evaluation(self, seed=8141):
        """No epoch, target or split-specific random seed; deterministic replay."""
        k = np.minimum(self.capacity, self.old_lengths//2)
        return self._encode(np.arange(self.n), k, 4*(20-self.capacity), np.random.default_rng(seed))[0]

    def epoch_summary(self, epoch):
        return {**super().epoch_summary(epoch), 'mode': self.mode,
            'rare_context_items_merged': self.rare_merged,
            'eligible_region': 'oldest_floor_4n_over_5', 'rare_threshold_train_users': 5}


def needed_pairs(contexts, nitems):
    h, lengths = compact20(contexts); valid = np.arange(15)[None, :]+1 < (lengths*4//5)[:, None]
    if np.any(h >= nitems): raise ValueError('item outside catalog')
    return np.unique((h[:, :15].astype(np.int64)*nitems+h[:, 1:16])[valid])


# Reuse the completed control's optimizer/selection and offline runtime contract.
import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from rhpr.gr.experiment import _event, _loss, _train_generator, _write


def validate(c):
    from rhpr.gr import kindle_mixed20 as previous
    if (c['kind'] != 'gr_kindle_frequency' or c['trials'] != TRIALS
        or c['frequency_policy'] != POLICY or c['seed_roles'] != SEEDS or c['windows'] != [20]
        or c['expected_new_pairs'] != 5055
        or c['cache_parent']['kernel'] != 'carasik/rhpr-kindle-mixed20'
        or c['cache_parent']['git_sha'] != '4026db0e479881586cad0cdaeae264e0ec4d06ef'
        or c['kaggle']['kernel_sources'] != [c['parent']['kernel'], c['cache_parent']['kernel']]):
        raise ValueError('expected frequency/random budget5 and aggressive random budget8 only')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_mixed20', trials=previous.TRIALS, windows=[20, 10],
        random_tokenization=previous.POLICY, expected_new_pairs=5011)
    normalized['kaggle']['kernel_sources'] = [c['parent']['kernel']]
    previous.validate(normalized)


def cache_plan(data, parent):
    from rhpr.gr.ml100k_pair_raw import cases
    idx = Path(parent)/'frozen-indexes'
    old = np.load(idx/'pair-keys.npy'); sid = np.load(idx/'items-42.npy')
    sets = {s: needed_pairs(cases(data, s)[0], len(sid)) for s in ('train', 'validation', 'test')}
    missing = np.setdiff1d(np.unique(np.concatenate(list(sets.values()))), old)
    return old, sets, missing


def prepare_cache(plan_path):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.pair_sid import predict_rq
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError('cache preparation requires isolated T4')
    parent = Path(p['parent'])/'frozen-indexes'; out = Path(p['indexes'])
    old, sets, missing = cache_plan(p['data'], p['cache_parent'])
    if len(missing) != p['config']['expected_new_pairs']:
        raise ValueError('new pair count differs from CPU preflight')
    _event('frequency_cache_start', new_pairs=len(missing))
    cache = Path(p['output'])/'pair-cache'; cache.mkdir()
    texts = json.loads((Path(p['data'])/'texts.json').read_text())
    encoder = SentenceTransformer(str(Path(p['resources'])/'sentence-t5-base'), device='cuda:0', local_files_only=True)
    encoder.max_seq_length = 2048; encoder.eval()
    encode_pair_cache(encoder, texts, missing, cache/'new-pair-embeddings.npy', ['First book', 'Second book'])
    del encoder; gc.collect(); torch.cuda.empty_cache()
    codes = predict_rq(np.load(cache/'new-pair-embeddings.npy', mmap_mode='r'),
        np.load(parent/'pair-centroids-42.npy'), 'cuda:0')
    np.save(cache/'new-pair-keys.npy', missing); np.save(cache/'new-pair-codes.npy', codes)
    from rhpr.gr.kindle_mixed20 import extend_by_split
    keys, pairs = extend_by_split(old, np.load(Path(p['cache_parent'])/'frozen-indexes/pairs-42.npy'), sets, missing, codes)
    np.save(out/'pair-keys.npy', keys); np.save(out/'pairs-42.npy', pairs)
    report = dict(new_pairs=len(missing), original_pairs=len(old), total_pairs=len(keys),
        needed_by_split={s: len(k) for s, k in sets.items()},
        original_sids_unchanged=True, tokenizer_refitted=False, suffix_allocation='train_then_validation_then_test',
        max_suffix=int(pairs[:, 3].max()))
    _write(cache/'cache-report.json', report); _event('frequency_cache_complete', **report)


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.kindle_pairs import test_selected
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
    raw = {s: cases(p['data'], s) for s in ('train', 'validation')}
    support = np.load(root/'train-user-support.npy')
    def make_view(h):
        return FrequencyTokenization(h, sid, keys, pairs, support, trial['mode'],
            seed=seed, augmentation_seed=c['seed_roles']['augmentation_seed'], max_pairs=trial['max_pairs'])
    views = make_view(raw['train'][0])
    encoded = {}; layout_audits = {}
    for s, (h, u, t) in raw.items():
        view = views if s == 'train' else make_view(h)
        encoded[s] = (view.evaluation(), u, t)
        if s != 'train':
            layout_audits[s] = dict(pair_starts=view.last_starts, rare_items_merged=view.last_rare_per_row)
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
    for s, (x, u, t) in encoded.items():
        if s != 'train': np.savez_compressed(out/(s+'-inputs.npz'), contexts=x, user_ids=u, targets=t, **layout_audits[s])
    _write(out/'training-view-provenance.json', dict(seed_roles=SEEDS, policy=POLICY,
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
    test_view = make_view(h); test_x = test_view.evaluation()
    np.savez_compressed(out/'test-inputs.npz', contexts=test_x, user_ids=u, targets=t,
        pair_starts=test_view.last_starts, rare_items_merged=test_view.last_rare_per_row)
    test = test_selected(model, (test_x, u, t), wrapped, out, out/'generator-best.pt', selection['best_epoch'])
    row = dict(trial=trial['id'], seed=seed, seed_roles=SEEDS, history_items=width, mode=trial['mode'],
        initial_state_sha256=initial, sid_file_sha256=file_sha256(root/'items-42.npy'),
        pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'), selection=selection, forward=forward,
        raw_case_sha256={s: case_digest(a) for s, a in raw.items()},
        case_targets_sha256={s: case_digest(a[1:]) for s, a in raw.items()},
        test_case_sha256=case_digest((h, u, t)), timing=timing, test=test, test_evaluated=True)
    row['frequency_policy'] = POLICY; row['max_pairs'] = trial['max_pairs']
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
        _event('frequency_trial_start', trial=trial['id'])
        try:
            row = run_trial(trial, p); state['results'][trial['id']] = row
            _event('frequency_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('frequency_trial_failed', **state['failures'][-1])
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
    cache_parent = locate_parent(c['cache_parent'])
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
    support = train_support(data, c['data']['items'])
    if hashlib.sha256(support.astype('<i4').tobytes()).hexdigest() != c['train_support_sha256']:
        raise ValueError('TRAIN-only popularity differs from CPU plan')
    np.save(indexes/'train-user-support.npy', support)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes),
        parent=str(parent), cache_parent=str(cache_parent), resources=str(resources), assignments=assignments(c, 2))
    path = output/'plan.json'; _write(path, p)
    subprocess.run([sys.executable, '-u', '-m', 'rhpr.gr.kindle_frequency', '--prepare-cache', str(path)],
        env=worker_environment(c, p['assignments'][0]), check=True, timeout=900)
    overlay = scratch/'modern-site'; overlay.mkdir()
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    summary = launch_workers(path, worker_module='rhpr.gr.kindle_frequency', event_prefix='frequency')
    if set(summary['results']) != {t['id'] for t in TRIALS}: raise RuntimeError('missing completed fit')
    rows = list(summary['results'].values())
    for key in ('initial_state_sha256', 'sid_file_sha256', 'case_targets_sha256', 'raw_case_sha256', 'test_case_sha256'):
        if any(row[key] != rows[0][key] for row in rows[1:]): raise ValueError('unmatched new controls: '+key)
    _write(output/'study-summary.json', dict(results=summary['results'], new_generator_fits=3, tokenizer_refitted=False,
        seed_roles=SEEDS, training_policy=POLICY, inference_formats=['frequency20_budget5', 'matched_random20_budget5', 'random20_budget8'],
        reused_baselines=['recent20_wide_42', 'recent32_wide_42', 'pair32_wide_42']))
    metrics = dict(experiment=c['name'], models=3, seed=42, test_evaluated=True)
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
