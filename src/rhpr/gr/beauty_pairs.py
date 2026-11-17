"""Beauty/P5 leave-last-two pilot with a fixed three/five-item unmerged tail."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _event, _loss, _train_generator, _write
from rhpr.gr.kindle_random_tokenization import RandomTokenization, matching_starts

TRIALS = [dict(id='recent20_42', mode='recent', recent=20, seed=42),
          dict(id='recent3_42', mode='recent', recent=3, seed=42),
          dict(id='random20_tail3_42', mode='random', recent=3, seed=42),
          dict(id='random20_tail5_42', mode='random', recent=5, seed=42)]
POLICY = dict(history_items=20, merge_probability=.5, protected_tails=[3, 5],
    training='batch_binomial_then_uniform_individual_prefix_matching',
    short_counts='conditional_uniform_virtual_slots_not_clipping',
    inference='fixed_oldest_adjacent_pairs_then_odd_single_then_protected_tail',
    count_rng_stream=7401, tiling_rng_stream=7402, compiler_recompile_limit=32,
    suffix_policy='train_then_validation_then_test', all_targets_single=True)


def compact(contexts, nitems):
    h = np.asarray(contexts)
    if h.ndim != 2 or h.shape[1] != 20 or h.dtype.kind not in 'iu' or np.any(h < -1) or np.any(h >= nitems):
        raise ValueError('expected20left-padded integer item contexts')
    lengths = (h >= 0).sum(1)
    if np.any((np.arange(20)[None, :] < 20-lengths[:, None]) != (h < 0)):
        raise ValueError('noncontiguous left padding')
    return np.take_along_axis(h, np.minimum(20-lengths[:, None]+np.arange(20), 19), axis=1), lengths


def pair_keys(contexts, nitems, tail, *, random_tiling):
    h, lengths = compact(contexts, nitems)
    if tail not in (3, 5): raise ValueError('protected tail must be3or5')
    starts = np.arange(20-tail-1) if random_tiling else np.arange(0, 20-tail-1, 2)
    valid = starts[None, :]+1 < np.maximum(lengths-tail, 0)[:, None]
    return np.unique((h[:, starts].astype(np.int64)*nitems+h[:, starts+1])[valid])


def encode(contexts, sids, keys=None, pairs=None, *, tail=3, recent=None):
    h, lengths = compact(contexts, len(sids))
    if recent is not None:
        if recent not in (3, 12, 20): raise ValueError('expectedRecent3,12or20')
        raw = np.asarray(contexts)[:, -recent:]
        tokens = sids[np.maximum(raw, 0)]+np.arange(4)*256
        return np.where(raw[:, :, None] >= 0, tokens, -1).reshape(len(raw), 4*recent).astype(np.int32)
    if tail not in (3, 5): raise ValueError('protected tail must be3or5')
    out = np.full((len(h), 4*(20-(20-tail)//2)), -1, dtype=np.int32)
    for row, n in enumerate(lengths):
        old = max(int(n)-tail, 0); tokens = []; j = 0
        while j+1 < old:
            key = int(h[row, j])*len(sids)+int(h[row, j+1]); at = np.searchsorted(keys, key)
            if at >= len(keys) or keys[at] != key: raise ValueError('missing fixed pair SID')
            tokens.extend(pairs[at]+(4+np.arange(4))*256); j += 2
        while j < n:
            tokens.extend(sids[h[row, j]]+np.arange(4)*256); j += 1
        if tokens: out[row, -len(tokens):] = tokens
    return out


class TailTokenization(RandomTokenization):
    """Reuse the audited epoch/shuffle/hash contract with a fixed raw-item tail."""
    def __init__(self, contexts, item_sids, keys, pair_sids, *, tail, batch_size=512,
                 seed=42, augmentation_seed=42, merge_fraction=.5):
        h, self.lengths = compact(contexts, len(item_sids))
        if tail not in (3, 5) or item_sids.shape != (len(item_sids), 4) or pair_sids.shape != (len(keys), 4):
            raise ValueError('expected3+1indexes and tail3or5')
        if batch_size <= 0 or not 0 <= merge_fraction <= 1 or np.any(keys[1:] <= keys[:-1]):
            raise ValueError('invalid sampling/index settings')
        self.tail = tail; self.width = 20; self.capacity = (20-tail)//2
        self.augmentation_seed = augmentation_seed
        self.n, self.batch_size, self.seed, self.probability = len(h), batch_size, seed, merge_fraction
        self.singles = (item_sids[np.maximum(h, 0)]+np.arange(4)*256).astype(np.int32)
        edges = 20-tail-1
        valid = np.arange(edges)[None, :]+1 < np.maximum(self.lengths-tail, 0)[:, None]
        query = h[:, :edges].astype(np.int64)*len(item_sids)+h[:, 1:edges+1]
        needed = query[valid]; at = np.searchsorted(keys, needed)
        if np.any(at >= len(keys)) or not np.array_equal(keys[at], needed):
            raise ValueError('missing random pair SID; never drop an event')
        self.pairs = np.zeros((len(h), edges, 4), dtype=np.int32)
        self.pairs[valid] = pair_sids[at]+(4+np.arange(4))*256
        self.unique_adjacent_candidates = len(np.unique(needed)); self.epoch = None

    def encode_batch(self, part, count, epoch, batch):
        if not 0 <= count <= self.capacity: raise ValueError('invalid merge count')
        out = np.full((len(part), 4*(20-count)), -1, dtype=np.int32)
        rng = np.random.default_rng(np.random.SeedSequence([self.augmentation_seed, epoch, batch, POLICY['tiling_rng_stream']]))
        slots = np.r_[0, self.merge_slots(count, epoch, batch).cumsum()]
        total = odd = 0
        for row, index in enumerate(part):
            n = int(self.lengths[index]); old = max(n-self.tail, 0); k = int(slots[old//2])
            starts = np.arange(0, 2*k, 2) if count == self.capacity else matching_starts(old, k, rng)
            merge = set(starts.tolist()); tokens = []; j = 0
            while j < n:
                if j in merge: tokens.append(self.pairs[index, j]); j += 2
                else: tokens.append(self.singles[index, j]); j += 1
            if tokens:
                x = np.concatenate(tokens)
                if len(x) > out.shape[1]: raise RuntimeError('native shape bound exceeded')
                out[row, -len(x):] = x
            total += k; odd += int((starts % 2).sum())
        return out, total, odd

    def epoch_summary(self, epoch):
        return {**super().epoch_summary(epoch), 'protected_tail_items': self.tail}


def validate(c):
    if c.get('architecture_pilot'):
        if (c.get('recent12_control') or c['architecture_pilot'] != dict(
                reference_kernel='carasik/rhpr-beauty-pairs/1', new_generator_fits=2,
                source_revision='cf2d58f014477ca3702d63c7937289cf3c2f9ade',
                adopted=['encoder_layers', 'decoder_layers', 'dropout'])
                or c['trials'] != [dict(id=f'mender_recent{n}_42', mode='recent', recent=n, seed=42) for n in (20, 3)]
                or c['execution']['worker_devices'] != ['cuda:0', 'cuda:1']
                or (c['generator']['encoder_layers'], c['generator']['decoder_layers'], c['generator']['dropout']) != (6, 6, .2)
                or c['expected_sid_file_sha256'] != '624a0e23fc8c4ca844972beb4718800121d23f9a12a85c8135a4c26767dd02ab'):
            raise ValueError('expected exactly two matched Mender-depth Beauty controls')
        c = copy.deepcopy(c); c['trials'] = TRIALS
        c['generator'].update(encoder_layers=4, decoder_layers=4, dropout=.1)
    if c.get('recent12_control'):
        if (c['recent12_control'] != dict(reference_kernel='carasik/rhpr-beauty-pairs/1', new_generator_fits=1)
                or c['trials'] != [dict(id='recent12_42', mode='recent', recent=12, seed=42)]
                or c['execution']['worker_devices'] != ['cuda:0']
                or c['expected_sid_file_sha256'] != '624a0e23fc8c4ca844972beb4718800121d23f9a12a85c8135a4c26767dd02ab'):
            raise ValueError('expected exactly one matched Beauty Recent12 control')
        c = copy.deepcopy(c); c['trials'] = TRIALS
    t = c['training']; g = c['generator']; d = c['data']
    if (c['kind'] != 'gr_beauty_pairs' or c['trials'] != TRIALS or c['policy'] != POLICY or c['seed'] != 42
        or (d['users'], d['items'], d['train_examples'], d['history_items']) != (22363, 12101, 131413, 20)
        or (t['min_epochs'], t['max_epochs'], t['validation_interval'], t['patience_epochs']) != (40, 120, 5, 20)
        or (t['batch_size'], t['physical_batch_size'], t['evaluation_batch_size']) != (512, 512, 64)
        or t['learning_rate'] != .003 or t['precision'] != 'fp16' or t['evaluation_precision'] != 'fp32'
        or t['scheduler'] != dict(kind='warmup_cosine', warmup_epochs=2, total_epochs=60, min_lr=.001)
        or t['early_stopping'] != dict(kind='ema', alpha=.5, min_delta=.0005)
        or (g['encoder_layers'], g['decoder_layers'], g['d_model'], g['d_ff']) != (4, 4, 128, 1024)
        or g['attn_implementation'] != 'sdpa' or c['quantizer']['kind'] != 'grid_rkmeans'
        or not c['evaluation']['test_after_selection'] or c['kaggle']['kernel_sources']):
        raise ValueError('Beauty pilot differs from declared four-fit protocol')


def prepare_cases(c, resources, output):
    from rhpr.gr.data import build_examples, load_beauty
    texts, users, _, counts = load_beauty(Path(resources)/'data', c['data'])
    full = np.array([len(u.train)+2 for u in users])
    if full.min() < 5: raise ValueError('Beauty user filter differs')
    root = Path(output)/'prepared-cases'; root.mkdir()
    report = dict(**counts, full_length_median=float(np.median(full)), splits={})
    for split in ('train', 'validation', 'test'):
        h, u, t = build_examples(users, 20, split); n = (h >= 0).sum(1)
        np.savez_compressed(root/(split+'-cases.npz'), contexts=h, user_ids=u, targets=t)
        report['splits'][split] = dict(examples=len(t), context_length_counts=np.bincount(n, minlength=21).tolist(),
            tails={str(tail): dict(compressible_examples=int((n >= tail+2).sum()),
                fixed_pair_occurrences=int((np.maximum(n-tail, 0)//2).sum())) for tail in (3, 5)})
    _write(Path(output)/'data-report.json', report)
    return texts, root, report


def cache_sets(data, nitems):
    from rhpr.gr.ml100k_pair_raw import cases
    return {s: np.unique(np.concatenate([pair_keys(cases(data, s)[0], nitems, tail,
        random_tiling=s == 'train') for tail in (3, 5)])) for s in ('train', 'validation', 'test')}


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.kindle_pairs import test_selected
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import attach4, InputIndex4
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c = copy.deepcopy(p['config']); c['active_trial'] = trial; c['active_training_view_policy'] = POLICY
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    root = Path(p['indexes']); sids = np.load(root/'items-42.npy'); keys = np.load(root/'pair-keys.npy'); pairs = np.load(root/'pairs-42.npy')
    idx = SidIndex.from_full_sids(sids, width=256); wrapped = InputIndex4(idx)
    model = attach4(build_grid_generator(sids, c['generator'], 42, 'cpu', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    if initial != c['expected_initial_state_sha256']:
        raise ValueError('initializer differs from CPU preflight')
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    def encoding(h):
        return encode(h, sids, keys, pairs, tail=trial['recent'], recent=trial['recent'] if trial['mode'] == 'recent' else None)
    raw = {s: cases(p['data'], s) for s in ('train', 'validation')}
    if c.get('recent12_control') or c.get('architecture_pilot'):
        reference = c['reference_result']
        if (file_sha256(root/'items-42.npy') != reference['sid_file_sha256']
                or initial != c['expected_initial_state_sha256']
                or {s: case_digest(v) for s, v in raw.items()} != reference['raw_case_sha256']
                or {s: case_digest(v[1:]) for s, v in raw.items()} != reference['case_targets_sha256']
                or case_digest(cases(p['data'], 'test')) != reference['test_case_sha256']):
            raise ValueError('Beauty control input/index/initializer mismatch before training')
    encoded = {s: (encoding(h), u, t) for s, (h, u, t) in raw.items()}
    views = TailTokenization(raw['train'][0], sids, keys, pairs, tail=trial['recent']) if trial['mode'] == 'random' else None
    model.eval(); probe = (encoded['validation'][0][:1], raw['validation'][1][:1], raw['validation'][2][:1])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss = _loss(model, wrapped.model_batch(*probe, 'cpu'))
    if not torch.isfinite(loss) or state_digest(dict(model.named_parameters())) != initial: raise RuntimeError('invalid profiler probe')
    forward = dict(parameters=sum(v.numel() for v in model.parameters()), encoder_positions=probe[0].shape[1]+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='Fixed padded model width; B1 CPU mathSDPA teacher-forced4code forward,FMA2; excludes beam/backward/text encoder')
    _write(out/'forward-flops.json', forward)
    model.to('cuda:0'); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    def builder(root, width, split):
        if split not in encoded: raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection = _train_generator(model, p['data'], wrapped, c, out, 'cuda:0', example_builder=builder, training_context_selector=views)
    if selection['stopping_reason'] not in ('early_stopping', 'epoch_limit'): raise RuntimeError('incomplete fit; preserve checkpoints')
    curve = json.loads((out/'training-curves.json').read_text())
    timing = dict(wall_seconds=time.monotonic()-started, first_epoch_training_seconds=curve[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([v['train_seconds'] for v in curve[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    h, u, t = cases(p['data'], 'test')
    test = test_selected(model, (encoding(h), u, t), wrapped, out, out/'generator-best.pt', selection['best_epoch'])
    row = dict(trial=trial['id'], seed=42, mode=trial['mode'], recent=trial['recent'], initial_state_sha256=initial,
        sid_file_sha256=file_sha256(root/'items-42.npy'), pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'),
        raw_case_sha256={s: case_digest(v) for s, v in raw.items()},
        case_targets_sha256={s: case_digest(v[1:]) for s, v in raw.items()},
        test_case_sha256=case_digest((h, u, t)), selection=selection, forward=forward, timing=timing, test=test)
    _write(out/'trial-result.json', row); return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    torch._dynamo.config.recompile_limit = 32; state = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('beauty_pairs_trial_start', trial=trial['id'])
        try:
            row = run_trial(trial, p); state['results'][trial['id']] = row; _event('beauty_pairs_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('beauty_pairs_trial_failed', **state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', state); gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.grid import fit_grid_rkmeans, prepare_grid_source
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.pair_sid import predict_rq
    from rhpr.gr.pair_unique4 import unique_pair_sids
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True); (output/'workers').mkdir()
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no fallback')
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    texts, data, report = prepare_cases(c, resources, output)
    control = bool(c.get('recent12_control') or c.get('architecture_pilot'))
    if not control:
        sets = cache_sets(data, len(texts)); keys = np.unique(np.concatenate(list(sets.values()))); mask = np.isin(keys, sets['train'])
        if {s: len(k) for s, k in sets.items()} != c['expected_pair_counts']: raise ValueError('pair candidates differ from CPU preflight')
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    cache = output/'pair-cache'; cache.mkdir(); indexes = output/'frozen-indexes'; indexes.mkdir()
    _event('beauty_pairs_embedding_start', items=len(texts), train_pairs=0 if control else int(mask.sum()), all_pairs=0 if control else len(keys))
    encoder = SentenceTransformer(str(resources/'sentence-t5-base'), device='cuda:0', local_files_only=True)
    encoder.max_seq_length = 256; encoder.eval()
    vec = np.asarray(encoder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    np.save(cache/'item-embeddings.npy', vec)
    if not control:
        encoder.max_seq_length = 2048
        encode_pair_cache(encoder, texts, keys, cache/'pair-embeddings.npy', ['First item', 'Second item'])
    del encoder; gc.collect(); torch.cuda.empty_cache()
    codes, centers, item_report = fit_grid_rkmeans(vec, c['quantizer'], 42, 'cuda:0', source)
    idx = SidIndex(codes, width=256); np.save(indexes/'items-42.npy', idx.sids); np.save(indexes/'item-centroids-42.npy', centers)
    if control:
        from rhpr.gr.data import file_sha256
        if file_sha256(indexes/'items-42.npy') != c['expected_sid_file_sha256']:
            raise ValueError('recomputed SID differs; no generator training allowed')
        # The control retains the original unused pair-input parameter table,
        # but never encodes or fits pair representations.
        np.save(indexes/'pair-keys.npy', np.empty(0, dtype=np.int64))
        np.save(indexes/'pairs-42.npy', np.empty((0, 4), dtype=np.int32))
        _write(output/'index-report.json', dict(item=item_report, item_diagnostics=idx.diagnostics(), pair_fitting_skipped=True))
    else:
        pairvec = np.load(cache/'pair-embeddings.npy', mmap_mode='r')
        fitcodes, centers, pair_report = fit_grid_rkmeans(np.asarray(pairvec[mask]), c['quantizer'], 42, 'cuda:0', source)
        codes = predict_rq(pairvec, centers, 'cuda:0')
        if not np.array_equal(codes[mask], fitcodes): raise RuntimeError('pair assignments differ')
        pairs = unique_pair_sids(keys, codes, sets['train'], np.union1d(sets['train'], sets['validation']))
        np.save(indexes/'pair-keys.npy', keys); np.save(indexes/'pairs-42.npy', pairs); np.save(indexes/'pair-centroids-42.npy', centers)
        _write(output/'index-report.json', dict(item=item_report, pair=pair_report, item_diagnostics=idx.diagnostics(),
            pair_counts={s: len(v) for s, v in sets.items()}, all_pairs=len(keys), max_pair_suffix=int(pairs[:, 3].max()),
            heldout_interactions_used_for_fit=False, heldout_pair_assignment='frozen centroids,append-only suffixes'))
        del pairvec, fitcodes
    del vec, codes, centers; gc.collect(); torch.cuda.empty_cache()
    overlay = scratch/'modern-site'; overlay.mkdir()
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log: subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    jobs = [dict(slot=0, device='cuda:0', trials=c['trials'])] if c.get('recent12_control') else assignments(c, 2)
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes), assignments=jobs)
    path = output/'plan.json'; _write(path, p)
    summary = launch_workers(path, worker_module='rhpr.gr.beauty_pairs', event_prefix='beauty_pairs')
    rows = list(summary['results'].values())
    for key in ('initial_state_sha256', 'sid_file_sha256', 'pair_sid_file_sha256', 'raw_case_sha256', 'case_targets_sha256', 'test_case_sha256'):
        if len(rows) != len(c['trials']) or any(v[key] != rows[0][key] for v in rows[1:]): raise ValueError('unmatched comparison: '+key)
    _write(output/'study-summary.json', dict(results=summary['results'], data=report, policy=POLICY,
        exact_TIGER_reproduction=False, new_generator_fits=len(c['trials'])))
    metrics = dict(experiment=c['name'], models=len(c['trials']), seed=42, test_evaluated=True)
    for name, row in summary['results'].items():
        metrics[name+'_validation_ndcg10'] = row['selection']['validation_ndcg10']; metrics[name+'_test_ndcg10'] = row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops'] = row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); worker(args.plan, args.slot)
