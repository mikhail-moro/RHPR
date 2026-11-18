"""Full MovieLens1M, eight fixed TRAIN targets, matched ordinary TIGER windows."""
from __future__ import annotations

import argparse
import copy
import gc
import io
import json
import random
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate, _event, _loss, _model_batch, _train_generator, _write
from rhpr.gr.movie_data import case_digest

WINDOWS = [8, 16, 32, 64, 128, 256]
TRIALS = [dict(id=f'recent{w}_42', history_items=w, seed=42) for w in (256, 128, 64, 32, 16, 8)]
REMAINING_WINDOWS = [16, 64]
REMAINING_TRIALS = [dict(id=f'recent{w}_42', history_items=w, seed=42) for w in (64, 16)]
REFERENCE_SID_SHA256 = '80422aa36fbcabbd0c1133a0695547513c1a795f908c72b4ed412e9cc2c2b672'
MICROBATCHES = {'8': 256, '16': 256, '32': 128, '64': 64, '128': 32, '256': 16}
EVAL_BATCHES = {'8': 64, '16': 64, '32': 32, '64': 32, '128': 16, '256': 8}
CONTROL_WINDOWS = [64, 128]
CONTROL_SEEDS = [101, 202]
CONTROL_TRIALS = [dict(id=f'recent{width}_{seed}', history_items=width, seed=seed)
                  for width in reversed(CONTROL_WINDOWS) for seed in CONTROL_SEEDS]
EXTENDED_CONTROL_WINDOWS = [96, 128]
# Continue Recent128 first on each T4, then fit the matched-cost Recent96.
EXTENDED_CONTROL_TRIALS = [dict(id=f'recent{width}_{seed}', history_items=width, seed=seed)
                          for width in reversed(EXTENDED_CONTROL_WINDOWS) for seed in CONTROL_SEEDS]
RESUME_CONTROL_KERNEL = 'carasik/rhpr-ml1m-mixed120-controls'
RESUME_CONTROL_COMMIT = '6754b3a1cd1631f77de0db01c532fe5d1023b0c7'
RESUME_CONTROL_FILES = {'run-manifest.json'} | {
    f'trials/recent128_{seed}/{name}' for seed in CONTROL_SEEDS
    for name in ('generator-best.pt','generator-last.pt','training-curves.json','training-selection.json',
                 *(f'validation-epoch-{epoch:03d}.npz' for epoch in range(5,41,5)))}


def validate_frozen_controls(c):
    from rhpr.gr.ml1m_last8_pairs import PARENT_COMMIT, PARENT_FILES, PARENT_KERNEL

    parent, initializers = c['frozen_parent'], c['control_initializers']
    extended = c.get('extended_budget')
    epochs = 60 if extended else 40
    windows, trials = ((EXTENDED_CONTROL_WINDOWS, EXTENDED_CONTROL_TRIALS) if extended else
                       (CONTROL_WINDOWS, CONTROL_TRIALS))
    sources = [PARENT_KERNEL, RESUME_CONTROL_KERNEL] if extended else [PARENT_KERNEL]
    if (c['frozen_controls'] is not True or c['windows'] != windows
            or ('extended_budget' in c and extended is not True)
            or c['seeds'] != CONTROL_SEEDS or c['trials'] != trials
            or (c['training']['min_epochs'], c['training']['max_epochs'], c['training']['max_seconds'])
                != (epochs, epochs, 9000 if extended else 7200)
            or c['suite_deadline_seconds'] != (14400 if extended else 10800)
            or parent['kernel'] != PARENT_KERNEL or parent['git_sha'] != PARENT_COMMIT
            or set(parent['files']) != PARENT_FILES
            or parent['files']['frozen-indexes/items-42.npy']['sha256'] != REFERENCE_SID_SHA256
            or c['kaggle']['kernel_sources'] != sources
            or set(initializers) != {str(seed) for seed in CONTROL_SEEDS}
            or any(re.fullmatch('[0-9a-f]{64}', v) is None for v in initializers.values())
            or len(set(initializers.values()) | {c['expected_initial_state_sha256']}) != 3):
        raise ValueError('ordinary controls differ from the registered frozen-index protocol')
    normalized = copy.deepcopy(c)
    del normalized['frozen_controls']
    normalized.update(windows=WINDOWS, trials=TRIALS, seeds=[42])
    normalized['training'].update(min_epochs=40,max_epochs=120)
    normalized['kaggle']['kernel_sources'] = []
    if extended:
        resume = c['resume_parent']
        if (resume['kernel'] != RESUME_CONTROL_KERNEL or resume['git_sha'] != RESUME_CONTROL_COMMIT
                or set(resume['files']) != RESUME_CONTROL_FILES
                or normalized['physical_batch_by_window'].pop('96', None) != 64
                or normalized['evaluation_batch_by_window'].pop('96', None) != 16):
            raise ValueError('expected exact Recent128 resume and matching Recent96 microbatches')
    validate(normalized)


def validate(c):
    if 'frozen_controls' in c:
        validate_frozen_controls(c)
        return
    t, g = c['training'], c['generator']
    remaining = c.get('remaining_windows')
    windows, trials = (REMAINING_WINDOWS, REMAINING_TRIALS) if remaining else (WINDOWS, TRIALS)
    if remaining is not None and (remaining != dict(source_kernel='carasik/rhpr-ml1m-last8-windows',
            source_version=2, completed_windows=[8, 32, 128], excluded_windows=[256])
            or c.get('expected_sid_file_sha256') != REFERENCE_SID_SHA256):
        raise ValueError('remaining-window run must preserve the completed index and exclude 256')
    if (c['kind'] != 'gr_ml1m_last8' or c['windows'] != windows or c['trials'] != trials
            or c['seed'] != 42 or c['seeds'] != [42] or c['data']['history_items'] != 256
            or c['data']['users'] != 6040 or c['data']['items'] != 3706 or c['data']['train_examples'] != 48320
            or c['sampling'] != dict(kind='fixed_last_train_targets', targets_per_user=8,
                split='leave_last_two_timestamp_then_movie_id', rating_filter=False, user_length_filter=False)
            or (t['min_epochs'], t['max_epochs'], t['validation_interval'], t['patience_epochs']) != (40, 120, 5, 20)
            or (t['batch_size'], t['physical_batch_size'], t['learning_rate']) != (512, 256, .003)
            or t['precision'] != 'fp16' or t['evaluation_precision'] != 'fp32'
            or t['scheduler'] != dict(kind='warmup_cosine', warmup_epochs=2, total_epochs=60, min_lr=.001)
            or t['early_stopping'] != dict(kind='ema', alpha=.5, min_delta=.0005)
            or (g['encoder_layers'], g['decoder_layers'], g['d_model'], g['d_ff'], g['d_kv'], g['num_heads']) != (4, 4, 128, 1024, 64, 6)
            or g['dropout'] != .1 or g['attn_implementation'] != 'sdpa' or g['separator_token']
            or c['quantizer']['kind'] != 'grid_rkmeans'
            or c['quantizer']['num_hierarchies'] != 3 or c['quantizer']['codebook_size'] != 256
            or g['num_user_bins'] != 2000 or g['beam_width'] != 20 or g['mlp_layers'] is not None
            or t['weight_decay'] != .0001 or t['grad_clip_norm'] != 1.
            or t['compile_training'] != dict(options={'triton.cudagraphs': False}, fullgraph=False, dynamic=False)
            or c['physical_batch_by_window'] != MICROBATCHES or c['evaluation_batch_by_window'] != EVAL_BATCHES
            or not c['evaluation']['test_after_selection'] or c['kaggle']['kernel_sources']
            or c['execution']['worker_devices'] != ['cuda:0', 'cuda:1']):
        raise ValueError('ML1M Last-8 differs from the registered window protocol')
    hashes = [c['expected_initial_state_sha256'], c['movie_input']['source_zip_sha256'], *c['expected_raw_case_sha256'].values()]
    if (set(c['expected_raw_case_sha256']) != {'train','validation','test'}
            or any(re.fullmatch('[0-9a-f]{64}', value) is None for value in hashes)):
        raise ValueError('CPU preflight hashes must be pinned before dispatch')


def build_last8_cases(histories, width=256, targets_per_user=8):
    """Input rows are chronological item IDs. Every context ends before its label."""
    if type(width) is not int or width < 1 or type(targets_per_user) is not int or targets_per_user < 1:
        raise ValueError('positive integer window and target count required')
    result = {}
    for split in ('train', 'validation', 'test'):
        n = len(histories)*(targets_per_user if split == 'train' else 1)
        h = np.full((n, width), -1, dtype=np.int32)
        u, t, positions = (np.empty(n, dtype=np.int64) for _ in range(3))
        j = 0
        for uid, sequence in sorted(histories.items()):
            s = np.asarray(sequence)
            if s.ndim != 1 or s.dtype.kind not in 'iu' or np.any(s < 0) or len(s) < targets_per_user+3:
                raise ValueError('history cannot supply distinct targets with nonempty past')
            train_end = len(s)-2
            selected = range(train_end-targets_per_user, train_end) if split == 'train' else [len(s)-(2 if split == 'validation' else 1)]
            for pos in selected:
                prefix = s[max(0, pos-width):pos]
                h[j, -len(prefix):] = prefix
                u[j], t[j], positions[j] = uid, s[pos], pos
                j += 1
        result[split] = (h, u, t, positions)
    return result


def prepare_raw(archive, output, expected=None):
    """No metadata join, positive-rating filter, core or long-user selection."""
    expected = expected or {}; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        metadata = {}
        for line in z.read('ml-1m/movies.dat').decode('latin-1').splitlines():
            movie, title, genres = line.split('::')
            metadata[int(movie)] = dict(title=title, genres=genres.split('|'))
        a = np.loadtxt(io.StringIO(z.read('ml-1m/ratings.dat').decode().replace('::', ',')), delimiter=',', dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 4 or np.any(a[:, 2] < 1) or np.any(a[:, 2] > 5):
        raise ValueError('invalid ML1M ratings')
    if len(np.unique(a[:, :2], axis=0)) != len(a): raise ValueError('duplicate user-movie ratings')
    ids = np.unique(a[:, 1]); users, lengths = np.unique(a[:, 0], return_counts=True)
    if not set(map(int, ids)).issubset(metadata): raise ValueError('missing native movie metadata')
    observed = dict(users=len(users), items=len(ids), events=len(a), train_examples=8*len(users))
    if any(observed[k] != value for k, value in expected.items() if k in observed): raise ValueError('source counts changed')
    if lengths.min() < 20: raise ValueError('original ML1M minimum history changed')
    # Lexical movie-ID tie break is explicit; it does not establish true order.
    a = a[np.lexsort((a[:, 1], a[:, 3], a[:, 0]))]
    boundaries = np.r_[0, np.flatnonzero(a[1:, 0] != a[:-1, 0])+1, len(a)]
    histories = {}; timestamps = {}; ties = dict(train_validation=0, validation_test=0)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        uid = int(a[lo, 0]); histories[uid] = np.searchsorted(ids, a[lo:hi, 1]).astype(np.int32)
        timestamps[uid] = a[lo:hi, 3]
        ties['train_validation'] += int(a[hi-3, 3] == a[hi-2, 3])
        ties['validation_test'] += int(a[hi-2, 3] == a[hi-1, 3])
    catalog = [dict(item_id=int(i), **metadata[int(i)], text=f"Title: {metadata[int(i)]['title']}. Genres: {', '.join(metadata[int(i)]['genres'])}.") for i in ids]
    _write(output/'catalog.json', catalog)
    cases = build_last8_cases(histories); reports = {}
    train_items = np.unique(np.concatenate([s[:-2] for s in histories.values()]))
    train_targets = np.unique(cases['train'][2])
    for split, (h, u, t, pos) in cases.items():
        ts = np.array([timestamps[int(uid)][p] for uid, p in zip(u, pos)], dtype=np.int64)
        previous_ts = np.array([timestamps[int(uid)][p-1] for uid, p in zip(u, pos)], dtype=np.int64)
        np.savez_compressed(output/(split+'-cases.npz'), contexts=h, user_ids=u, targets=t)
        np.savez_compressed(output/(split+'-anchors.npz'), user_ids=u, target_positions=pos,
            target_timestamps=ts, preceding_timestamps=previous_ts)
        reports[split] = dict(examples=len(t), raw_case_sha256=case_digest((h, u, t)), targets_sha256=case_digest((u, t)),
            unknown_to_all_train_events=int((~np.isin(t, train_items)).sum()),
            absent_from_last8_train_labels=int((~np.isin(t, train_targets)).sum()),
            target_preceding_timestamp_ties=int((ts == previous_ts).sum()),
            context_length_counts=np.bincount((h >= 0).sum(1), minlength=257).tolist())
    np.savez_compressed(output/'user-histories.npz', user_ids=users, offsets=boundaries,
        items=np.searchsorted(ids, a[:, 1]).astype(np.int32), timestamps=a[:, 3])
    report = dict(**observed, source_zip_sha256=file_sha256(Path(archive)), catalogue='all rated native movies; no un-rated catalogue candidates',
        train_events=int((lengths-2).sum()), last8_train_target_items=len(train_targets), all_train_observed_items=len(train_items),
        full_length_quantiles={str(q):float(np.quantile(lengths, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)},
        mean_full_length=float(lengths.mean()), boundary_timestamp_ties=ties, splits=reports,
        filtering='none; all ratings and users retained; timestamp ties retained with movie-ID tie break',
        windows={str(w):{s:int((v[0][:, -w:] >= 0).sum()) for s, v in cases.items()} for w in WINDOWS})
    _write(output/'data-report.json', report)
    return [r['text'] for r in catalog], report


def trial_config(config, trial):
    c = copy.deepcopy(config); width = trial['history_items']
    c['data']['history_items'] = width; c['active_trial'] = trial
    c['training']['physical_batch_size'] = c['physical_batch_by_window'][str(width)]
    c['training']['evaluation_batch_size'] = c['evaluation_batch_by_window'][str(width)]
    if c.get('frozen_controls'):
        c['seed'] = trial['seed']
        c['expected_initial_state_sha256'] = c['control_initializers'][str(trial['seed'])]
    return c


def val_best_within(curve, max_epoch):
    best = max((row for row in curve if row['epoch'] <= max_epoch and 'validation' in row),
               key=lambda row: row['validation']['ndcg@10'])
    return dict(epoch=best['epoch'], ndcg10=best['validation']['ndcg@10'])


def run_trial(trial, p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.profiling import benchmark_training, ProfileDeadline
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c = trial_config(p['config'], trial); width = trial['history_items']; seed = trial['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    sidpath = Path(p['indexes'])/'items-42.npy'; index = SidIndex.from_full_sids(np.load(sidpath), width=256)
    model = build_grid_generator(index.sids, c['generator'], seed, 'cpu', p['source'])
    initial = state_digest(dict(model.named_parameters()))
    if initial != c['expected_initial_state_sha256']: raise ValueError('initializer differs from preflight')
    raw = {s:cases(p['data'], s) for s in ('train', 'validation')}
    visible = {s:(h[:, -width:].copy(), u, t) for s, (h, u, t) in raw.items()}
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    model.eval()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss = _loss(model, _model_batch(*(a[:1] for a in visible['validation']), index.sids, 'cpu'))
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite preflight forward')
    forward = dict(parameters=sum(v.numel() for v in model.parameters()), encoder_positions=4*width+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='B1 fixed padded width, CPU mathSDPA teacher-forced4code forward,FMA2; excludes beam/backward/text encoder')
    _write(out/'forward-flops.json', forward)
    model.to('cuda:0')
    # Exercise the actual compiled forward/backward with accumulation before a
    # long fit. Eager-only profiling missed the native GRID CUDAGraph failure.
    # Keep Inductor fusion, but explicitly disable graph-owned output buffers.
    profile_model = torch.compile(model, **c['training']['compile_training'])
    profile = benchmark_training(profile_model, visible['train'], index.sids,
        np.random.default_rng(seed).permutation(len(visible['train'][2]))[:512],
        c['training']['physical_batch_size'], 'fp16', 'full_pipeline', c['training'],
        dict(seed=seed, warmup=1, iterations=2), ProfileDeadline(180), 'cuda:0')
    profile['compile_training'] = c['training']['compile_training']
    profile['estimate_scope'] = 'compiled train updates after one warmup; excludes validation and cold compilation'
    if state_digest(dict(model.named_parameters())) != initial: raise RuntimeError('profile did not restore weights')
    _write(out/'training-profile.json', profile)
    _event('ml1m_last8_profile_complete', trial=trial['id'], history_items=width, **profile)
    # The profile uses fork_rng and restores weights. Reset all streams explicitly.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    def builder(root, ignored, split):
        if split not in visible: raise ValueError('TEST forbidden during training')
        return visible[split]
    resume_from = None
    if c.get('extended_budget') and width == 128:
        resume_from = Path(p['resume_parent'])/'trials'/trial['id']
    selection = _train_generator(model, p['data'], index, c, out, 'cuda:0', example_builder=builder,
                                 resume_from=resume_from)
    epochs = selection['training_epochs']
    if selection['optimizer_updates']+selection['amp_skipped_updates'] != ((len(raw['train'][2])+511)//512)*epochs:
        raise RuntimeError('update accounting mismatch')
    cap = c['training']['max_epochs']
    if c.get('frozen_controls') and (epochs, selection['validation_checks']) != (cap, cap//5):
        raise RuntimeError(f'incomplete {cap}-epoch control fit')
    if resume_from is not None and (selection['resumed_from_epoch'], selection['continuation_epochs']) != (40, 20):
        raise RuntimeError('Recent128 did not resume at epoch 40')
    curve = json.loads((out/'training-curves.json').read_text())
    timing = dict(wall_seconds=time.monotonic()-started, first_epoch_training_seconds=curve[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([r['train_seconds'] for r in curve[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    if resume_from is not None:
        timing.update(wall_seconds_scope='continuation session; curve median covers all epochs',
                      warm_continuation_epoch_seconds=float(np.median([r['train_seconds'] for r in curve[41:]])))
    h, u, t = cases(p['data'], 'test')
    checkpoint = out/'generator-best.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck['epoch'] != selection['best_epoch']: raise ValueError('selected checkpoint epoch differs')
    model.load_state_dict(ck['model'], strict=True)
    metrics, ids, scores = _evaluate(model, (h[:, -width:].copy(), u, t), index,
        c['training']['evaluation_batch_size'], 'cuda:0', 'fp32')
    np.savez_compressed(out/'test-best-predictions.npz', semantic_ids=ids, scores=scores, user_ids=u, target_item_ids=t)
    test = dict(epoch=selection['best_epoch'], metrics=metrics, checkpoint_sha256=file_sha256(checkpoint),
        selection='raw validation NDCG@10 only', seen_items_filtered=False)
    _write(out/'test-metrics.json', test)
    row = dict(trial=trial['id'], seed=seed, history_items=width, initial_state_sha256=initial,
        sid_file_sha256=file_sha256(sidpath), raw_case_sha256={s:case_digest(v) for s, v in raw.items()},
        case_targets_sha256={s:case_digest(v[1:]) for s, v in raw.items()}, test_case_sha256=case_digest((h, u, t)),
        selection=selection, forward=forward, timing=timing, profile=profile, test=test,
        budget_censored=selection['stopping_reason'] != 'early_stopping')
    if c.get('extended_budget'):
        row['validation_best_within_40'] = val_best_within(curve, 40)
    _write(out/'trial-result.json', row); return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    status = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('ml1m_last8_trial_start', trial=trial['id'], **trial)
        try:
            row = run_trial(trial, p); status['results'][trial['id']] = row
            _event('ml1m_last8_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('ml1m_last8_trial_failed', **status['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', status)
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()


def verify_results(c, results):
    if set(results) != {t['id'] for t in c['trials']}: raise ValueError('incomplete registered-window comparison')
    rows = list(results.values())
    for key in ('sid_file_sha256', 'raw_case_sha256', 'case_targets_sha256', 'test_case_sha256'):
        if any(r[key] != rows[0][key] for r in rows[1:]): raise ValueError('unmatched comparison: '+key)
    for row in rows:
        expected = (c['control_initializers'][str(row['seed'])] if c.get('frozen_controls')
                    else rows[0]['initial_state_sha256'])
        if row['initial_state_sha256'] != expected:
            raise ValueError('unmatched initialization for its seed')


def prepare_indexes(c, output, resources, source):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory

    root = resolve_dataset_directory(c['movie_input']['path'])
    archive = root/'ml-1m-source.bin'
    if file_sha256(archive) != c['movie_input']['source_zip_sha256']: raise ValueError('raw ML1M source changed')
    data = output/'prepared-cases'; texts, report = prepare_raw(archive, data, c['data'])
    for split, digest in c['expected_raw_case_sha256'].items():
        if report['splits'][split]['raw_case_sha256'] != digest: raise ValueError('prepared cases differ from CPU preflight')
    _event('ml1m_last8_data_ready', users=report['users'], items=report['items'], train_examples=report['train_examples'],
        boundary_timestamp_ties=report['boundary_timestamp_ties'], last8_train_target_items=report['last8_train_target_items'])
    indexes = output/'frozen-indexes'; indexes.mkdir()
    encoder = SentenceTransformer(str(resources/'sentence-t5-base'), device='cuda:0', local_files_only=True)
    encoder.max_seq_length = 256; encoder.eval()
    vec = np.asarray(encoder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if vec.shape != (3706, 768) or not np.isfinite(vec).all(): raise ValueError('invalid title+genres embeddings')
    np.save(indexes/'item-embeddings.npy', vec); del encoder; gc.collect(); torch.cuda.empty_cache()
    codes, centers, fit = fit_grid_rkmeans(vec, c['quantizer'], 42, 'cuda:0', source)
    index = SidIndex(codes, width=256); np.save(indexes/'items-42.npy', index.sids); np.save(indexes/'centroids-42.npy', centers)
    if c.get('expected_sid_file_sha256') and file_sha256(indexes/'items-42.npy') != c['expected_sid_file_sha256']:
        raise ValueError('recomputed frozen index differs from the completed windows; no training started')
    _write(indexes/'index-report.json', dict(fit=fit, sid=index.diagnostics(), scope='frozen title+genres, full rated static catalogue; no behavior fitting'))
    del vec, codes, centers; gc.collect(); torch.cuda.empty_cache()
    return data, indexes, report


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True); (output/'workers').mkdir()
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no fallback')
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    if c.get('frozen_controls'):
        from rhpr.gr.ml1m_last8_pairs import frozen_inputs
        _, data, indexes, report, _, _ = frozen_inputs(c, ordinary_controls=True)
        _write(output/'frozen-parent-verification.json', c['frozen_parent'])
        _event('ml1m_last8_frozen_parent_verified', kernel=c['frozen_parent']['kernel'],
               revision=c['frozen_parent']['git_sha'], files=len(c['frozen_parent']['files']))
    else:
        data, indexes, report = prepare_indexes(c, output, resources, source)
    extra = {}
    if c.get('extended_budget'):
        from rhpr.gr.pair_test import locate_parent
        resume = locate_parent(c['resume_parent'])
        _write(output/'resume-parent-verification.json', c['resume_parent'])
        extra['resume_parent'] = str(resume)
    overlay = scratch/'modern-site'; overlay.mkdir()
    with (output/'modern-install.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
            '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern),
            '-r', str(modern/'requirements-lock.txt')], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes),
             assignments=assignments(c, 2), **extra)
    path = output/'plan.json'; _write(path, p)
    summary = launch_workers(path, worker_module='rhpr.gr.ml1m_last8', event_prefix='ml1m_last8')
    verify_results(c, summary['results'])
    _write(output/'study-summary.json', dict(results=summary['results'], data=report, new_generator_fits=len(c['trials']), exact_TIGER_reproduction=False))
    metrics = dict(experiment=c['name'], models=len(c['trials']), seed=42, test_evaluated=True, train_examples=48320, validation_examples=6040, test_examples=6040)
    if c.get('frozen_controls'):
        del metrics['seed']
    for name, row in summary['results'].items():
        metrics[name+'_validation_ndcg10'] = row['selection']['validation_ndcg10']
        metrics[name+'_test_ndcg10'] = row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops'] = row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); worker(args.plan, args.slot)
