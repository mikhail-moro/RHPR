"""Matched TIGER history windows on frozen Steam cohorts and generator seeds."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _event, _loss, _train_generator, _write

WINDOWS = [10, 20, 30]
# Start the longest fit first; one worker then also runs the cheapest control.
TRIALS = [dict(id=f'{dataset}_recent{width}_42', dataset=dataset, history_items=width, seed=42)
          for dataset, width in [('steam', 30), ('steam', 20), ('steam', 10)]]
REPLICATION_TRIALS = TRIALS[:2]
CROSSED_TRIALS = [dict(id=f'{dataset}_recent{width}_{seed}', dataset=dataset, history_items=width, seed=seed)
    for dataset, width, seed in [('steam_a', 20, 101), ('steam_a', 10, 101),
        ('steam_b', 10, 101), ('steam_b', 20, 101), ('steam_a', 20, 202), ('steam_a', 10, 202),
        ('steam_b', 10, 202), ('steam_b', 20, 202), ('steam_b', 10, 42)]]


def trial_config(config, trial):
    """Use the trial seed for initialization AND epoch shuffling/dropout training."""
    c = copy.deepcopy(config); c['seed'] = trial['seed']
    c['data'] = {**c['cohorts'][trial['dataset']], 'history_items': trial['history_items']}
    c['active_trial'] = trial
    if c.get('crossed_recent10_20'):
        c['expected_initial_state_sha256'] = c['expected_initial_state_by_seed'][str(trial['seed'])]
    return c


def validate(c):
    t, g = c['training'], c['generator']
    replication = c.get('replication')
    trials, windows = (REPLICATION_TRIALS, [20, 30]) if replication is not None else (TRIALS, WINDOWS)
    cohorts = {'steam'}
    if c.get('crossed_recent10_20') is not None:
        if (replication is not None or c['crossed_recent10_20'] != dict(data_seeds=[4242, 1729], generator_seeds=[42, 101, 202], reused_models=3)
                or c['seeds'] != [42, 101, 202] or set(c['cohort_inputs']) != {'steam_a', 'steam_b'}
                or set(c['expected_initial_state_by_seed']) != {'42', '101', '202'}
                or c['expected_initial_state_by_seed']['42'] != c['expected_initial_state_sha256']
                or set(c['reused_results']) != {'steam_a_recent10_42', 'steam_a_recent20_42', 'steam_b_recent20_42'}):
            raise ValueError('unexpected crossed Recent10/20 replication')
        trials, windows, cohorts = CROSSED_TRIALS, [10, 20], {'steam_a', 'steam_b'}
    if replication is not None and replication != dict(data_seed=1729, excluded_data_seed=4242, excluded_users=2000,
            overlapping_users=0, original_kernel='carasik/rhpr-steam-history/2'):
        raise ValueError('unexpected Steam replication selection')
    if (c['kind'] != 'gr_history_windows' or c['trials'] != trials or c['windows'] != windows
        or c['seed'] != 42 or c['data']['history_items'] != 30
        or set(c['cohorts']) != cohorts
        or (t['min_epochs'], t['max_epochs'], t['validation_interval'], t['patience_epochs']) != (40, 120, 5, 20)
        or (t['batch_size'], t['physical_batch_size'], t['evaluation_batch_size']) != (512, 512, 64)
        or (t['learning_rate'], t['weight_decay'], t['grad_clip_norm']) != (.003, .0001, 1.)
        or t['precision'] != 'fp16' or t['evaluation_precision'] != 'fp32'
        or t['scheduler'] != dict(kind='warmup_cosine', warmup_epochs=2, total_epochs=60, min_lr=.001)
        or t['early_stopping'] != dict(kind='ema', alpha=.5, min_delta=.0005)
        or (g['encoder_layers'], g['decoder_layers'], g['d_model'], g['d_ff']) != (4, 4, 128, 1024)
        or g['attn_implementation'] != 'sdpa' or c['quantizer']['kind'] != 'grid_rkmeans'
        or c['execution']['worker_devices'] != ['cuda:0', 'cuda:1']
        or not c['evaluation']['test_after_selection'] or c['kaggle']['kernel_sources']):
        raise ValueError('history pilot differs from the declared Steam protocol')


def encode_window(contexts, sids, width):
    """Truncate raw events before SID encoding; never change targets or suffixes."""
    h, sid = np.asarray(contexts), np.asarray(sids)
    if (width not in WINDOWS or h.ndim != 2 or h.shape[1] != 30 or h.dtype.kind not in 'iu'
        or sid.ndim != 2 or sid.shape[1] != 4 or sid.dtype.kind not in 'iu'
        or np.any(h < -1) or np.any(h >= len(sid)) or np.any(sid < 0) or np.any(sid >= 256)):
        raise ValueError('expected30-item contexts and fixed3+1SIDs')
    lengths = (h >= 0).sum(1)
    if np.any((np.arange(30)[None, :] < 30-lengths[:, None]) != (h < 0)):
        raise ValueError('noncontiguous left padding')
    h = h[:, -width:]
    tokens = sid[np.maximum(h, 0)] + np.arange(4)*256
    return np.where(h[:, :, None] >= 0, tokens, -1).reshape(len(h), 4*width).astype(np.int32)


def prepare_cases(directory, specification, output):
    from rhpr.gr.data import build_examples, load_beauty
    texts, users, _, counts = load_beauty(Path(directory), specification)
    if min(len(u.train)+2 for u in users) < 32:
        raise ValueError('cohort requires at least30 events before validation')
    output = Path(output); output.mkdir(parents=True)
    report = dict(**counts, splits={})
    for split in ('train', 'validation', 'test'):
        h, u, t = build_examples(users, 30, split)
        np.savez_compressed(output/(split+'-cases.npz'), contexts=h, user_ids=u, targets=t)
        report['splits'][split] = dict(examples=len(t), context_length_counts=np.bincount((h >= 0).sum(1), minlength=31).tolist())
    _write(output/'data-report.json', report)
    return texts, report


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
    c = trial_config(p['config'], trial); width = trial['history_items']; dataset = trial['dataset']; seed = trial['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    sidpath = Path(p['indexes'])/dataset/'items-42.npy'; sid = np.load(sidpath, allow_pickle=False)
    wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256))
    model = attach4(build_grid_generator(sid, c['generator'], seed, 'cpu', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    if initial != c['expected_initial_state_sha256']:
        raise ValueError('generator initializer differs from CPU preflight')
    data = Path(p['data'])/dataset
    raw = {s: cases(data, s) for s in ('train', 'validation')}
    encoded = {s: (encode_window(h, sid, width), u, t) for s, (h, u, t) in raw.items()}
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    model.eval(); probe = tuple(a[:1] for a in encoded['validation'])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        loss = _loss(model, wrapped.model_batch(*probe, 'cpu'))
    if not torch.isfinite(loss) or state_digest(dict(model.named_parameters())) != initial:
        raise RuntimeError('invalid CPU profiling probe')
    forward = dict(parameters=sum(v.numel() for v in model.parameters()), encoder_positions=4*width+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='Fixed padded width,B1 CPU mathSDPA teacher-forced4code forward,FMA2; excludes beam/backward/text encoder')
    _write(out/'forward-flops.json', forward)
    model.to('cuda:0'); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    def builder(root, ignored, split):
        if split not in encoded: raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection = _train_generator(model, data, wrapped, c, out, 'cuda:0', example_builder=builder)
    epochs = selection['training_epochs']
    if (not 40 <= epochs <= 120 or selection['stopping_reason'] not in ('early_stopping', 'epoch_limit')
        or selection['optimizer_updates']+selection['amp_skipped_updates'] != ((len(raw['train'][2])+511)//512)*epochs):
        raise RuntimeError('incomplete fit; checkpoints retained without automatic retry')
    curve = json.loads((out/'training-curves.json').read_text())
    timing = dict(wall_seconds=time.monotonic()-started, first_epoch_training_seconds=curve[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([v['train_seconds'] for v in curve[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    h, u, t = cases(data, 'test')
    test = test_selected(model, (encode_window(h, sid, width), u, t), wrapped, out, out/'generator-best.pt', selection['best_epoch'])
    row = dict(trial=trial['id'], dataset=dataset, seed=seed, history_items=width, initial_state_sha256=initial,
        sid_file_sha256=file_sha256(sidpath), raw_case_sha256={s: case_digest(v) for s, v in raw.items()},
        case_targets_sha256={s: case_digest(v[1:]) for s, v in raw.items()}, test_case_sha256=case_digest((h, u, t)),
        selection=selection, forward=forward, timing=timing, test=test)
    _write(out/'trial-result.json', row); return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    state = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('history_windows_trial_start', trial=trial['id'], **trial)
        try:
            row = run_trial(trial, p); state['results'][trial['id']] = row
            _event('history_windows_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('history_windows_trial_failed', **state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', state)
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import fit_grid_rkmeans, prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True); (output/'workers').mkdir()
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no fallback')
    input_specs = c.get('cohort_inputs') or {dataset: c['pilot_input'] for dataset in c['cohorts']}
    inputs = {}
    for dataset, spec in input_specs.items():
        inputs[dataset] = resolve_dataset_directory(spec['path'])
        for name, digest in spec['sha256'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or file_sha256(inputs[dataset]/path) != digest:
                raise ValueError('pilot input digest mismatch: '+dataset+'/'+name)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    data = output/'prepared-cases'; indexes = output/'frozen-indexes'; indexes.mkdir()
    reports = {}; texts = {}
    for dataset, spec in c['cohorts'].items():
        texts[dataset], reports[dataset] = prepare_cases(inputs[dataset], spec, data/dataset)
    provenance_by_dataset = {k: json.loads((v/'source-provenance.json').read_text()) for k, v in inputs.items()}
    provenance = provenance_by_dataset.get('steam')
    if c.get('replication') is not None:
        selected, excluded = provenance['selected_source_user_ids'], provenance['excluded_source_user_ids']
        if (len(selected) != 2000 or len(set(selected)) != 2000 or len(excluded) != 2000 or len(set(excluded)) != 2000
                or set(selected) & set(excluded) or provenance['data_seed'] != 1729 or provenance['excluded_data_seed'] != 4242
                or provenance['overlapping_users'] != 0 or provenance['selection_uses_model_outcomes'] is not False
                or c['selection']['data_seed'] != 1729):
            raise ValueError('replication must use the fixed disjoint cohort')
    if c.get('crossed_recent10_20'):
        a = json.loads((inputs['steam_a']/'user-audit.json').read_text())
        b = json.loads((inputs['steam_b']/'user-audit.json').read_text())
        old, new = ({r['source_sasrec_user_id'] for r in rows} for rows in (a, b))
        if (len(old) != 2000 or len(new) != 2000 or old & new or texts['steam_a'] != texts['steam_b']
                or old != set(provenance_by_dataset['steam_b']['excluded_source_user_ids'])
                or new != set(provenance_by_dataset['steam_b']['selected_source_user_ids'])):
            raise ValueError('crossed experiment requires the two pinned disjoint cohorts and identical catalogue')
    _write(output/'cohort-provenance.json', provenance if provenance is not None else provenance_by_dataset)
    encoder = SentenceTransformer(str(resources/'sentence-t5-base'), device='cuda:0', local_files_only=True)
    encoder.max_seq_length = 256; encoder.eval()
    for dataset in c['cohorts']:
        root = indexes/dataset; root.mkdir()
        _event('history_windows_embedding_start', dataset=dataset, items=len(texts[dataset]))
        vec = np.asarray(encoder.encode(texts[dataset], batch_size=32, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if vec.shape != (len(texts[dataset]), 768) or not np.isfinite(vec).all(): raise ValueError('invalid item embeddings')
        np.save(root/'item-embeddings.npy', vec)
    del encoder; gc.collect(); torch.cuda.empty_cache()
    for dataset in c['cohorts']:
        root = indexes/dataset; vec = np.load(root/'item-embeddings.npy')
        codes, centers, fit = fit_grid_rkmeans(vec, c['quantizer'], 42, 'cuda:0', source)
        idx = SidIndex(codes, width=256)
        np.save(root/'items-42.npy', idx.sids); np.save(root/'centroids-42.npy', centers)
        if c.get('expected_sid_file_sha256') and file_sha256(root/'items-42.npy') != c['expected_sid_file_sha256']:
            raise ValueError('recomputed SID differs from the pinned original index; no generator training allowed')
        _write(root/'index-report.json', dict(fit=fit, sid=idx.diagnostics(), scope='full static source-core catalogue; no behavior fitting'))
    del vec, codes, centers; gc.collect(); torch.cuda.empty_cache()
    overlay = scratch/'modern-site'; overlay.mkdir()
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    p = dict(config=c, output=str(output), source=str(source), data=str(data), indexes=str(indexes), assignments=assignments(c, 2))
    path = output/'plan.json'; _write(path, p)
    summary = launch_workers(path, worker_module='rhpr.gr.history_windows', event_prefix='history_windows')
    verify_matched_results(c, summary['results'])
    _write(output/'study-summary.json', dict(results=summary['results'], data=reports, new_generator_fits=len(c['trials']),
        reused_results=c.get('reused_results', {}), exact_TIGER_reproduction=False))
    metrics = dict(experiment=c['name'], models=len(c['trials']), seed=42, test_evaluated=True)
    if c.get('crossed_recent10_20'):
        metrics.update(generator_seed_count=3, cohort_count=2, reused_models=3, total_comparison_models=12)
    for name, row in summary['results'].items():
        metrics[name+'_validation_ndcg10'] = row['selection']['validation_ndcg10']
        metrics[name+'_test_ndcg10'] = row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops'] = row['forward']['supported_forward_flops_batch1']
    return metrics


def verify_matched_results(c, results):
    if set(results) != {t['id'] for t in c['trials']}: raise ValueError('incomplete registered trials')
    matched = dict(results)
    for name, reference in c.get('reused_results', {}).items():
        row = reference['result']
        if name in matched or row['trial'] != name or row['seed'] != 42 or reference['best_checkpoint_sha256'] != row['test']['checkpoint_sha256']:
            raise ValueError('invalid reused control reference')
        matched[name] = row
    for dataset in c['cohorts']:
        rows = [r for r in matched.values() if r['dataset'] == dataset]
        for key in ('sid_file_sha256', 'raw_case_sha256', 'case_targets_sha256', 'test_case_sha256'):
            if not rows or any(v[key] != rows[0][key] for v in rows[1:]): raise ValueError('unmatched comparison: '+key)
    if c.get('crossed_recent10_20'):
        expected_cells = {(cohort, seed, width) for cohort in c['cohorts'] for seed in c['seeds'] for width in c['windows']}
        if {(r['dataset'], r['seed'], r['history_items']) for r in matched.values()} != expected_cells or len(matched) != 12:
            raise ValueError('incomplete crossed comparison')
    for seed in {r['seed'] for r in matched.values()}:
        rows = [r for r in matched.values() if r['seed'] == seed]
        expected = c.get('expected_initial_state_by_seed', {}).get(str(seed), c['expected_initial_state_sha256'])
        if any(r['initial_state_sha256'] != expected for r in rows): raise ValueError('unmatched seed initializer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); worker(args.plan, args.slot)
