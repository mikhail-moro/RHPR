"""Matched Recent10 baseline for the 50-epoch pair-SID study."""
from __future__ import annotations
import argparse, copy, gc, json, random, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event, _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_sid import InputIndex, attach_pair_encoder, preencode

TRIALS = [dict(id=f'recent10_{seed}', seed=seed) for seed in (42, 101, 202)]


def validate(c):
    t, d, g = c['training'], c['data'], c['generator']
    if (c['kind'] != 'gr_ml1m_pair_recent10' or c['trials'] != TRIALS or c['recent_items'] != 10
        or c['seeds'] != [42, 101, 202] or d['history_items'] != 20 or d['users'] != 2786
        or d['items'] != 2874 or d['train_examples'] != 151166 or t['min_epochs'] != 50
        or t['max_epochs'] != 50 or t['validation_interval'] != 5 or t['validate_first_epoch']
        or t['batch_size'] != 512 or t['physical_batch_size'] != 512 or t['precision'] != 'fp16'
        or t['evaluation_precision'] != 'fp32' or t['learning_rate'] != .001 or t['weight_decay'] != .0001
        or t['patience_epochs'] != 30 or 'scheduler' in t or g['d_model'] != 64
        or g['encoder_layers'] != 4 or g['decoder_layers'] != 4 or g['attn_implementation'] != 'sdpa'
        or c['evaluation']['test_after_selection'] or c['parent']['git_sha'] != 'c61c27ac9cd8fe8e1c863b2430d0288f3e38c26c'):
        raise ValueError('Recent10 control differs from matched protocol')


def recent_preencode(contexts, item_sids, recent_items=10):
    """Retain the last observed items, regardless of input padding layout."""
    contexts = np.asarray(contexts)
    if contexts.ndim != 2 or recent_items < 1:
        raise ValueError('invalid context array or window')
    retained = np.full((len(contexts), recent_items), -1, dtype=contexts.dtype)
    for i, row in enumerate(contexts):
        items = row[row >= 0][-recent_items:]
        if len(items): retained[i, -len(items):] = items
    return preencode(retained, item_sids)[:, -4 * recent_items:]


def resolve_parent(c, input_root=Path('/kaggle/input')):
    from rhpr.gr.data import file_sha256
    slug = 'rhpr-ml1m-pair-sid-50'
    candidates = [input_root / slug, input_root / 'carasik' / slug,
                  input_root / 'notebooks' / 'carasik' / slug,
                  input_root / 'kernels' / 'carasik' / slug]
    if input_root.exists():
        candidates.extend(p.parent for p in input_root.rglob('run-manifest.json') if slug in p.parts)
    for root in candidates:
        if not all((root / f).is_file() for f in c['parent']['files']):
            continue
        for name, expected in c['parent']['files'].items():
            path = root / name
            if path.stat().st_size != expected['size_bytes'] or file_sha256(path) != expected['sha256']:
                raise ValueError(f'parent artifact mismatch: {name}')
        if json.loads((root / 'run-manifest.json').read_text())['git_sha'] != c['parent']['git_sha']:
            raise ValueError('parent code revision mismatch')
        return root
    raise FileNotFoundError('pinned prior pair-SID kernel output not mounted')


def run_trial(trial, p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    seed = trial['seed']; c = copy.deepcopy(p['config']); c['seed'] = seed
    out = Path(p['output']) / 'trials' / trial['id']; out.mkdir(parents=True, exist_ok=True)
    idx = SidIndex.from_full_sids(np.load(Path(p['parent']) / 'frozen-indexes' / f'items-{seed}.npy'), width=256)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = attach_pair_encoder(build_grid_generator(idx.sids, c['generator'], seed, 'cuda:0', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    parent = json.loads((Path(p['parent']) / 'study-summary.json').read_text())['results']
    if initial != parent[f'control_{seed}']['initial_state_sha256'] or initial != parent[f'pair_{seed}']['initial_state_sha256']:
        raise RuntimeError('initialization differs from saved pair and control runs')
    def builder(root, width, split):
        contexts, users, targets = case_examples(root, width, split)
        return recent_preencode(contexts, idx.sids), users, targets
    selection = _train_generator(model, p['data_root'], InputIndex(idx), c, out, 'cuda:0', example_builder=builder)
    curve = json.loads((out / 'training-curves.json').read_text())
    checks = [r['epoch'] for r in curve if 'validation' in r]
    if (selection['training_epochs'] != 50 or len(curve) != 50 or checks != list(range(5, 51, 5))
        or selection['optimizer_updates'] + selection['amp_skipped_updates'] != 50 * 296):
        raise RuntimeError('incomplete 50-epoch training')
    row = dict(trial=trial['id'], seed=seed, initial_state_sha256=initial, selection=selection,
               epochs={str(r['epoch']): r['validation']['ndcg@10'] for r in curve if 'validation' in r})
    _write(out / 'trial-result.json', row)
    return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config'])
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('worker requires one T4 and HF 5.22.1')
    torch.set_num_threads(1); state = dict(results={}, failures=[])
    path = Path(p['output']) / 'workers' / f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('recent10_trial_start', trial=trial['id'], worker=slot)
        try:
            row = run_trial(trial, p); state['results'][trial['id']] = row
            _event('recent10_trial_complete', **row)
        except Exception as e:
            root = Path(p['output']) / 'trials' / trial['id']; root.mkdir(parents=True, exist_ok=True)
            (root / 'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'], exception_type=type(e).__name__, message=str(e)))
            _event('recent10_trial_failed', trial=trial['id'], exception_type=type(e).__name__, message=str(e))
        _write(path, state); gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / 'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4 GPUs')
    torch.set_num_threads(1); roots = {}
    for name in ('resources', 'movie_input', 'modern_runtime'):
        roots[name] = resolve_dataset_directory(c[name]['path']); verify_resources(roots[name], c[name]['manifest_sha256'])
    for split in ('train', 'validation'):
        if case_digest(case_examples(roots['movie_input'], 20, split)) != c['data']['case_sha256'][split]:
            raise ValueError('source cases differ from prior experiment')
    parent = resolve_parent(c)
    parent_config = json.loads((parent / 'run-manifest.json').read_text())['config']
    for name in ('training', 'generator', 'data', 'evaluation'):
        if c[name] != parent_config[name]:
            raise ValueError(f'{name} differs from the pinned prior experiment')
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(roots['resources'] / 'grid-source.bin', scratch / 'original'), scratch / 'grid')
    overlay = scratch / 'modern-site'; overlay.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
               '--require-hashes', '--only-binary=:all:', '--target', str(overlay), '--find-links',
               str(roots['modern_runtime']), '-r', str(roots['modern_runtime'] / 'requirements-lock.txt')]
    with (output / 'modern-install.log').open('w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=180)
    plan = dict(config=c, output=str(output), source=str(source), data_root=str(roots['movie_input']),
                parent=str(parent), assignments=assignments(c, 2))
    _write(output / 'plan.json', plan); sys.path.insert(0, str(overlay))
    _event('recent10_suite_start', trials=len(c['trials']))
    summary = launch_workers(output / 'plan.json', worker_module='rhpr.gr.pair_recent10', event_prefix='recent10')
    records = summary['results']
    if set(records) != {t['id'] for t in TRIALS}: raise RuntimeError('missing model results')
    _write(output / 'study-summary.json', dict(results=records, parent_git_sha=c['parent']['git_sha'], test_evaluated=False))
    return dict(experiment=c['name'], models=len(records), seeds=3, epochs=50, test_evaluated=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--plan', required=True); ap.add_argument('--slot', required=True, type=int)
    args = ap.parse_args(); worker(args.plan, args.slot)
