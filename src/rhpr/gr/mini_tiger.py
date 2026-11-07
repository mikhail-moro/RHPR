"""Frozen-index mini-TIGER calibration; validation only, no tokenizer fitting."""
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

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples


ARCHITECTURES = {
    'shallow': dict(encoder_layers=1, decoder_layers=1),
    'narrow': dict(encoder_layers=4, decoder_layers=4, d_model=64,
                   d_ff=512, d_kv=32),
    'compact': dict(encoder_layers=2, decoder_layers=2, d_model=128,
                    d_ff=256, num_heads=4, d_kv=32),
}


REDUCED_ARCHITECTURES = {
    'ffn_small': dict(encoder_layers=4, decoder_layers=4, d_model=64,
                      d_ff=256, num_heads=6, d_kv=32),
    'attention_small': dict(encoder_layers=4, decoder_layers=4, d_model=64,
                            d_ff=256, num_heads=4, d_kv=16),
}
ALL_ARCHITECTURES = {**ARCHITECTURES, **REDUCED_ARCHITECTURES}
ALL_ARCHITECTURES['full'] = {}


def protocol_shape(c):
    phase = c.get('calibration_phase', 'pilot')
    if phase == 'pilot':
        return list(ARCHITECTURES), [42, 101], 20
    if phase == 'seed202_transfer':
        return ['narrow'], [202], 10
    if phase == 'width_reduction':
        return list(REDUCED_ARCHITECTURES), [42, 101, 202], 10
    if phase == 'full_seed_replication':
        return ['full'], [42, 101, 202], 30
    raise ValueError('unknown mini-TIGER calibration phase')


def validate(c):
    if c.get('calibration_phase') == 'full_seed_replication':
        from rhpr.gr.proxy_target import validate_target
        return validate_target(c)
    architectures, index_seeds, epochs = protocol_shape(c)
    expected = {(a, i, s) for a in architectures for i in c['index_ids']
                for s in c['generator_seeds']}
    actual = [(r['architecture'], r['index_id'], r['seed']) for r in c['trials']]
    t = c['training']
    indexes = {f'{m}_seed{s}' for m in ['sum', 'uw', 'gradnorm', 'pcgrad'] for s in index_seeds}
    if (c['kind'] != 'gr_mini_tiger' or set(c['index_ids']) != indexes
            or len(c['index_ids']) != len(indexes) or c['generator_seeds'] != [31415, 27182]
            or len(actual) != len(expected) or set(actual) != expected
            or len({r['id'] for r in c['trials']}) != len(actual)
            or t['max_epochs'] != epochs or t['min_epochs'] != epochs
            or t['patience_epochs'] != epochs
            or t['validation_interval'] != 5 or t['validate_first_epoch']
            or t['batch_size'] != 512 or t['physical_batch_size'] != 512
            or t['precision'] != 'fp16' or t['evaluation_precision'] != 'fp32'
            or t['learning_rate'] != .001 or 'scheduler' in t
            or c['data']['history_items'] != 20
            or c['generator']['attn_implementation'] != 'sdpa'
            or c['generator']['beam_width'] != 20
            or c['generator']['should_check_prefix']
            or 'index_input' in c or set(c['parents']) != {str(s) for s in index_seeds}
            or set(c['kaggle']['kernel_sources']) != {p['kernel'] for p in c['parents'].values()}
            or c['evaluation'].get('test_after_selection') is not False):
        raise ValueError('mini-TIGER differs from the frozen-index calibration contract')
    if (c.get('calibration_phase') == 'seed202_transfer' and (c.get('primary_proxy_epoch') != 10
                          or c.get('suite_deadline_seconds') != 3000)):
        raise ValueError('transfer fixes narrow epoch10 and its compute budget in advance')
    if c.get('calibration_phase') == 'width_reduction' and (
            c.get('primary_proxy_epoch') != 10 or c.get('suite_deadline_seconds') != 9900):
        raise ValueError('width reduction fixes epoch10 and the compute budget')
    if set(c['initialization_pins']) != set(architectures) or any(
            len(c['initialization_pins'][a][str(s)]) != 64
            for a in architectures for s in c['generator_seeds']):
        raise ValueError('every architecture/seed needs an initialization pin')


def stage_indexes(c, data, output):
    """Read artifacts already on Kaggle; no new research-data upload needed."""
    from rhpr.gr.resources import verify_parent_artifacts
    ids = json.loads((Path(data) / 'source-ids.json').read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    refs = {}
    for seed, parent in c['parents'].items():
        source = verify_parent_artifacts(parent)
        plan = json.loads((source / 'plan.json').read_text())
        if plan['source_ids'] != ids or plan['config']['data'] != c['data']:
            raise ValueError('completed parent data/catalogue differs')
        for method in ['sum', 'uw', 'gradnorm', 'pcgrad']:
            name = f'{method}_seed{seed}'
            if name not in c['index_ids']:
                continue
            for filename, digest in parent['artifacts'][name].items():
                if filename not in {'semantic-ids.npy', 'training-curves.json', 'training-selection.json', 'generator.json'}:
                    raise ValueError('unexpected frozen reference artifact')
                path = source / 'trials' / name / filename
                if file_sha256(path) != digest:
                    raise ValueError('completed frozen reference artifact changed')
                shutil.copyfile(path, output / (name + '-' + filename))
            curves = json.loads((output / (name + '-training-curves.json')).read_text())
            epochs = {str(r['epoch']): r['validation']['ndcg@10'] for r in curves if 'validation' in r}
            if set(epochs) != {'5', '10', '15', '20', '25', '30'}:
                raise ValueError('full reference needs all six validation checkpoints')
            selection = json.loads((output / (name + '-training-selection.json')).read_text())
            if selection['training_epochs'] != 30 or selection['training_examples_per_epoch'] != 151166:
                raise ValueError('full reference budget changed')
            refs[name] = dict(full_seed=int(seed), method=method, epochs=epochs,
                sids_sha256=parent['artifacts'][name]['semantic-ids.npy'], source_kernel=parent['kernel'])
    _write(output / 'references.json', refs)
    return refs


def rank_correlation(x, y):
    """Spearman with average ties; report undefined instead of a fake zero."""
    def ranks(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite correlation input')
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('aligned correlation vectors required')
    a, b = ranks(x), ranks(y)
    if not np.std(a) or not np.std(b):
        return None
    return float(np.corrcoef(a, b)[0, 1])


def selection_stats(scores, target):
    order = np.argsort(-np.asarray(scores), kind='stable')
    best = np.argsort(-np.asarray(target), kind='stable')
    return dict(spearman=rank_correlation(scores, target),
                top2_overlap=len(set(order[:2]) & set(best[:2])) / 2,
                selected_target_regret=float(max(target) - target[int(order[0])]))


def correlate(c, records, references):
    """Candidate rankings; epochs are repeated measurements, not samples."""
    index_ids = c['index_ids']
    architectures, index_seeds, max_epochs = protocol_shape(c)
    epochs = list(range(5, max_epochs + 1, 5))
    out = dict(index_ids=index_ids, reference_seeds={i: references[i]['full_seed'] for i in index_ids},
               target='fixed full-TIGER epoch30 VAL NDCG@10; best-of30 secondary',
               test_used=False, groups={}, limitations=[
                   f'{len(index_ids)} related indexes; each full target has one generator seed',
                   'same validation users, exploratory calibration, no causal/significance claim',
                   'mini generator seeds are distinct from all full pipeline seeds'])
    for group, ids in [('all', index_ids)] + [
            (f'index_seed{s}', [i for i in index_ids if references[i]['full_seed'] == s])
            for s in index_seeds] + [
            (f'method_{m}', [i for i in index_ids if references[i].get('method') == m])
            for m in ['sum', 'uw', 'gradnorm', 'pcgrad']
            if sum(references[i].get('method') == m for i in index_ids) >= 2]:
        target = [references[i]['epochs']['30'] for i in ids]
        selected = [max(references[i]['epochs'].values()) for i in ids]
        report = dict(index_ids=ids, early_full={}, mini={})
        for epoch in epochs:
            report['early_full'][str(epoch)] = selection_stats(
                [references[i]['epochs'][str(epoch)] for i in ids], target)
        for architecture in architectures:
            for epoch in epochs:
                by_seed = {}
                for seed in c['generator_seeds']:
                    values = []
                    for i in ids:
                        matches = [r for r in records.values() if
                                   (r['architecture'], r['index_id'], r['seed']) == (architecture, i, seed)]
                        if len(matches) != 1:
                            raise ValueError('missing/duplicate mini fit; cannot compute calibration')
                        values.append(matches[0]['epochs'][str(epoch)])
                    by_seed[str(seed)] = values
                mean = np.mean(list(by_seed.values()), axis=0).tolist()
                report['mini'][f'{architecture}_epoch{epoch}'] = dict(
                    mean=selection_stats(mean, target),
                    mean_vs_full_selected=selection_stats(mean, selected),
                    seeds={s: selection_stats(v, target) for s, v in by_seed.items()},
                    scores_by_seed=by_seed, mean_scores=mean)
        out['groups'][group] = report
    if c.get('calibration_phase') == 'seed202_transfer':
        out['primary_proxy_epoch'] = 10
        out['primary'] = out['groups']['all']['mini']['narrow_epoch10']
    if c.get('calibration_phase') == 'width_reduction':
        out['primary_proxy_epoch'] = 10
        out['primary'] = {a: out['groups']['all']['mini'][f'{a}_epoch10'] for a in architectures}
    return out


def run_trial(trial, c, plan, device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    c = copy.deepcopy(c)
    seed = trial['seed']
    c['seed'] = seed
    c['generator'].update(ALL_ARCHITECTURES[trial['architecture']])
    root = Path(plan['output']) / 'trials' / trial['id']
    root.mkdir(parents=True, exist_ok=True)
    sid_path = Path(plan['index_root']) / (trial['index_id'] + '-semantic-ids.npy')
    if file_sha256(sid_path) != plan['references'][trial['index_id']]['sids_sha256']:
        raise ValueError('frozen SID bytes changed')
    sids = np.load(sid_path, allow_pickle=False)
    index = SidIndex.from_full_sids(sids, width=256)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_grid_generator(index.sids, c['generator'], seed, device, plan['source'])
    initial = state_digest(dict(model.named_parameters()))
    if initial != c['initialization_pins'][trial['architecture']][str(seed)]:
        raise ValueError('mini initialization pin changed')
    _write(root / 'generator.json', dict(config=c['generator'], seed=seed,
           parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial,
           sids_sha256=file_sha256(sid_path), frozen_index=True, test_evaluated=False))
    selection = _train_generator(model, plan['data_root'], index, c, root, device,
                                 example_builder=case_examples)
    epochs = protocol_shape(c)[2]
    if selection['training_epochs'] != epochs or selection['validation_checks'] != epochs // 5:
        raise RuntimeError('incomplete mini fit; do not treat as a full-budget result')
    curves = json.loads((root / 'training-curves.json').read_text())
    row = dict(trial=trial['id'], architecture=trial['architecture'], index_id=trial['index_id'],
               seed=seed, selection=selection, initial_state_sha256=initial,
               initialization_group=f"{trial['architecture']}_{seed}",
               epochs={str(r['epoch']): r['validation']['ndcg@10'] for r in curves if 'validation' in r})
    _write(root / 'trial-result.json', row)
    return row


def run_worker(plan_path, slot):
    import torch
    import transformers
    plan = json.loads(Path(plan_path).read_text()); c = plan['config']
    if transformers.__version__ != '5.22.1' or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError('mini worker requires native HF5.22.1 and exactly one T4')
    torch.set_num_threads(1)
    status = dict(results={}, failures=[])
    for trial in plan['assignments'][slot]['trials']:
        _event('mini_tiger_trial_start', trial=trial['id'], worker=slot)
        try:
            if time.time() > plan['deadline']:
                raise TimeoutError('suite compute budget exhausted; no extra fits authorized')
            row = run_trial(trial, c, plan, 'cuda:0')
            status['results'][trial['id']] = row
            _event('mini_tiger_trial_complete', **row)
        except Exception as error:
            root = Path(plan['output']) / 'trials' / trial['id']; root.mkdir(parents=True, exist_ok=True)
            (root / 'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('mini_tiger_trial_failed', **status['failures'][-1])
            _write(Path(plan['output']) / 'workers' / f'worker-{slot}.json', status)
            # Do not retry, change LR/precision, or silently shorten the matrix.
            return
        _write(Path(plan['output']) / 'workers' / f'worker-{slot}.json', status)
        gc.collect(); torch.cuda.empty_cache()


def run_mini_tiger(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / 'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in [0, 1]):
        raise RuntimeError('mini calibration requires2T4; no hardware fallback')
    roots = {}
    for name in ['resources', 'movie_input', 'modern_runtime']:
        root = resolve_dataset_directory(c[name]['path'])
        verify_resources(root, c[name]['manifest_sha256']); roots[name] = root
    # The experiment never opens TEST cases or evaluates TEST predictions.
    for split in ['train', 'validation']:
        if case_digest(case_examples(roots['movie_input'], 20, split)) != c['data']['case_sha256'][split]:
            raise ValueError('frozen ML1M cases changed')
    index_root = output / 'frozen-indexes'
    refs = stage_indexes(c, roots['movie_input'], index_root)
    if set(refs) != set(c['index_ids']):
        raise ValueError('reference index matrix changed')
    scratch = Path(c['resources']['scratch_path'])
    original = unpack_grid(roots['resources'] / 'grid-source.bin', scratch / 'original')
    source = prepare_grid_source(original, scratch / 'grid')
    overlay = scratch / 'modern-site'; overlay.mkdir(parents=True, exist_ok=True)
    modern = roots['modern_runtime']
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
               '--no-compile', '--require-hashes', '--only-binary=:all:', '--target', str(overlay),
               '--find-links', str(modern), '-r', str(modern / 'requirements-lock.txt')]
    with (output / 'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    plan = dict(config=c, output=str(output), source=str(source), data_root=str(roots['movie_input']),
                index_root=str(index_root), references=refs,
                deadline=time.time() + c.get('suite_deadline_seconds', 10000), assignments=assignments(c, 2))
    _write(output / 'plan.json', plan); sys.path.insert(0, str(overlay))
    summary = launch_workers(output / 'plan.json', worker_module='rhpr.gr.mini_tiger', event_prefix='mini_tiger')
    if c.get('calibration_phase') == 'full_seed_replication':
        from rhpr.gr.proxy_target import correlate_target
        analysis = correlate_target(c, summary['results'], refs)
        _write(output / 'correlations.json', analysis)
        result = dict(experiment=c['name'], completed=len(summary['results']),
                      indexes=len(c['index_ids']), generator_seeds=1, architectures=1,
                      test_evaluated=False)
        for key, estimator in [('mini_mean_spearman', 'mini_mean10'),
                               ('historical_full_spearman', 'historical_full30')]:
            rho = analysis['groups']['all']['estimators'][estimator]['spearman']
            if rho is not None:
                result[key] = rho
        return result
    analysis = correlate(c, summary['results'], refs); _write(output / 'correlations.json', analysis)
    result = dict(experiment=c['name'], completed=len(summary['results']), indexes=len(c['index_ids']),
                  generator_seeds=2, architectures=len(protocol_shape(c)[0]), test_evaluated=False)
    for name, row in analysis['groups']['all']['mini'].items():
        rho = row['mean']['spearman']
        if rho is not None: result[name + '_spearman'] = rho
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True); parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args(); run_worker(args.plan, args.slot)
