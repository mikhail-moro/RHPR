"""Bounded ML1M tokenizer-by-generator stability study, validation only.

All six indexes are staged before any generator starts.  Parent artifacts are
verified by manifest/source/SID/content hashes; fresh text indexes use the
same aligned frozen T5 inputs.  The workers train isolated narrow generators.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.generator_stability import validate_training_protocol


INDEX_IDS = tuple(f'{method}_seed{s}' for method in
                  ('meta_kmeans', 'meta_rqvae', 'multimodal_rqvae') for s in (42, 101))
FEATURES_SHA256 = 'fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4'
PARENT_KERNELS = {'42': 'carasik/rhpr-ml1m-clean-mtl',
                  '101': 'carasik/rhpr-ml1m-clean-mtl-replicates'}


def validate(c):
    """Reject changes to the authorized matrix, data, budget, and provenance."""
    validate_training_protocol(c)
    if (c.get('kind') != 'gr_ml1m_stability' or c.get('index_ids') != list(INDEX_IDS)
            or c.get('feature_sha256') != FEATURES_SHA256
            or c.get('tokenizer_seeds') != [42,101]
            or c.get('compare_initialization_by_seed') is not True
            or c.get('suite_deadline_seconds', 0) < 25000
            or c['suite_deadline_seconds'] > 28000
            or not 3200 <= c['training']['max_seconds'] <= 3600
            or set(c['parents']) != set(PARENT_KERNELS)
            or any(c['parents'][s]['kernel'] != name for s, name in PARENT_KERNELS.items())
            or c['data'].get('split') != 'raw_contiguous_20_val_test'
            or c['training'].get('device') != 'cuda' or c['training'].get('amp') is not True
            or c['training'].get('sampling') != 'precomputed_raw_clean_windows'):
        raise ValueError('ML1M stability matrix, input or budget differs from authorized protocol')
    q = c['quantizer']
    if (q.get('kind') != 'grid_multimodal_rqvae' or q.get('num_hierarchies') != 3
            or q.get('codebook_size') != 256 or q.get('max_steps') != 3000
            or q.get('batch_size') != 2048 or q.get('init_buffer_size') != 2874
            or q.get('initialization_steps') != 1000 or q.get('distance_backend') != 'torch_cdist'
            or q.get('head_dim') != 128 or q.get('head_hidden') != 256
            or q.get('zero_inactive_slots') != ['people']
            or q.get('ae_warmup') != {'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
            or set(q.get('expected_initial_state_sha256_by_seed', {})) != {'42', '101'}):
        raise ValueError('text and existing multimodal RQ-VAE need the matched pinned recipe')
    k = c['kaggle']
    if (set(k['kernel_sources']) != set(PARENT_KERNELS.values())
            or set(k['dataset_sources']) != {'carasik/rhpr-gr-beauty-phase0',
                 'carasik/rhpr-ml1m-clean-mtl-input','carasik/rhpr-t5-sdpa-runtime'}
            or k.get('machine_shape') != 'NvidiaTeslaT4'
            or k.get('competition_sources') or k.get('model_sources')):
        raise ValueError('stability sources or T4 machine shape differ from the fixed protocol')
    # The two controls are exact, immutable SUM artifacts from the original
    # parent runs, independent of whichever local paths resolve those parents.
    for seed in ('42', '101'):
        parent = c['parents'][seed]
        if ('run-manifest.json' not in parent['files'] or 'plan.json' not in parent['files']
                or f'sum_seed{seed}' not in parent['artifacts']
                or len(parent['artifacts'][f'sum_seed{seed}']['semantic-ids.npy']) != 64):
            raise ValueError('missing pinned parent provenance or frozen SUM SID')


def summarize(c, records, output):
    """Descriptive finite output; fixed epoch30 and selected scores are distinct."""
    if set(records) != {t['id'] for t in c['trials']}:
        raise ValueError('all 18 generator fits are required')
    seeds = c['generator_seeds']; groups = {}
    for i in c['index_ids']:
        rows = [records[next(t['id'] for t in c['trials'] if t['index_id'] == i and t['seed'] == s)]
                for s in seeds]
        for seed, row in zip(seeds, rows):
            if (row['index_id'] != i or row['seed'] != seed or row['architecture'] != 'narrow'
                    or '30' not in row['epochs']
                    or row['plateau_reached'] == row['budget_censored']):
                raise ValueError('trial identity, fixed checkpoint, or stopping flags changed')
        fixed = np.asarray([float(r['epochs']['30']) for r in rows])
        selected = np.asarray([float(r['selection']['validation_ndcg10']) for r in rows])
        if not np.isfinite(fixed).all() or not np.isfinite(selected).all():
            raise ValueError('nonfinite per-index NDCG')
        groups[i] = dict(generator_seeds=seeds,
            fixed_epoch30_by_seed=dict(zip(map(str,seeds), fixed.tolist())),
            fixed_epoch30_mean=float(fixed.mean()),fixed_epoch30_sample_std=float(fixed.std(ddof=1)),
            selected_by_seed=dict(zip(map(str,seeds),selected.tolist())),
            selected_mean=float(selected.mean()),selected_sample_std=float(selected.std(ddof=1)),
            best_epochs=[r['selection']['best_epoch'] for r in rows],
            stop_epochs=[r['selection']['training_epochs'] for r in rows],
            plateau_reached=sum(bool(r['plateau_reached']) for r in rows),
            budget_censored=sum(bool(r['budget_censored']) for r in rows))
    report = dict(indexes=groups,comparison='fixed epoch30 and selected VAL reported separately',
        caveats=['Three generator seeds per fixed index and two tokenizer seeds per pipeline are descriptive.',
                 'Same validation cohort; selected checkpoints are correlated with selection noise.',
                 'RQ-KMeans versus RQ-VAE is a pipeline comparison, not an isolated quantizer intervention.',
                 'Early stopping is an operational rule, not proof of convergence.'],test_evaluated=False)
    _write(Path(output)/'stability-summary.json', report)
    return dict(experiment=c['name'],completed=len(records),indexes=len(groups),generator_seeds=len(seeds),
        plateau_reached=sum(x['plateau_reached'] for x in groups.values()),
        budget_censored=sum(x['budget_censored'] for x in groups.values()),test_evaluated=False)


def run_stability(c, output):
    """Stage pinned/fresh indexes, then launch isolated T4 workers."""
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ml1m_clean_mtl import case_examples
    from rhpr.gr.ml1m_stability_indexes import stage_stability_indexes
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.resources import unpack_grid, verify_parent_artifacts, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7,5) for i in (0,1)):
        raise RuntimeError('stability requires exactly two T4 GPUs, no hardware fallback')
    roots={}
    for name in ('resources','movie_input','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path'])
        verify_resources(roots[name],c[name]['manifest_sha256'])
    data=roots['movie_input']; ids=json.loads((data/'source-ids.json').read_text())
    for split in ('train','validation'):
        if case_digest(case_examples(data,20,split)) != c['data']['case_sha256'][split]:
            raise ValueError('frozen training or validation cases differ')
    parents={s:verify_parent_artifacts(c['parents'][s]) for s in ('42','101')}
    sources={}
    for seed in ('42','101'):
        source=parents[seed]; plan=json.loads((source/'plan.json').read_text())
        if plan['source_ids'] != ids or plan['config']['data'] != c['data']:
            raise ValueError('frozen control parent uses a different dataset/catalogue')
        sources[int(seed)]={'path':source/'trials'/f'sum_seed{seed}'/'semantic-ids.npy',
            'sha256':c['parents'][seed]['artifacts'][f'sum_seed{seed}']['semantic-ids.npy']}
        feature=source/'features'/'shared-modalities'/'features.npz'
        if file_sha256(feature) != c['feature_sha256'] or plan['features_sha256'] != c['feature_sha256']:
            raise ValueError('shared item embeddings differ between control parents')
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    original=unpack_grid(roots['resources']/'grid-source.bin',scratch/'original')
    source=prepare_grid_source(original,scratch/'grid')
    features=parents['42']/'features'/'shared-modalities';index_root=output/'frozen-indexes'
    refs=stage_stability_indexes(index_root,features,data,source,original,c['quantizer'],sources,
        expected_features_sha256=c['feature_sha256'],device='cuda:0')
    if set(refs) != set(c['index_ids']): raise ValueError('staged index matrix differs')
    # Quantizer fitting intentionally happens before the modern Transformers
    # overlay, preserving the original validated RQ-VAE dependency stack.
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir',
        '--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),
        '--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(data),index_root=str(index_root),
        references=refs,deadline=time.time()+c['suite_deadline_seconds'],assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    _event('stability_suite_start',indexes=len(refs),trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.generator_stability',event_prefix='stability')
    return summarize(c,summary['results'],output)
