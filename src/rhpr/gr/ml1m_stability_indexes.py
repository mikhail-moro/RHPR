"""Build frozen ML1M text indexes and stage the pinned multimodal controls.

All methods use a three-level vocabulary of 256 plus the same collision suffix.
KMeans follows the earlier Beauty SID study (Lloyd on normalized residuals);
the text RQ-VAE keeps all historical head slots but activates only metadata.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.sid import SidIndex, validate_item_embeddings


def _checked_sids(path, expected_sha, items):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_sha:
        raise ValueError('multimodal control bytes do not match the pinned index')
    sids = np.load(path, allow_pickle=False)
    if sids.shape != (items, 4):
        raise ValueError('multimodal control catalogue or SID depth changed')
    index = SidIndex.from_full_sids(sids, width=256)
    return index


def _aligned_features(features_root, data_root, expected_features_sha256):
    root, data = Path(features_root), Path(data_root)
    path = root / 'features.npz'
    if file_sha256(path) != expected_features_sha256:
        raise ValueError('shared item representations do not match their pinned hash')
    ids = json.loads((data / 'source-ids.json').read_text())
    if len(ids) != 2874 or len(set(ids)) != len(ids) or ids != json.loads((root / 'source-ids.json').read_text()):
        raise ValueError('shared representation catalogue and original movie order differ')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in ('text', 'poster', 'plot', 'people')}
        masks = {k: f[k + '_observed'] for k in arrays}
    for name, values in arrays.items():
        validate_item_embeddings(values, len(ids))
        if masks[name].dtype != np.bool_ or masks[name].shape != (len(ids),):
            raise ValueError('modality observation masks must be aligned boolean arrays')
    if arrays['text'].shape != (2874, 768) or not masks['text'].all():
        raise ValueError('all catalogue items must have their pinned T5 metadata vector')
    return ids, arrays, masks


def stage_stability_indexes(output, features_root, data_root, source, original,
                            quantizer_spec, multimodal_sources, *, expected_features_sha256,
                            device='cuda:0'):
    """Stage six index arrays, returning references used by generator workers.

    ``multimodal_sources`` maps each integer seed 42/101 to a dict with ``path``
    and a pre-pinned ``sha256``. The caller must verify the parent execution and
    catalogue provenance when resolving those sources. Quantizer seeds and
    all regularization/protocol choices are fixed by the experiment contract.
    This function does not inspect validation or test, nor train any generator.
    """
    from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
    from rhpr.gr.sid_study import fit_quantizer

    seeds = (42, 101)
    if set(multimodal_sources) != set(seeds):
        raise ValueError('two pinned multimodal indexes, seeds 42 and 101, required')
    q = copy.deepcopy(quantizer_spec)
    if (q['num_hierarchies'] != 3 or q['codebook_size'] != 256 or q['max_steps'] != 3000
            or q['head_dim'] != 128 or q['head_hidden'] != 256
            or q['ae_warmup'] != {'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
            or set(q['expected_initial_state_sha256_by_seed']) != {'42','101'}):
        raise ValueError('text RQ-VAE must use the matched 3500 AE/3000 RQ recipe and seed pins')
    ids, arrays, masks = _aligned_features(features_root, data_root, expected_features_sha256)
    controls = {s: _checked_sids(multimodal_sources[s]['path'], multimodal_sources[s]['sha256'], len(ids)) for s in seeds}
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    if (root / 'references.json').exists() or any(root.glob('*-semantic-ids.npy')):
        raise FileExistsError('index output is not empty; refusing to overwrite existing research artifacts')
    kspec = {'levels': 3, 'codebook_size': 256, 'n_init': 1, 'max_iter': 100, 'cpu_threads': 2}
    refs = {}
    for seed in seeds:
        for kind in ('meta_kmeans', 'meta_rqvae', 'multimodal_rqvae'):
            name = f'{kind}_seed{seed}'
            work = root / name; work.mkdir()
            _event('stability_index_start', index_id=name, tokenizer_seed=seed)
            if kind == 'meta_kmeans':
                index, _ = fit_quantizer(validate_item_embeddings(arrays['text'], len(ids)), kspec, seed, work)
                report = {'method': 'beauty_residual_lloyd_kmeans', 'normalization': 'L2 each incoming residual',
                          'active_modalities': ['text'], 'spec': kspec,
                          'sklearn_version': importlib.metadata.version('scikit-learn'),
                          'numpy_version': np.__version__}
            elif kind == 'meta_rqvae':
                qq = copy.deepcopy(q)
                qq['mtl_mode'] = 'sum'
                qq['zero_inactive_slots'] = ['poster','plot','people']
                codes, qr = fit_multimodal_rqvae(arrays, masks, qq, seed, device, source,
                    Path(original) / 'configs/experiment/rqvae_train_flat.yaml', work, ['text'])
                index = SidIndex(codes, width=256)
                report = {'method': 'text_ae_warmup_grid_rqvae', 'active_modalities': ['text'],
                          'quantizer_initial_state_sha256': qr['quantizer_initial_state_sha256'],
                          'steps': qr['steps'], 'warmup': qr['warmup']}
            else:
                index = controls[seed]
                report = {'method': 'reuse_multimodal_sum_rqvae', 'active_modalities': ['text','poster','plot'],
                          'source_path': str(multimodal_sources[seed]['path']),
                          'source_sha256': multimodal_sources[seed]['sha256']}
            flat = root / (name + '-semantic-ids.npy')
            if kind == 'multimodal_rqvae':
                shutil.copyfile(multimodal_sources[seed]['path'], flat)
                if file_sha256(flat) != multimodal_sources[seed]['sha256']:
                    raise RuntimeError('control copy is not byte-identical')
            else:
                np.save(flat, index.sids, allow_pickle=False)
            if np.load(flat, allow_pickle=False).shape != (len(ids), 4):
                raise ValueError('generated SID catalogue is incomplete')
            report.update(index_id=name, tokenizer_seed=seed, items=len(ids),
                          sids_sha256=file_sha256(flat), features_sha256=expected_features_sha256,
                          suffix='catalogue-row collision ordinal; fixed for singleton and collisions',
                          diagnostics=index.diagnostics())
            refs[name] = report
            _write(work / 'stability-index.json', report)
            _event('stability_index_complete', index_id=name, sids_sha256=report['sids_sha256'])
    _write(root / 'references.json', refs)
    return refs
