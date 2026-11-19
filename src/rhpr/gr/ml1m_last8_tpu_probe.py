"""Bounded XLA compatibility probe for the frozen ML1M depth pilot."""
from __future__ import annotations

import importlib.metadata
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from rhpr.gr.experiment import _autocast, _event, _loss, _write


def depth_config(config):
    from rhpr.gr.ml1m_last8_pairs import validate

    depth = {**config, 'kind': 'gr_ml1m_last8_pairs'}
    validate(depth)
    if (config['generator']['encoder_layers'] != 8 or 'runtime' in config
            or config['tpu_probe']['updates_per_width'] != 3):
        raise ValueError('expected the 8+4 pilot, native XLA and three updates per width')
    return depth


def install_dependencies(config, scratch):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.tpu_probe import install_probe_dependencies
    from rhpr.offline_runtime import resolve_dataset_directory

    base = config['resources']
    resources = resolve_dataset_directory(base['path'])
    verify_resources(resources, base['manifest_sha256'], include_prefixes=base['include_prefixes'])
    install_probe_dependencies(resources, scratch / 'base-site', config['tpu_probe']['wheels_manifest_sha256'])
    modern = config['modern_runtime']
    directory = resolve_dataset_directory(modern['path'])
    manifest = verify_resources(directory, modern['manifest_sha256'])
    supported = set(sys_tags())
    compatible = []
    for entry in manifest['files']:
        filename = entry['path']
        if not filename.endswith('.whl'):
            continue
        name, _, _, tags = parse_wheel_filename(filename)
        if name in {'torch', 'torch-xla', 'libtpu'}:
            raise ValueError('TPU probe must retain the native Torch/XLA stack')
        if name != 'numpy' and tags & supported:
            compatible.append(str(directory / filename))
    overlay = scratch / 'modern-site'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
                    '--target', str(overlay), *compatible], capture_output=True, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    return prepare_grid_source(unpack_grid(resources / 'grid-source.bin', scratch / 'source'), scratch / 'grid')


def prepare_inputs(config):
    import numpy as np
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.ml1m_last8_pairs import depth_pilot_reference, encode, encode_full, frozen_inputs
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import InputIndex4
    from rhpr.gr.sid import SidIndex

    _, data, indexes, _, _, reference_views = frozen_inputs(depth_pilot_reference(config))
    sids, keys, pairs = (np.load(indexes / name, allow_pickle=False)
                         for name in ('items-42.npy', 'pair-keys.npy', 'pairs-42.npy'))
    inputs = {}
    for split in ('train', 'validation'):
        histories, users, targets = cases(data, split)
        full = encode_full(histories, sids)
        thinned = encode(histories, sids, keys, pairs, 64, keep_second=True)
        if case_digest((thinned, users, targets)) != config['expected_thinned_case_sha256'][split]:
            raise ValueError(f'probe thinning differs from the registered {split} inputs')
        if split == 'train' and case_digest((full, users, targets)) != reference_views['full_case_sha256']:
            raise ValueError('probe full histories differ from the frozen training inputs')
        eligible = np.flatnonzero(np.count_nonzero(histories >= 0, axis=1) >= 128)
        size = config['pair_batches']['physical' if split == 'train' else 'evaluation']
        used = size * (config['tpu_probe']['updates_per_width'] if split == 'train' else 1)
        chosen = np.random.default_rng(config['seeds'][0]).choice(eligible, size=used, replace=False)
        inputs[split] = {385: (thinned[chosen], users[chosen], targets[chosen]),
                         513: (full[chosen], users[chosen], targets[chosen])}
    return sids, InputIndex4(SidIndex.from_full_sids(sids, width=256)), inputs


def probe_width(model, index, train, validation, config, device, xm, record):
    import torch

    settings = config['training']
    size = config['pair_batches']['physical']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    weight = next(model.decoder.decoder_mlp[0].parameters())
    before = weight.detach().cpu().clone()
    model.train()
    for start in range(0, len(train[0]), size):
        started = time.monotonic()
        batch = index.model_batch(*(value[start:start + size] for value in train), device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast('bf16', str(device)):
            loss = _loss(model, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip_norm'])
        xm.optimizer_step(optimizer, barrier=True)
        loss_value, norm_value = (float(value.detach().float().cpu()) for value in (loss, norm))
        if not math.isfinite(loss_value) or not math.isfinite(norm_value):
            raise FloatingPointError('nonfinite TPU update')
        record('bf16_update', loss=loss_value, grad_norm=norm_value, seconds=time.monotonic() - started)
    if torch.equal(weight.detach().cpu(), before):
        raise RuntimeError('probe updates did not change the decoder output head')

    model.eval()
    record('fp32_beam_start')
    started = time.monotonic()
    batch = index.model_batch(*validation, device)
    with torch.no_grad():
        ids, scores = model.generate(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask_encoder'], user_id=batch['user_id'])
    xm.mark_step()
    ids, scores = ids.cpu(), scores.cpu()
    expected = (len(validation[0]), config['generator']['beam_width'])
    if ids.shape != (*expected, 4) or scores.shape != expected or not torch.isfinite(scores).all():
        raise RuntimeError('unexpected or nonfinite FP32 beam output')
    record('fp32_beam', queries=len(ids), seconds=time.monotonic() - started)


def run(config, output):
    report = dict(compatible=False, quality_not_evaluated=True, python=platform.python_version(), steps=[])
    started = time.monotonic()
    destination = output / 'tpu-compatibility.json'

    def stage(phase, **details):
        report['last_stage'] = phase
        report['steps'].append(dict(phase=phase, **details))
        _write(destination, report)
        _event('ml1m_tpu_probe', stage=phase, **details)

    try:
        stage('native_xla_import')
        os.environ.setdefault('PJRT_DEVICE', 'TPU')
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        if xr.device_type() != 'TPU':
            raise RuntimeError('probe requires actual TPU hardware')
        torch.set_num_threads(1)
        device = xm.xla_device()
        report.update(torch=str(torch.__version__), torch_xla=str(torch_xla.__version__),
                      device_type=xr.device_type(), addressable_devices=xr.addressable_device_count(), devices_used=1)
        stage('offline_dependencies')
        depth = depth_config(config)
        scratch = Path(config['resources']['scratch_path'])
        scratch.mkdir(parents=True, exist_ok=True)
        source = install_dependencies(config, scratch)
        report['packages'] = {name: importlib.metadata.version(name)
                              for name in ('transformers', 'numpy', 'regex', 'PyYAML')}
        if report['packages']['transformers'] != '5.22.1':
            raise RuntimeError('probe requires the depth pilot Transformers version')
        from rhpr.gr.ml1m_last8_pairs import build_trial_model
        from rhpr.gr.pair_unique4 import attach4
        from rhpr.gr.sid_study import state_digest

        stage('frozen_inputs')
        sids, index, inputs = prepare_inputs(depth)
        seed = config['seeds'][0]
        model = attach4(build_trial_model(sids, depth, seed, source))
        report['matches_gpu_initializer'] = (state_digest(dict(model.named_parameters()))
                                             == depth['depth_pilot']['initializers']['pair'])
        report['attention'] = depth['generator']['attn_implementation']
        torch.manual_seed(seed)
        xm.set_rng_state(seed, device)
        model.to(device)
        for width in (385, 513):
            stage('shape_start', positions=width)
            def record(phase, **details):
                stage(phase, positions=width, **details)
            probe_width(model, index, inputs['train'][width], inputs['validation'][width],
                        depth, device, xm, record)
        report['compatible'] = True
        stage('complete')
    except Exception as error:
        report['failure'] = dict(type=type(error).__name__, message=str(error)[:1500])
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = error.stderr or error.stdout
            if diagnostic:
                report['failure']['process_output'] = diagnostic.decode(errors='replace')[-1500:]
    report['seconds'] = time.monotonic() - started
    _write(destination, report)
    return dict(experiment=config['name'], generator_tpu_compatible=report['compatible'],
                quality_not_evaluated=True, last_stage=report['last_stage'], probe_seconds=report['seconds'])
