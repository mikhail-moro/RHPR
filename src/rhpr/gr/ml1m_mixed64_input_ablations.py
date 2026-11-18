"""Inference-only comparison of full and reassigned pair histories on ML1M VAL."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.ml1m_last8_pairs import encode
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex


def read(path):
    return json.loads(path.read_text())


def load_arrays(path, *names):
    with np.load(path) as saved:
        return tuple(saved[name] for name in names)


def prepare(config, frozen):
    data = frozen / 'prepared-cases'
    val = load_arrays(data / 'validation-cases.npz', 'contexts', 'user_ids', 'targets')
    assert case_digest(val) == config['validation_case_sha256']
    anchor_users, past = load_arrays(data / 'validation-anchors.npz', 'user_ids', 'target_positions')
    np.testing.assert_array_equal(anchor_users, val[1])
    selected = past >= config['minimum_history']
    contexts, users, targets = [x[selected] for x in val]
    lengths = (contexts[:, -128:] >= 0).sum(1)
    np.testing.assert_array_equal(lengths, np.minimum(past[selected], 128))
    index_path = frozen / 'frozen-indexes'
    sids, keys, pair_sids = [np.load(index_path / f'{name}.npy')
                            for name in ('items-42', 'pair-keys', 'pairs-42')]
    thinned = config.get('history_reduction') == 'keep_second'
    original = encode(contexts, sids, keys, pair_sids, 64, keep_second=thinned)
    recent = contexts[:, -128:]
    singleton_tokens = sids[np.maximum(recent, 0)] + np.arange(4) * 256
    full = np.where(recent[:, :, None] >= 0, singleton_tokens, -1).reshape(len(recent), 512)

    index = InputIndex4(SidIndex.from_full_sids(sids, width=256))
    views = {'original':original, 'full':full}
    if thinned:
        return index, selected, views, users, targets, past[selected], None

    # A seeded one-to-one reassignment within identical clipped history lengths
    # preserves pair counts, padding, and the overall distribution of pair tokens.
    rng = np.random.default_rng(config['shuffle_seed'])
    donors = np.empty(len(lengths), dtype=int)
    for length in np.unique(lengths):
        rows = rng.permutation(np.flatnonzero(lengths == length))
        if len(rows) < 2:
            raise ValueError(f'no distinct donor for length {length}')
        donors[rows] = np.roll(rows, 1)
    pair_token = original >= 1024
    np.testing.assert_array_equal(np.sort(donors), np.arange(len(users)))
    np.testing.assert_array_equal(pair_token, pair_token[donors])
    assert np.all(users != users[donors])
    swapped = np.where(pair_token, original[donors], original)
    assert np.all(((original != swapped) & pair_token).any(1))
    np.testing.assert_array_equal(original[~pair_token], swapped[~pair_token])
    views['shuffled'] = swapped
    return index, selected, views, users, targets, past[selected], users[donors]


def evaluate_checkpoint(seed, config, prepared, checkpoints, source, output, device):
    import torch
    from rhpr.gr.sid import evaluate_ranked_sids

    index, selected, views, users, targets, past, donors = prepared
    prefix = 'thin128' if config.get('history_reduction') == 'keep_second' else 'mixed128'
    trial = checkpoints / f'trials/{prefix}_merge64_{seed}'
    epoch = read(trial / 'training-selection.json')['best_epoch']
    weights = trial / 'generator-best.pt'
    sha = file_sha256(weights)
    if sha != read(trial / 'test-metrics.json')['checkpoint_sha256']:
        raise ValueError('selected checkpoint hash differs')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if state['epoch'] != epoch:
        raise ValueError('selected checkpoint epoch differs')
    model = attach4(build_grid_generator(index.sids, config['generator'], seed, 'cpu', source))
    model.load_state_dict(state['model'], strict=True)
    model.to(device)
    saved_users, saved_targets, ids = load_arrays(
        trial / f'validation-epoch-{epoch:03d}.npz', 'user_ids', 'target_item_ids', 'semantic_ids')
    np.testing.assert_array_equal(saved_users[selected], users)
    np.testing.assert_array_equal(saved_targets[selected], targets)
    prefix = f'seed_{seed}'
    metrics = {}
    batch_size = config['evaluation_batch_size']

    # Replay a sample of the original view on the actual hardware before comparing inputs.
    rows = np.random.default_rng(seed).choice(len(users), 96, replace=False)
    _, replay, _ = _evaluate(model, (views['original'][rows], users[rows], targets[rows]),
                             index, batch_size, device, 'fp32')
    original_top10 = ids[selected][rows, :10]
    match = float((replay[:, :10] == original_top10).all(-1).mean())
    wanted = index.sids[targets[rows], None, :]
    np.testing.assert_array_equal((replay[:, :10] == wanted).all(-1), (original_top10 == wanted).all(-1))
    if match < .99:
        raise ValueError(f'saved top 10 replay agreement too low: {match:.4f}')
    metrics[f'{prefix}_original_top10_replay_agreement'] = match
    original = evaluate_ranked_sids(index, ids[selected], targets)
    for key in ('ndcg@10', 'recall@10'):
        metrics[f'{prefix}_original_{key}'] = original[key]
    print(f'{prefix} original replay agreement {match:.4f}', flush=True)
    for name in (v for v in views if v != 'original'):
        values, ids, scores = _evaluate(model, (views[name], users, targets),
                                        index, batch_size, device, 'fp32')
        path = output / f'{name}-{seed}.npz'
        donor_data = {'donor_user_ids':donors} if donors is not None else {}
        np.savez_compressed(path, user_ids=users, target_item_ids=targets, semantic_ids=ids,
                            scores=scores, past=past, checkpoint_sha256=sha, epoch=epoch, **donor_data)
        for key in ('ndcg@10', 'recall@10', 'seconds'):
            metrics[f'{prefix}_{name}_{key}'] = values[key]
        print(json.dumps(dict(seed=seed, view=name, sha256=file_sha256(path), **values)), flush=True)
    return metrics


def run(config, output):
    import torch
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory

    if (config['seeds'], config['minimum_history'], config['evaluation_batch_size'], config['shuffle_seed']) != (
            [101, 202], 97, 16, 64128):
        raise ValueError('expected the pinned Mixed64 validation ablation protocol')
    if config.get('history_reduction') not in (None, 'keep_second'):
        raise ValueError('unsupported history reduction')
    devices = config['execution']['worker_devices']
    if (devices != ['cuda:0', 'cuda:1'] or torch.cuda.device_count() != len(devices)
            or any(torch.cuda.get_device_capability(i) != (7, 5) for i in range(len(devices)))):
        raise RuntimeError('this run requires two T4s')
    torch.set_num_threads(1)
    print(f'Kaggle allocated {torch.cuda.device_count()} GPUs', flush=True)
    frozen = locate_parent(config['frozen_parent'])
    checkpoints = locate_parent(config['checkpoints_parent'])
    prepared = prepare(config, frozen)
    roots = {name: resolve_dataset_directory(config[name]['path']) for name in ('resources', 'modern_runtime')}
    for name, root in roots.items():
        verify_resources(root, config[name]['manifest_sha256'])
    scratch = Path(config['resources']['scratch_path'])
    scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(roots['resources'] / 'grid-source.bin', scratch / 'original'),
                                 scratch / 'grid')
    overlay = scratch / 'modern-site'
    with (output / 'modern-install.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--no-cache-dir', '--no-compile', '--require-hashes', '--only-binary=:all:',
                        '--target', str(overlay), '--find-links', str(roots['modern_runtime']),
                        '-r', str(roots['modern_runtime'] / 'requirements-lock.txt')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    plan = scratch / 'plan.json'
    plan.write_text(json.dumps(dict(config=config, frozen=str(frozen), checkpoints=str(checkpoints),
                                    source=str(source), output=str(output))))
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        runs = [
            pool.submit(subprocess.run, [sys.executable, '-u', '-m', __name__, str(plan), str(seed)],
                        env=worker_environment(config, {'device':device}), check=True, timeout=480)
            for seed, device in zip(config['seeds'], devices)
        ]
        for job in runs:
            job.result()
    metrics = {'validation_users':len(prepared[3])}
    for seed in config['seeds']:
        metrics.update(read(output / f'worker-{seed}.json'))
    return metrics


def worker(plan_path, seed):
    import torch
    import transformers

    if (torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5)
            or transformers.__version__ != '5.22.1'):
        raise RuntimeError('worker requires one isolated T4 and Transformers 5.22.1')
    torch.set_num_threads(1)
    plan = read(plan_path)
    config = plan['config']
    prepared = prepare(config, Path(plan['frozen']))
    output = Path(plan['output'])
    metrics = evaluate_checkpoint(seed, config, prepared, Path(plan['checkpoints']),
                                  Path(plan['source']), output, 'cuda:0')
    (output / f'worker-{seed}.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    worker(Path(sys.argv[1]), int(sys.argv[2]))
