"""Frozen Sentence-T5 field ladder, native RQ-VAE and matched temporal TIGER."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _write, _event
from rhpr.gr.movie_data import load_temporal_users, temporal_examples, case_digest


def validate(config):
    if (config['data']['split'] != 'timestamp_group_leave_two'
            or config['representation']['max_length'] != 2048
            or config['training']['max_epochs'] != 20
            or config['training']['min_epochs'] != 20
            or config['training']['validation_interval'] != 5
            or config['training']['validate_first_epoch']
            or config['quantizer']['init_buffer_size'] > config['data']['items']):
        raise ValueError('movie ladder differs from approved fixed-budget protocol')
    seeds = config.get('replicate_seeds', [config['seed']])
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds) or config['seed'] not in seeds):
        raise ValueError('movie ladder requires distinct nonnegative pipeline seeds')
    pairs = [(trial.get('seed', config['seed']), trial['stage']) for trial in config['trials']]
    warm = config.get('kind') == 'gr_movie_multimodal_ae'
    multimodal = config.get('kind') in {'gr_movie_multimodal', 'gr_movie_multimodal_ae', 'gr_movie_multimodal_ae_ladder'}
    stages = [1] if warm else range(1, 5) if multimodal else range(1, 8)
    expected = {(seed, stage) for seed in seeds for stage in stages}
    if (any(type(seed) is not int or type(stage) is not int for seed, stage in pairs)
            or len(pairs) != len(expected) or set(pairs) != expected):
        raise ValueError('each pipeline seed must have exactly the declared stages')
    if multimodal:
        from rhpr.gr.movie_multimodal import validate_modalities
        validate_modalities(config)
    if 'replicate_seeds' in config:
        if config.get('compare_initialization_by_seed') is not True:
            raise ValueError('replicated movie ladders must compare initialization within each seed')
        hashes = config.get('expected_initial_state_sha256_by_seed', {})
        if (set(hashes) != {str(seed) for seed in seeds}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value)
                       for value in hashes.values())
                or hashes[str(config['seed'])] != config['expected_initial_state_sha256']):
            raise ValueError('each pipeline seed needs its pinned generator initialization')


def movie_assignments(config, available_gpus):
    from rhpr.gr.ae_suite import assignments
    if config.get('kind') != 'gr_movie_multimodal_ae':
        return assignments(config, available_gpus)
    validate(config)
    devices = config['execution']['worker_devices']
    if config['training']['device'] == 'cuda':
        if devices != ['cuda:0'] or available_gpus < 1:
            raise RuntimeError('single AE experiment needs its assigned CUDA GPU; no hardware fallback')
    elif config['training']['device'] != 'cpu' or devices != ['cpu']:
        raise ValueError('CPU verification needs one explicit CPU worker')
    if config['execution']['cpu_threads_per_worker'] != 1:
        raise ValueError('single AE experiment requires one CPU thread')
    if len(config['trials']) != 1 or config['trials'][0]['id'] != 'seed42_text_ae':
        raise ValueError('single AE experiment requires one fixed safe trial ID')
    return [{'slot': 0, 'device': devices[0], 'trials': config['trials']}]


def run_trial(trial, config, plan, users, source, device):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.rqvae import fit_grid_rqvae
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.experiment import _train_generator, _evaluate

    stage = trial['stage']; seed = trial.get('seed', config['seed'])
    # The trainer also consumes config.seed for the epoch permutations and saved
    # checkpoints. Override the whole trial config, not only torch.manual_seed.
    config = {**config, 'seed': seed}
    root = Path(plan['output']) / 'trials' / trial['id']; root.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    path = Path(plan['data_root']) / f'catalog-stage-{stage:02d}.json'
    rows = json.loads(path.read_text()); texts = [r['text'] for r in rows]
    if [r['item_id'] for r in rows] != plan['source_ids']:
        raise ValueError('trial catalogue order differs')
    before = time.monotonic()
    encoder = SentenceTransformer(str(Path(plan['resources']) / 'sentence-t5-base'), device=device, local_files_only=True)
    encoder.max_seq_length = config['representation']['max_length']
    lengths = [len(encoder.tokenizer.encode(t, truncation=False)) for t in texts]
    if max(lengths) > encoder.max_seq_length:
        raise ValueError('text would be truncated')
    probe = encoder.tokenize([texts[int(np.argmax(lengths))]])['attention_mask']
    if int(probe.sum()) != max(lengths):
        raise ValueError('encoder tokenization silently shortened the longest text')
    embeddings = encoder.encode(texts, batch_size=config['representation']['batch_size'], convert_to_numpy=True, show_progress_bar=False)
    if embeddings.shape != (len(rows), 768) or not np.isfinite(embeddings).all():
        raise ValueError('invalid Sentence-T5 embeddings')
    np.save(root / 'item-embeddings.npy', embeddings, allow_pickle=False)
    _write(root / 'representation.json', {'stage': stage, 'fields': config['fields'][:stage],
        'model': config['representation'], 'frozen': True, 'seconds': time.monotonic() - before,
        'catalogue_sha256': file_sha256(path), 'embedding_sha256': file_sha256(root / 'item-embeddings.npy'),
        'tokens_max': max(lengths), 'truncated_items': 0,
        'modules': [type(m).__name__ for m in encoder]})
    del encoder; gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    codes, quantizer = fit_grid_rqvae(embeddings, config['quantizer'], seed, device, source,
        Path(plan['original']) / 'configs/experiment/rqvae_train_flat.yaml', root)
    del embeddings; gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    index = SidIndex(codes, width=config['quantizer']['codebook_size'])
    np.save(root / 'semantic-ids.npy', index.sids, allow_pickle=False)
    _write(root / 'sid-diagnostics.json', index.diagnostics())
    _write(root / 'sid-catalogue.json', {'original_item_ids': plan['source_ids']})
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_grid_generator(index.sids, config['generator'], seed, device, source)
    initial = state_digest(dict(model.named_parameters()))
    expected = (plan['initial_state_sha256_by_seed'][str(seed)]
                if 'initial_state_sha256_by_seed' in plan else plan['initial_state_sha256'])
    if initial != expected:
        raise ValueError('generator initialization differs between feature configurations')
    _write(root / 'generator.json', {'parameters': sum(p.numel() for p in model.parameters()),
        'initial_state_sha256': initial, 'config': config['generator']})
    selection = _train_generator(model, users, index, config, root, device, example_builder=temporal_examples)
    if selection['training_epochs'] != 20 or selection['validation_checks'] != 4:
        raise RuntimeError('incomplete declared20epoch fit')
    examples = temporal_examples(users, config['data']['history_items'], 'test')
    scores, predictions, probabilities = _evaluate(model, examples, index,
        config['training']['evaluation_batch_size'], device, precision='fp32')
    np.savez_compressed(root / 'test-predictions.npz', semantic_ids=predictions, scores=probabilities,
        user_ids=examples[1], target_item_ids=examples[2])
    result = {'trial': trial['id'], 'stage': stage, 'seed': seed, 'initial_state_sha256': initial,
        'quantizer_steps': quantizer['steps'], 'selection': selection, 'test': scores}
    _write(root / 'trial-result.json', result)
    del model; gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    return result


def run_worker(plan_path, slot):
    import torch
    plan = json.loads(Path(plan_path).read_text()); config = plan['config']; assignment = plan['assignments'][slot]
    torch.set_num_threads(config['execution']['cpu_threads_per_worker'])
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    device = 'cpu'
    if assignment['device'] != 'cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('movie worker requires exactly its assigned GPU')
        torch.cuda.set_device(0); device = 'cuda:0'
    users = load_temporal_users(Path(plan['data_root']) / 'sequences.json', plan['source_ids'])
    status = {'slot': slot, 'status': 'running', 'results': {}, 'failures': []}
    status_path = Path(plan['output']) / 'workers' / f'worker-{slot}.json'
    _write(status_path, status)
    for trial in assignment['trials']:
        _event('movie_suite_trial_start', trial=trial['id'], stage=trial['stage'],
               seed=trial.get('seed', config['seed']), worker=slot)
        status['current_trial'] = trial['id']; _write(status_path, status)
        try:
            fit = run_trial
            if config.get('kind') in {'gr_movie_multimodal', 'gr_movie_multimodal_ae', 'gr_movie_multimodal_ae_ladder'}:
                from rhpr.gr.movie_multimodal import run_multimodal_trial
                fit = run_multimodal_trial
            result = fit(trial, config, plan, users, plan['source'], device)
            status['results'][trial['id']] = result
            _event('movie_suite_trial_complete', trial=trial['id'], test=result['test'])
        except Exception as e:
            root = Path(plan['output']) / 'trials' / trial['id']; root.mkdir(parents=True, exist_ok=True)
            (root / 'failure.log').write_text(traceback.format_exc())
            status['failures'].append({'trial': trial['id'], 'exception_type': type(e).__name__, 'message': str(e)})
            _event('movie_suite_trial_failed', **status['failures'][-1])
        _write(status_path, status); gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
    status['status'] = 'complete'; _write(status_path, status)


def run_movie_suite(config, output):
    import torch
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.resources import verify_resources, unpack_grid
    from rhpr.gr.grid import prepare_grid_source, build_grid_generator
    from rhpr.gr.sid_study import state_digest
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(config); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    jobs = movie_assignments(config, torch.cuda.device_count())
    resources = resolve_dataset_directory(config['resources']['path'])
    verify_resources(resources, config['resources']['manifest_sha256'])
    data_root = resolve_dataset_directory(config['movie_input']['path'])
    verify_resources(data_root, config['movie_input']['manifest_sha256'])
    modality_root = None
    if config.get('kind') in {'gr_movie_multimodal_ae', 'gr_movie_multimodal_ae_ladder'}:
        from rhpr.gr.movie_multimodal import frozen_control_input
        modality_root = frozen_control_input(config['modality_input'])
    elif config.get('kind') == 'gr_movie_multimodal':
        modality_root = resolve_dataset_directory(config['modality_input']['path'])
        verify_resources(modality_root, config['modality_input']['manifest_sha256'])
    rows = json.loads((data_root / 'catalog-stage-01.json').read_text()); source_ids = [r['item_id'] for r in rows]
    users = load_temporal_users(data_root / 'sequences.json', source_ids)
    split = json.loads((data_root / 'split-report.json').read_text())
    if (len(source_ids) != config['data']['items'] or len(users) != config['data']['users']
            or split['items'] != len(source_ids) or split['users'] != len(users)
            or split['train_examples'] != config['data']['train_examples']):
        raise ValueError('movie input counts changed')
    for name in ('train', 'validation', 'test'):
        examples = temporal_examples(users, config['data']['history_items'], name)
        if case_digest(examples) != split['case_sha256'][name]:
            raise ValueError('temporal case schedule differs from prepared input')
    scratch = Path(config['resources']['scratch_path'])
    original = unpack_grid(resources / 'grid-source.bin', scratch / 'original')
    source = prepare_grid_source(original, scratch / 'runtime')
    torch.set_num_threads(config['execution']['cpu_threads_per_worker'])
    dummy = np.zeros((len(source_ids), 4), dtype=np.int64)
    dummy[:, 0] = np.arange(len(source_ids)) // 256; dummy[:, 1] = np.arange(len(source_ids)) % 256
    expected_by_seed = {}
    for seed in config.get('replicate_seeds', [config['seed']]):
        model = build_grid_generator(dummy, config['generator'], seed, 'cpu', source)
        observed = state_digest(dict(model.named_parameters())); del model; gc.collect()
        expected = config.get('expected_initial_state_sha256_by_seed', {}).get(
            str(seed), config['expected_initial_state_sha256'])
        if observed != expected:
            raise ValueError(f'generator initialization differs for pipeline seed {seed}')
        expected_by_seed[str(seed)] = observed
    if config.get('kind') in {'gr_movie_multimodal_ae', 'gr_movie_multimodal_ae_ladder'}:
        if file_sha256(modality_root / 'features.npz') != config['modality_input']['features_sha256']:
            raise ValueError('frozen modality vectors differ from completed direct-training control')
        if json.loads((modality_root / 'source-ids.json').read_text()) != source_ids:
            raise ValueError('frozen modality catalogue order differs')
    elif modality_root is not None:
        from rhpr.gr.movie_modal_features import prepare_shared_modalities
        modality_root = prepare_shared_modalities(config, modality_root, resources,
            output / 'shared-modalities', 'cuda:0' if config['training']['device'] == 'cuda' else 'cpu')
    (output / 'workers').mkdir(exist_ok=True)
    _write(output / 'split-report.json', split)
    _write(output / 'input-provenance.json', {'source_ids': source_ids, 'split': split,
        'case_order': 'shared strict-past target cases; pipeline-seed+epoch permutation, matched across stages within each seed',
        'config': config})
    plan = {'config': config, 'output': str(output), 'resources': str(resources), 'data_root': str(data_root),
        'source': str(source), 'original': str(original), 'source_ids': source_ids,
        'initial_state_sha256': expected_by_seed[str(config['seed'])],
        'initial_state_sha256_by_seed': expected_by_seed, 'assignments': jobs}
    if modality_root is not None:
        plan['modality_root'] = str(modality_root)
    path = output / 'workers' / 'plan.json'; _write(path, plan)
    summary = launch_workers(path, worker_module='rhpr.gr.movie_suite', event_prefix='movie_suite')
    metrics = {'experiment': config['name'], 'seed': config['seed'], 'trials': len(summary['results']),
               'items': len(source_ids), 'users': len(users), 'epochs_per_trial': 20,
               'pipeline_seed_count': len(expected_by_seed)}
    for name, result in summary['results'].items():
        metrics.update({f'{name}_test_{k.replace("@", "")}': v for k, v in result['test'].items()})
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); p.add_argument('--slot', type=int, required=True)
    a = p.parse_args(); run_worker(a.plan, a.slot)
