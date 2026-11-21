"""Train on the same eight ML1M targets with shared encoder memory or separate prefixes."""
from __future__ import annotations

import argparse
from functools import partial
import gc
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import traceback

import numpy as np

from rhpr.gr.experiment import _evaluate, _event, _loss, _model_batch, _train_generator, _write
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

HISTORY_ITEMS = 128
TARGETS_PER_USER = 8
SID_LEVELS = 4


def load_cases(parent, config):
    splits = {}
    for split, expected in config['expected_raw_case_sha256'].items():
        with np.load(parent / 'prepared-cases' / f'{split}-cases.npz', allow_pickle=False) as saved:
            cases = tuple(saved[key] for key in ('contexts', 'user_ids', 'targets'))
        if case_digest(cases) != expected:
            raise ValueError(f'Frozen ML1M cases changed: {split}')
        splits[split] = cases
    return splits


def group_training_cases(cases):
    """Take one common historical prefix and verify each successive training anchor."""
    contexts, users, targets = cases
    order = np.argsort(users, kind='stable')
    grouped = users[order].reshape(-1, TARGETS_PER_USER)
    targets = targets[order].reshape(-1, TARGETS_PER_USER)
    users = grouped[:, 0]
    if len(np.unique(users)) != len(users) or not (grouped == users[:, None]).all():
        raise ValueError('Expected eight chronological targets per user')
    prefix_width = HISTORY_ITEMS - TARGETS_PER_USER + 1
    prefix = contexts[order[::TARGETS_PER_USER], -prefix_width:]
    for step in range(TARGETS_PER_USER):
        expected = np.concatenate((prefix, targets[:, :step]), axis=1)
        np.testing.assert_array_equal(contexts[order[step::TARGETS_PER_USER], -expected.shape[1]:], expected)
    return prefix, users, targets


class WindowInput:
    """Materialize the per-target control or the shared-encoder training batch."""

    def __init__(self, sids, window):
        self.catalogue = SidIndex.from_full_sids(sids, width=256)
        self.sids = self.catalogue.sids
        self.window = window

    def evaluate(self, predictions, targets, contexts, *, cutoffs):
        return evaluate_ranked_sids(self.catalogue, predictions, targets, cutoffs=cutoffs)

    def model_batch(self, contexts, users, targets, device):
        import torch

        if targets.ndim == 1:
            return _model_batch(contexts, users, targets, self.sids, device)
        if self.window is None:
            prefixes = np.full((len(users), TARGETS_PER_USER, HISTORY_ITEMS), -1, dtype=contexts.dtype)
            for step in range(TARGETS_PER_USER):
                observed = np.concatenate((contexts, targets[:, :step]), axis=1)
                prefixes[:, step, -observed.shape[1]:] = observed
            return _model_batch(prefixes.reshape(-1, HISTORY_ITEMS), np.repeat(users, TARGETS_PER_USER),
                                targets.reshape(-1), self.sids, device)
        recent_items = self.window - TARGETS_PER_USER
        old, recent = contexts[:, :-recent_items], contexts[:, -recent_items:]
        decoder_items = np.concatenate((recent, targets), axis=1)
        decoder_sids = self.sids[np.maximum(decoder_items, 0)]
        decoder_sids[decoder_items < 0] = -1
        tokens = torch.as_tensor(decoder_sids.reshape(len(users), -1), dtype=torch.long, device=device)
        batch = _model_batch(old, users, targets[:, 0], self.sids, device)
        batch['future_ids'] = tokens[:, :-1]
        batch['attention_mask_decoder'] = (tokens[:, :-1] >= 0).long()
        batch['labels'] = tokens[:, -TARGETS_PER_USER * SID_LEVELS:].reshape(-1, SID_LEVELS)
        return batch


def window_loss(model, batch):
    import torch.nn.functional as F

    labels = batch['labels']
    hidden = model(**{key: value for key, value in batch.items() if key != 'labels'})
    target_states = hidden[:, -TARGETS_PER_USER * SID_LEVELS:].reshape(-1, SID_LEVELS, hidden.shape[-1])
    return sum(F.cross_entropy(head(target_states[:, level]), labels[:, level])
               for level, head in enumerate(model.decoder.decoder_mlp))


def generate_next(model, attention_mask, input_ids, user_id=None, *, window):
    """Use GRID's same four-step beam with a cached, observed decoder prefix."""
    import torch
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    recent_codes = SID_LEVELS * (window - 1)
    memory, memory_mask = model.encoder_forward_pass(
        attention_mask[:, :-recent_codes], input_ids[:, :-recent_codes], user_id)
    recent_ids = input_ids[:, -recent_codes:]
    recent_mask = attention_mask[:, -recent_codes:]
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    hidden, cache = model.decoder_forward_pass(
        attention_mask=recent_mask, future_ids=recent_ids, encoder_output=memory,
        attention_mask_for_encoder=memory_mask, use_cache=True, past_key_values=cache)

    batch_size = len(input_ids)
    beam = model.top_k_for_generation
    ids, scores, _ = model._beam_search_one_step(
        model.decoder.decoder_mlp[0](hidden[:, -1]), None, None, None, 0, batch_size)
    # GRID creates a fresh cache on the first beam expansion. Here all beams
    # extend the same already encoded decoder prefix.
    cache.reorder_cache(torch.arange(batch_size, device=input_ids.device).repeat_interleave(beam))
    memory = memory.repeat_interleave(beam, dim=0)
    memory_mask = memory_mask.repeat_interleave(beam, dim=0)
    mask = torch.cat((torch.ones_like(recent_mask[:, :1]), recent_mask), dim=1).repeat_interleave(beam, dim=0)
    embedding = model.get_embedding_table('decoder')
    for level in range(1, SID_LEVELS):
        last_code = ids[:, :, -1].reshape(-1, 1) + (level - 1) * model.num_embeddings_per_hierarchy
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        hidden, cache = model.decoder(
            attention_mask=mask, sequence_embedding=embedding(last_code),
            encoder_output=memory, encoder_attention_mask=memory_mask,
            use_cache=True, past_key_values=cache)
        ids, scores, cache = model._beam_search_one_step(
            model.decoder.decoder_mlp[level](hidden[:, -1]), ids, scores, cache, level, batch_size)
    return ids, scores


def attach_window_generation(model, window):
    model.generate = partial(generate_next, model, window=window)


def run_trial(trial, plan):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid_study import state_digest

    config = plan['config']
    trial_config = {**config, 'active_trial': trial,
                    'training': {**config['training'], 'physical_batch_size': trial['physical_batch_size']}}
    window = trial['decoder_window']
    output = Path(plan['output']) / 'trials' / trial['id']
    output.mkdir(parents=True, exist_ok=True)
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parent = Path(plan['parent'])
    raw = load_cases(parent, config)
    train = group_training_cases(raw['train'])
    validation, test = ((raw[split][0][:, -HISTORY_ITEMS:], *raw[split][1:])
                        for split in ('validation', 'test'))
    if (len(train[1]), len(validation[1]), len(test[1]), train[2].size) != (6040, 6040, 6040, 48320):
        raise ValueError('Incorrect Last-8 target counts')

    sids = np.load(parent / 'frozen-indexes' / 'items-42.npy', allow_pickle=False)
    index = WindowInput(sids, window)
    model = build_grid_generator(sids, config['generator'], seed, 'cpu', plan['source'])
    initial = state_digest(dict(model.named_parameters()))
    if initial != config['expected_initial_state_sha256']:
        raise ValueError('GRID initialization differs from the frozen initializer')
    if window:
        attach_window_generation(model, window)
    loss_fn = _loss if window is None else window_loss
    cpu_batch = index.model_batch(*(part[:1] for part in train), 'cpu')
    model.eval()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
            loss = loss_fn(model, cpu_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite CPU forward')
    flops_per_user = sum(entry.flops for entry in profile.key_averages())
    forward = dict(encoder_calls_per_user=8 if window is None else 1,
                   encoder_positions=cpu_batch['input_ids'].shape[1] + 1,
                   decoder_positions=5 if window is None else SID_LEVELS * window,
                   supported_flops_per_user=flops_per_user,
                   supported_flops_per_target=flops_per_user / TARGETS_PER_USER,
                   scope='padded teacher-forced CPU math SDPA with loss, FMA2, no backward or beam')
    _write(output / 'forward-flops.json', forward)
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    examples = {'train': train, 'validation': validation}

    def build_examples(_root, _width, split):
        return examples[split]

    selection = _train_generator(model, parent, index, trial_config, output, 'cuda:0',
                                 example_builder=build_examples, loss_fn=window_loss if window else None)
    if selection['training_epochs'] != config['training']['max_epochs']:
        raise RuntimeError('Pilot hit its time limit; selected and recent checkpoints were saved')
    if selection['optimizer_updates'] + selection['amp_skipped_updates'] != 95 * selection['training_epochs']:
        raise ValueError('Incorrect number of target batches')
    curves = json.loads((output / 'training-curves.json').read_text())
    timing = dict(first_epoch_seconds=curves[0]['train_seconds'],
                  warm_epoch_median_seconds=median(row['train_seconds'] for row in curves[1:]),
                  train_wall_seconds=selection['training_seconds'],
                  peak_training_allocated_bytes=torch.cuda.max_memory_allocated())
    metrics, predictions, scores = _evaluate(model, test, index, config['training']['evaluation_batch_size'],
                                              'cuda:0', precision='fp32')
    np.savez_compressed(output / 'test-best-predictions.npz', semantic_ids=predictions, scores=scores,
                        user_ids=test[1], target_item_ids=test[2])
    test_report = dict(metrics=metrics, epoch=selection['best_epoch'],
                       checkpoint_sha256=file_sha256(output / 'generator-best.pt'))
    _write(output / 'test-metrics.json', test_report)
    result = dict(trial=trial['id'], seed=seed, window=window, initial_state_sha256=initial,
                  selection=selection, forward=forward, timing=timing, test=test_report)
    _write(output / 'trial-result.json', result)
    return result


def worker(plan_path, slot):
    import torch
    import transformers
    from rhpr.gr.ml1m_devices import check_worker

    plan = json.loads(plan_path.read_text())
    check_worker(plan['config'], transformers.__version__)
    torch.set_num_threads(1)
    status = dict(results={}, failures=[])
    for trial in plan['assignments'][slot]['trials']:
        name = trial['id']
        _event('window_tiger_trial_start', trial=name)
        try:
            result = run_trial(trial, plan)
            status['results'][name] = result
            _event('window_tiger_trial_complete', **result)
        except Exception as error:
            output = Path(plan['output']) / 'trials' / name
            output.mkdir(parents=True, exist_ok=True)
            (output / 'failure.log').write_text(traceback.format_exc())
            failure = dict(trial=name, exception_type=type(error).__name__, message=str(error))
            status['failures'].append(failure)
            _event('window_tiger_trial_failed', **failure)
        _write(Path(plan['output']) / 'workers' / f'worker-{slot}.json', status)
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def summarize(results, output):
    control = results['prefix_control']
    threshold = control['selection']['validation_ndcg10']
    metrics = {'models': len(results), 'seed': control['seed'], 'control_validation_threshold': threshold}
    comparison = {}
    for name, row in results.items():
        curves = json.loads((output / 'trials' / name / 'training-curves.json').read_text())
        first_match = next((epoch for epoch in curves if epoch.get('validation', {}).get('ndcg@10', -1)
                            >= threshold), None)
        relative = dict(
            warm_epoch_speedup=control['timing']['warm_epoch_median_seconds'] / row['timing']['warm_epoch_median_seconds'],
            validation_ndcg10=row['selection']['validation_ndcg10'],
            test_ndcg10=row['test']['metrics']['ndcg@10'],
            forward_flops_per_target=row['forward']['supported_flops_per_target'],
            reached_control_validation=first_match is not None)
        if first_match is not None:
            relative['wall_seconds_to_control_validation'] = first_match['elapsed_seconds']
        comparison[name] = relative
        metrics.update({f'{name}_{key}': value for key, value in relative.items()})
    _write(output / 'study-summary.json', dict(results=results, comparison=comparison))
    return metrics


def run(config, output):
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ml1m_devices import assignments_for, devices_for
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    if (config['data']['history_items'], config['targets_per_user'], config['training']['targets_per_row'],
            config['training']['batch_size']) != (HISTORY_ITEMS, TARGETS_PER_USER, TARGETS_PER_USER, 64):
        raise ValueError('The pilot requires 128 history items and 512 targets per optimizer update')
    if {trial['id']: trial['decoder_window'] for trial in config['trials']} != {
            'prefix_control': None, 'window16': 16, 'window32': 32}:
        raise ValueError('Expected the common-prefix control and windows 16 and 32')
    devices = devices_for(config)
    parent = locate_parent(config['frozen_parent'])
    roots = {name: resolve_dataset_directory(config[name]['path']) for name in ('resources', 'modern_runtime')}
    for name, root in roots.items():
        verify_resources(root, config[name]['manifest_sha256'])
    scratch = Path(config['resources']['scratch_path'])
    source = prepare_grid_source(unpack_grid(roots['resources'] / 'grid-source.bin', scratch / 'original'),
                                 scratch / 'grid')
    (output / 'workers').mkdir(parents=True, exist_ok=True)
    overlay = scratch / 'modern-site'
    with (output / 'modern-install.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
                        '--no-compile', '--require-hashes', '--only-binary=:all:', '--target', str(overlay),
                        '--find-links', str(roots['modern_runtime']), '-r', str(roots['modern_runtime'] / 'requirements-lock.txt')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    plan = output / 'plan.json'
    _write(plan, dict(config=config, parent=str(parent), source=str(source), output=str(output),
                      assignments=assignments_for(config, devices)))
    summary = launch_workers(plan, worker_module='rhpr.gr.ml1m_window_tiger', event_prefix='window_tiger')
    return summarize(summary['results'], output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--slot', type=int, required=True)
    arguments = parser.parse_args()
    worker(arguments.plan, arguments.slot)
