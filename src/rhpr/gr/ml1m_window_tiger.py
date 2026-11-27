"""Train on the same eight ML1M targets with shared encoder memory or separate prefixes."""
from __future__ import annotations

import argparse
from functools import partial
import gc
import json
from pathlib import Path
import random
import shutil
from statistics import mean, median, stdev
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
            raise ValueError(f'Frozen cases changed: {split}')
        splits[split] = cases
    return splits


def group_training_cases(cases, targets_per_user=TARGETS_PER_USER):
    """Take one causal prefix per complete block of eight consecutive targets."""
    contexts, users, targets = cases
    order = np.argsort(users, kind='stable')
    if targets_per_user < TARGETS_PER_USER or targets_per_user % TARGETS_PER_USER:
        raise ValueError('Expected complete blocks of eight targets per user')
    grouped = users[order].reshape(-1, targets_per_user)
    targets = targets[order].reshape(-1, TARGETS_PER_USER)
    users = grouped[:, 0]
    if len(np.unique(users)) != len(users) or not (grouped == users[:, None]).all():
        raise ValueError('Incorrect number of chronological targets per user')
    prefix_width = HISTORY_ITEMS - TARGETS_PER_USER + 1
    prefix = contexts[order[::TARGETS_PER_USER], -prefix_width:]
    for step in range(TARGETS_PER_USER):
        expected = np.concatenate((prefix, targets[:, :step]), axis=1)
        np.testing.assert_array_equal(contexts[order[step::TARGETS_PER_USER], -expected.shape[1]:], expected)
    return prefix, np.repeat(users, targets_per_user // TARGETS_PER_USER), targets


def build_last16_training(parent, raw, expected_sha256):
    from rhpr.gr.ml1m_last8 import build_last8_cases

    with np.load(parent / 'prepared-cases' / 'user-histories.npz', allow_pickle=False) as saved:
        histories = dict(zip(saved['user_ids'], np.split(saved['items'], saved['offsets'][1:-1])))
    expanded = build_last8_cases(histories, width=raw['train'][0].shape[1], targets_per_user=16)
    for split in ('validation', 'test'):
        if case_digest(expanded[split][:3]) != case_digest(raw[split]):
            raise ValueError(f'Last-16 changed the frozen {split} cases')
    later = np.arange(len(histories) * 16).reshape(-1, 16)[:, TARGETS_PER_USER:].ravel()
    for original, extended in zip(raw['train'], expanded['train'][:3]):
        np.testing.assert_array_equal(extended[later], original)
    train = group_training_cases(expanded['train'][:3], targets_per_user=16)
    if case_digest(train) != expected_sha256:
        raise ValueError('Last-16 train block schedule differs from preflight')
    return train


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
        block_size = targets.shape[1]
        old_width = contexts.shape[1] - (self.window - block_size)
        old, recent = contexts[:, :old_width], contexts[:, old_width:]
        decoder_items = np.concatenate((recent, targets), axis=1)
        decoder_sids = self.sids[np.maximum(decoder_items, 0)]
        decoder_sids[decoder_items < 0] = -1
        tokens = torch.as_tensor(decoder_sids.reshape(len(users), -1), dtype=torch.long, device=device)
        batch = _model_batch(old, users, targets[:, 0], self.sids, device)
        batch['future_ids'] = tokens[:, :-1]
        batch['attention_mask_decoder'] = (tokens[:, :-1] >= 0).long()
        batch['labels'] = tokens[:, -block_size * SID_LEVELS:].reshape(-1, SID_LEVELS)
        return batch


def window_loss(model, batch):
    hidden = model(**{key: value for key, value in batch.items() if key != 'labels'})
    return window_target_loss(model, hidden, batch['labels'])


def window_target_loss(model, hidden, labels):
    import torch.nn.functional as F

    target_codes_per_row = labels.numel() // hidden.shape[0]
    target_states = hidden[:, -target_codes_per_row:].reshape(-1, SID_LEVELS, hidden.shape[-1])
    return sum(F.cross_entropy(head(target_states[:, level]), labels[:, level])
               for level, head in enumerate(model.decoder.decoder_mlp))


def generate_next(model, attention_mask, input_ids, user_id=None, *, window):
    """Use GRID's same four-step beam with a cached, observed decoder prefix."""
    import torch
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache

    recent_codes = SID_LEVELS * (window - 1)
    split_at = max(0, input_ids.shape[1] - recent_codes)
    memory, memory_mask = model.encoder_forward_pass(
        attention_mask[:, :split_at], input_ids[:, :split_at], user_id)
    recent_ids = input_ids[:, split_at:]
    recent_mask = attention_mask[:, split_at:]
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    hidden, cache = model.decoder_forward_pass(
        attention_mask=recent_mask if recent_codes else None, future_ids=recent_ids if recent_codes else None,
        encoder_output=memory,
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
    mask = torch.cat((recent_mask.new_ones((batch_size, 1)), recent_mask), dim=1).repeat_interleave(beam, dim=0)
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


def evaluate_window_tails(model, index, splits, output, batch_size, device):
    """Keep positions fixed and remove the oldest observed events."""
    results = {}
    for split, (contexts, users, targets) in splits.items():
        results[split] = {}
        for tail in (HISTORY_ITEMS, 32, 16, 8):
            view = contexts[:, -HISTORY_ITEMS:].copy()
            removed = int(np.any(view[:, :-tail] >= 0, axis=1).sum())
            view[:, :-tail] = -1
            metrics, predictions, scores = _evaluate(
                model, (view, users, targets), index, batch_size, device, precision='fp32')
            np.savez_compressed(output / f'{split}-recent{tail}-predictions.npz',
                                user_ids=users, target_item_ids=targets,
                                semantic_ids=predictions, scores=scores)
            results[split][str(tail)] = dict(**metrics, users_with_history_removed=removed)
            _event('window_tail_evaluated', trial=output.name, split=split, tail=tail,
                   ndcg10=metrics['ndcg@10'], users_with_history_removed=removed)
    _write(output / 'tail-evaluation.json', results)
    return results


def run_saved_window_tails(trial, plan):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator

    config, seed, window = plan['config'], trial['seed'], trial['decoder_window']
    parent = Path(plan['parent'])
    sids = np.load(parent / 'frozen-indexes' / 'items-42.npy', allow_pickle=False)
    raw = load_cases(parent, config)
    splits = {split: raw[split] for split in ('validation', 'test')}
    old = Path(plan['window_tail_parent']) / 'trials' / f'window{window}'
    report = json.loads((old / 'trial-result.json').read_text())
    path = old / 'generator-best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    digest = file_sha256(path)
    if (report['seed'] != seed or report['window'] != window
            or checkpoint['epoch'] != report['selection']['best_epoch']
            or digest != report['test']['checkpoint_sha256']):
        raise ValueError('The saved Window checkpoint differs from the selected model')
    model = build_grid_generator(sids, config['generator'], seed, 'cpu', plan['source'])
    model.load_state_dict(checkpoint['model'], strict=True)
    attach_window_generation(model, window)
    model.to('cuda:0')
    output = Path(plan['output']) / 'trials' / trial['id']
    output.mkdir(parents=True, exist_ok=True)
    views = evaluate_window_tails(model, WindowInput(sids, window), splits, output,
                                  config['training']['evaluation_batch_size'], 'cuda:0')
    return dict(trial=trial['id'], seed=seed, window=window, checkpoint_sha256=digest,
                checkpoint_epoch=checkpoint['epoch'], tail_views=views,
                original_ndcg10=dict(validation=report['selection']['validation_ndcg10'],
                                     test=report['test']['metrics']['ndcg@10']))


def run_trial(trial, plan):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid_study import state_digest

    if trial.get('predictor') == 'window_checkpoint':
        return run_saved_window_tails(trial, plan)
    if trial.get('predictor') == 'frozen_ntp':
        from rhpr.gr.yambda_ntp import run_ntp_trial

        return run_ntp_trial(trial, plan)
    if trial.get('predictor') == 'd4_mask_finetune':
        from rhpr.gr.yambda_masked_decoder import run_trial as run_mask_finetune

        return run_mask_finetune(trial, plan)
    config = plan['config']
    seed = trial.get('seed', config['seed'])
    sasrec = trial.get('predictor') == 'sasrec'
    full_decoder = trial.get('predictor') == 'full_decoder'
    item_decoder = trial.get('predictor') == 'item_decoder'
    causal_decoder = full_decoder or item_decoder
    trial_config = {**config, 'active_trial': trial, 'seed': seed,
                    'training': {**config['training'], 'physical_batch_size': trial['physical_batch_size']}}
    block_size = trial.get('targets_per_block', TARGETS_PER_USER)
    trial_config['training'].update(
        targets_per_row=block_size,
        batch_size=config['training']['batch_size'] * TARGETS_PER_USER // block_size,
        rows_per_shuffle_group=TARGETS_PER_USER // block_size)
    window = trial['decoder_window']
    output = Path(plan['output']) / 'trials' / trial['id']
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parent = Path(plan['parent'])
    raw = load_cases(parent, config)
    targets_per_user = config['targets_per_user']
    last16 = targets_per_user == 16
    if config['kind'] == 'gr_yambda_window_tiger':
        train = raw['train']
        if block_size != TARGETS_PER_USER:
            from rhpr.gr.yambda_window_tiger import split_training_blocks

            train = split_training_blocks(train, block_size)
            if case_digest(train) != config['expected_reblocked_train_sha256']:
                raise ValueError('Reblocked Yambda train cases differ from preflight')
    else:
        train = (build_last16_training(parent, raw, config['expected_last16_grouped_sha256'])
                 if last16 else group_training_cases(raw['train']))
    history_items = config['data']['history_items']
    validation, test = ((raw[split][0][:, -history_items:], *raw[split][1:])
                        for split in ('validation', 'test'))
    users = config['data']['users']
    if (len(train[1]), len(validation[1]), len(test[1]), train[2].size) != (
            users * targets_per_user // block_size, users, users, users * targets_per_user):
        raise ValueError('Incorrect target and block counts')

    sids = np.load(parent / 'frozen-indexes' / 'items-42.npy', allow_pickle=False)
    if sasrec:
        from rhpr.gr.movie_sasrec import Last8ItemIDIndex, build_movie_sasrec, grouped_sasrec_loss

        trial_config['training'].update(config['sasrec_training_overrides'])
        index = Last8ItemIDIndex(len(sids))
        model = build_movie_sasrec(config, seed, 'cpu', grouped=True)
        loss_fn = grouped_sasrec_loss
    elif causal_decoder:
        from rhpr.gr.ml1m_full_decoder import build_full_decoder, full_decoder_loss

        index = SidIndex.from_full_sids(sids, width=256)
        trial_config['generator'] = {**config['generator'], **trial.get('generator_overrides', {})}
        if item_decoder:
            from rhpr.gr.yambda_item_decoder import build_item_decoder, item_decoder_loss

            model = build_item_decoder(sids, trial_config['generator'], seed, plan['source'], trial['history_mode'])
            loss_fn = item_decoder_loss
        else:
            model = build_full_decoder(sids, trial_config['generator'], seed, plan['source'])
            loss_fn = full_decoder_loss
    else:
        index = WindowInput(sids, window)
        model = build_grid_generator(sids, config['generator'], seed, 'cpu', plan['source'])
        loss_fn = _loss if window is None else window_loss
    initial = state_digest(dict(model.named_parameters()))
    if not (sasrec or causal_decoder):
        pinned = config.get('window_initializers', {}).get(str(seed), config['expected_initial_state_sha256'])
        if initial != pinned:
            raise ValueError('GRID initialization differs from the frozen initializer')
    if window:
        attach_window_generation(model, window)
    cpu_batch = (_model_batch(*(part[:1] for part in train), sids, 'cpu') if causal_decoder else
                 index.model_batch(*(part[:1] for part in train), 'cpu'))
    # Training mode disables the fused SASRec inference path in the CPU FLOP trace.
    model.train(sasrec)
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        profile_loss = loss_fn
        profile_batch = cpu_batch
        if config.get('frozen_encoder_pilot'):
            from rhpr.gr.yambda_frozen_encoder import cached_window_loss, decoder_batch

            memory, mask = model.encoder_forward_pass(cpu_batch['attention_mask_encoder'],
                cpu_batch['input_ids'], cpu_batch['user_id'])
            profile_loss = cached_window_loss
            profile_batch = decoder_batch(cpu_batch, memory, mask)
        with torch.random.fork_rng(devices=[]), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
            loss = profile_loss(model, profile_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite CPU forward')
    flops_per_user = sum(entry.flops for entry in profile.key_averages())
    forward = dict(encoder_calls_per_user=1 if sasrec or window else 8,
                   encoder_positions=cpu_batch['item_ids' if sasrec else 'input_ids'].shape[1] + (not sasrec),
                   decoder_positions=0 if sasrec else 5 if window is None else SID_LEVELS * window,
                   supported_flops_per_user=flops_per_user,
                   supported_flops_per_target=flops_per_user / block_size,
                   scope='padded teacher-forced CPU math SDPA with loss, FMA2, no backward or beam')
    if 'targets_per_block' in trial:
        blocks_per_user = targets_per_user // block_size
        forward.update(targets_per_block=block_size, encoder_calls_per_user=blocks_per_user,
                       supported_flops_per_block=flops_per_user,
                       supported_flops_per_user=flops_per_user * blocks_per_user)
    if config.get('frozen_encoder_pilot'):
        forward.update(encoder_calls_per_user=0, cached_encoder_positions=forward['encoder_positions'],
                       encoder_positions=0, scope=forward['scope'] + '; decoder on cached memory')
    if causal_decoder:
        from rhpr.gr.movie_sasrec import parameter_counts

        loss_fn(model, cpu_batch).backward()
        counts = parameter_counts(model)
        expected = trial.get('expected_active_nonembedding_parameters', 1967360)
        if counts['inactive_nonembedding_names'] or (full_decoder and counts['active_nonembedding_parameters'] != expected):
            raise ValueError('Unexpected active parameter count for the full-history decoder')
        model.zero_grad(set_to_none=True)
        forward.update(encoder_calls_per_user=0, encoder_positions=0,
                       decoder_positions=cpu_batch['input_ids'].shape[1] + SID_LEVELS * TARGETS_PER_USER + 1,
                       parameters=counts)
        if item_decoder:
            forward.update(decoder_positions=cpu_batch['input_ids'].shape[1] // SID_LEVELS + block_size,
                           history_mode=trial['history_mode'],
                           history_passes=SID_LEVELS if trial['history_mode'] == 'residual' else 1)
    _write(output / 'forward-flops.json', forward)
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    examples = {'train': train, 'validation': validation}

    def build_examples(_root, _width, split):
        return examples[split]

    resume_from = plan.get('resume_trials', {}).get(trial['id'])
    train_batch_builder = None
    if config.get('frozen_encoder_pilot'):
        from rhpr.gr.yambda_frozen_encoder import cache_training_memory, cached_window_loss, freeze_encoder

        checkpoint = torch.load(Path(resume_from) / 'generator-last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        frozen_digest = freeze_encoder(model)
        train_batch_builder, cache_report = cache_training_memory(
            model, index, train, 'cuda:0', trial_config['training'])
        _write(output / 'encoder-cache.json', cache_report)
        loss_fn = cached_window_loss
    selection = _train_generator(model, parent, index, trial_config, output, 'cuda:0',
                                 example_builder=build_examples, resume_from=resume_from,
                                 loss_fn=loss_fn if sasrec or window or causal_decoder else None,
                                 train_batch_builder=train_batch_builder)
    if train_batch_builder is not None:
        from rhpr.gr.yambda_frozen_encoder import frozen_parameters_digest

        if frozen_parameters_digest(model) != frozen_digest:
            raise ValueError('Cached encoder inputs or weights changed during training')
        del train_batch_builder
    stopped_at_plateau = config.get('train_to_plateau') and selection['stopping_reason'] == 'early_stopping'
    if selection['training_epochs'] != config['training']['max_epochs'] and not stopped_at_plateau:
        raise RuntimeError('Pilot hit its time limit; selected and recent checkpoints were saved')
    batch_size = trial_config['training']['batch_size']
    updates_per_epoch = (len(train[1]) + batch_size - 1) // batch_size
    if selection['optimizer_updates'] + selection['amp_skipped_updates'] != updates_per_epoch * selection['training_epochs']:
        raise ValueError('Incorrect number of target batches')
    curves = json.loads((output / 'training-curves.json').read_text())
    timing = dict(first_epoch_seconds=curves[0]['train_seconds'],
                  warm_epoch_median_seconds=median(row['train_seconds'] for row in curves[1:]),
                  train_wall_seconds=selection['training_seconds'],
                  peak_training_allocated_bytes=torch.cuda.max_memory_allocated())
    if resume_from is not None:
        continued = curves[selection['resumed_from_epoch']:]
        timing['continuation_warm_epoch_median_seconds'] = median(r['train_seconds'] for r in continued[1:])
    metrics, predictions, scores = _evaluate(model, test, index, config['training']['evaluation_batch_size'],
                                              'cuda:0', precision='fp32')
    prediction_key = 'item_ids' if sasrec else 'semantic_ids'
    np.savez_compressed(output / 'test-best-predictions.npz', scores=scores,
                        user_ids=test[1], target_item_ids=test[2], **{prediction_key: predictions})
    test_report = dict(metrics=metrics, epoch=selection['best_epoch'],
                       checkpoint_sha256=file_sha256(output / 'generator-best.pt'))
    _write(output / 'test-metrics.json', test_report)
    result = dict(trial=trial['id'], seed=seed, window=window, initial_state_sha256=initial,
                  selection=selection, forward=forward, timing=timing, test=test_report)
    if config.get('frozen_encoder_pilot'):
        result['encoder_cache'] = {**cache_report, 'frozen_parameters_sha256': frozen_digest}
    if 'window_tail_parent' in config:
        result['tail_views'] = evaluate_window_tails(
            model, index, dict(validation=validation, test=test), output,
            config['training']['evaluation_batch_size'], 'cuda:0')
    elif last16:
        result['initialization_group'] = trial['id']
    elif config.get('decoder_depth_replication'):
        result['initialization_group'] = f"full_decoder{trial_config['generator']['decoder_layers']}_{seed}"
    elif 'pilot_parent' in config:
        result['initialization_group'] = f"{trial['predictor']}_{seed}"
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


def validate_replication(config):
    predictors = {'window16': 16, 'window32': 32, 'sasrec': None}
    fresh = {(trial['predictor'], trial['seed'], trial['decoder_window']) for trial in config['trials']}
    expected = {(name, seed, window) for name, window in predictors.items()
                for seed in (42, 101, 202) if seed != 101 or name == 'sasrec'}
    if (fresh != expected or len(config['trials']) != len(expected)
            or any(trial['id'] != f"{trial['predictor']}_{trial['seed']}" for trial in config['trials'])):
        raise ValueError('Expected fresh SASRec on three seeds and Window TIGER replication on two')


def validate_sasrec_recovery(config):
    expected = [dict(id=f'sasrec_{seed}', predictor='sasrec', seed=seed,
                     decoder_window=None, physical_batch_size=64) for seed in (101, 42, 202)]
    training = {**config['training'], **config['sasrec_training_overrides']}
    if (config['trials'] != expected or 'pilot_parent' in config or training.get('precision') != 'fp32'
            or training.get('compile_training') or not config.get('compare_initialization_by_seed')):
        raise ValueError('Expected only the three missing SASRec fits in eager FP32')


def validate_wide_decoder_replication(config):
    expected = [
        dict(id=f'full_decoder_wide_{seed}', predictor='full_decoder', seed=seed,
             decoder_window=None, physical_batch_size=16,
             generator_overrides=dict(d_model=256, d_ff=1344),
             expected_active_nonembedding_parameters=4590080)
        for seed in (101, 42, 202)
    ]
    if (config['trials'] != expected or not config.get('compare_initialization_by_seed')
            or any(key in config for key in ('pilot_parent', 'decoder_pilot_parent'))):
        raise ValueError('Expected only the three wide full-history decoder fits')


def validate_decoder_depth_replication(config):
    expected = [
        dict(id=f'{name}_{seed}', predictor='full_decoder', seed=seed,
             decoder_window=None, physical_batch_size=16, **architecture)
        for name, architecture in (
            ('full_decoder', dict(expected_active_nonembedding_parameters=1967360)),
            ('full_decoder8', dict(generator_overrides=dict(decoder_layers=8, d_ff=1408),
                                   expected_active_nonembedding_parameters=4589824)),
        )
        for seed in (42, 202)
    ]
    if (config['trials'] != expected or not config.get('compare_initialization_by_shape')
            or any(key in config for key in ('pilot_parent', 'decoder_pilot_parent'))):
        raise ValueError('Expected only the original D4 and D8 on the two missing seeds')


def validate_window_tail_pilot(config, parent):
    expected = [dict(id='window8_101', predictor='window8', seed=101,
                     decoder_window=8, physical_batch_size=16)]
    expected += [dict(id=f'window{window}_tails_101', predictor='window_checkpoint',
                      seed=101, decoder_window=window) for window in (16, 32)]
    reference = json.loads((parent / 'run-manifest.json').read_text())['config']
    common = ('seed', 'generator', 'training', 'data', 'evaluation', 'expected_raw_case_sha256',
              'frozen_parent', 'expected_initial_state_sha256')
    if config['trials'] != expected or any(config[key] != reference[key] for key in common):
        raise ValueError('Window8 and the tail evaluation must use the original Window101 protocol')


def validate_last16_replication(config):
    expected = {
        f'{name}_{seed}': dict(id=f'{name}_{seed}', predictor=name, seed=seed,
                               decoder_window=window, physical_batch_size=16)
        for name, window in (('full_decoder', None), ('window16', 16)) for seed in (42, 101, 202)
    }
    actual = {trial['id']: trial for trial in config['trials']}
    training = config['training']
    if (actual != expected or len(config['trials']) != len(expected)
            or config['data']['train_examples'] != 96640
            or training['sampling'] != 'fixed_last16_train_targets_after_leave_last_two'
            or training['scheduler'] != dict(kind='warmup_cosine', warmup_epochs=2, total_epochs=60,
                                             min_lr=.001, reference_steps_per_epoch=95)
            or (training['min_epochs'], training['max_epochs'], training['validation_interval']) != (40, 40, 5)
            or not config.get('compare_initialization_by_shape')
            or any(key in config for key in ('pilot_parent', 'decoder_pilot_parent', 'window_tail_parent'))):
        raise ValueError('Expected the original D4 and Window16 with a three-seed Last-16 schedule')


def import_pilot(config, parent, output):
    manifest = json.loads((parent / 'run-manifest.json').read_text())
    pilot = manifest['config']
    common = ('generator', 'data', 'evaluation', 'expected_raw_case_sha256', 'frozen_parent')
    if (any(config[key] != pilot[key] for key in common) or pilot['seed'] != 101
            or pilot['training'] != config['training']
            or pilot['expected_initial_state_sha256'] != config['window_initializers']['101']):
        raise ValueError('The frozen seed 101 differs from the replication protocol')
    imported = {}
    for predictor in ('window16', 'window32'):
        name = f'{predictor}_101'
        source = parent / 'trials' / predictor
        report = json.loads((source / 'trial-result.json').read_text())
        if (report['seed'] != 101 or report['window'] != int(predictor[6:])
                or report['initial_state_sha256'] != config['window_initializers']['101']
                or report['selection']['training_epochs'] != config['training']['max_epochs']):
            raise ValueError(f'Invalid pilot result: {predictor}')
        dest = output / 'trials' / name
        dest.mkdir(parents=True)
        for path in config['pilot_parent']['files']:
            relative = Path(path)
            if relative.parent == Path('trials', predictor):
                shutil.copy2(parent / relative, dest / relative.name)
        imported[name] = {**report, 'trial': name, 'reused_from': config['pilot_parent']['kernel']}
    return imported


def import_decoder_pilot(config, parent, output):
    manifest = json.loads((parent / 'run-manifest.json').read_text())
    pilot = manifest['config']
    common = ('seed', 'generator', 'training', 'data', 'evaluation', 'expected_raw_case_sha256',
              'frozen_parent', 'pilot_parent')
    if any(config[key] != pilot[key] for key in common):
        raise ValueError('The four-layer decoder uses a different comparison protocol')
    name = 'full_decoder_101'
    trial = Path('trials', name)
    report = json.loads((parent / trial / 'trial-result.json').read_text())
    if (report['trial'] != name or report['seed'] != config['seed'] or report['window'] is not None
            or report['forward']['parameters']['active_nonembedding_parameters'] != 1967360
            or report['selection']['training_epochs'] != config['training']['max_epochs']):
        raise ValueError('Invalid four-layer decoder result')
    (output / trial).mkdir(parents=True)
    spec = config['decoder_pilot_parent']
    for relative in map(Path, spec['files']):
        if relative.parent == trial:
            shutil.copy2(parent / relative, output / relative)
    return {name: {**report, 'reused_from': spec['kernel']}}


def summarize_replication(results, output):
    aggregate = {}
    seeds = (42, 101, 202)
    metrics = {'models': len(results), 'seeds': len(seeds),
               'new_fits': sum('reused_from' not in row for row in results.values())}
    for name in ('window16', 'window32', 'sasrec'):
        rows = [results[f'{name}_{seed}'] for seed in seeds]
        test = [row['test']['metrics']['ndcg@10'] for row in rows]
        valid = [row['selection']['validation_ndcg10'] for row in rows]
        delta = [score - results[f'sasrec_{seed}']['test']['metrics']['ndcg@10']
                 for seed, score in zip(seeds, test)]
        summary = {
            'test_ndcg10_mean': mean(test), 'test_ndcg10_std': stdev(test),
            'val_ndcg10_mean': mean(valid),
            'warm_epoch_seconds_median': median(row['timing']['warm_epoch_median_seconds'] for row in rows),
            'wall_seconds_median': median(row['timing']['train_wall_seconds'] for row in rows),
            'test_ndcg10_delta_vs_sasrec_mean': mean(delta),
        }
        aggregate[name] = {**summary, 'test_ndcg10_by_seed': dict(zip(seeds, test)),
                           'test_ndcg10_delta_vs_sasrec_by_seed': dict(zip(seeds, delta))}
        metrics.update({f'{name}_{key}': value for key, value in summary.items()})
    common_time = {seed: same_time_validation({name: f'{name}_{seed}' for name in aggregate}, output)
                   for seed in seeds}
    _write(output / 'study-summary.json', dict(results=results, aggregate=aggregate,
                                              same_time_validation=common_time))
    return metrics


def same_time_validation(trials, output):
    curves = {}
    for name, trial in trials.items():
        rows = json.loads((output / 'trials' / trial / 'training-curves.json').read_text())
        curves[name] = [row for row in rows if 'validation' in row]
    time_limit = min(rows[-1]['elapsed_seconds'] for rows in curves.values())
    best = {}
    for name, rows in curves.items():
        eligible = [row for row in rows if row['elapsed_seconds'] <= time_limit]
        if eligible:
            row = max(eligible, key=lambda row: row['validation']['ndcg@10'])
            best[name] = dict(epoch=row['epoch'], ndcg10=row['validation']['ndcg@10'],
                              elapsed_seconds=row['elapsed_seconds'])
    return dict(time_limit_seconds=time_limit, best_validation=best)


def summarize_full_decoder(results, output):
    time_matched = same_time_validation({name: name for name in results}, output)
    _write(output / 'study-summary.json', dict(results=results, same_time_validation=time_matched))
    return trial_metrics(results)


def trial_metrics(results):
    metrics = {}
    for name, row in results.items():
        metrics.update({f'{name}_val_ndcg10': row['selection']['validation_ndcg10'],
                        f'{name}_test_ndcg10': row['test']['metrics']['ndcg@10'],
                        f'{name}_warm_epoch_seconds': row['timing']['warm_epoch_median_seconds'],
                        f'{name}_forward_flops_per_target': row['forward']['supported_flops_per_target']})
    return metrics


def run(config, output):
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ml1m_devices import assignments_for, devices_for
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    yambda = config['kind'] == 'gr_yambda_window_tiger'
    frozen_yambda = yambda and 'frozen_parent' in config
    resume_trials = {}
    last16 = config['targets_per_user'] == 16
    expected_width = config['data']['history_items'] if yambda else HISTORY_ITEMS
    if (config['data']['history_items'], config['training']['targets_per_row'],
            config['training']['batch_size']) != (expected_width, TARGETS_PER_USER, 64) or (
            not last16 and config['targets_per_user'] != TARGETS_PER_USER):
        raise ValueError(f'The pilot requires {expected_width} history items and 512 targets per optimizer update')
    replication = 'pilot_parent' in config
    sasrec_recovery = config.get('sasrec_control_only', False)
    wide_decoder = config.get('wide_decoder_replication', False)
    decoder_depth = config.get('decoder_depth_replication', False)
    tail_pilot = 'window_tail_parent' in config
    full_decoder = any(trial.get('predictor') == 'full_decoder' for trial in config['trials'])
    if frozen_yambda:
        parent = locate_parent(config['frozen_parent'])
        if 'finetune_parent' in config:
            from rhpr.gr.yambda_masked_decoder import validate_finetune

            finetune_checkpoint = validate_finetune(config, locate_parent(config['finetune_parent']))
        elif config.get('item_decoder_pilot'):
            from rhpr.gr.yambda_item_decoder import validate_pilot as validate_item_decoder_pilot

            baseline = validate_item_decoder_pilot(config, parent)
        elif 'continuation_parent' in config:
            ablation_parent = locate_parent(config['continuation_parent'])
            if config.get('frozen_encoder_pilot'):
                from rhpr.gr.yambda_frozen_encoder import validate_frozen_encoder_pilot

                resume_trials = validate_frozen_encoder_pilot(config, parent, ablation_parent)
            else:
                from rhpr.gr.yambda_window_tiger import validate_window_continuation

                resume_trials = validate_window_continuation(config, parent, ablation_parent)
        else:
            from rhpr.gr.yambda_window_tiger import validate_window_ablation

            baseline = validate_window_ablation(config, parent)
    elif yambda:
        from rhpr.gr.yambda_window_tiger import validate_pilot

        validate_pilot(config)
    elif last16:
        validate_last16_replication(config)
    elif sasrec_recovery:
        validate_sasrec_recovery(config)
    elif wide_decoder:
        validate_wide_decoder_replication(config)
    elif decoder_depth:
        validate_decoder_depth_replication(config)
    elif tail_pilot:
        tail_parent = locate_parent(config['window_tail_parent'])
        validate_window_tail_pilot(config, tail_parent)
    elif full_decoder:
        expected = dict(id='full_decoder_101', predictor='full_decoder', seed=101,
                        decoder_window=None, physical_batch_size=16)
        if 'decoder_pilot_parent' in config:
            expected.update(id='full_decoder8_101', generator_overrides=dict(decoder_layers=8, d_ff=1408),
                            expected_active_nonembedding_parameters=4589824)
        if not replication or config['trials'] != [expected]:
            raise ValueError('Expected one registered full-history decoder and frozen seed 101 controls')
    elif replication:
        validate_replication(config)
    elif {trial['id']: trial['decoder_window'] for trial in config['trials']} != {
            'prefix_control': None, 'window16': 16, 'window32': 32}:
        raise ValueError('Expected the common-prefix control and windows 16 and 32')
    devices = devices_for(config)
    if not frozen_yambda:
        parent = output if yambda else locate_parent(config['frozen_parent'])
    reused = import_pilot(config, locate_parent(config['pilot_parent']), output) if replication else {}
    if 'decoder_pilot_parent' in config:
        reused.update(import_decoder_pilot(config, locate_parent(config['decoder_pilot_parent']), output))
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
    if yambda and not frozen_yambda:
        from rhpr.gr.yambda_window_tiger import prepare

        prepare(config, output, source)
    plan = output / 'plan.json'
    execution_plan = dict(config=config, parent=str(parent), source=str(source), output=str(output),
                          assignments=assignments_for(config, devices))
    if resume_trials:
        execution_plan['resume_trials'] = resume_trials
    if 'finetune_parent' in config:
        execution_plan['finetune_checkpoint'] = str(finetune_checkpoint)
    if tail_pilot:
        execution_plan['window_tail_parent'] = str(tail_parent)
    _write(plan, execution_plan)
    summary = launch_workers(plan, worker_module='rhpr.gr.ml1m_window_tiger', event_prefix='window_tiger')
    if 'finetune_parent' in config:
        from rhpr.gr.yambda_masked_decoder import summarize as summarize_finetune

        return summarize_finetune(summary['results'], output)
    if tail_pilot:
        results = summary['results']
        metrics = trial_metrics({'window8_101': results['window8_101']})
        for name, row in results.items():
            for split, views in row['tail_views'].items():
                for tail, values in views.items():
                    metrics[f'{name}_{split}_recent{tail}_ndcg10'] = values['ndcg@10']
        _write(output / 'study-summary.json', dict(results=results))
        return metrics
    if frozen_yambda and not resume_trials:
        results = {baseline['trial']: baseline, **summary['results']}
        report = dict(control=baseline['trial'], results=results)
        if config.get('item_decoder_pilot'):
            from rhpr.gr.yambda_item_decoder import compare_predictions

            report['test_cohorts'] = compare_predictions(results, config, parent, output)
        _write(output / 'study-summary.json', report)
        return trial_metrics(results)
    if config.get('frozen_encoder_pilot'):
        results = summary['results']
        metrics = trial_metrics(results)
        for name, result in results.items():
            metrics[f'{name}_encoder_cache_seconds'] = result['encoder_cache']['seconds']
            metrics[f'{name}_encoder_cache_bytes'] = result['encoder_cache']['bytes']
            metrics[f'{name}_cached_epoch_seconds'] = result['timing']['continuation_warm_epoch_median_seconds']
        return metrics
    if sasrec_recovery or wide_decoder or decoder_depth or last16:
        return trial_metrics(summary['results'])
    if full_decoder:
        return summarize_full_decoder({**summary['results'], **reused}, output)
    if replication:
        return summarize_replication({**summary['results'], **reused}, output)
    return summarize(summary['results'], output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--slot', type=int, required=True)
    arguments = parser.parse_args()
    worker(arguments.plan, arguments.slot)
