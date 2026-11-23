"""Reuse frozen Window32 encoder outputs during the decoder continuation."""
import json
from time import monotonic

import numpy as np

from rhpr.gr.experiment import _autocast, _event
from rhpr.gr.ml1m_window_tiger import window_loss, window_target_loss
from rhpr.gr.sid_study import state_digest


def validate_frozen_encoder_pilot(config, data_parent, checkpoint_parent):
    data = json.loads((data_parent / 'run-manifest.json').read_text())['config']
    checkpoint = json.loads((checkpoint_parent / 'run-manifest.json').read_text())['config']
    shared = ('generator', 'data', 'evaluation', 'expected_raw_case_sha256', 'targets_per_user')
    name = 'window32_b8_101'
    trial = next(t for t in checkpoint['trials'] if t['id'] == name)
    source = checkpoint_parent / 'trials' / name
    result = json.loads((source / 'trial-result.json').read_text())
    if (any(config[key] != previous[key] for key in shared for previous in (data, checkpoint))
            or config['training'] != {**checkpoint['training'], 'max_epochs': 60, 'min_epochs': 60}
            or config['seed'] != 101 or config['trials'] != [trial]
            or config['execution']['worker_devices'] != ['cuda:0'] or config['execution']['kaggle_gpu'] != 'T4'
            or config['expected_initial_state_sha256'] != result['initial_state_sha256']
            or not config.get('compare_initialization_by_shape')
            or (result['selection']['training_epochs'], result['selection']['best_epoch']) != (30, 30)
            or result['selection']['optimizer_updates'] != 8610):
        raise ValueError('Expected cached Window32/8 on seed 101 continuing the exact Last-16 run from epoch 30')
    return {name: str(source)}


def frozen_parameters_digest(model):
    return state_digest({name: value for name, value in model.named_parameters() if not value.requires_grad})


def freeze_encoder(model):
    model.encoder.requires_grad_(False)
    model.get_embedding_table('encoder').requires_grad_(False)
    model.user_embedding.requires_grad_(False)
    if model.sep_token is not None:
        model.sep_token.requires_grad_(False)
    return frozen_parameters_digest(model)


def cached_window_loss(model, batch):
    hidden = model.decoder_forward_pass(
        future_ids=batch['future_ids'], attention_mask=batch['attention_mask_decoder'],
        encoder_output=batch['encoder_output'], attention_mask_for_encoder=batch['encoder_memory_mask'])
    return window_target_loss(model, hidden, batch['labels'])


def decoder_batch(batch, memory, mask):
    return dict(future_ids=batch['future_ids'], attention_mask_decoder=batch['attention_mask_decoder'],
                labels=batch['labels'], encoder_output=memory, encoder_memory_mask=mask)


def cache_training_memory(model, index, train, device, training):
    import torch

    model.eval()
    count = len(train[1])
    rows_per_batch = training['physical_batch_size']
    started = monotonic()
    _event('yambda_encoder_cache_started', blocks=count, batch_size=rows_per_batch)
    memory_cache = mask_cache = None
    # no_grad leaves ordinary tensors available for the trainable K/V layers
    # to save as inputs to their backward pass.
    with torch.no_grad(), _autocast(training['precision'], device):
        for begin in range(0, count, rows_per_batch):
            part = slice(begin, begin + rows_per_batch)
            batch = index.model_batch(*(array[part] for array in train), device)
            memory, mask = model.encoder_forward_pass(
                batch['attention_mask_encoder'], batch['input_ids'], batch['user_id'])
            if memory_cache is None:
                memory_cache = memory.new_empty((count, *memory.shape[1:]))
                mask_cache = mask.new_empty((count, *mask.shape[1:]))
            memory_cache[part].copy_(memory)
            mask_cache[part].copy_(mask)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = monotonic() - started

    def build(part, contexts, users, targets):
        batch = index.model_batch(contexts, users, targets, device)
        rows = torch.as_tensor(part, device=device, dtype=torch.long)
        return decoder_batch(batch, memory_cache[rows], mask_cache[rows])

    part = slice(0, rows_per_batch)
    examples = tuple(a[part] for a in train)
    with torch.no_grad(), _autocast(training['precision'], device):
        exact = window_loss(model, index.model_batch(*examples, device))
        cached = cached_window_loss(model, build(np.arange(len(examples[1])), *examples))
    torch.testing.assert_close(cached, exact, rtol=1e-5, atol=1e-6)
    report = dict(blocks=count, seconds=seconds, memory_dtype=str(memory_cache.dtype),
                  memory_shape=list(memory_cache.shape), device=str(memory_cache.device),
                  bytes=memory_cache.nbytes + mask_cache.nbytes,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    _event('yambda_encoder_cache_ready', **report)
    return build, report
