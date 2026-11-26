"""Causal item sequences with an autoregressive SID head and optional level passes."""
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rhpr.gr.ml1m_full_decoder import build_full_decoder

MODES = ('sum', 'residual', 'cumulative')


def build_item_decoder(sids, config, seed, source, mode):
    model = build_full_decoder(sids, config, seed, source)
    if len(model.decoder.decoder.block) != sids.shape[1] or mode not in MODES:
        raise ValueError('Expected four decoder layers for four SID levels and a registered mode')
    dim = config['d_model']
    model.item_sid_fusion = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
    model.item_history_mode = mode
    model.generate = partial(generate_next, model)
    return model


def code_embeddings(model, codes):
    offset = torch.arange(codes.shape[-1], device=codes.device) * model.num_embeddings_per_hierarchy
    return model.get_embedding_table('decoder')(codes.clamp_min(0) + offset)


def item_tape(model, codes, observed, users):
    embeddings = code_embeddings(model, codes) * observed[:, :, None, None]
    embeddings = F.pad(embeddings, (0, 0, 0, 0, 1, 0))
    user_slots = torch.arange(embeddings.shape[1], device=codes.device) == (~observed).sum(1, keepdim=True)
    user = model.user_embedding(users.reshape(-1) % model.user_embedding.num_embeddings)
    start = user_slots.unsqueeze(-1) * (user + model.decoder.bos_token).unsqueeze(1)
    embeddings = torch.cat((embeddings[:, :, :1] + start.unsqueeze(2), embeddings[:, :, 1:]), dim=2)
    return embeddings, F.pad(observed, (1, 0)) | user_slots


def history_states(model, codes, observed, users):
    from transformers.masking_utils import create_causal_mask

    stack = model.decoder.decoder
    tape, mask = item_tape(model, codes, observed, users)
    mode = model.item_history_mode
    if mode == 'sum':
        state = stack(inputs_embeds=tape.sum(dim=2), attention_mask=mask, use_cache=False).last_hidden_state
        return (state,) * tape.shape[2]
    if mode == 'residual':
        states = []
        state = torch.zeros_like(tape[:, :, 0])
        for embeddings in tape.unbind(dim=2):
            # Each pass shares the whole stack. Its output is already RMS-normalized.
            state = stack(inputs_embeds=embeddings + state, attention_mask=mask, use_cache=False).last_hidden_state
            states.append(state)
        return tuple(states)

    # The pinned T5 stack uses one causal mask and the first block's position
    # bias throughout. Inject a new SID embedding before each native block.
    attention = create_causal_mask(config=stack.config, inputs_embeds=tape[:, :, 0],
                                   attention_mask=mask, past_key_values=None)
    state = torch.zeros_like(tape[:, :, 0])
    bias = None
    states = []
    for block, embeddings in zip(stack.block, tape.unbind(dim=2), strict=True):
        state, bias, _ = block(state + stack.dropout(embeddings), attention_mask=attention,
                               position_bias=bias, past_key_values=None)
        states.append(stack.dropout(stack.final_layer_norm(state)))
    return tuple(states)


def sid_logits(model, level, context, within_item):
    hidden = context + model.item_sid_fusion(torch.cat((context, within_item), dim=-1))
    return model.decoder.decoder_mlp[level](hidden)


def teacher_forced_logits(model, batch):
    labels = batch['future_ids']
    observed = batch['attention_mask_encoder'][:, ::labels.shape[-1]].bool()
    history = batch['input_ids'].unflatten(1, (-1, labels.shape[-1]))
    # Predict target j from the last completed item before j. The whole SID of
    # target j is available to the temporal stack starting at target j+1 only.
    items = torch.cat((history, labels[:, :-1]), dim=1)
    observed = F.pad(observed, (0, labels.shape[1] - 1), value=True)
    states = history_states(model, items, observed, batch['user_id'])
    target_embeddings = code_embeddings(model, labels)
    prefix = torch.zeros_like(target_embeddings[:, :, 0])
    logits = []
    for level, state in enumerate(states):
        logits.append(sid_logits(model, level, state[:, -labels.shape[1]:], prefix))
        prefix = prefix + target_embeddings[:, :, level]
    return tuple(logits)


def item_decoder_loss(model, batch):
    labels = batch['future_ids']
    return sum(F.cross_entropy(logits.flatten(0, 1), labels[:, :, level].flatten())
               for level, logits in enumerate(teacher_forced_logits(model, batch)))


def generate_next(model, attention_mask, input_ids, user_id):
    levels = len(model.decoder.decoder_mlp)
    observed = attention_mask[:, ::levels].bool()
    states = history_states(model, input_ids.unflatten(1, (-1, levels)), observed, user_id)
    batch_size = len(input_ids)
    ids = scores = None
    for level, state in enumerate(states):
        context = state[:, -1]
        if level:
            context = context.repeat_interleave(model.top_k_for_generation, dim=0)
        prefix = torch.zeros_like(context)
        if level:
            for embeddings in code_embeddings(model, ids.flatten(0, 1)).unbind(dim=1):
                prefix = prefix + embeddings
        logits = sid_logits(model, level, context, prefix)
        ids, scores, _ = model._beam_search_one_step(logits, ids, scores, None, level, batch_size)
    return ids, scores


def validate_pilot(config, parent):
    reference = json.loads((parent / 'run-manifest.json').read_text())['config']
    shared = ('generator', 'training', 'data', 'evaluation', 'execution', 'expected_raw_case_sha256',
              'targets_per_user', 'quantizer', 'quantizer_seed')
    expected = [dict(id=f'item_{mode}_d4_101', predictor='item_decoder', history_mode=mode, seed=101,
                     decoder_window=None, physical_batch_size=16) for mode in MODES]
    if (any(config[key] != reference[key] for key in shared) or config['trials'] != expected
            or config['seed'] != 101 or config['generator_seeds'] != [101]
            or reference['training']['max_epochs'] != 30):
        raise ValueError('Expected three seed-101 item-Decoder pilots on the pinned 30-epoch Last-16 protocol')
    baseline = json.loads((parent / 'trials/full_decoder_101/trial-result.json').read_text())
    selection = baseline['selection']
    if (baseline['trial'] != 'full_decoder_101' or baseline['seed'] != 101 or baseline['window'] is not None
            or selection['training_epochs'] != 30
            or selection['training_examples_per_epoch'] != config['data']['train_examples']):
        raise ValueError('The archived token D4 does not match this pilot')
    return {**baseline, 'reused_from': config['frozen_parent']['kernel']}


def compare_predictions(results, config, parent, output):
    from rhpr.gr.ml1m_window_tiger import load_cases

    test = load_cases(parent, config)['test']
    contexts, users, targets = test
    sids = np.load(parent / 'frozen-indexes/items-42.npy', allow_pickle=False)
    target_sids = sids[targets]
    recent = np.any(contexts[:, -8:] == targets[:, None], axis=1)
    summary = {}
    for name, result in results.items():
        root = parent if 'reused_from' in result else output
        with np.load(root / 'trials' / name / 'test-best-predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['user_ids'], users)
            np.testing.assert_array_equal(saved['target_item_ids'], targets)
            predictions = saved['semantic_ids'][:, :10]
        triple = (predictions[:, :, :3] == target_sids[:, None, :3]).all(axis=-1)
        full = triple & (predictions[:, :, 3] == target_sids[:, None, 3])
        ndcg = np.where(full.any(axis=1), 1 / np.log2(full.argmax(axis=1) + 2), 0)
        np.testing.assert_allclose(ndcg.mean(), result['test']['metrics']['ndcg@10'], atol=1e-10, rtol=0)
        summary[name] = dict(recent_repeat_users=int(recent.sum()), recent_repeat_ndcg10=float(ndcg[recent].mean()),
                             correct_triple_top10=int(triple.any(axis=1).sum()),
                             exact_track_top10=int(full.any(axis=1).sum()))
    return summary
