"""Continue a token D4 with a causal SID-level attention mask."""
from functools import partial
import json
from pathlib import Path
import random
from statistics import median

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate, _model_batch, _train_generator, _write
from rhpr.gr.ml1m_full_decoder import build_full_decoder, embed_prefix, full_decoder_loss, generate_next
from rhpr.gr.ml1m_window_tiger import load_cases
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest

RECENT_ITEMS = 31
CHECKPOINT_EPOCH = 110
TRIALS = ('masked_d4_101', 'dense_d4_101')


def sid_attention_mask(observed, queries):
    """Route each causal query by the level of the code it predicts next."""
    observed = observed.bool()
    length = observed.shape[-1]
    keys = torch.arange(length, device=observed.device)
    positions = torch.arange(length - queries, length, device=observed.device)
    target_level = ((positions - 1) % 4)[:, None]
    target_item = ((positions - 1) // 4)[:, None]
    key_level = (keys - 2) % 4
    key_item = (keys - 2) // 4
    start = (~observed).sum(dim=1, keepdim=True)
    special = (keys == start) | (keys == start + 1)
    route = (target_item - key_item <= RECENT_ITEMS) | (target_level == key_level) | (target_level == 3)
    return ((route | special[:, None, :]) & observed[:, None, :] & (keys <= positions[:, None]))[:, None]


def training_spec(reference):
    return {**reference, 'min_epochs': 10, 'max_epochs': 10, 'learning_rate': .0003,
            'scheduler': None, 'max_seconds': 3600}


def validate_finetune(config, parent):
    reference = json.loads((parent / 'run-manifest.json').read_text())['config']
    shared = ('generator', 'data', 'execution', 'evaluation', 'expected_raw_case_sha256', 'targets_per_user',
              'quantizer', 'quantizer_seed')
    expected = [dict(id=name, predictor='d4_mask_finetune', seed=101, decoder_window=None,
                     physical_batch_size=16) for name in TRIALS]
    if (any(config[key] != reference[key] for key in shared) or config['seed'] != 101
            or config['generator_seeds'] != [101] or config['training'] != training_spec(reference['training'])
            or config['trials'] != expected):
        raise ValueError('Expected paired ten-epoch D4 mask adaptation from the pinned reference')
    source = parent / 'trials/full_decoder_101'
    previous = json.loads((source / 'trial-result.json').read_text())
    weights = source / 'generator-best.pt'
    if (previous['seed'] != 101 or previous['selection']['best_epoch'] != CHECKPOINT_EPOCH
            or previous['test']['checkpoint_sha256'] != file_sha256(weights)):
        raise ValueError('Wrong validation-selected D4 initialization')
    return weights


def packed_query_plan(length, device):
    """Group queries with overlapping key sets for unpadded prefill."""
    allowed = sid_attention_mask(torch.ones((1, length), dtype=torch.bool), length)[0, 0]
    positions = torch.arange(length)
    levels = (positions - 1) % 4
    groups = [positions[levels == 3]]
    for block in positions.split(128):
        groups.extend(block[levels[block] == level] for level in range(3))
    plan = []
    for queries in groups:
        if queries.numel():
            keys = torch.where(allowed[queries].any(dim=0))[0]
            plan.append((queries.to(device), keys.to(device)))
    return tuple(plan)


def packed_attention(plan, module, query, key, value, attention_mask, position_bias=None, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    result = torch.empty_like(query.transpose(1, 2))
    for rows, columns in plan:
        output, _ = sdpa_attention_forward(
            module, query.index_select(2, rows), key.index_select(2, columns), value.index_select(2, columns),
            attention_mask.index_select(-2, rows).index_select(-1, columns),
            position_bias=position_bias.index_select(-2, rows).index_select(-1, columns), **kwargs)
        result.index_copy_(1, rows, output)
    return result, None


def prefill_benchmark(model, cases, sids, device):
    """Measure dense, masked, and physically packed attention on full windows."""
    from transformers.models.t5.modeling_t5 import ALL_ATTENTION_FUNCTIONS

    selected = np.flatnonzero((cases[0] >= 0).all(axis=1))[:16]
    if len(selected) != 16:
        raise ValueError('The prefill benchmark requires 16 fully occupied validation windows')
    batch = _model_batch(*(array[selected] for array in cases), sids, device)
    tape, padding = embed_prefix(model, batch['input_ids'], batch['attention_mask_encoder'], batch['user_id'])
    masked = sid_attention_mask(padding, tape.shape[1])
    plan = packed_query_plan(tape.shape[1], device)
    backend = model.decoder.decoder.config
    original = backend._attn_implementation
    custom_backend = 'rhpr_d4_packed_prefill'
    ALL_ATTENTION_FUNCTIONS[custom_backend] = partial(packed_attention, plan)

    def run(mask):
        return model.decoder(sequence_embedding=tape, attention_mask=mask, encoder_output=None,
                             encoder_attention_mask=None, use_cache=True, past_key_values=None)

    try:
        model.eval()
        with torch.inference_mode():
            expected, cache = run(masked)
            backend._attn_implementation = custom_backend
            packed, packed_cache = run(masked)
            torch.testing.assert_close(packed, expected, atol=2e-4, rtol=2e-4)
            for dense_layer, packed_layer in zip(cache.layers, packed_cache.layers, strict=True):
                torch.testing.assert_close(packed_layer.keys, dense_layer.keys, atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(packed_layer.values, dense_layer.values, atol=2e-4, rtol=2e-4)
            timings = {}
            for label, implementation, mask in (('dense', original, padding), ('masked_dense', original, masked),
                                                ('masked_packed', custom_backend, masked)):
                backend._attn_implementation = implementation
                for _ in range(3):
                    run(mask)
                torch.cuda.synchronize()
                samples = []
                for _ in range(12):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    run(mask)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                timings[label] = dict(median_ms=median(samples), samples_ms=samples)
        return dict(timings=timings, positions=tape.shape[1], batch_size=len(selected),
                    dense_query_key_pairs=tape.shape[1]**2,
                    packed_query_key_pairs=sum(rows.numel() * cols.numel() for rows, cols in plan),
                    scope='FP32 warm prefill with KV cache, unpadded batch; no beam search')
    finally:
        backend._attn_implementation = original
        del ALL_ATTENTION_FUNCTIONS[custom_backend]


def assess(model, cases, index, config, directory, name, masked):
    model.generate = partial(generate_next, model, mask_builder=sid_attention_mask if masked else None)
    metrics, predictions, scores = _evaluate(model, cases, index, config['training']['evaluation_batch_size'],
                                              'cuda:0', precision='fp32')
    contexts, users, targets = cases
    np.savez_compressed(directory / f'{name}.npz', scores=scores, semantic_ids=predictions,
                        user_ids=users, target_item_ids=targets)
    hits = np.all(predictions[:, :10] == index.sids[targets, None, :], axis=-1)
    contributions = np.where(hits.any(axis=1), 1 / np.log2(hits.argmax(axis=1) + 2), 0)
    repeats = np.any(contexts[:, -8:] == targets[:, None], axis=1)
    assert abs(contributions.mean() - metrics['ndcg@10']) < 1e-10
    return dict(metrics=metrics, repeat_last8_users=int(repeats.sum()),
                repeat_last8_ndcg10=float(contributions[repeats].mean()))


def run_trial(trial, plan):
    config = plan['config']
    root = Path(plan['parent'])
    output = Path(plan['output']) / 'trials' / trial['id']
    output.mkdir(parents=True)
    seed = trial['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cases = load_cases(root, config)
    sids = np.load(root / 'frozen-indexes/items-42.npy', allow_pickle=False)
    index = SidIndex.from_full_sids(sids, width=256)
    model = build_full_decoder(sids, config['generator'], seed, plan['source'])
    weights = Path(plan['finetune_checkpoint'])
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != CHECKPOINT_EPOCH:
        raise ValueError('Wrong initial D4 epoch')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    initial_state = state_digest(dict(model.named_parameters()))
    model.to('cuda:0')
    masked = trial['id'] == TRIALS[0]
    result = dict(trial=trial['id'], seed=seed, initial_checkpoint_sha256=file_sha256(weights),
                  initial_state_sha256=initial_state, initialization_group='pretrained_token_d4_101',
                  source_epoch=CHECKPOINT_EPOCH, train_with_mask=masked)
    if masked:
        result['zero_shot'] = {f'{split}_{label}': assess(model, cases[split], index, config, output,
                                                        f'zero-shot-{split}-{label}', label == 'masked')
                               for split in ('validation', 'test') for label in ('dense', 'masked')}
        _write(output / 'zero-shot.json', result['zero_shot'])
    # Select both arms using the same serving mask. The dense control differs
    # from the masked arm only in which mask its training queries use.
    model.generate = partial(generate_next, model, mask_builder=sid_attention_mask)
    active = {**config, 'active_trial': trial}
    selection = _train_generator(model, root, index, active, output, 'cuda:0',
                                 example_builder=lambda _, __, split: cases[split],
                                 loss_fn=partial(full_decoder_loss, mask_builder=sid_attention_mask if masked else None))
    expected_steps = (len(cases['train'][1]) + 63) // 64 * 10
    if selection['training_epochs'] != 10 or selection['optimizer_updates'] + selection['amp_skipped_updates'] != expected_steps:
        raise ValueError('Fine-tuning ended before the matched 10-epoch budget')
    result.update(selection=selection, best_checkpoint_sha256=file_sha256(output / 'generator-best.pt'),
                  test_masked=assess(model, cases['test'], index, config, output, 'test-masked', True),
                  test_dense=assess(model, cases['test'], index, config, output, 'test-dense', False))
    _write(output / 'trial-result.json', result)
    if masked:
        result['prefill'] = prefill_benchmark(model, cases['validation'], sids, 'cuda:0')
        _write(output / 'trial-result.json', result)
    return result


def summarize(results, output):
    _write(output / 'study-summary.json', {'results': results})
    metrics = {}
    for name, row in results.items():
        metrics[f'{name}_val_masked_ndcg10'] = row['selection']['validation_ndcg10']
        for usage in ('test_masked', 'test_dense'):
            metrics[f'{name}_{usage}_ndcg10'] = row[usage]['metrics']['ndcg@10']
            metrics[f'{name}_{usage}_repeat8_ndcg10'] = row[usage]['repeat_last8_ndcg10']
    for name, row in results[TRIALS[0]]['zero_shot'].items():
        metrics[f'zero_shot_{name}_ndcg10'] = row['metrics']['ndcg@10']
    for name, timing in results[TRIALS[0]]['prefill']['timings'].items():
        metrics[f'prefill_{name}_median_ms'] = timing['median_ms']
    return metrics
