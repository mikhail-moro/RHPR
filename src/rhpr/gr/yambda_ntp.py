"""Single-target encoder–decoder finetuning on the frozen W32 encoder."""
import hashlib
import json
from pathlib import Path
import random
import shutil
from statistics import median
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _autocast, _evaluate, _event, _model_batch, _train_generator, _write
from rhpr.gr.ml1m_window_tiger import attach_window_generation, load_cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest
from rhpr.gr.yambda_frozen_encoder import cached_window_loss, freeze_encoder, frozen_parameters_digest


def validate_ntp_phase(config):
    phase = config['ntp_phase']
    if phase != dict(source_trial='window32_b8_101', max_epochs=120, min_epochs=5,
                     learning_rate=.001, physical_batch_size=32, cache_batch_size=64,
                     max_seconds=30_000):
        raise ValueError('Expected a single-target frozen Window32 continuation to plateau')
    return dict(id='frozen_ntp_101', predictor='frozen_ntp', seed=101, decoder_window=None,
                physical_batch_size=phase['physical_batch_size'])


def single_target_cases(cases, history_items):
    """Expand each original block into the same targets and exact causal prefixes."""
    contexts, users, targets = cases
    block_size = targets.shape[1]
    if contexts.shape[1] + block_size - 1 != history_items:
        raise ValueError('The saved grouped examples have an unexpected history width')
    expanded = np.full((len(users), block_size, history_items), -1, dtype=contexts.dtype)
    for offset in range(block_size):
        prefix = np.concatenate((contexts, targets[:, :offset]), axis=1)
        expanded[:, offset, -prefix.shape[1]:] = prefix
    return expanded.reshape(-1, history_items), np.repeat(users, block_size), targets.reshape(-1)


def _finish_decoder_batch(batch, memory, memory_mask):
    future = batch['future_ids'][:, :-1]
    return dict(future_ids=future, attention_mask_decoder=future.new_ones(future.shape),
                labels=batch['future_ids'], encoder_output=memory, encoder_memory_mask=memory_mask)


def precompute_memory(model, index, train, device, precision, phase, directory):
    """Write the frozen per-prefix states to an ephemeral FP16 memory map."""
    import torch

    model.eval()
    n = len(train[1])
    size = phase['cache_batch_size']
    cache = None
    start = monotonic()
    expected_shape = None
    with torch.no_grad(), _autocast(precision, device):
        for begin in range(0, n, size):
            piece = slice(begin, begin + size)
            batch = _model_batch(*(a[piece] for a in train), index.sids, device)
            memory, _ = model.encoder_forward_pass(
                batch['attention_mask_encoder'], batch['input_ids'], batch['user_id'])
            if cache is None:
                expected_shape = (n, *memory.shape[1:])
                read_dtype = torch.float16 if precision == 'fp16' else memory.dtype
                required = n * memory.shape[1] * memory.shape[2] * 2
                if shutil.disk_usage(directory).free < required:
                    raise RuntimeError(f'Insufficient scratch disk for {required} bytes of NTP encoder memory')
                cache = np.memmap(directory / 'memory.f16', mode='w+', dtype=np.float16, shape=expected_shape)
                _event('yambda_ntp_cache_started', shape=list(expected_shape), bytes=required)
            cache[piece] = memory.to(dtype=torch.float16).cpu().numpy()
            if begin % 8192 == 0:
                _event('yambda_ntp_cache_progress', completed=min(begin+size,n), examples=n)
    cache.flush()

    def batches(rows, contexts, users, targets):
        batch = _model_batch(contexts, users, targets, index.sids, device)
        mask = batch['attention_mask_encoder']
        user_mask = torch.ones((len(users), 1), device=device)
        memory_mask = torch.cat((user_mask,mask),dim=1)
        memory = torch.from_numpy(cache[rows]).to(device=device, dtype=read_dtype)
        return _finish_decoder_batch(batch,memory,memory_mask)

    # Validate that storing FP16 does not materially perturb the starting
    # decoder under the same AMP policy used during training.
    with torch.no_grad(), _autocast(precision, device):
        span = slice(0, size)
        inputs = tuple(a[span] for a in train)
        batch = _model_batch(*inputs, index.sids, device)
        exact_memory, exact_mask = model.encoder_forward_pass(
            batch['attention_mask_encoder'], batch['input_ids'], batch['user_id'])
        exact_loss = cached_window_loss(model,_finish_decoder_batch(batch,exact_memory,exact_mask))
        cached_loss = cached_window_loss(model,batches(np.arange(len(inputs[1])),*inputs))
        torch.testing.assert_close(cached_loss,exact_loss,atol=2e-3,rtol=2e-3)
    report = dict(examples=n, shape=list(expected_shape), bytes=cache.nbytes, dtype='float16',
                  seconds=monotonic()-start, path_policy='ephemeral scratch memory map',
                  train_case_sha256=case_digest(train))
    _event('yambda_ntp_cache_ready', **report)
    return batches, report


def run_ntp_trial(trial, plan):
    import torch

    from rhpr.gr.grid import build_grid_generator

    config = plan['config']
    phase = config['ntp_phase']
    output_root = Path(plan['output'])
    parent = Path(plan['parent'])
    source = output_root / 'trials' / phase['source_trial']
    output = output_root / 'trials' / trial['id']
    output.mkdir()
    mtp = json.loads((source / 'trial-result.json').read_text())
    checkpoint_path = source / 'generator-best.pt'
    if (mtp['selection']['stopping_reason'] not in ('epoch_limit','early_stopping')
            or file_sha256(checkpoint_path) != mtp['test']['checkpoint_sha256']):
        raise ValueError('NTP needs the validation-selected completed MTP checkpoint')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint['epoch'] != mtp['selection']['best_epoch']:
        raise ValueError('Incorrect MTP checkpoint epoch')
    seed = trial['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sids = np.load(parent / 'frozen-indexes/items-42.npy',allow_pickle=False)
    index = SidIndex.from_full_sids(sids,width=256)
    model = build_grid_generator(sids, config['generator'], seed, 'cpu', plan['source'])
    model.load_state_dict(checkpoint['model'],strict=True)
    initial = state_digest(dict(model.named_parameters()))
    frozen = freeze_encoder(model)
    attach_window_generation(model,1)
    del checkpoint
    raw = load_cases(parent,config)
    train = single_target_cases(raw['train'],config['data']['history_items'])
    if len(train[2]) != config['data']['train_examples']:
        raise ValueError('Unexpected NTP target count')
    # One optimizer batch contains exactly the same 512 targets as in MTP.
    spec = {**config['training'], 'targets_per_row':1, 'rows_per_shuffle_group':8,
            'batch_size':512, 'physical_batch_size':phase['physical_batch_size'],
            'min_epochs':phase['min_epochs'], 'max_epochs':phase['max_epochs'],
            'max_seconds':phase['max_seconds'], 'learning_rate':phase['learning_rate'],
            'scheduler':None}
    run_config = {**config,'seed':seed,'active_trial':trial,'training':spec}
    cpu_batch = _model_batch(*(a[:1] for a in train),sids,'cpu')
    model.eval()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        h,m = model.encoder_forward_pass(cpu_batch['attention_mask_encoder'],cpu_batch['input_ids'],cpu_batch['user_id'])
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profile:
            loss = cached_window_loss(model,_finish_decoder_batch(cpu_batch,h,m))
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite decoder forward')
    forward = dict(supported_flops_per_target=sum(entry.flops for entry in profile.key_averages()),
                   cached_encoder_positions=h.shape[1], decoder_positions=4,
                   scope='padded teacher-forced decoder with cached memory, CPU math SDPA, FMA2')
    _write(output/'forward-flops.json',forward)
    device = 'cuda:0'
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    splits = {'train':train,'validation':raw['validation']}
    with TemporaryDirectory(prefix='ntp-cache-',dir=config['resources']['scratch_path']) as tmp:
        batches, report = precompute_memory(model,index,train,device,spec['precision'],phase,Path(tmp))
        _write(output/'encoder-cache.json',report)
        selection = _train_generator(model,parent,index,run_config,output,device,
            example_builder=lambda _root,_width,split:splits[split],
            loss_fn=cached_window_loss,train_batch_builder=batches)
        del batches
    if frozen_parameters_digest(model) != frozen:
        raise ValueError('Frozen MTP encoder weights changed during NTP')
    elapsed = monotonic()-started
    metrics, predictions, scores = _evaluate(model,raw['test'],index,spec['evaluation_batch_size'],
                                              device,precision=spec['evaluation_precision'])
    np.savez_compressed(output/'test-best-predictions.npz',scores=scores,user_ids=raw['test'][1],
                        target_item_ids=raw['test'][2],semantic_ids=predictions)
    test = dict(metrics=metrics,epoch=selection['best_epoch'],
                checkpoint_sha256=file_sha256(output/'generator-best.pt'))
    _write(output/'test-metrics.json',test)
    curves = json.loads((output/'training-curves.json').read_text())
    timing = dict(warm_epoch_median_seconds=median(r['train_seconds'] for r in curves[1:] or curves),
                  train_wall_seconds=selection['training_seconds'], total_with_cache_seconds=elapsed,
                  peak_training_allocated_bytes=torch.cuda.max_memory_allocated())
    result = dict(trial=trial['id'], seed=seed, window=None, initialization_group=trial['id'],
                  initial_state_sha256=initial, forward=forward, selection=selection, test=test, timing=timing,
                  encoder_cache=report, frozen_parameters_sha256=frozen, mtp_source=dict(
                      trial=phase['source_trial'], selected_epoch=mtp['selection']['best_epoch'],
                      training_epochs=mtp['selection']['training_epochs'],
                      checkpoint_sha256=mtp['test']['checkpoint_sha256']))
    _write(output/'trial-result.json',result)
    return result
