"""Fixed-prefix Pair128 screens with optional full/pair training views."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate, _event, _loss, _train_generator, _write
from rhpr.gr.ml1m_last8 import REFERENCE_SID_SHA256, prepare_raw, val_best_within
from rhpr.gr.movie_data import case_digest

MERGE_PREFIXES = [64, 96, 112, 120]
TRIALS = [dict(id=f'pair128_merge{m}_42', merge_items=m, seed=42) for m in MERGE_PREFIXES]
MIXED_TRIALS = [dict(id=f'mixed128_merge{m}_42', merge_items=m, seed=42) for m in MERGE_PREFIXES]
REPLICATION_SEEDS = [101, 202]
REPLICATION_TRIALS = [dict(id=f'mixed128_merge120_{seed}', merge_items=120, seed=seed)
                      for seed in REPLICATION_SEEDS]
EXTENDED_TRIALS = [dict(id=f'mixed128_merge64_{seed}', merge_items=64, seed=seed)
                   for seed in REPLICATION_SEEDS]
THINNED_TRIALS = [dict(id=f'thin128_merge64_{seed}', merge_items=64, seed=seed)
                  for seed in REPLICATION_SEEDS]
MEAN_TRIALS = [dict(id=f'mean128_merge64_{seed}', merge_items=64, seed=seed)
               for seed in REPLICATION_SEEDS]
MEAN_ITEM_VECTORS = 'pair-cache/item-embeddings.npy'
PARENT_KERNEL = 'carasik/rhpr-ml1m-mixed128-screen'
PARENT_COMMIT = '1074a7599c52875d66eb83716491f377c2310f33'
PARENT_TRIAL = 'trials/mixed128_merge120_42'
PARENT_FILES = {
    'run-manifest.json', 'index-report.json', 'prepared-cases/data-report.json',
    *(f'prepared-cases/{split}-cases.npz' for split in ('train','validation','test')),
    *(f'frozen-indexes/{name}.npy' for name in ('items-42','pair-keys','pairs-42')),
    f'{PARENT_TRIAL}/training-view-provenance.json', f'{PARENT_TRIAL}/trial-result.json',
}
MIXED_VIEWS = dict(full_probability=.5, unit='optimizer_batch', inference='fixed_pairs',
                   rng_stream=7301, preserve_native_positions=True)
POLICY = dict(history_items=128, merge_prefix_items=MERGE_PREFIXES,
    protected_tail_items=[64,32,16,8], training='fixed_oldest_valid_adjacent_pairs',
    padding='left_pad_compacted_3plus1_units', odd_prefix='keep_single_before_tail',
    pair_index='one_shared_train_pair_union', suffix='train_then_validation_then_test',
    pair_text_labels=['First movie','Second movie'], pair_max_tokens=2048,
    all_targets_single=True, random_training=False, fixed_epochs=40)
MIXED_POLICY = {**POLICY, 'training': 'full_or_fixed_pairs_per_optimizer_batch',
                'random_training': True}
EXTENDED_MIXED_POLICY = {**MIXED_POLICY, 'fixed_epochs': 60}
THINNED_POLICY = {**EXTENDED_MIXED_POLICY, 'training': 'full_or_keep_second_per_optimizer_batch',
                  'merge_prefix_items': [64], 'protected_tail_items': [64]}
THINNED_VIEWS = {**MIXED_VIEWS, 'inference': 'keep_second'}
DEPTH_PILOT_SEED = 101
DEPTH_PILOT_RESOURCES = ["grid-source.bin", "wheels/"]
DEPTH_T4_TRAIN_SECONDS = 16800


def depth_pilot_trials(depth):
    return [dict(id=f'd{depth}_{view}_{DEPTH_PILOT_SEED}', seed=DEPTH_PILOT_SEED,
                 history_view=view, merge_items=merge)
            for view, merge in (('recent128', 0), ('thinned64', 64))]


def single_depth_t4_trial(config):
    return config['execution'].get('colab_gpu') == 'T4' and len(config['trials']) == 1


def snapshot_colab_checkpoint(trial_output, output, epoch):
    pending = output / 'checkpoint.partial'
    with tarfile.open(pending, 'w') as archive:
        for name in ('generator-last.pt', 'generator-best.pt', 'training-curves.json'):
            archive.add(trial_output / name, arcname=name)
    pending.replace(output / 'checkpoint.tar')
    pending_epoch = output / 'checkpoint-epoch.partial'
    pending_epoch.write_text(f'{epoch}\n')
    pending_epoch.replace(output / 'checkpoint-epoch')


def depth_pilot_reference(config):
    """Validate and read the frozen data using its original registered protocol."""
    reference = copy.deepcopy(config)
    reference.pop('depth_pilot')
    reference['generator']['encoder_layers'] = 4
    reference['trials'] = THINNED_TRIALS
    reference['seeds'] = REPLICATION_SEEDS
    reference['pair_batches']['physical'] = 64
    reference['resources'].pop('include_prefixes')
    return reference


def build_trial_model(sids, c, seed, source):
    from rhpr.gr.grid import build_grid_generator

    model = build_grid_generator(sids, c['generator'], seed, 'cpu', source)
    if c.get('depth_pilot') and c['generator']['encoder_layers'] == 8:
        shallow = build_grid_generator(sids, {**c['generator'], 'encoder_layers': 4}, seed, 'cpu', source)
        missing, unexpected = model.load_state_dict(shallow.state_dict(), strict=False)
        if unexpected or not missing or any(not re.search(r'\.encoder\.block\.[4-7]\.', name) for name in missing):
            raise ValueError('only the four added encoder blocks may have new initial weights')
    return model


def mixed_parent_trial(config):
    return 'trials/mixed128_merge64_42' if config.get('extended_budget') else PARENT_TRIAL


def mixed_parent_files(config):
    trial = mixed_parent_trial(config)
    files = {name.replace(PARENT_TRIAL, trial) for name in PARENT_FILES}
    if config.get('pair_vector_method') == 'mean':
        files.add(MEAN_ITEM_VECTORS)
    return files


def validate(c):
    if 'depth_pilot' in c:
        depth = c['generator']['encoder_layers']
        initial = c['depth_pilot']['initializers']
        registered = depth_pilot_trials(depth)
        trial_sets = [[trial] for trial in registered] if single_depth_t4_trial(c) else [registered]
        if (depth not in (4, 8) or set(c['depth_pilot']) != {'initializers'}
                or c.get('history_reduction') != 'keep_second'
                or c['resources'].get('include_prefixes') != DEPTH_PILOT_RESOURCES
                or c['seeds'] != [DEPTH_PILOT_SEED]
                or c['trials'] not in trial_sets
                or c['pair_batches'] != dict(physical=32, evaluation=16)
                or set(initial) != {'base', 'pair'}
                or any(not re.fullmatch('[0-9a-f]{64}', v) for v in initial.values())):
            raise ValueError('invalid depth-by-history pilot protocol')
        validate(depth_pilot_reference(c))
        return
    from rhpr.gr.ml1m_last8 import WINDOWS, TRIALS as BASE_TRIALS, validate as base_validate
    t = c['training']; reference = c['reference_control']
    mixed = c.get('mixed_views')
    parent = c.get('frozen_parent')
    extended = c.get('extended_budget')
    thinned = c.get('history_reduction') == 'keep_second'
    mean_pairs = c.get('pair_vector_method') == 'mean'
    expected_views = THINNED_VIEWS if thinned else MIXED_VIEWS
    epochs = 60 if extended else 40
    trials = (EXTENDED_TRIALS if extended else REPLICATION_TRIALS) if parent else MIXED_TRIALS if mixed else TRIALS
    policy = EXTENDED_MIXED_POLICY if extended else MIXED_POLICY if mixed else POLICY
    if thinned:
        trials, policy = THINNED_TRIALS, THINNED_POLICY
    if mean_pairs:
        trials = MEAN_TRIALS
    deadline = 10800 if extended else 7200 if parent else 14400
    if (('mixed_views' in c and mixed != expected_views)
            or ('pair_vector_method' in c and (not mean_pairs or not extended or thinned))
            or ('history_reduction' in c and (not thinned or not extended or not mixed))
            or ('extended_budget' in c and (extended is not True or not parent or not mixed))
            or c['kind'] != 'gr_ml1m_last8_pairs' or c['windows'] != [128] or c['trials'] != trials
            or c['policy'] != policy
            or c['expected_sid_file_sha256'] != REFERENCE_SID_SHA256
            or (t['min_epochs'],t['max_epochs'],t['max_seconds']) != (epochs,epochs,9000 if extended else 7200)
            or c['pair_batches'] != dict(physical=64,evaluation=16)
            or c['pair_embedding_batch_size'] != 128
            or reference['kernel'] != 'carasik/rhpr-ml1m-last8-windows/2'
            or reference['source_commit'] != '5c1af46bddb1a848877acead858ec855e77201ed'
            or reference['best_epoch'] != 35 or reference['matched_max_epoch'] != 40
            or not re.fullmatch('[0-9a-f]{64}',c['expected_pair_initial_state_sha256'])
            or c['suite_deadline_seconds'] != deadline):
        raise ValueError('Pair128 configuration differs from the registered protocol')
    if parent:
        initializations = c['replication_initializers']
        if (mixed != expected_views or c['seeds'] != REPLICATION_SEEDS
                or parent['kernel'] != PARENT_KERNEL or parent['git_sha'] != PARENT_COMMIT
                or c['kaggle']['kernel_sources'] != [PARENT_KERNEL]
                or set(parent['files']) != mixed_parent_files(c)
                or parent['files']['frozen-indexes/items-42.npy']['sha256'] != REFERENCE_SID_SHA256
                or set(initializations) != {str(seed) for seed in REPLICATION_SEEDS}
                or any(set(v) != {'base','pair'} or
                       any(re.fullmatch('[0-9a-f]{64}', h) is None for h in v.values())
                       for v in initializations.values())
                or len({v['pair'] for v in initializations.values()} | {c['expected_pair_initial_state_sha256']}) != 3):
            raise ValueError('expected two new Mixed128 seeds on the completed frozen index')
    if thinned:
        hashes = c.get('expected_thinned_case_sha256', {})
        if (set(hashes) != {'train', 'validation', 'test'}
                or any(not re.fullmatch('[0-9a-f]{64}', v) for v in hashes.values())):
            raise ValueError('expected pinned cases for thinning each split')
    if (len(reference['validation_curve']) != 8
            or [r['epoch'] for r in reference['validation_curve']] != list(range(5,41,5))
            or max(reference['validation_curve'],key=lambda r:r['ndcg10'])['epoch'] != reference['best_epoch']
            or reference['validation_ndcg10'] != max(r['ndcg10'] for r in reference['validation_curve'])):
        raise ValueError('existing control checkpoint must also be the best within40epochs')
    normalized=copy.deepcopy(c);normalized.update(kind='gr_ml1m_last8',windows=WINDOWS,trials=BASE_TRIALS)
    normalized['training'].update(min_epochs=40,max_epochs=120)
    if parent:
        normalized['seeds'] = [42]
        normalized['kaggle']['kernel_sources'] = []
    base_validate(normalized)


def checked_contexts(contexts, nitems):
    h = np.asarray(contexts)
    if h.ndim != 2 or h.shape[1] < 128 or h.dtype.kind not in 'iu': raise ValueError('integer histories with128slots required')
    h=h[:, -128:]; n=(h >= 0).sum(1)
    if (np.any(h < -1) or np.any(h >= nitems)
            or np.any((np.arange(128)[None,:] < 128-n[:,None]) != (h < 0))):
        raise ValueError('invalid catalogue ID or noncontiguous left padding')
    return h,n


def pair_keys(contexts, nitems, merge_items):
    if merge_items not in MERGE_PREFIXES: raise ValueError('unregistered merge prefix')
    h,n=checked_contexts(contexts,nitems);tail=128-merge_items;parts=[]
    for j in range(merge_items//2):
        rows=np.flatnonzero(2*j+1 < np.maximum(n-tail,0))
        if len(rows):
            pos=128-n[rows]+2*j
            parts.append(h[rows,pos].astype(np.int64)*nitems+h[rows,pos+1])
    return np.unique(np.concatenate(parts)) if parts else np.empty(0,dtype=np.int64)


def encode(contexts, sids, keys, pairs, merge_items, *, keep_second=False):
    """Represent each old pair by its SID or the second item's SID."""
    if (merge_items not in MERGE_PREFIXES or sids.shape != (len(sids),4)
            or pairs.shape != (len(keys),4) or np.any(keys[1:] <= keys[:-1])):
        raise ValueError('expected sorted pair keys and two3+1SIDtables')
    h,n=checked_contexts(contexts,len(sids));p=np.maximum(n-(128-merge_items),0)//2
    capacity=128-merge_items//2;units=n-p
    out=np.full((len(h),capacity,4),-1,dtype=np.int32)
    for col in range(capacity):
        local=col-(capacity-units);valid=local >= 0;ispair=valid & (local < p)
        source=128-n+np.where(local < p,2*local,local+p)
        rows=np.flatnonzero(valid & ~ispair)
        if len(rows):out[rows,col]=sids[h[rows,source[rows]]]+np.arange(4)*256
        rows=np.flatnonzero(ispair)
        if keep_second:
            out[rows, col] = sids[h[rows, source[rows] + 1]] + np.arange(4) * 256
        elif len(rows):
            wanted=h[rows,source[rows]].astype(np.int64)*len(sids)+h[rows,source[rows]+1]
            at=np.searchsorted(keys,wanted)
            if np.any(at >= len(keys)) or not np.array_equal(keys[at],wanted): raise ValueError('missing pair SID; never drop events')
            out[rows,col]=pairs[at]+(4+np.arange(4))*256
    return out.reshape(len(h),capacity*4)


def cache_sets(data):
    from rhpr.gr.ml100k_pair_raw import cases
    # Smaller protected tails contain every other variant's oldest pairs.
    return {s:pair_keys(cases(data,s)[0],3706,120) for s in ('train','validation','test')}


def encode_full(contexts, sids):
    history, _ = checked_contexts(contexts, len(sids))
    tokens = sids[np.maximum(history, 0)] + 256 * np.arange(4, dtype=np.int32)
    return np.where(history[:, :, None] >= 0, tokens, -1).reshape(len(history), 512)


def full_context_selector(contexts, sids, pair_contexts, config, merge_items):
    """Switch between the same histories at their respective native widths."""
    from rhpr.gr.kindle_mixed_views import MixedViews

    compressed_name = 'thin128' if config.get('history_reduction') == 'keep_second' else 'pair128'
    return MixedViews(encode_full(contexts, sids), pair_contexts, batch_size=config['training']['batch_size'],
                      seed=config['seed'], full_probability=config['mixed_views']['full_probability'],
                      layout=(('full128', 512), (compressed_name, 4 * (128 - merge_items // 2))))


def profiling_batch(index, contexts, users, targets, ignored_sids, device):
    """Profiling's five-argument builder contract, using the preencoded index."""
    return index.model_batch(contexts,users,targets,device)


def frozen_inputs(c, input_root=None, *, ordinary_controls=False):
    """Load pinned cases and tokens from the completed Mixed128 pilot."""
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_test import locate_parent

    parent = locate_parent(c['frozen_parent'], input_root)
    original = json.loads((parent/'run-manifest.json').read_text())['config']
    shared_keys = ('data','generator','quantizer','sampling','evaluation',
                   'reference_control','expected_raw_case_sha256')
    if not ordinary_controls:
        shared_keys += ('mixed_views','pair_batches')
    thinned = c.get('history_reduction') == 'keep_second'
    comparison = {**c, 'mixed_views': MIXED_VIEWS} if thinned else c
    if original['trials'] != MIXED_TRIALS or any(original[k] != comparison[k] for k in shared_keys):
        raise ValueError('frozen parent uses a different data, model or training protocol')
    limits = dict(min_epochs=40,max_epochs=40,max_seconds=7200)
    training = {**c['training'], **limits} if c.get('extended_budget') else c['training']
    policy = MIXED_POLICY if thinned else {**c['policy'], 'fixed_epochs': 40}
    if original['training'] != training or (not ordinary_controls and original['policy'] != policy):
        raise ValueError('frozen parent differs outside the allowed training budget')
    data, indexes = parent/'prepared-cases', parent/'frozen-indexes'
    report = json.loads((data/'data-report.json').read_text())
    for split, expected in c['expected_raw_case_sha256'].items():
        if case_digest(cases(data, split)) != expected:
            raise ValueError(f'parent case data changed: {split}')
    prior = parent/(mixed_parent_trial(c) if not ordinary_controls else PARENT_TRIAL)
    reference = json.loads((prior/'trial-result.json').read_text())
    views = json.loads((prior/'training-view-provenance.json').read_text())
    expected_merge = 64 if c.get('extended_budget') and not ordinary_controls else 120
    if (reference['seed'] != 42 or reference['merge_items'] != expected_merge
            or reference['selection']['training_epochs'] != 40
            or reference['pair_sid_file_sha256'] != c['frozen_parent']['files']['frozen-indexes/pairs-42.npy']['sha256']):
        raise ValueError('frozen parent lacks the completed seed 42 comparison')
    return parent, data, indexes, report, reference, views


def run_trial(trial,p):
    import torch
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_unique4 import attach4,InputIndex4
    from rhpr.gr.profiling import benchmark_training,ProfileDeadline
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=copy.deepcopy(p['config']);c['active_trial']=trial;c['data']['history_items']=128
    seed = trial['seed']
    c['seed'] = seed
    depth_pilot = c.get('depth_pilot')
    expected_init = depth_pilot['initializers'] if depth_pilot else c.get('replication_initializers', {}).get(str(seed), {
        'base': c['expected_initial_state_sha256'], 'pair': c['expected_pair_initial_state_sha256']})
    c['training'].update(physical_batch_size=c['pair_batches']['physical'],evaluation_batch_size=c['pair_batches']['evaluation'])
    if depth_pilot and single_depth_t4_trial(c):
        c['training']['max_seconds'] = DEPTH_T4_TRAIN_SECONDS
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    root=Path(p['indexes']);sids=np.load(root/'items-42.npy');keys=np.load(root/'pair-keys.npy');pairs=np.load(root/'pairs-42.npy')
    index=SidIndex.from_full_sids(sids,width=256);wrapped=InputIndex4(index)
    model=build_trial_model(sids,c,seed,p['source'])
    base=state_digest(dict(model.named_parameters()))
    if base != expected_init['base']: raise ValueError('base parameters differ from CPU preflight')
    before={k:v.detach().clone() for k,v in model.named_parameters()};attach4(model)
    if any(not torch.equal(v,dict(model.named_parameters())[k]) for k,v in before.items()): raise ValueError('pair adapter changed common parameters')
    del before
    initial=state_digest(dict(model.named_parameters()))
    if initial != expected_init['pair']: raise ValueError('pair initialization differs from CPU preflight')
    m = trial['merge_items']
    full_only = trial.get('history_view') == 'recent128'
    thinned = c.get('history_reduction') == 'keep_second' and not full_only
    mean_pairs = c.get('pair_vector_method') == 'mean'

    def build_cases(split):
        h, u, t = cases(p['data'], split)
        x = encode_full(h, sids) if full_only else encode(h, sids, keys, pairs, m, keep_second=thinned)
        if thinned:
            if np.any(x >= 1024) or case_digest((x, u, t)) != c['expected_thinned_case_sha256'][split]:
                raise ValueError(f'thinning produced unexpected inputs in {split}')
        return (h, u, t), (x, u, t)
    splits = ('train', 'validation')
    built = {s:build_cases(s) for s in splits}
    raw = {s:built[s][0] for s in splits}
    encoded = {s:built[s][1] for s in splits}
    if full_only and case_digest(encoded['train']) != p['reference_views']['full_case_sha256']:
        raise ValueError('full training inputs differ from the frozen 128-item history')
    views = None
    if 'mixed_views' in c and not full_only:
        views = full_context_selector(raw['train'][0], sids, encoded['train'][0], c, m)
    out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True)
    if views is not None:
        provenance = dict(
            policy=c['mixed_views'], native_sid_widths=views.widths,
            full_case_sha256=case_digest((views.full, *encoded['train'][1:])))
        compressed_digest = case_digest(encoded['train'])
        if thinned:
            reference = p['reference_views']
            provenance['thinned_case_sha256'] = compressed_digest
            if (provenance['full_case_sha256'] != reference['full_case_sha256']
                    or list(views.widths.values()) != list(reference['native_sid_widths'].values())):
                raise ValueError('full training inputs or widths differ from Mixed64')
        else:
            provenance['pair_case_sha256'] = compressed_digest
            expected = p.get('reference_views', provenance)
            if mean_pairs:
                expected = {**expected, 'pair_case_sha256':compressed_digest}
            if expected != provenance:
                raise ValueError('full inputs or training view protocol differ from the reference')
        _write(out/'training-view-provenance.json', provenance)
    model.eval()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        loss=_loss(model,wrapped.model_batch(*(v[:1] for v in encoded['validation']),'cpu'))
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite CPU preflight')
    forward=dict(parameters=sum(v.numel() for v in model.parameters()),added_pair_parameters=4*256*128,
        encoder_positions=encoded['validation'][0].shape[1]+1,supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='B1 fixed-padded CPU mathSDPA teacher-forced4code forward,FMA2; excludes beam/backward/text encoding')
    if depth_pilot:
        forward['active_parameters'] = forward['parameters'] - forward['added_pair_parameters']
    _write(out/'forward-flops.json',forward);model.to('cuda:0')
    compiled=torch.compile(model,**c['training']['compile_training'])
    profile=benchmark_training(compiled,encoded['train'],sids,np.random.default_rng(42).permutation(48320)[:512],
        c['training']['physical_batch_size'],'fp16','full_pipeline',c['training'],dict(seed=42,warmup=1,iterations=2),
        ProfileDeadline(360 if depth_pilot else 180),'cuda:0',batch_builder=partial(profiling_batch,wrapped))
    if state_digest(dict(model.named_parameters())) != initial:raise RuntimeError('profile changed initial weights')
    _write(out/'training-profile.json',profile);_event('ml1m_pairs_profile_complete',trial=trial['id'],**profile)
    if views is not None:
        full_profile = benchmark_training(
            compiled, (views.full, *encoded['train'][1:]), sids,
            np.random.default_rng(42).permutation(48320)[:512], c['training']['physical_batch_size'],
            'fp16', 'full_pipeline', c['training'], dict(seed=42, warmup=1, iterations=2),
            ProfileDeadline(360 if depth_pilot else 180), 'cuda:0', batch_builder=partial(profiling_batch, wrapped))
        if state_digest(dict(model.named_parameters())) != initial:
            raise RuntimeError('full-view profile changed initial weights')
        _write(out/'training-profile-full.json', full_profile)
        _event('ml1m_pairs_full_profile_complete', trial=trial['id'], **full_profile)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    def builder(root,width,split):
        if split not in encoded:raise ValueError('TEST forbidden during training')
        return encoded[split]
    checkpoint = (partial(snapshot_colab_checkpoint, out, Path(p['output']))
                  if depth_pilot and single_depth_t4_trial(c) else None)
    selection=_train_generator(model,p['data'],wrapped,c,out,'cuda:0',example_builder=builder,
                               training_context_selector=views,after_validation_checkpoint=checkpoint)
    epochs = c['training']['max_epochs']
    if (selection['training_epochs'] != epochs or selection['validation_checks'] != epochs//5
            or selection['optimizer_updates']+selection['amp_skipped_updates'] != 95*epochs):
        raise RuntimeError(f'incomplete {epochs}-epoch run; checkpoints are preserved')
    curve=json.loads((out/'training-curves.json').read_text())
    timing=dict(wall_seconds=time.monotonic()-started,warm_epoch_training_median_seconds=float(np.median([v['train_seconds'] for v in curve[1:]])),
        first_epoch_training_seconds=curve[0]['train_seconds'],cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    (h, u, t), (x, _, _) = build_cases('test')
    checkpoint = out / 'generator-best.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['epoch'] != selection['best_epoch']:raise ValueError('selected checkpoint mismatch')
    model.load_state_dict(ck['model'],strict=True)
    metrics,pred,scores=_evaluate(model,(x,u,t),wrapped,c['training']['evaluation_batch_size'],'cuda:0','fp32')
    np.savez_compressed(out/'test-best-predictions.npz',semantic_ids=pred,scores=scores,user_ids=u,target_item_ids=t)
    test=dict(epoch=selection['best_epoch'],metrics=metrics,checkpoint_sha256=file_sha256(checkpoint),selection='raw validation NDCG@10 only',seen_items_filtered=False)
    _write(out/'test-metrics.json',test)
    row=dict(trial=trial['id'],seed=seed,merge_items=m,protected_tail=128-m,base_initial_state_sha256=base,initial_state_sha256=initial,
        sid_file_sha256=file_sha256(root/'items-42.npy'),pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'),
        raw_case_sha256={s:case_digest(v) for s,v in raw.items()},case_targets_sha256={s:case_digest(v[1:]) for s,v in raw.items()},
        test_case_sha256=case_digest((h,u,t)),selection=selection,forward=forward,timing=timing,profile=profile,test=test,
        epoch_capped_screen=True,training_pair_view='fixed',test_pair_view='fixed_oldest_valid_pairs')
    if views is not None:
        row.update(training_pair_view='batch_mixture', mixed_views=c['mixed_views'],
                   full_training_profile=full_profile,
                   train_view_examples={name: sum(v['training_views']['examples'][name] for v in curve)
                                        for name in views.widths})
    if thinned:
        row.update(history_reduction='keep_second', training_pair_view='batch_mixture_with_thinning',
                   test_pair_view='keep_second_of_oldest_valid_pairs', used_pair_embeddings=False)
    if full_only:
        row.update(training_pair_view='full128', test_pair_view='full128', used_pair_embeddings=False)
    if depth_pilot:
        row.update(history_view=trial['history_view'], encoder_layers=c['generator']['encoder_layers'],
                   training_time_limit_seconds=c['training']['max_seconds'])
    if mean_pairs:
        row['pair_vector_method'] = 'mean'
        row['item_embeddings_sha256'] = c['frozen_parent']['files'][MEAN_ITEM_VECTORS]['sha256']
    if c.get('extended_budget'):
        row['validation_best_within_40'] = val_best_within(curve, 40)
    _write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    from rhpr.gr.ml1m_devices import check_worker
    p=json.loads(Path(plan_path).read_text());validate(p['config']);torch.set_num_threads(1)
    check_worker(p['config'], transformers.__version__)
    status=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('ml1m_pairs_trial_start',trial=trial['id'])
        try:
            row=run_trial(trial,p);status['results'][trial['id']]=row;_event('ml1m_pairs_trial_complete',**row)
        except Exception as error:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc());status['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('ml1m_pairs_trial_failed',**status['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',status)
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()


def prepare_indexes(c, output, source):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.pair_sid import predict_rq
    from rhpr.gr.pair_unique4 import unique_pair_sids
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    resources = resolve_dataset_directory(c['resources']['path'])
    archive = resolve_dataset_directory(c['movie_input']['path'])/'ml-1m-source.bin'
    if file_sha256(archive)!=c['movie_input']['source_zip_sha256']:raise ValueError('raw source changed')
    data=output/'prepared-cases';texts,report=prepare_raw(archive,data,c['data'])
    if {s:v['raw_case_sha256'] for s,v in report['splits'].items()} != c['expected_raw_case_sha256']:raise ValueError('raw cases differ')
    sets=cache_sets(data);keys=np.unique(np.concatenate(list(sets.values())));mask=np.isin(keys,sets['train'])
    if {s:len(k) for s,k in sets.items()} != c['expected_pair_counts'] or len(keys)!=c['expected_pair_union']:raise ValueError('pair schedule differs from preflight')
    cache=output/'pair-cache';cache.mkdir();indexes=output/'frozen-indexes';indexes.mkdir()
    np.save(indexes/'pair-keys.npy',keys);np.save(cache/'train-mask.npy',mask)
    _event('ml1m_pairs_embedding_start',items=len(texts),pair_counts={s:len(v) for s,v in sets.items()},pair_union=len(keys))
    encoder=SentenceTransformer(str(resources/'sentence-t5-base'),device='cuda:0',local_files_only=True);encoder.max_seq_length=256;encoder.eval()
    vec=np.asarray(encoder.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
    np.save(cache/'item-embeddings.npy',vec)
    codes,centers,item_fit=fit_grid_rkmeans(vec,c['quantizer'],42,'cuda:0',source)
    item_index=SidIndex(codes,width=256);np.save(indexes/'items-42.npy',item_index.sids);np.save(indexes/'item-centroids-42.npy',centers)
    if file_sha256(indexes/'items-42.npy') != c['expected_sid_file_sha256']:raise ValueError('single-item index differs from control; no pair embedding or training')
    encoder.max_seq_length=POLICY['pair_max_tokens'];encode_pair_cache(encoder,texts,keys,cache/'pair-embeddings.npy',POLICY['pair_text_labels'],
        batch_size=c['pair_embedding_batch_size'],event_prefix='ml1m_pairs')
    del encoder;gc.collect();torch.cuda.empty_cache()
    pairvec=np.load(cache/'pair-embeddings.npy',mmap_mode='r')
    fitcodes,centers,pair_fit=fit_grid_rkmeans(np.asarray(pairvec[mask]),c['quantizer'],42,'cuda:0',source)
    codes=predict_rq(pairvec,centers,'cuda:0')
    if not np.array_equal(codes[mask],fitcodes):raise ValueError('pair quantizer assignment mismatch')
    # Preserve the fitted index even if the collision-suffix guard rejects it.
    np.save(indexes/'pair-codes-42.npy',codes);np.save(indexes/'pair-centroids-42.npy',centers)
    collision_counts=np.unique(codes,axis=0,return_counts=True)[1]
    _write(indexes/'pair-fit-report.json',dict(fit=pair_fit,semantic_groups=len(collision_counts),max_collision_group=int(collision_counts.max()),pair_union=len(keys)))
    pairs=unique_pair_sids(keys,codes,sets['train'],np.union1d(sets['train'],sets['validation']))
    np.save(indexes/'pairs-42.npy',pairs)
    _write(output/'index-report.json',dict(item=item_fit,pair=pair_fit,pair_counts={s:len(v) for s,v in sets.items()},pair_union=len(keys),
        max_pair_suffix=int(pairs[:,3].max()),unique_pair_sids=len(np.unique(pairs,axis=0)),pair_fit_on_train_only=True,
        suffix_policy=POLICY['suffix'],all_targets_single=True,pair_quantizer_shared_across_variants=True))
    del pairvec,vec,codes,centers,pairs,fitcodes;gc.collect();torch.cuda.empty_cache()
    return data, indexes, report


def prepare_mean_pair_indexes(config, parent, data, original_indexes, output, source):
    """Refit only the pair quantizer on arithmetic means of cached item vectors."""
    import torch
    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.gr.pair_sid import predict_rq
    from rhpr.gr.pair_unique4 import unique_pair_sids

    item_path = parent / MEAN_ITEM_VECTORS
    item_vectors = np.load(item_path, allow_pickle=False)
    if item_vectors.shape != (config['data']['items'], 768) or item_vectors.dtype != np.float32:
        raise ValueError('mean pairs require the original FP32 Sentence-T5 item embeddings')
    if not np.isfinite(item_vectors).all():
        raise ValueError('nonfinite original item embeddings')
    sets = cache_sets(data)
    keys = np.load(original_indexes / 'pair-keys.npy', allow_pickle=False)
    if not np.array_equal(keys,np.unique(np.concatenate(list(sets.values())))):
        raise ValueError('ordered pair universe differs from the concatenated-text index')
    train = np.isin(keys, sets['train'])
    first, second = np.divmod(keys, len(item_vectors))
    means = (item_vectors[first] + item_vectors[second]) * np.float32(.5)
    train_means = means[train]
    fit_codes, centers, fit = fit_grid_rkmeans(train_means,config['quantizer'],42,'cuda:0',source)
    del train_means
    codes = predict_rq(means,centers,'cuda:0')
    del means
    if not np.array_equal(codes[train],fit_codes):
        raise ValueError('mean pair quantizer assignments differ from train fit')

    indexes = output / 'mean-pair-indexes'
    indexes.mkdir()
    for name in ('items-42.npy','pair-keys.npy'):
        shutil.copyfile(original_indexes/name,indexes/name)
    np.save(indexes/'pair-codes-42.npy',codes)
    np.save(indexes/'pair-centroids-42.npy',centers)
    counts = np.unique(codes,axis=0,return_counts=True)[1]
    report = dict(method='arithmetic_mean_of_two_original_fp32_t5_vectors',
                  fit=fit,item_embeddings_sha256=file_sha256(item_path),
                  pair_counts={s:len(ids) for s,ids in sets.items()},pair_union=len(keys),
                  semantic_groups=len(counts),max_collision_group=int(counts.max()),
                  pair_fit_on_train_only=True,collision_suffix='ordered_keys_train_then_validation_then_test')
    _write(output/'mean-pair-index-report.json',report)
    pair_sids = unique_pair_sids(keys,codes,sets['train'],np.union1d(sets['train'],sets['validation']))
    pair_path = indexes/'pairs-42.npy'
    np.save(pair_path,pair_sids)
    report.update(pair_sid_file_sha256=file_sha256(pair_path),max_pair_suffix=int(pair_sids[:,3].max()))
    _write(output/'mean-pair-index-report.json',report)
    _event('ml1m_pairs_mean_index_complete',**report)
    torch.cuda.empty_cache()
    return indexes,report


def depth_paired_summary(rows, output, data, indexes):
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

    histories, users, targets = cases(data, 'test')
    lengths = (histories >= 0).sum(axis=1)
    index = SidIndex.from_full_sids(np.load(indexes / 'items-42.npy'), width=256)
    cohorts = {'all': lengths >= 0, 'at_least_128': lengths >= 128}
    summary = {name: {'users': int(selected.sum())} for name, selected in cohorts.items()}
    for row in rows:
        view = row['history_view']
        with np.load(output / 'trials' / row['trial'] / 'test-best-predictions.npz') as saved:
            np.testing.assert_array_equal(users, saved['user_ids'])
            np.testing.assert_array_equal(targets, saved['target_item_ids'])
            predictions = saved['semantic_ids']
            for name, selected in cohorts.items():
                metrics = evaluate_ranked_sids(index, predictions[selected], targets[selected])
                summary[name][view] = {key: metrics[key] for key in ('ndcg@10', 'recall@10')}
        if not np.isclose(summary['all'][view]['ndcg@10'], row['test']['metrics']['ndcg@10']):
            raise ValueError('depth pilot predictions do not replay the saved score')
    for result in summary.values():
        if 'recent128' in result and 'thinned64' in result:
            result['recent_minus_thinned_ndcg10'] = result['recent128']['ndcg@10'] - result['thinned64']['ndcg@10']
    return summary


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ml1m_devices import assignments_for, devices_for
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir()
    torch.set_num_threads(1)
    devices = devices_for(c)
    roots={name:resolve_dataset_directory(c[name]['path']) for name in ('resources','modern_runtime')}
    for name in roots:
        prefixes = c[name].get('include_prefixes', ()) if c.get('depth_pilot') and name == 'resources' else ()
        verify_resources(roots[name], c[name]['manifest_sha256'], include_prefixes=prefixes)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    extra = {}
    if 'frozen_parent' in c:
        parent, data, indexes, report, reference, views = frozen_inputs(
            depth_pilot_reference(c) if c.get('depth_pilot') else c)
        extra = dict(reference_views=views, reference_mixed42=reference)
        _write(output/'index-report.json', json.loads((parent/'index-report.json').read_text()))
        _write(output/'frozen-parent-verification.json', c['frozen_parent'])
        _event('ml1m_pairs_frozen_parent_verified', kernel=c['frozen_parent']['kernel'],
               revision=c['frozen_parent']['git_sha'], files=len(c['frozen_parent']['files']))
        if c.get('pair_vector_method') == 'mean':
            indexes, mean_report = prepare_mean_pair_indexes(c, parent, data, indexes, output, source)
            extra['mean_pair_index'] = mean_report
    else:
        data, indexes, report = prepare_indexes(c, output, source)
    _write(output/'reference-control.json',c['reference_control'])
    overlay=scratch/'modern-site';overlay.mkdir()
    with (output/'modern-install.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes',
            '--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    sys.path.insert(0,str(overlay));p=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=assignments_for(c, devices),**extra)
    path=output/'plan.json';_write(path,p);summary=launch_workers(path,worker_module='rhpr.gr.ml1m_last8_pairs',event_prefix='ml1m_pairs')
    rows=list(summary['results'].values())
    shared = ('sid_file_sha256','pair_sid_file_sha256','raw_case_sha256','case_targets_sha256','test_case_sha256')
    if extra:
        keys = shared
        anchor = extra['reference_mixed42']
        if 'mean_pair_index' in extra:
            anchor = {**anchor, 'pair_sid_file_sha256':extra['mean_pair_index']['pair_sid_file_sha256']}
        if c.get('depth_pilot'):
            positions = {row['history_view']: row['forward']['encoder_positions'] for row in rows}
            registered_positions = dict(recent128=513, thinned64=385)
            expected_positions = {trial['history_view']: registered_positions[trial['history_view']]
                                  for trial in c['trials']}
            if positions != expected_positions:
                raise ValueError('depth pilot encoder widths differ from the registered histories')
            extra['depth_comparison'] = depth_paired_summary(rows, output, data, indexes)
        else:
            reference_forward = anchor['forward']
            flop_key = 'supported_forward_flops_batch1'
            for row in rows:
                profile = row['forward']
                if (any(profile[k] != reference_forward[k] for k in profile if k != flop_key)
                        or abs(profile[flop_key] / reference_forward[flop_key] - 1) > 1e-6):
                    raise ValueError('forward dimensions or FLOPs differ from the reference')
    else:
        keys = (*shared, 'base_initial_state_sha256', 'initial_state_sha256')
        anchor = rows[0]
    for key in keys:
        if any(r[key]!=anchor[key] for r in rows):raise ValueError('unmatched pair comparison: '+key)
    _write(output/'study-summary.json',dict(results=summary['results'],data=report,reference_control=c['reference_control'],
        new_generator_fits=len(rows),epoch_capped_screen=True,**extra))
    metrics=dict(experiment=c['name'],models=len(rows),epochs=c['training']['max_epochs'],test_evaluated=True)
    if not extra:
        metrics['seed'] = 42
    for name,row in summary['results'].items():
        metrics[name+'_validation_ndcg10']=row['selection']['validation_ndcg10'];metrics[name+'_test_ndcg10']=row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops']=row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True)
    a=p.parse_args();worker(a.plan,a.slot)
