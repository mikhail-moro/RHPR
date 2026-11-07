"""Frozen-SID narrow mini-TIGER fits for the ML1M stability experiment.

The orchestrator prepares/pins the six indexes and writes a plan with two
worker assignments.  This module never fits a tokenizer or opens TEST.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples


def validate_training_protocol(c):
    """Fail closed rather than silently changing the authorized training."""
    t = c['training']; g = c['generator']; e = c['evaluation']
    expected_training = dict(min_epochs=30, max_epochs=60, patience_epochs=20,
        validation_interval=10, validate_first_epoch=False, batch_size=512,
        physical_batch_size=512, precision='fp16', evaluation_precision='fp32',
        learning_rate=.001, weight_decay=.0001, grad_clip_norm=1.0,
        evaluation_batch_size=32)
    expected_generator = dict(encoder_layers=4, decoder_layers=4, d_model=64,
        d_ff=512, d_kv=32, num_heads=6, dropout=.1, attn_implementation='sdpa',
        beam_width=20, should_check_prefix=False, num_user_bins=2000,
        codebook_size=256)
    if (any(t.get(k) != v for k,v in expected_training.items())
            or any(g.get(k) != v for k,v in expected_generator.items())
            or c['data']['history_items'] != 20 or c['data']['train_examples'] != 151166
            or c['data']['users'] != 2786 or c['data']['items'] != 2874
            or 'scheduler' in t or t['max_seconds'] < 3200
            or e.get('save_validation_predictions') is not True
            or e.get('save_validation_losses') is not True
            or e.get('test_after_selection') is not False):
        raise ValueError('stability generator protocol differs from authorized 30/60/10/20 experiment')
    seeds = [31415, 27182, 161803]
    if c['generator_seeds'] != seeds or set(c['initialization_pins']) != {'narrow'}:
        raise ValueError('three prespecified generator seeds and narrow pins required')
    pins = c['initialization_pins']['narrow']
    if set(pins) != {str(s) for s in seeds} or any(
            len(p) != 64 or any(x not in '0123456789abcdef' for x in p) for p in pins.values()):
        raise ValueError('all narrow generator initial states must be pinned')
    if c['execution']['worker_devices'] != ['cuda:0','cuda:1'] or c['execution']['cpu_threads_per_worker'] != 1:
        raise ValueError('two isolated single-threaded T4 workers required')
    expected = {(i,s) for i in c['index_ids'] for s in seeds}
    actual = [(r['index_id'],r['seed']) for r in c['trials']]
    if (len(c['index_ids']) != 6 or len(set(c['index_ids'])) != 6
            or len(actual) != 18 or set(actual) != expected
            or len({r['id'] for r in c['trials']}) != 18
            or any(r.get('architecture') != 'narrow' for r in c['trials'])
            or any(not i or not i.replace('_','').isalnum() for i in c['index_ids'])
            or any(not r['id'] or not r['id'].replace('_','').isalnum() for r in c['trials'])):
        raise ValueError('stability requires exactly six fixed indexes x three seeds')


def validate_completion(selection, curves, c, output=None):
    """Check real epoch budget, counters, selected checkpoint, and saved VAL."""
    t = c['training']; epoch = selection['training_epochs']; best = selection['best_epoch']
    stop = selection['stopping_reason']; skipped = selection['amp_skipped_updates']
    if (epoch not in [30,40,50,60] or best not in range(10,epoch+1,10)
            or not 0 <= skipped <= 16
            or selection['optimizer_updates'] + skipped != 296*epoch
            or selection['training_examples_per_epoch'] != 151166
            or selection['training_precision'] != 'fp16' or selection['evaluation_precision'] != 'fp32'
            or selection['validation_checks'] != epoch//10
            or stop not in ['early_stopping','epoch_limit']
            or (stop == 'early_stopping' and epoch-best < t['patience_epochs'])
            or (stop == 'epoch_limit' and (epoch != 60 or epoch-best >= t['patience_epochs']))):
        raise RuntimeError('incomplete fit or inconsistent stopping rule')
    if len(curves) != epoch or [r['epoch'] for r in curves] != list(range(1,epoch+1)):
        raise RuntimeError('training curves do not cover completed epochs')
    vals = []
    for r in curves:
        if (r['updates'] + r['amp_skipped_updates'] != 296*r['epoch']
                or r['train_examples'] != 151166
                or not np.isfinite(r['train_loss_sum_over_codes'])):
            raise RuntimeError('nonfinite loss or inconsistent per-epoch AMP counters')
        if 'validation' in r:
            if (r['epoch'] % 10 or 'validation_loss' not in r
                    or not np.isfinite(r['validation_loss']['total_ce'])
                    or not np.isfinite(r['validation']['ndcg@10'])):
                raise RuntimeError('invalid validation curve')
            vals.append(r)
    if [r['epoch'] for r in vals] != list(range(10,epoch+1,10)):
        raise RuntimeError('validation must run exactly every ten epochs')
    chosen = max(vals, key=lambda r:r['validation']['ndcg@10'])
    if chosen['epoch'] != best or chosen['validation']['ndcg@10'] != selection['validation_ndcg10']:
        raise RuntimeError('selected validation checkpoint is inconsistent')
    if output is not None:
        output = Path(output)
        for name in ['generator-best.pt','generator-last.pt','generator.json','training-selection.json']:
            if not (output/name).is_file(): raise RuntimeError('missing checkpoint or provenance: '+name)
        for r in vals:
            if not (output/f"validation-epoch-{r['epoch']:03d}.npz").is_file():
                raise RuntimeError('missing saved validation predictions')
    return dict(plateau_reached=stop == 'early_stopping', budget_censored=stop == 'epoch_limit')


def run_trial(trial, c, plan, device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    c = copy.deepcopy(c); seed = trial['seed']; c['seed'] = seed
    root = Path(plan['output'])/'trials'/trial['id']; root.mkdir(parents=True, exist_ok=True)
    sid_path = Path(plan['index_root'])/(trial['index_id']+'-semantic-ids.npy')
    sid_hash = file_sha256(sid_path)
    if sid_hash != plan['references'][trial['index_id']]['sids_sha256']:
        raise ValueError('frozen SID bytes changed')
    index = SidIndex.from_full_sids(np.load(sid_path,allow_pickle=False),width=256)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_grid_generator(index.sids,c['generator'],seed,device,plan['source'])
    initial = state_digest(dict(model.named_parameters()))
    if initial != c['initialization_pins']['narrow'][str(seed)]:
        raise ValueError('initial generator state differs from pinned seed')
    _write(root/'generator.json',dict(config=c['generator'],seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,
        sids_sha256=sid_hash,frozen_index=True,test_evaluated=False))
    selection = _train_generator(model,plan['data_root'],index,c,root,device,example_builder=case_examples)
    curves = json.loads((root/'training-curves.json').read_text())
    flags = validate_completion(selection,curves,c,root)
    row = dict(trial=trial['id'],architecture='narrow',index_id=trial['index_id'],seed=seed,
        initial_state_sha256=initial,initialization_group=f'narrow_{seed}',sids_sha256=sid_hash,
        selection=selection,epochs={str(r['epoch']):r['validation']['ndcg@10']
                                    for r in curves if 'validation' in r},**flags)
    _write(root/'trial-result.json',row)
    return row


def run_worker(plan_path, slot):
    import torch
    import transformers
    plan = json.loads(Path(plan_path).read_text()); c = plan['config']
    validate_training_protocol(c)
    if (transformers.__version__ != '5.22.1' or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability() != (7,5)):
        raise RuntimeError('stability worker requires native HF5.22.1 and exactly one T4')
    torch.set_num_threads(1)
    status = dict(results={},failures=[])
    path = Path(plan['output'])/'workers'/f'worker-{slot}.json'
    for trial in plan['assignments'][slot]['trials']:
        _event('stability_trial_start',trial=trial['id'],worker=slot)
        try:
            if time.time() > plan['deadline']:
                raise TimeoutError('suite deadline reached; no hidden extensions')
            row = run_trial(trial,c,plan,'cuda:0')
            status['results'][trial['id']] = row
            _event('stability_trial_complete',**row)
        except Exception as error:
            root = Path(plan['output'])/'trials'/trial['id']; root.mkdir(parents=True,exist_ok=True)
            (root/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('stability_trial_failed',**status['failures'][-1]); _write(path,status)
            return
        _write(path,status); gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True)
    a = p.parse_args();run_worker(a.plan,a.slot)
