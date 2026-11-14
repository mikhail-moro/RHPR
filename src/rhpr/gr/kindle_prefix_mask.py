"""Frozen Pair32/Recent32 prefix-mask intervention, preserving every position."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _evaluate, _event, _write

TRIALS=[dict(id=m,model=m,seed=42) for m in ('pair32','recent32')]
VARIANTS=['full','masked_old']


def validate(c):
    if (c['kind']!='gr_kindle_prefix_mask' or c['trials']!=TRIALS or c['seed']!=42
        or c['training']!=dict(device='cuda',cpu_threads=1,evaluation_batch_size=64,evaluation_precision='fp32')
        or c['evaluation']!=dict(splits=['validation','test'],variants=VARIANTS,selected_epoch=55,
            seen_items_filtered=False,training_forbidden=True,recent_popularity_cutoff=2.300820444821813)
        or c['execution']!=dict(worker_devices=['cuda:0','cuda:1'],cpu_threads_per_worker=1)
        or set(c['parents'])!={'pair','control'}
        or c['parents']['pair']['kernel']!='carasik/rhpr-kindle-long-pairs'
        or c['parents']['pair']['git_sha']!='33754d70acc0e4cf826f139202632bf0ffe6ac5b'
        or c['parents']['control']['kernel']!='carasik/rhpr-kindle-capacity-pilot'
        or c['parents']['control']['git_sha']!='6d2919a155893e4c6bbb859a1fbcc0c800573c6c'
        or c['kaggle']['kernel_sources']!=[c['parents'][k]['kernel'] for k in ('pair','control')]):
        raise ValueError('expected only frozen epoch55 Pair32/Recent32 full/masked VAL/TEST')


def mask_old(tokens,model):
    """Mark old keys invisible via existing padding mask; never compact the tail."""
    widths={'pair32':(96,32),'recent32':(128,64)}
    if model not in widths:raise ValueError('unknown model')
    total,old=widths[model];x=np.asarray(tokens)
    if x.ndim!=2 or x.shape[1]!=total or np.any(x<0):raise ValueError('full32 evaluation inputs required')
    out=x.copy();out[:,:old]=-1
    assert np.array_equal(out[:,old:],x[:,old:]) and out.shape==x.shape
    return out


def load_frozen_model(c,model,parents,source,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.pair_unique4 import attach4
    pair=Path(parents['pair']);parent=pair if model=='pair32'else Path(parents['control'])
    ckpath=parent/'trials'/(model+'_wide_42')/'generator-best.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if ck['epoch']!=55 or ck['config']['generator']!=c['generator'] or ck['config']['seed']!=42:
        raise ValueError('selected checkpoint provenance differs')
    sids=np.load(pair/'frozen-indexes/items-42.npy')
    net=attach4(build_grid_generator(sids,c['generator'],42,'cpu',source))
    net.load_state_dict(ck['model'],strict=True);net.requires_grad_(False);net.eval();net.to(device)
    return net,ckpath


def inputs(c,model,parents,data,split):
    from rhpr.gr.kindle_pairs import encode
    from rhpr.gr.kindle_windows import encode_window
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_unique4 import InputIndex4
    from rhpr.gr.sid import SidIndex
    root=Path(parents['pair']);sid=np.load(root/'frozen-indexes/items-42.npy')
    idx=SidIndex.from_full_sids(sid,width=256);wrapped=InputIndex4(idx)
    h,u,t=cases(data,split)
    if len(t)!=958 or not np.all(h[:,-32:]>=0):raise ValueError('expected full32 heldout windows')
    if model=='pair32':
        x=encode(h,sid,np.load(root/'frozen-indexes/pair-keys.npy'),np.load(root/'frozen-indexes/pairs-42.npy'),32)
    else:x=encode_window(h,sid,32)
    return (x,u,t),wrapped


def original_prediction(parents,model,split):
    root=Path(parents['pair'])
    if split=='validation':
        folder='trials'if model=='pair32'else'baselines'
        return root/folder/(model+'_wide_42')/'validation-epoch-055.npz'
    folder='trials'if model=='pair32'else'control-tests'
    return root/folder/(model+'_wide_42')/'test-best-predictions.npz'


def run_trial(trial,p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.sid_study import state_digest
    c=p['config'];name=trial['model'];out=Path(p['output'])/'trials'/name;out.mkdir(parents=True)
    model,ckpath=load_frozen_model(c,name,p['parents'],p['source'],'cuda:0')
    before=state_digest(model.state_dict());results={}
    _event('prefix_mask_checkpoint_loaded',model=name,epoch=55)
    for split in c['evaluation']['splits']:
        examples,wrapped=inputs(c,name,p['parents'],p['data'],split);x,u,t=examples
        results[split]={}
        for variant in VARIANTS:
            tokens=x if variant=='full'else mask_old(x,name)
            metrics,pred,scores=_evaluate(model,(tokens,u,t),wrapped,64,'cuda:0','fp32')
            replay={}
            if variant=='full':
                original=original_prediction(p['parents'],name,split)
                with np.load(original)as z:
                    np.testing.assert_array_equal(u,z['user_ids']);np.testing.assert_array_equal(t,z['target_item_ids'])
                    np.testing.assert_array_equal(pred,z['semantic_ids'])
                    replay=dict(all_ranked_sids_identical=True,max_abs_score_difference=float(np.max(np.abs(scores-z['scores']))))
            np.savez_compressed(out/f'{split}-{variant}.npz',semantic_ids=pred,scores=scores,user_ids=u,target_item_ids=t)
            results[split][variant]=dict(metrics=metrics,case_sha256=case_digest((tokens,u,t)),
                original_case_sha256=case_digest(examples),encoder_positions=x.shape[1]+1,
                visible_sid_positions=int((tokens[0]>=0).sum()),replay=replay)
            _event('prefix_mask_evaluation_complete',model=name,split=split,variant=variant,metrics=metrics)
    if state_digest(model.state_dict())!=before or any(v.requires_grad or v.grad is not None for v in model.parameters()):
        raise RuntimeError('frozen inference changed weights or enabled gradients')
    row=dict(trial=name,seed=42,selected_epoch=55,checkpoint_sha256=file_sha256(ckpath),loaded_state_sha256=before,
        weights_unchanged=True,training_performed=False,results=results)
    _write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config']);torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    state=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('prefix_mask_trial_start',trial=trial['id'])
        try:
            row=run_trial(trial,p);state['results'][trial['id']]=row;_event('prefix_mask_trial_complete',**row)
        except Exception as error:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('prefix_mask_trial_failed',**state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',state);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):
        raise RuntimeError('requires two T4s; no fallback')
    torch.set_num_threads(1);parents={k:locate_parent(v)for k,v in c['parents'].items()}
    for root in parents.values():
        old=json.loads((root/'run-manifest.json').read_text())['config']
        for key in ('generator','data','movie_input'):
            if c[key]!=old[key]:raise ValueError('parent differs: '+key)
    if file_sha256(parents['pair']/'frozen-indexes/items-42.npy')!=file_sha256(parents['control']/'frozen-indexes/items-42.npy'):
        raise ValueError('item SID differs')
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,sha in c['movie_input']['sha256'].items():
        if file_sha256(data/name)!=sha:raise ValueError('input hash differs: '+name)
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:',
        '--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),parents={k:str(v)for k,v in parents.items()},assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_prefix_mask',event_prefix='prefix_mask')
    _write(output/'study-summary.json',dict(results=summary['results'],training_performed=False,tokenizer_refitted=False,encoder_positions_preserved=True))
    metrics=dict(experiment=c['name'],models=2,evaluations=8,training_performed=False)
    for model,row in summary['results'].items():
        for split,variants in row['results'].items():
            for variant,record in variants.items():metrics[f'{model}_{split}_{variant}_ndcg10']=record['metrics']['ndcg@10']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True)
    a=p.parse_args();worker(a.plan,a.slot)
