"""Verified continuation of unique-pair models to val-NDCG plateau."""
from __future__ import annotations
import argparse,copy,gc,json,random,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_evaluate,_train_generator,_write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_sid import preencode
from rhpr.gr.pair_recent10 import recent_preencode
from rhpr.gr.pair_test import locate_parent
from rhpr.gr.pair_unique4 import TRIALS,InputIndex4,attach4,pair_preencode


def validate(c):
    t=c['training']
    if (c['kind']!='gr_ml1m_pair_unique4_plateau' or c['trials']!=TRIALS or c['seeds']!=[42,101,202]
        or t['max_epochs']!=120 or t['min_epochs']!=55 or t['patience_epochs']!=30 or t['validation_interval']!=5
        or t['precision']!='fp16' or t['evaluation_precision']!='fp32' or 'scheduler' in t
        or not c['evaluation']['test_after_selection'] or c['resume_epoch']!=50):raise ValueError('invalid plateau continuation protocol')


def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=copy.deepcopy(p['config']);seed=trial['seed'];mode=trial['mode'];c['seed']=seed;out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    idx=SidIndex.from_full_sids(np.load(Path(p['parents']['pair'])/'frozen-indexes'/f'items-{seed}.npy'),width=256)
    src=Path(p['parents']['continuation']);keys=np.load(src/'frozen-pairs/pair-keys.npy') if mode=='pair' else None;codes=np.load(src/'frozen-pairs'/f'pairs-{seed}.npy') if mode=='pair' else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=attach4(build_grid_generator(idx.sids,c['generator'],seed,'cuda:0',p['source']));initial=state_digest(dict(model.named_parameters()))
    original=json.loads((src/'study-summary.json').read_text())['results'][trial['id']]
    if initial!=original['initial_state_sha256']:raise RuntimeError('initialization differs from source model')
    wrapped=InputIndex4(idx)
    def builder(root,width,split):
        h,u,t=case_examples(root,width,split)
        if mode=='pair':x=pair_preencode(h,idx.sids,keys,codes)
        elif mode=='recent15':x=recent_preencode(h,idx.sids,recent_items=15)
        else:x=preencode(h,idx.sids)
        return x,u,t
    selection=_train_generator(model,p['data_root'],wrapped,c,out,'cuda:0',example_builder=builder,resume_from=src/'trials'/trial['id'])
    curves=json.loads((out/'training-curves.json').read_text());verification=json.loads((out/'resume-verification.json').read_text())
    if verification['source_epoch']!=50 or not verification['rng_restored_and_verified'] or selection['training_epochs']<55 or selection['optimizer_updates']+selection['amp_skipped_updates']!=selection['training_epochs']*296:raise RuntimeError('continuation verification failed')
    examples=builder(p['data_root'],20,'test');test={}
    for which in ('last','best'):
        ck=torch.load(out/f'generator-{which}.pt',map_location='cpu',weights_only=False);epoch=selection['training_epochs'] if which=='last' else selection['best_epoch']
        if ck['epoch']!=epoch:raise ValueError('checkpoint epoch mismatch')
        model.load_state_dict(ck['model'],strict=True);metrics,pred,scores=_evaluate(model,examples,wrapped,32,'cuda:0','fp32')
        np.savez_compressed(out/f'test-{which}-predictions.npz',semantic_ids=pred,scores=scores,user_ids=examples[1],target_item_ids=examples[2]);test[which]=dict(epoch=epoch,metrics=metrics)
    row=dict(trial=trial['id'],mode=mode,seed=seed,initial_state_sha256=initial,selection=selection,test=test,epochs={str(r['epoch']):r['validation']['ndcg@10'] for r in curves if 'validation'in r},source_epoch=50)
    _write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('worker requires T4 and HF 5.22.1')
    torch.set_num_threads(1);state=dict(results={},failures=[]);path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('unique4_plateau_trial_start',trial=trial['id'],worker=slot)
        try:row=run_trial(trial,p);state['results'][trial['id']]=row;_event('unique4_plateau_trial_complete',**row)
        except Exception as e:
            root=Path(p['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True);(root/'failure.log').write_text(traceback.format_exc());state['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)));_event('unique4_plateau_trial_failed',**state['failures'][-1])
        _write(path,state);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('requires two T4 GPUs')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation','test'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('case split differs')
    parents={k:locate_parent(v) for k,v in c['parents'].items()};original=json.loads((parents['continuation']/'run-manifest.json').read_text())['config']
    for name in ('generator','data','trials','seeds'):
        if c[name]!=original[name]:raise ValueError('source protocol differs: '+name)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True);source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),parents={k:str(v) for k,v in parents.items()},assignments=assignments(c,2));_write(output/'plan.json',plan);sys.path.insert(0,str(overlay));gc.collect();torch.cuda.empty_cache();_event('unique4_plateau_suite_start',trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_unique4_plateau',event_prefix='unique4_plateau');records=summary['results']
    _write(output/'study-summary.json',dict(results=records,resumed_from=50,test_evaluated=True));return dict(experiment=c['name'],models=len(records),source_epoch=50,test_evaluated=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--slot',type=int,required=True);a=ap.parse_args();worker(a.plan,a.slot)
