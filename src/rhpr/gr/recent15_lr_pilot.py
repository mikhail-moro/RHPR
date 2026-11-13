"""One fixed-index Recent15 LR pilot with val-loss logging."""
from __future__ import annotations
import argparse,gc,json,random,subprocess,sys
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_evaluate,_train_generator,_write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_recent10 import recent_preencode
from rhpr.gr.pair_test import locate_parent
from rhpr.gr.pair_unique4 import InputIndex4,attach4


def validate(c):
    t=c['training'];s=t['scheduler']
    if (c['kind']!='gr_ml1m_recent15_lr_pilot' or c['seed']!=42 or t['learning_rate']!=.002
        or t['max_epochs']!=100 or t['min_epochs']!=30 or t['patience_epochs']!=30 or t['validation_interval']!=5
        or t['precision']!='fp16' or t['batch_size']!=512 or t['physical_batch_size']!=512
        or s!={'kind':'reduce_on_plateau','factor':.5,'patience_epochs':3,'min_lr':.0001,'threshold':0.,'start_epoch':5}
        or not c['evaluation']['save_validation_losses'] or not c['evaluation']['test_after_selection']):raise ValueError('invalid LR pilot protocol')


def worker(plan_path):
    import torch,transformers
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    p=json.loads(Path(plan_path).read_text());c=p['config'];validate(c)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('worker requires T4 and HF 5.22.1')
    torch.set_num_threads(1);parent=Path(p['parents']['pair']);source=Path(p['parents']['source']);idx=SidIndex.from_full_sids(np.load(parent/'frozen-indexes/items-42.npy'),width=256)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=attach4(build_grid_generator(idx.sids,c['generator'],42,'cuda:0',p['grid']));initial=state_digest(dict(model.named_parameters()));old=json.loads((source/'study-summary.json').read_text())['results']['recent15_42']
    if initial!=old['initial_state_sha256']:raise RuntimeError('pilot initialization differs from control')
    def builder(root,width,split):
        h,u,t=case_examples(root,width,split);return recent_preencode(h,idx.sids,recent_items=15),u,t
    out=Path(p['output'])/'pilot';out.mkdir(parents=True,exist_ok=True);wrapped=InputIndex4(idx)
    _event('lr_pilot_training_start',lr=.002,seed=42)
    selection=_train_generator(model,p['data_root'],wrapped,c,out,'cuda:0',example_builder=builder)
    rows=json.loads((out/'training-curves.json').read_text());epochs=selection['training_epochs'];checks=[r for r in rows if 'validation' in r]
    if epochs<30 or any('validation_loss' not in r for r in checks) or selection['optimizer_updates']+selection['amp_skipped_updates']!=epochs*296:raise RuntimeError('incomplete pilot')
    examples=builder(p['data_root'],20,'test');test={}
    for kind in ('last','best'):
        ck=torch.load(out/f'generator-{kind}.pt',map_location='cpu',weights_only=False);epoch=epochs if kind=='last' else selection['best_epoch']
        if ck['epoch']!=epoch:raise ValueError('checkpoint mismatch')
        model.load_state_dict(ck['model'],strict=True);metrics,pred,scores=_evaluate(model,examples,wrapped,32,'cuda:0','fp32');np.savez_compressed(out/f'test-{kind}-predictions.npz',semantic_ids=pred,scores=scores,user_ids=examples[1],target_item_ids=examples[2]);test[kind]=dict(epoch=epoch,metrics=metrics)
    result=dict(seed=42,initial_state_sha256=initial,selection=selection,test=test,control_kernel='rhpr-ml1m-pair-unique4-plateau',validation_loss_logged=True)
    _write(out/'trial-result.json',result);_event('lr_pilot_complete',**result)


def run(c,output):
    import torch
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if torch.cuda.device_count()<1 or torch.cuda.get_device_capability(0)!=(7,5):raise RuntimeError('requires a T4')
    roots={}
    for name in ('resources','movie_input','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation','test'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('case split differs')
    parents={k:locate_parent(v) for k,v in c['parents'].items()};old=json.loads((parents['source']/'run-manifest.json').read_text())['config']
    for k in ('generator','data'):
        if c[k]!=old[k]:raise ValueError('control differs: '+k)
    for k in ('batch_size','physical_batch_size','precision','evaluation_precision','weight_decay','grad_clip_norm','sampling'):
        if c['training'][k]!=old['training'][k]:raise ValueError('control training differs: '+k)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True);grid=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid');overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),grid=str(grid),data_root=str(roots['movie_input']),parents={k:str(v) for k,v in parents.items()});_write(output/'plan.json',plan);sys.path.insert(0,str(overlay));gc.collect();torch.cuda.empty_cache()
    env=worker_environment(c,{'device':'cuda:0'});subprocess.run([sys.executable,'-u','-m','rhpr.gr.recent15_lr_pilot','--plan',str(output/'plan.json')],env=env,check=True,timeout=5100)
    result=json.loads((output/'pilot/trial-result.json').read_text());_write(output/'study-summary.json',result)
    return dict(experiment=c['name'],models=1,epochs=result['selection']['training_epochs'],test_evaluated=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();worker(a.plan)
