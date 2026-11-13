"""Two fresh Recent20 LR schedules on a pinned Kindle index; VAL only."""
from __future__ import annotations
import argparse,copy,gc,json,random,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_train_generator,_write

TRIALS=[dict(id='cosine_003',seed=42,peak_lr=.003,min_lr=.0003),
        dict(id='cosine_010',seed=42,peak_lr=.01,min_lr=.001)]
EMA_TRIALS=[dict(id='cosine_003_floor001_ema',seed=42,peak_lr=.003,min_lr=.001)]


def validate(c):
    t=c['training'];s=t['scheduler']
    ema=c['kind']=='gr_kindle_lr_ema_pilot'
    if (c['kind']not in ('gr_kindle_lr_pilot','gr_kindle_lr_ema_pilot') or c['trials']!=(EMA_TRIALS if ema else TRIALS) or c['seed']!=42
            or c['data']!={'history_items':20,'users':2625,'items':4969,'train_examples':92133,
                'split':'positive4plus_popular_catalogue_active_users_leave_last_two_timestamp_asin'}
            or c['evaluation']['test_after_selection'] is not False
            or c['execution']['worker_devices']!=['cuda:0','cuda:1']
            or (t['min_epochs'],t['max_epochs'],t['validation_interval'])!=((40,60,5)if ema else (60,60,5))
            or s!={'kind':'warmup_cosine','warmup_epochs':2,'total_epochs':60,'min_lr':.001 if ema else .0003}
            or t['learning_rate']!=.003 or t['precision']!='fp16'
            or t['evaluation_precision']!='fp32' or t['batch_size']!=512
            or t['physical_batch_size']!=512 or t['weight_decay']!=.0001
            or t['grad_clip_norm']!=1. or c['generator']['attn_implementation']!='sdpa'):
        raise ValueError('Kindle LR pilot differs from authorized two-arm protocol')
    if ema and (t.get('early_stopping')!={'kind':'ema','alpha':.5,'min_delta':.0005} or t['patience_epochs']!=15):
        raise ValueError('EMA pilot differs from declared stopping policy')


def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.pair_unique4 import attach4,InputIndex4
    from rhpr.gr.pair_sid import preencode
    from rhpr.gr.ml100k_pair_raw import cases
    c=copy.deepcopy(p['config']);c['training']['learning_rate']=trial['peak_lr']
    c['training']['scheduler']['min_lr']=trial['min_lr']
    seed=trial['seed'];device='cuda:0' if c['training']['device']=='cuda' else 'cpu'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    sid=np.load(Path(p['indexes'])/'items-42.npy',allow_pickle=False)
    idx=SidIndex.from_full_sids(sid,width=256)
    model=attach4(build_grid_generator(idx.sids,c['generator'],seed,device,p['source']))
    digest=state_digest(dict(model.named_parameters()))
    if digest!=c['expected_initial_state_sha256']:raise RuntimeError('initialization differs from fixed-LR control')
    wrapped=InputIndex4(idx);out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    def builder(root,width,split):
        if split not in ('train','validation'):raise ValueError('TEST is closed for LR tuning')
        h,u,t=cases(root,split);return preencode(h,idx.sids),u,t
    selection=_train_generator(model,p['data'],wrapped,c,out,device,example_builder=builder)
    epochs=selection['training_epochs'];ema=c['kind']=='gr_kindle_lr_ema_pilot'
    if (not ema and epochs!=60)or (ema and not 40<=epochs<=60)or selection['validation_checks']!=epochs//5:
        raise RuntimeError('LR pilot did not complete its declared epoch budget')
    if selection['optimizer_updates']+selection['amp_skipped_updates']!=180*epochs:
        raise RuntimeError('LR pilot update/AMP accounting differs')
    row=dict(trial=trial['id'],seed=seed,initial_state_sha256=digest,selection=selection,
        peak_lr=trial['peak_lr'],min_lr=trial['min_lr'],test_evaluated=False)
    _write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':
        raise RuntimeError('requires one T4 and HF5.22.1')
    state=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('kindle_lr_trial_start',trial=trial['id'])
        try:
            row=run_trial(trial,p);state['results'][trial['id']]=row;_event('kindle_lr_trial_complete',**row)
        except Exception as e:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)))
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',state);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.pair_test import locate_parent
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):
        raise RuntimeError('requires two T4 GPUs; no fallback')
    torch.set_num_threads(1);parent=locate_parent(c['parent'])
    previous=json.loads((parent/'run-manifest.json').read_text())['config']
    for key in ('generator','data'):
        if c[key]!=previous[key]:raise ValueError('changed parent '+key)
    for key in ('weight_decay','batch_size','physical_batch_size','precision','evaluation_precision','grad_clip_norm','sampling'):
        if c['training'][key]!=previous['training'][key]:raise ValueError('changed parent training.'+key)
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if name=='test-cases.npz':continue
        if file_sha256(data/name)!=digest:raise ValueError('input digest mismatch: '+name)
    indexes=output/'frozen-indexes';indexes.mkdir();shutil.copy2(parent/'frozen-indexes/items-42.npy',indexes/'items-42.npy')
    # Only the old validation trajectory is copied; no old/new TEST predictions are read.
    shutil.copy2(parent/'trials/control_42/training-curves.json',output/'fixed-lr-reference-curves.json')
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
         '--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    jobs=assignments(c,2) if len(c['trials'])==2 else [dict(slot=0,device='cuda:0',trials=c['trials'])]
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=jobs)
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_lr_pilot',event_prefix='kindle_lr')
    _write(output/'study-summary.json',dict(results=summary['results'],parent=c['parent'],test_evaluated=False))
    return dict(experiment=c['name'],models=len(summary['results']),test_evaluated=False,
                **({'max_epochs_per_model':60}if c['kind']=='gr_kindle_lr_ema_pilot'else {'epochs_per_model':60}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True)
    a=p.parse_args();worker(a.plan,a.slot)
