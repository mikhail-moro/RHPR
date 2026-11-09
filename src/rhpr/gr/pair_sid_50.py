"""Three-seed 50-epoch pair-SID study with exact 101 checkpoint continuation."""
from __future__ import annotations
import argparse,copy,gc,json,os,random,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_train_generator,_write
from rhpr.gr.movie_data import case_digest
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.pair_sid import InputIndex,attach_pair_encoder,preencode,predict_rq

TRIALS=[dict(id=f'{m}_{s}',mode=m,seed=s,resume=(s==101)) for m,s in [
    ('control',42),('pair',42),('pair',101),('control',101),('pair',202),('control',202)]]

def validate(c):
    t=c['training'];q=c['quantizer'];d=c['data'];g=c['generator']
    if (c['kind']!='gr_ml1m_pair_sid_50' or c['seeds']!=[42,101,202] or c['trials']!=TRIALS
        or d['history_items']!=20 or d['items']!=2874 or d['users']!=2786 or d['train_examples']!=151166
        or t['min_epochs']!=50 or t['max_epochs']!=50 or t['validation_interval']!=5 or t['validate_first_epoch']
        or t['batch_size']!=512 or t['physical_batch_size']!=512 or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
        or t['learning_rate']!=0.001 or t['weight_decay']!=0.0001 or t['patience_epochs']!=30
        or 'scheduler' in t or g['d_model']!=64 or g['encoder_layers']!=4 or g['decoder_layers']!=4
        or g['attn_implementation']!='sdpa' or q['kind']!='grid_rkmeans' or q['num_hierarchies']!=3
        or q['codebook_size']!=256 or q['passes_per_level']!=10 or c['evaluation']['test_after_selection']
        or c['continuation_input']['source_commit']!='22d06b2d528291de0cabcd7a9c2f6e8c8ff5356d'):
        raise ValueError('50-epoch protocol differs from authorized study')

def fit_new_index(plan_path):
    import torch
    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.gr.sid import SidIndex
    p=json.loads(Path(plan_path).read_text());c=p['config'];validate(c)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5):raise RuntimeError('new index requires one T4')
    torch.set_num_threads(1); root=Path(p['input_root']); out=Path(p['index_root']);out.mkdir(parents=True,exist_ok=True)
    item=np.load(root/'item-embeddings.npy');pairs=np.load(root/'pair-embeddings.npy',mmap_mode='r');mask=np.load(root/'train-mask.npy')
    if item.shape!=(2874,768) or pairs.shape!=(124812,768) or mask.shape!=(124812,) or mask.dtype!=np.bool_ or int(mask.sum())!=121129:raise ValueError('unexpected cached embeddings')
    _event('pair50_index_start',seed=202)
    codes,itemcent,ireport=fit_grid_rkmeans(item,c['quantizer'],202,'cuda:0',p['source']);index=SidIndex(codes,width=256)
    np.save(out/'items-202.npy',index.sids);np.save(out/'item-centroids-202.npy',itemcent)
    pcodes,pcent,preport=fit_grid_rkmeans(np.asarray(pairs[mask]),c['quantizer'],202,'cuda:0',p['source'])
    allcodes=predict_rq(pairs,pcent,'cuda:0')
    if not np.array_equal(allcodes[mask],pcodes):raise RuntimeError('202 pair code export mismatch')
    np.save(out/'pairs-202.npy',allcodes);np.save(out/'pair-centroids-202.npy',pcent)
    _write(out/'index-202.json',dict(item=ireport,pair=preport,item_diagnostics=index.diagnostics(),
        pair_unique_sids=len(np.unique(pcodes,axis=0)),pair_total=len(pcodes)))
    _event('pair50_index_complete',seed=202,pairs=len(pcodes),unique_pair_sids=len(np.unique(pcodes,axis=0)))

def run_trial(trial,p,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=copy.deepcopy(p['config']);seed=trial['seed'];c['seed']=seed;out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    idx=SidIndex.from_full_sids(np.load(Path(p['index_root'])/f'items-{seed}.npy'),width=256)
    keys=np.load(Path(p['index_root'])/'pair-keys.npy') if trial['mode']=='pair' else None
    codes=np.load(Path(p['index_root'])/f'pairs-{seed}.npy') if trial['mode']=='pair' else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=attach_pair_encoder(build_grid_generator(idx.sids,c['generator'],seed,device,p['source']))
    init=state_digest(dict(model.named_parameters()));wrapped=InputIndex(idx)
    def builder(root,width,split):
        h,u,t=case_examples(root,width,split);return preencode(h,idx.sids,keys,codes),u,t
    resume=None
    if trial['resume']:
        if init!=p['initial_101'][trial['mode']]:raise RuntimeError('101 generator initial hash mismatch')
        resume=Path(p['resume_root'])/trial['id']
    selection=_train_generator(model,p['data_root'],wrapped,c,out,device,example_builder=builder,resume_from=resume)
    curves=json.loads((out/'training-curves.json').read_text());checks=[r['epoch'] for r in curves if 'validation' in r]
    if (selection['training_epochs']!=50 or len(curves)!=50 or checks!=list(range(5,51,5))
        or selection['optimizer_updates']+selection['amp_skipped_updates']!=50*296):raise RuntimeError('incomplete 50-epoch training')
    if resume:
        verification=json.loads((out/'resume-verification.json').read_text())
        if verification['source_epoch']!=30 or not verification['rng_restored_and_verified']:raise RuntimeError('101 continuation not verified')
    result=dict(trial=trial['id'],mode=trial['mode'],seed=seed,resumed_from_epoch=30 if resume else 0,
        initial_state_sha256=init,selection=selection,epochs={str(r['epoch']):r['validation']['ndcg@10'] for r in curves if 'validation' in r})
    _write(out/'trial-result.json',result);return result

def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('worker requires one T4 and HF 5.22.1')
    torch.set_num_threads(1);state=dict(results={},failures=[]);path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('pair50_trial_start',trial=trial['id'],worker=slot)
        try:
            row=run_trial(trial,p,'cuda:0');state['results'][trial['id']]=row;_event('pair50_trial_complete',**row)
        except Exception as e:
            root=Path(p['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True);(root/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)));_event('pair50_trial_failed',trial=trial['id'],exception_type=type(e).__name__,message=str(e))
        _write(path,state);gc.collect();torch.cuda.empty_cache()

def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('requires two T4 GPUs')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime','continuation_input'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('source cases differ')
    src=roots['continuation_input'];source_meta=json.loads((src/'source.json').read_text())
    if source_meta['source_commit']!=c['continuation_input']['source_commit'] or source_meta['epoch']!=30:raise ValueError('checkpoint provenance mismatch')
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    indexes=output/'frozen-indexes';indexes.mkdir();resume_root=scratch/'checkpoint-sources';resume_root.mkdir(parents=True,exist_ok=True)
    for s in (42,101):
        for prefix in ('items','pairs'):
            shutil.copy2(src/f'{prefix}-{s}.npy',indexes/f'{prefix}-{s}.npy')
    shutil.copy2(src/'pair-keys.npy',indexes/'pair-keys.npy')
    initial={}
    for mode in ('control','pair'):
        trial=f'{mode}_101';folder=resume_root/trial;folder.mkdir(parents=True,exist_ok=True)
        for name in ['generator-best.pt','generator-last.pt','training-curves.json']+[f'validation-epoch-{i:03d}.npz' for i in range(5,31,5)]:
            shutil.copy2(src/f'{trial}--{name}',folder/name)
        initial[mode]=json.loads((src/f'{trial}--trial-result.json').read_text())['initial_state_sha256']
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:',
        '--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),input_root=str(src),source=str(source),data_root=str(roots['movie_input']),
        index_root=str(indexes),resume_root=str(resume_root),initial_101=initial,assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    # Exit the index-building process before launching generators. This releases
    # all CUDA allocations on its T4, independent of PyTorch's allocator cache.
    env=worker_environment(c,plan['assignments'][0]);cmd=[sys.executable,'-u','-m','rhpr.gr.pair_sid_50','--fit-202',str(output/'plan.json')]
    subprocess.run(cmd,env=env,check=True,timeout=1800)
    gc.collect();torch.cuda.empty_cache();free=[torch.cuda.mem_get_info(i)[0] for i in (0,1)];_event('pair50_gpu_ready',free_bytes=free)
    if min(free)<8*1024**3:raise RuntimeError('insufficient free GPU memory before generator workers')
    _event('pair50_suite_start',trials=len(c['trials']),resumed_seed=101)
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_sid_50',event_prefix='pair50')
    records=summary['results']
    if set(records)!={t['id'] for t in TRIALS}:raise RuntimeError('missing model results')
    for s in (42,101,202):
        if records[f'control_{s}']['initial_state_sha256']!=records[f'pair_{s}']['initial_state_sha256']:raise RuntimeError('initial weights differ within seed')
    _write(output/'study-summary.json',dict(results=records,test_evaluated=False,resumed_seed101=True))
    return dict(experiment=c['name'],models=len(records),seeds=3,epochs=50,test_evaluated=False,resumed_seed101=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--slot',type=int);ap.add_argument('--fit-202');a=ap.parse_args()
    if a.fit_202:fit_new_index(a.fit_202)
    elif a.plan and a.slot is not None:worker(a.plan,a.slot)
    else:ap.error('specify --fit-202 or --plan and --slot')
