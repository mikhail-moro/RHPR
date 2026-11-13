"""MovieLens100K ordinary leave-last-two-out, three matched 3+1 SID models."""
from __future__ import annotations
import argparse,copy,gc,json,random,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_evaluate,_train_generator,_write
from rhpr.gr import pair_sid,pair_unique4
from rhpr.gr.pair_recent10 import recent_preencode

N=1682
pair_sid.NITEMS=N
pair_unique4.NITEMS=N

def cases(root,split):
    with np.load(Path(root)/(split+'-cases.npz'),allow_pickle=False) as z:return tuple(z[k] for k in ('contexts','user_ids','targets'))

def encode(contexts,item_sids,mode,keys=None,pairs=None):
    if mode=='pair':return pair_unique4.pair_preencode(contexts,item_sids,keys,pairs)
    if mode=='recent15':return recent_preencode(contexts,item_sids,recent_items=15)
    return pair_sid.preencode(contexts,item_sids)

def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    seed=trial['seed'];mode=trial['mode'];c=copy.deepcopy(p['config']);c['seed']=seed
    out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    root=Path(p['indexes']);idx=SidIndex.from_full_sids(np.load(root/f'items-{seed}.npy'),width=256)
    keys=np.load(root/'pair-keys.npy');pairs=np.load(root/f'pairs-{seed}.npy')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=pair_unique4.attach4(build_grid_generator(idx.sids,c['generator'],seed,'cuda:0',p['source']))
    initial=state_digest(dict(model.named_parameters()));wrapped=pair_unique4.InputIndex4(idx)
    def builder(root,width,split):
        h,u,t=cases(root,split);return encode(h,idx.sids,mode,keys,pairs),u,t
    selection=_train_generator(model,p['data'],wrapped,c,out,'cuda:0',example_builder=builder)
    test=builder(p['data'],20,'test');metrics,pred,scores=_evaluate(model,test,wrapped,32,'cuda:0','fp32')
    np.savez_compressed(out/'test-best-predictions.npz',semantic_ids=pred,scores=scores,user_ids=test[1],target_item_ids=test[2])
    row=dict(trial=trial['id'],seed=seed,mode=mode,initial_state_sha256=initial,selection=selection,test=metrics)
    _write(out/'trial-result.json',row);return row

def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('requires one T4 and HF 5.22.1')
    state=dict(results={},failures=[]);path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for t in p['assignments'][slot]['trials']:
        _event('ml100k_pair_trial_start',trial=t['id'])
        try:
            row=run_trial(t,p);state['results'][t['id']]=row;_event('ml100k_pair_trial_complete',**row)
        except Exception as e:
            root=Path(p['output'])/'trials'/t['id'];root.mkdir(parents=True,exist_ok=True);(root/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=t['id'],exception_type=type(e).__name__,message=str(e)))
        _write(path,state);gc.collect();torch.cuda.empty_cache()

def run(c,output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import fit_grid_rkmeans,prepare_grid_source
    from rhpr.gr.data import file_sha256
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    if c['data']['history_items']!=20 or c['seeds']!=[42,101,202] or not c['evaluation']['test_after_selection']:raise ValueError('invalid experiment')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):raise RuntimeError('requires two T4s')
    torch.set_num_threads(1)
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    data=resolve_dataset_directory(c['movie_input']['path']);report=json.loads((data/'data-report.json').read_text())
    for name,digest in c['movie_input']['sha256'].items():
        if file_sha256(data/name)!=digest:raise ValueError('input digest mismatch: '+name)
    if (report['users'],report['items'],report['train_examples'])!=(943,N,97171):raise ValueError('unexpected dataset')
    texts=json.loads((data/'texts.json').read_text());splits={s:cases(data,s)for s in ('train','validation','test')}
    sets={s:pair_sid.collect_pairs(splits[s][0])for s in splits};keys=np.array(sorted(set.union(*sets.values())),dtype=np.int64)
    trainkeys=np.array(sorted(sets['train']),dtype=np.int64);oldkeys=np.array(sorted(sets['train']|sets['validation']),dtype=np.int64);trainmask=np.isin(keys,trainkeys)
    cache=output/'pair-cache';cache.mkdir();np.save(cache/'pair-keys.npy',keys);np.save(cache/'train-mask.npy',trainmask)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    model=SentenceTransformer(str(resources/'sentence-t5-base'),device='cuda:0',local_files_only=True);model.max_seq_length=2048;model.eval()
    vec=np.asarray(model.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=False),np.float32)
    np.save(cache/'item-embeddings.npy',vec);pair_sid.encode_pairs(model,texts,keys,cache/'pair-embeddings.npy')
    del model;gc.collect();torch.cuda.empty_cache()
    pairvec=np.load(cache/'pair-embeddings.npy',mmap_mode='r');indexes=output/'frozen-indexes';indexes.mkdir();np.save(indexes/'pair-keys.npy',keys)
    for seed in c['seeds']:
        _event('ml100k_pair_index_start',seed=seed,train_pairs=int(trainmask.sum()),all_pairs=len(keys))
        codes,centers,item_report=fit_grid_rkmeans(vec,c['quantizer'],seed,'cuda:0',source);idx=SidIndex(codes,width=256)
        np.save(indexes/f'items-{seed}.npy',idx.sids);np.save(indexes/f'item-centroids-{seed}.npy',centers)
        fitcodes,centers,pair_report=fit_grid_rkmeans(np.asarray(pairvec[trainmask]),c['quantizer'],seed,'cuda:0',source)
        codes=pair_sid.predict_rq(pairvec,centers,'cuda:0')
        if not np.array_equal(codes[trainmask],fitcodes):raise RuntimeError('inconsistent pair export')
        unique=pair_unique4.unique_pair_sids(keys,codes,trainkeys,oldkeys);np.save(indexes/f'pairs-{seed}.npy',unique);np.save(indexes/f'pair-centroids-{seed}.npy',centers)
        _write(indexes/f'index-{seed}.json',dict(item=item_report,pair=pair_report,item_diagnostics=idx.diagnostics(),train_pairs=len(trainkeys),val_pairs=len(sets['validation']),test_pairs=len(sets['test']),all_pairs=len(keys),max_suffix=int(unique[:,3].max())))
        _event('ml100k_pair_index_complete',seed=seed)
    del pairvec,vec,codes,centers,fitcodes;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=assignments(c,2));_write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.ml100k_pair_raw',event_prefix='ml100k_pair');records=summary['results']
    if set(records)!={t['id'] for t in c['trials']}:raise RuntimeError('missing trials')
    for seed in c['seeds']:
        if len({records[f'{m}_{seed}']['initial_state_sha256']for m in ('control','recent15','pair')})!=1:raise RuntimeError('unmatched initialization')
    _write(output/'study-summary.json',dict(results=records,data=report));return dict(experiment=c['name'],models=len(records),users=943,test_evaluated=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();worker(a.plan,a.slot)
