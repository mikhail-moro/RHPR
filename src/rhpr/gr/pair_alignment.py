"""Validation-only frozen-model comparison of pair boundaries and context size."""
from __future__ import annotations
import argparse, gc, json, random, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event, _evaluate, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_sid import NITEMS, WIDTH, InputIndex, attach_pair_encoder, encode_pairs, predict_rq, preencode
from rhpr.gr.pair_test import locate_parent

VARIANTS=('20_reference','20_edges','18_first','18_last','22_first','22_last')
TRIALS=[dict(id=f'pair_{s}_{w}',seed=s,checkpoint=w) for s in (42,101,202) for w in ('last','best')]


def validate(c):
    if (c['kind']!='gr_ml1m_pair_alignment' or c['trials']!=TRIALS or c['variants']!=list(VARIANTS)
        or c['seeds']!=[42,101,202] or c['data']['history_items']!=20 or c['evaluation_split']!='validation'
        or c['training']['evaluation_precision']!='fp32' or c['training']['evaluation_batch_size']!=32
        or not c['training_forbidden'] or c['generator']['attn_implementation']!='sdpa'):
        raise ValueError('alignment protocol differs from registered study')


def recover22(train, validation):
    """Find two observed predecessors in clean train windows; never use test."""
    h,u,t=train; vh,vu,vt=validation;by={}
    for i,user in enumerate(u):by.setdefault(int(user),[]).append(i)
    rows=[];prefix=[]
    for row,(target,userid) in enumerate(zip(vh,vu)):
        candidates=set()
        for i in reversed(by.get(int(userid),[])):
            # A target from a tied timestamp group may not immediately follow
            # this context. Only the saved context is known to be contiguous.
            seq=h[i][h[i]>=0]
            for j in np.flatnonzero(seq==target[0]):
                k=min(len(seq)-j,len(target))
                if j>=2 and k>=12 and np.array_equal(seq[j:j+k],target[:k]):candidates.add((int(seq[j-2]),int(seq[j-1])))
        if len(candidates)>1:raise ValueError('conflicting recovered prefixes')
        if candidates:rows.append(row);prefix.append(next(iter(candidates)))
    rows=np.asarray(rows,dtype=np.int64);prefix=np.asarray(prefix,dtype=np.int64).reshape(-1,2)
    return rows,np.concatenate([prefix,vh[rows]],axis=1)


def segments(row,variant):
    """A flat list of singleton items (i) and ordered two-item pairs (p)."""
    row=np.asarray(row);row=row[row>=0]; length=int(variant[:2])
    if len(row)<length:raise ValueError('shorter history than declared')
    row=row[-length:];old=length//2
    if variant=='20_reference':parts=[('p',int(row[i]),int(row[i+1])) for i in range(0,old,2)]
    elif variant=='20_edges':parts=[('i',int(row[0]))]+[('p',int(row[i]),int(row[i+1])) for i in range(1,old-1,2)]+[('i',int(row[old-1]))]
    elif variant.endswith('_first'):parts=[('i',int(row[0]))]+[('p',int(row[i]),int(row[i+1])) for i in range(1,old,2)]
    elif variant.endswith('_last'):parts=[('p',int(row[i]),int(row[i+1])) for i in range(0,old-1,2)]+[('i',int(row[old-1]))]
    else:raise ValueError('unknown variant')
    return parts+[('i',int(item)) for item in row[old:]]


def pair_keys(contexts,variant):
    return {p[1]*NITEMS+p[2] for row in contexts for p in segments(row,variant) if p[0]=='p'}


def encode_layout(contexts,variant,sids,keys,codes):
    widths={'20_reference':56,'20_edges':60,'18_first':52,'18_last':52,'22_first':63,'22_last':63}
    out=np.full((len(contexts),widths[variant]),-1,dtype=np.int32);itemoffs=np.arange(4)*WIDTH;pairoffs=(4+np.arange(3))*WIDTH
    for n,row in enumerate(contexts):
        tokens=[]
        for p in segments(row,variant):
            if p[0]=='i':tokens.extend(sids[p[1]]+itemoffs)
            else:
                key=p[1]*NITEMS+p[2];j=int(np.searchsorted(keys,key))
                if j>=len(keys) or keys[j]!=key:raise ValueError('pair missing from frozen cache')
                tokens.extend(codes[j]+pairoffs)
        if len(tokens)>out.shape[1]:raise ValueError('encoded layout exceeds fixed width')
        out[n,-len(tokens):]=tokens
    return out


def prepare_cache(plan_path):
    import torch
    from sentence_transformers import SentenceTransformer
    p=json.loads(Path(plan_path).read_text());c=p['config'];validate(c)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5):raise RuntimeError('requires T4')
    torch.set_num_threads(1);out=Path(p['cache']);out.mkdir(parents=True,exist_ok=True);parent=Path(p['parent'])
    train=case_examples(p['data_root'],20,'train');val=case_examples(p['data_root'],20,'validation')
    rows,h22=recover22(train,val)
    if len(rows)!=2333 or not np.array_equal(h22[:,2:],val[0][rows]):raise ValueError('unexpected 22-history cohort or ordering')
    np.savez_compressed(out/'history22.npz',rows=rows,contexts=h22,user_ids=val[1][rows],targets=val[2][rows])
    needed=set()
    for v in VARIANTS:needed.update(pair_keys(h22 if v.startswith('22') else val[0],v))
    needed=np.array(sorted(needed),dtype=np.int64);old=np.load(parent/'frozen-indexes/pair-keys.npy');missing=np.setdiff1d(needed,old,assume_unique=True);keys=np.union1d(old,missing)
    np.save(out/'pair-keys.npy',keys);_event('alignment_cache_start',cohort_22=len(rows),needed_pairs=len(needed),missing_pairs=len(missing))
    if len(missing):
        texts=json.loads((Path(p['data_root'])/'metadata-texts.json').read_text());encoder=SentenceTransformer(str(Path(p['resources'])/'sentence-t5-base'),device='cuda:0',local_files_only=True)
        encoder.max_seq_length=2048;encoder.eval();encode_pairs(encoder,texts,missing,out/'new-embeddings.npy')
        del encoder;gc.collect();torch.cuda.empty_cache();emb=np.load(out/'new-embeddings.npy',mmap_mode='r')
    for seed in (42,101,202):
        oldcodes=np.load(parent/'frozen-indexes'/f'pairs-{seed}.npy');values=np.empty((len(keys),3),dtype=oldcodes.dtype);values[np.searchsorted(keys,old)]=oldcodes
        if len(missing):
            source=Path(p['centroids']) if seed!=202 else parent/'frozen-indexes';center=np.load(source/f'pair-centroids-{seed}.npy');values[np.searchsorted(keys,missing)]=predict_rq(emb,center,'cuda:0')
        if not np.array_equal(values[np.searchsorted(keys,old)],oldcodes):raise RuntimeError('old codes changed')
        np.save(out/f'pairs-{seed}.npy',values)
    if len(missing):del emb;(out/'new-embeddings.npy').unlink()
    _write(out/'cache-report.json',dict(cohort_22=len(rows),validation_users=len(val[1]),unique_pairs=len(needed),newly_encoded_pairs=len(missing),training_performed=False,train_context_used_for_recover22=True,test_used=False))
    _event('alignment_cache_complete',cohort_22=len(rows),newly_encoded_pairs=len(missing))


def replay(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex,evaluate_ranked_sids
    from rhpr.gr.sid_study import state_digest
    c=p['config'];seed=trial['seed'];which=trial['checkpoint'];root=Path(p['parent']);ident=f'pair_{seed}'
    record=json.loads((root/'study-summary.json').read_text())['results'][ident];epoch=50 if which=='last' else record['selection']['best_epoch']
    ck=torch.load(root/'trials'/ident/f'generator-{which}.pt',map_location='cpu',weights_only=False)
    if ck['epoch']!=epoch or ck['config']['seed']!=seed or ck['config']['generator']!=c['generator'] or ck['config']['data']!=c['data']:raise ValueError('checkpoint differs')
    idx=SidIndex.from_full_sids(np.load(root/'frozen-indexes'/f'items-{seed}.npy'),width=256)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=attach_pair_encoder(build_grid_generator(idx.sids,c['generator'],seed,'cuda:0',p['source']));model.load_state_dict(ck['model'],strict=True);del ck;before=state_digest(model.state_dict())
    cache=Path(p['cache']);keys=np.load(cache/'pair-keys.npy');codes=np.load(cache/f'pairs-{seed}.npy');h,u,t=case_examples(p['data_root'],20,'validation')
    with np.load(cache/'history22.npz') as d:rows=d['rows'];h22=d['contexts'];u22=d['user_ids'];t22=d['targets']
    if not np.array_equal(u[rows],u22) or not np.array_equal(t[rows],t22):raise ValueError('cohort targets differ')
    out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True);results={}
    for v in VARIANTS:
        is22=v.startswith('22');hist,users,targets=(h22,u22,t22) if is22 else (h,u,t)
        x=encode_layout(hist,v,idx.sids,keys,codes);metrics,pred,prob=_evaluate(model,(x,users,targets),InputIndex(idx),32,'cuda:0','fp32')
        if v=='20_reference' and abs(metrics['ndcg@10']-record['epochs'][str(epoch)])>0.00005:raise RuntimeError('original validation replay differs')
        common=metrics if is22 else evaluate_ranked_sids(idx,pred[rows],targets[rows],cutoffs=(5,10))
        np.savez_compressed(out/f'{v}-predictions.npz',semantic_ids=pred,scores=prob,user_ids=users,target_item_ids=targets)
        results[v]=dict(full=metrics,common22=common,encoder_length_including_user=x.shape[1]+1)
        _event('alignment_variant_complete',trial=trial['id'],variant=v,metrics_common22=common,metrics_full=metrics)
    if state_digest(model.state_dict())!=before:raise RuntimeError('model mutated')
    result=dict(trial=trial['id'],seed=seed,checkpoint=which,epoch=epoch,variants=results,training_performed=False,test_used=False)
    _write(out/'result.json',result);return result


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('worker needs T4 and HF 5.22.1')
    torch.set_num_threads(1);status=dict(results={},failures=[]);path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('alignment_trial_start',trial=trial['id'],worker=slot)
        try:row=replay(trial,p);status['results'][trial['id']]=row;_event('alignment_trial_complete',trial=trial['id'],epoch=row['epoch'])
        except Exception as e:
            folder=Path(p['output'])/'trials'/trial['id'];folder.mkdir(parents=True,exist_ok=True);(folder/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)));_event('alignment_trial_failed',**status['failures'][-1])
        _write(path,status);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('two T4 GPUs required')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime','centroids'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    if file_sha256(roots['movie_input']/'metadata-texts.json')!=c['metadata_texts_sha256']:raise ValueError('metadata changed')
    for split in ('train','validation'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('data changed')
    parent=locate_parent(c['parent']);old=json.loads((parent/'run-manifest.json').read_text())['config']
    for key in ('training','generator','data'):
        if c[key]!=old[key]:raise ValueError('parent protocol changed: '+key)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True);source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),resources=str(roots['resources']),centroids=str(roots['centroids']),parent=str(parent),cache=str(output/'pair-cache'),assignments=assignments(c,2));_write(output/'plan.json',plan)
    env=worker_environment(c,plan['assignments'][0]);subprocess.run([sys.executable,'-u','-m','rhpr.gr.pair_alignment','--prepare-cache',str(output/'plan.json')],env=env,check=True,timeout=1800)
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    sys.path.insert(0,str(overlay));gc.collect();torch.cuda.empty_cache();_event('alignment_suite_start',trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_alignment',event_prefix='alignment')
    _write(output/'study-summary.json',dict(results=summary['results'],training_performed=False,test_used=False));return dict(experiment=c['name'],models=6,variants=len(VARIANTS),training_performed=False,test_used=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--slot',type=int);ap.add_argument('--prepare-cache');a=ap.parse_args()
    if a.prepare_cache:prepare_cache(a.prepare_cache)
    elif a.plan and a.slot is not None:worker(a.plan,a.slot)
    else:ap.error('specify --prepare-cache or --plan and --slot')
