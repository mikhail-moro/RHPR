"""ML1M pilot: separately quantized two-item history tokens and matched controls."""
from __future__ import annotations
import argparse, collections, copy, gc, json, random, subprocess, sys, time, traceback
from pathlib import Path
from types import MethodType
import numpy as np
from rhpr.gr.experiment import _event, _train_generator, _write
from rhpr.gr.movie_data import case_digest
from rhpr.gr.ml1m_clean_mtl import case_examples

NITEMS = 2874
WIDTH = 256

def pairs_for_context(row):
    h = np.asarray(row); h = h[h >= 0]
    old = len(h) // 2
    return [(int(h[i]), int(h[i + 1])) for i in range(0, old - 1, 2)]

def collect_pairs(contexts):
    return {a * NITEMS + b for row in contexts for a, b in pairs_for_context(row)}

def preencode(contexts, item_sids, keys=None, pair_codes=None):
    """Left-pad compact already-offset token IDs, preserving any unpaired item."""
    is_pair = keys is not None
    # A 19-item history has 4 pairs and 11 singles: 56 tokens, one more
    # than a full 20-item history (5 pairs and 10 singles = 55 tokens).
    ncols = 56 if is_pair else 80
    out = np.full((len(contexts), ncols), -1, dtype=np.int32)
    single_offsets = np.arange(4, dtype=np.int32) * WIDTH
    pair_offsets = (4 + np.arange(3, dtype=np.int32)) * WIDTH
    for rowno, row in enumerate(contexts):
        h = row[row >= 0].astype(np.int64); old = len(h) // 2 if is_pair else 0
        tokens = []
        i = 0
        while i + 1 < old:
            key = int(h[i]) * NITEMS + int(h[i + 1])
            k = int(np.searchsorted(keys, key))
            if k >= len(keys) or keys[k] != key: raise ValueError('pair absent from frozen cache')
            tokens.extend(pair_codes[k] + pair_offsets); i += 2
        while i < len(h):
            tokens.extend(item_sids[h[i]] + single_offsets); i += 1
        if len(tokens) > ncols: raise AssertionError('compact history exceeds bound')
        if tokens: out[rowno, -len(tokens):] = tokens
    return out

class InputIndex:
    def __init__(self, sid_index):
        self.sids = sid_index.sids
        self.sid_to_item = sid_index.sid_to_item
        self.width = sid_index.width
        self.num_items = sid_index.num_items
        self.sid_length = sid_index.sid_length
    def model_batch(self, contexts, users, targets, device):
        import torch
        c=np.asarray(contexts); mask=c>=0
        if c.ndim!=2 or np.any(c >= 7*WIDTH): raise ValueError('invalid preencoded tokens')
        return dict(input_ids=torch.as_tensor(np.maximum(c,0),dtype=torch.long,device=device),
            attention_mask_encoder=torch.as_tensor(mask,dtype=torch.long,device=device),
            user_id=torch.as_tensor(np.asarray(users)[:,None],dtype=torch.long,device=device),
            future_ids=torch.as_tensor(self.sids[targets],dtype=torch.long,device=device))

def _encode(self, attention_mask, input_ids, user_id):
    import torch
    m=attention_mask.bool(); pair=input_ids >= 4*WIDTH
    old_ids=input_ids.clamp(0,4*WIDTH-1)
    new_ids=(input_ids-4*WIDTH).clamp(0,3*WIDTH-1)
    e=torch.where(pair[:,:,None], self.pair_input_embedding(new_ids),
                  self.get_embedding_table(table_name='encoder')(old_ids))
    if user_id is not None and self.user_embedding is not None:
        user=self.user_embedding(torch.remainder(user_id[:,0],self.user_embedding.num_embeddings))
        e=torch.cat([user[:,None],e],dim=1)
        attention_mask=torch.cat([torch.ones((len(e),1),dtype=attention_mask.dtype,device=attention_mask.device),attention_mask],dim=1)
    return self.encoder(sequence_embedding=e,attention_mask=attention_mask),attention_mask

def attach_pair_encoder(model):
    import torch
    if model.sep_token is not None: raise ValueError('pair encoder requires no separator')
    model.pair_input_embedding=torch.nn.Embedding(3*WIDTH,model.embedding_dim).to(model.get_embedding_table('encoder').weight.device)
    model.encoder_forward_pass=MethodType(_encode,model)
    return model

def encode_pairs(encoder, texts, keys, dest, labels=('First movie','Second movie')):
    start=time.monotonic(); n=len(keys)
    result=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(n,768))
    for i in range(0,n,2048):
        chunk=keys[i:i+2048]
        rows=[f'{labels[0]}: {texts[int(k)//NITEMS]} {labels[1]}: {texts[int(k)%NITEMS]}' for k in chunk]
        values=np.asarray(encoder.encode(rows,batch_size=32,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
        if values.shape!=(len(chunk),768) or not np.isfinite(values).all(): raise ValueError('invalid pair embeddings')
        result[i:i+len(chunk)]=values; result.flush()
        _event('pair_encoding_progress',completed=i+len(chunk),total=n,seconds=time.monotonic()-start)
    del result
    _event('pair_encoding_complete',pairs=n,seconds=time.monotonic()-start)

def predict_rq(embeddings, centers, device, batch_size=2048):
    import torch
    import torch.nn.functional as F
    from rhpr.gr.distances import TorchCdistSquaredEuclideanDistance
    distance = TorchCdistSquaredEuclideanDistance()
    result=[]; centers=torch.as_tensor(centers,dtype=torch.float32,device=device)
    with torch.no_grad():
        for begin in range(0,len(embeddings),batch_size):
            x=torch.as_tensor(np.asarray(embeddings[begin:begin+batch_size]).copy(),device=device)
            levels=[]
            for center in centers:
                x=F.normalize(x,dim=-1)
                # Match GRID fitting exactly: direct squared cdist in chunks,
                # avoiding the default GEMM path's cancellation on near ties.
                ids=distance.compute(x,center).argmin(1); levels.append(ids); x=x-center[ids]
            result.append(torch.stack(levels,dim=1).cpu().numpy())
    return np.concatenate(result)

def validate(c):
    t=c['training'];q=c['quantizer'];d=c['data']
    if (c['kind']!='gr_ml1m_pair_sid' or c['seeds'] != [42,101]
        or d['history_items']!=20 or d['train_examples']!=151166 or d['users']!=2786 or d['items']!=NITEMS
        or t['max_epochs']!=30 or t['min_epochs']!=30 or t['validation_interval']!=5
        or t['batch_size']!=512 or t['physical_batch_size']!=512 or t['precision']!='fp16'
        or q['num_hierarchies']!=3 or q['codebook_size']!=256
        or c['evaluation']['test_after_selection']): raise ValueError('invalid pilot protocol')

def worker(plan_path,slot):
    import torch
    import transformers
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    p=json.loads(Path(plan_path).read_text());c=p['config'];validate(c)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1': raise RuntimeError('requires one visible T4 and HF 5.22.1')
    torch.set_num_threads(1); status={'results':{},'failures':[]}; path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        name=trial['id']; seed=trial['seed']; mode=trial['mode'];out=Path(p['output'])/'trials'/name;out.mkdir(parents=True,exist_ok=True)
        _event('pair_trial_start',trial=name,worker=slot)
        try:
            index=SidIndex.from_full_sids(np.load(Path(p['index_root'])/f'items-{seed}.npy'),width=WIDTH)
            keys=np.load(Path(p['index_root'])/'pair-keys.npy') if mode=='pair' else None
            codes=np.load(Path(p['index_root'])/f'pairs-{seed}.npy') if mode=='pair' else None
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=attach_pair_encoder(build_grid_generator(index.sids,c['generator'],seed,'cuda:0',p['source']))
            init=state_digest(dict(model.named_parameters())); wrapped=InputIndex(index)
            def builder(root,history_items,split):
                contexts, users, targets=case_examples(root,history_items,split)
                return preencode(contexts,index.sids,keys,codes),users,targets
            conf=copy.deepcopy(c);conf['seed']=seed
            selection=_train_generator(model,p['data_root'],wrapped,conf,out,'cuda:0',example_builder=builder)
            curves=json.loads((out/'training-curves.json').read_text())
            if selection['training_epochs']!=30 or selection['validation_checks']!=6 or selection['stopping_reason']!='epoch_limit': raise RuntimeError('trial did not complete required 30 epochs')
            row=dict(trial=name,mode=mode,seed=seed,initial_state_sha256=init,selection=selection,
                epochs={str(r['epoch']):r['validation']['ndcg@10'] for r in curves if 'validation' in r})
            _write(out/'trial-result.json',row);status['results'][name]=row;_event('pair_trial_complete',**row)
            del model;gc.collect();torch.cuda.empty_cache()
        except Exception as e:
            (out/'failure.log').write_text(traceback.format_exc());status['failures'].append(dict(trial=name,error=type(e).__name__,message=str(e)));_write(path,status);return
        _write(path,status)

def run(c,output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.field_learning import make_texts
    from rhpr.gr.grid import prepare_grid_source,fit_grid_rkmeans
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('requires 2 T4')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    cases={s:case_examples(roots['movie_input'],20,s) for s in ('train','validation')}
    for name, arrays in cases.items():
        if case_digest(arrays)!=c['data']['case_sha256'][name]:raise ValueError('case hash mismatch')
    meta_path=roots['movie_input']/'metadata-texts.json'
    if file_sha256(meta_path)!=c['metadata_texts_sha256']: raise ValueError('metadata hash mismatch')
    meta=json.loads(meta_path.read_text()); texts=make_texts(meta,20261109)[0]['genres']
    trainkeys=collect_pairs(cases['train'][0]);valkeys=collect_pairs(cases['validation'][0]); keys=np.array(sorted(trainkeys|valkeys),dtype=np.int64)
    if len(trainkeys)!=121129 or len(valkeys)!=13134 or len(keys)!=124812:raise ValueError('pair count differs from verified source')
    cache=output/'pair-cache';cache.mkdir();np.save(cache/'pair-keys.npy',keys);trainmask=np.isin(keys,np.array(sorted(trainkeys),dtype=np.int64));np.save(cache/'train-mask.npy',trainmask)
    _write(cache/'provenance.json',dict(train_unique=len(trainkeys),validation_unique=len(valkeys),union=len(keys),training_pairs_only_for_fit=True,
        template='First movie: {title+genres} Second movie: {title+genres}',single_ordered=True,train_distribution='uniform over unique pairs',case_hash=c['data']['case_sha256']))
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    encoder=SentenceTransformer(str(roots['resources']/'sentence-t5-base'),device='cuda:0',local_files_only=True);encoder.max_seq_length=2048;encoder.eval()
    vec=np.asarray(encoder.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=False),np.float32)
    np.save(cache/'item-embeddings.npy',vec); encode_pairs(encoder,texts,keys,cache/'pair-embeddings.npy')
    del encoder;gc.collect();torch.cuda.empty_cache()
    pairs=np.load(cache/'pair-embeddings.npy',mmap_mode='r');index_root=output/'frozen-indexes';index_root.mkdir()
    np.save(index_root/'pair-keys.npy',keys)
    for seed in c['seeds']:
        _event('pair_index_start',seed=seed)
        item_codes,item_centers,item_report=fit_grid_rkmeans(vec,c['quantizer'],seed,'cuda:0',source)
        idx=SidIndex(item_codes,width=WIDTH);np.save(index_root/f'items-{seed}.npy',idx.sids)
        pair_train,pair_centers,pair_report=fit_grid_rkmeans(np.asarray(pairs[trainmask]),c['quantizer'],seed,'cuda:0',source)
        all_codes=predict_rq(pairs,pair_centers,'cuda:0')
        if not np.array_equal(all_codes[trainmask],pair_train):raise RuntimeError('pair code export mismatch')
        np.save(index_root/f'pairs-{seed}.npy',all_codes);np.save(index_root/f'pair-centroids-{seed}.npy',pair_centers)
        np.save(index_root/f'item-centroids-{seed}.npy',item_centers)
        _write(index_root/f'index-{seed}.json',dict(item=item_report,pair=pair_report,item_diagnostics=idx.diagnostics(),
            pair_unique_sids=len(np.unique(all_codes[trainmask],axis=0)),pair_total=len(pair_train)))
        _event('pair_index_complete',seed=seed,train_pairs=len(pair_train),unique_pair_sids=len(np.unique(pair_train,axis=0)))
    # Parent and workers share physical devices. KMeans can leave a large CUDA
    # allocator cache in the parent; return it before starting GPU workers.
    del pairs, vec, pair_train, pair_centers, item_centers, all_codes
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:',
        '--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),index_root=str(index_root),assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay));_event('pair_suite_start',trials=4)
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_sid',event_prefix='pair')
    records=summary['results'];expected={t['id'] for t in c['trials']}
    if set(records)!=expected:raise RuntimeError('missing pilot trials')
    for seed in c['seeds']:
        if records[f'control_{seed}']['initial_state_sha256']!=records[f'pair_{seed}']['initial_state_sha256']:raise RuntimeError('initial weights differ')
    _write(output/'pilot-summary.json',dict(results=records,test_evaluated=False));return dict(experiment=c['name'],models=len(records),test_evaluated=False,train_pairs=len(trainkeys))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--slot',required=True,type=int);args=ap.parse_args();worker(args.plan,args.slot)
