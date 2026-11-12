"""Matched unique 3+1 pair SIDs vs item and recent-history controls."""
from __future__ import annotations
import argparse,copy,gc,json,random,subprocess,sys,traceback
from pathlib import Path
from types import MethodType
import numpy as np
from rhpr.gr.experiment import _event,_evaluate,_train_generator,_write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_sid import NITEMS,WIDTH,InputIndex,preencode
from rhpr.gr.pair_recent10 import recent_preencode
from rhpr.gr.pair_test import locate_parent

TRIALS=[dict(id=f'{mode}_{seed}',mode=mode,seed=seed) for seed in (42,101,202) for mode in ('pair','recent15','control')]


def validate(c):
    t=c['training'];d=c['data'];g=c['generator']
    if (c['kind']!='gr_ml1m_pair_unique4' or c['trials']!=TRIALS or c['seeds']!=[42,101,202]
        or d['history_items']!=20 or d['users']!=2786 or d['items']!=2874 or d['train_examples']!=151166
        or t['min_epochs']!=50 or t['max_epochs']!=50 or t['validation_interval']!=5 or t['validate_first_epoch']
        or t['batch_size']!=512 or t['physical_batch_size']!=512 or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
        or t['learning_rate']!=.001 or t['weight_decay']!=.0001 or t['patience_epochs']!=30 or 'scheduler' in t
        or g['d_model']!=64 or g['encoder_layers']!=4 or g['decoder_layers']!=4 or g['attn_implementation']!='sdpa'
        or not c['evaluation']['test_after_selection'] or c['suffix_policy']!='train_then_val_then_test_sorted_pair_keys'
        or set(c['parents'])!={'pair','test_cache'}
        or {v['kernel'] for v in c['parents'].values()}!=set(c['kaggle']['kernel_sources'])):
        raise ValueError('unique pair study differs from declared protocol')


def unique_pair_sids(keys,codes,train_keys,old_keys,width=256):
    """Assign deterministic per-prefix ordinals without changing earlier splits."""
    keys=np.asarray(keys);codes=np.asarray(codes);train_keys=np.asarray(train_keys);old_keys=np.asarray(old_keys)
    if keys.ndim!=1 or codes.shape!=(len(keys),3) or np.any(keys[1:]<=keys[:-1]) or np.any(codes<0) or np.any(codes>=width):raise ValueError('invalid pair input')
    is_train=np.isin(keys,train_keys);is_old=np.isin(keys,old_keys)
    if not is_train.any() or not np.isin(train_keys,keys).all() or not np.isin(old_keys,keys).all() or np.any(is_train&~is_old):raise ValueError('invalid split membership')
    groups={};suffix=np.empty(len(keys),dtype=np.int32)
    for mask in (is_train,is_old&~is_train,~is_old):
        for i in np.flatnonzero(mask):
            prefix=tuple(map(int,codes[i]));value=groups.get(prefix,0)
            if value>=width:raise ValueError('pair collision group exceeds suffix vocabulary')
            suffix[i]=value;groups[prefix]=value+1
    result=np.column_stack([codes,suffix]).astype(np.int32)
    if len(np.unique(result,axis=0))!=len(keys):raise RuntimeError('pair SID collision remains')
    return result


def pair_preencode(contexts,item_sids,keys,pair_sids):
    out=np.full((len(contexts),60),-1,dtype=np.int32);itemoffs=np.arange(4)*WIDTH;pairoffs=(4+np.arange(4))*WIDTH
    for i,row in enumerate(contexts):
        h=row[row>=0];old=len(h)//2;tokens=[];j=0
        while j+1<old:
            key=int(h[j])*NITEMS+int(h[j+1]);pos=int(np.searchsorted(keys,key))
            if pos>=len(keys) or keys[pos]!=key:raise ValueError('pair absent from frozen index')
            tokens.extend(pair_sids[pos]+pairoffs);j+=2
        while j<len(h):tokens.extend(item_sids[h[j]]+itemoffs);j+=1
        if len(tokens)>60:raise ValueError('pair history exceeds width')
        if tokens:out[i,-len(tokens):]=tokens
    return out


class InputIndex4(InputIndex):
    def model_batch(self,contexts,users,targets,device):
        import torch
        x=np.asarray(contexts);mask=x>=0
        if x.ndim!=2 or np.any(x>=8*WIDTH):raise ValueError('invalid input token')
        return dict(input_ids=torch.as_tensor(np.maximum(x,0),dtype=torch.long,device=device),
            attention_mask_encoder=torch.as_tensor(mask,dtype=torch.long,device=device),
            user_id=torch.as_tensor(np.asarray(users)[:,None],dtype=torch.long,device=device),
            future_ids=torch.as_tensor(self.sids[targets],dtype=torch.long,device=device))


def _encode4(self,attention_mask,input_ids,user_id):
    import torch
    is_pair=input_ids>=4*WIDTH;old_ids=input_ids.clamp(0,4*WIDTH-1);pair_ids=(input_ids-4*WIDTH).clamp(0,4*WIDTH-1)
    x=torch.where(is_pair[:,:,None],self.pair_input_embedding(pair_ids),self.get_embedding_table(table_name='encoder')(old_ids))
    if user_id is not None and self.user_embedding is not None:
        user=self.user_embedding(torch.remainder(user_id[:,0],self.user_embedding.num_embeddings));x=torch.cat([user[:,None],x],dim=1)
        attention_mask=torch.cat([torch.ones((len(x),1),dtype=attention_mask.dtype,device=attention_mask.device),attention_mask],dim=1)
    return self.encoder(sequence_embedding=x,attention_mask=attention_mask),attention_mask


def attach4(model):
    import torch
    if model.sep_token is not None:raise ValueError('separator not supported')
    model.pair_input_embedding=torch.nn.Embedding(4*WIDTH,model.embedding_dim).to(model.get_embedding_table('encoder').weight.device)
    model.encoder_forward_pass=MethodType(_encode4,model);return model


def prepare_indexes(c,parents,source,out,data_root=None):
    oldroot=parents['pair'];testroot=parents['test_cache'];old=np.load(oldroot/'frozen-indexes/pair-keys.npy');keys=np.load(testroot/'pair-cache/pair-keys.npy');mask=np.load(source/'train-mask.npy')
    if mask.shape!=(len(old),) or mask.dtype!=np.bool_ or int(mask.sum())!=121129:raise ValueError('training pair mask differs')
    positions=np.searchsorted(keys,old)
    if np.any(positions>=len(keys)) or not np.array_equal(keys[positions],old):raise ValueError('original keys missing')
    out.mkdir(parents=True,exist_ok=True);np.save(out/'pair-keys.npy',keys);diagnostics={}
    for seed in (42,101,202):
        original=np.load(oldroot/'frozen-indexes'/f'pairs-{seed}.npy');codes=np.load(testroot/'pair-cache'/f'pairs-{seed}.npy')
        if not np.array_equal(codes[positions],original):raise ValueError('existing pair codes changed')
        sids=unique_pair_sids(keys,codes,old[mask],old);np.save(out/f'pairs-{seed}.npy',sids)
        counts=np.unique(codes,axis=0,return_counts=True)[1];diagnostics[str(seed)]=dict(pairs=len(keys),train_pairs=int(mask.sum()),max_collision_group=int(counts.max()),suffix_max=int(sids[:,3].max()),unique_pair_sids=len(np.unique(sids,axis=0)))
        if data_root is not None:
            train_suffix=np.unique(sids[np.searchsorted(keys,old[mask]),3]);diagnostics[str(seed)]['train_suffix_max']=int(train_suffix.max())
            for split in ('validation','test'):
                h,_,_=case_examples(data_root,20,split);pairkey=h[:,0:10:2]*NITEMS+h[:,1:10:2];suffix=sids[np.searchsorted(keys,pairkey),3]
                diagnostics[str(seed)][split]=dict(pair_occurrences=int(suffix.size),unseen_suffix_occurrences=int((~np.isin(suffix,train_suffix)).sum()),nonzero_suffix_fraction=float(np.mean(suffix>0)))
    _write(out/'index-diagnostics.json',diagnostics);return diagnostics


def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=copy.deepcopy(p['config']);seed=trial['seed'];mode=trial['mode'];c['seed']=seed;out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    idx=SidIndex.from_full_sids(np.load(Path(p['parents']['pair'])/'frozen-indexes'/f'items-{seed}.npy'),width=256);keys=np.load(Path(p['indexes'])/'pair-keys.npy') if mode=='pair' else None;codes=np.load(Path(p['indexes'])/f'pairs-{seed}.npy') if mode=='pair' else None
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=attach4(build_grid_generator(idx.sids,c['generator'],seed,'cuda:0',p['source']));initial=state_digest(dict(model.named_parameters()));wrapped=InputIndex4(idx)
    def builder(root,width,split):
        h,u,t=case_examples(root,width,split)
        if mode=='pair':x=pair_preencode(h,idx.sids,keys,codes)
        elif mode=='recent15':x=recent_preencode(h,idx.sids,recent_items=15)
        else:x=preencode(h,idx.sids)
        return x,u,t
    selection=_train_generator(model,p['data_root'],wrapped,c,out,'cuda:0',example_builder=builder)
    curves=json.loads((out/'training-curves.json').read_text())
    if selection['training_epochs']!=50 or [r['epoch'] for r in curves if 'validation' in r]!=list(range(5,51,5)) or selection['optimizer_updates']+selection['amp_skipped_updates']!=50*296:raise RuntimeError('training incomplete')
    examples=builder(p['data_root'],20,'test');test={}
    for which in ('last','best'):
        ck=torch.load(out/f'generator-{which}.pt',map_location='cpu',weights_only=False);epoch=50 if which=='last' else selection['best_epoch']
        if ck['epoch']!=epoch:raise ValueError('checkpoint mismatch')
        model.load_state_dict(ck['model'],strict=True);metrics,pred,scores=_evaluate(model,examples,wrapped,32,'cuda:0','fp32')
        np.savez_compressed(out/f'test-{which}-predictions.npz',semantic_ids=pred,scores=scores,user_ids=examples[1],target_item_ids=examples[2]);test[which]=dict(epoch=epoch,metrics=metrics)
    row=dict(trial=trial['id'],mode=mode,seed=seed,initial_state_sha256=initial,selection=selection,test=test,epochs={str(r['epoch']):r['validation']['ndcg@10'] for r in curves if 'validation'in r});_write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('worker requires T4 and HF 5.22.1')
    torch.set_num_threads(1);state=dict(results={},failures=[]);path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('unique4_trial_start',trial=trial['id'],worker=slot)
        try:
            row=run_trial(trial,p);state['results'][trial['id']]=row;_event('unique4_trial_complete',**row)
        except Exception as e:
            root=Path(p['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True);(root/'failure.log').write_text(traceback.format_exc());state['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)));_event('unique4_trial_failed',**state['failures'][-1])
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
    for name in ('resources','movie_input','modern_runtime','continuation_input'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation','test'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('case split differs')
    parents={k:locate_parent(v) for k,v in c['parents'].items()};original=json.loads((parents['pair']/'run-manifest.json').read_text())['config']
    for name in ('training','generator','data'):
        if c[name]!=original[name]:raise ValueError('control protocol differs: '+name)
    indexes=output/'frozen-pairs';diagnostics=prepare_indexes(c,parents,roots['continuation_input'],indexes,roots['movie_input']);_event('unique4_index_ready',diagnostics=diagnostics)
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True);source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),parents={k:str(v) for k,v in parents.items()},indexes=str(indexes),assignments=assignments(c,2));_write(output/'plan.json',plan);sys.path.insert(0,str(overlay));gc.collect();torch.cuda.empty_cache();_event('unique4_suite_start',trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_unique4',event_prefix='unique4');records=summary['results']
    for seed in (42,101,202):
        if len({records[f'{mode}_{seed}']['initial_state_sha256'] for mode in ('pair','recent15','control')})!=1:raise RuntimeError('initial weights differ')
    _write(output/'study-summary.json',dict(results=records,index_diagnostics=diagnostics,test_evaluated=True));return dict(experiment=c['name'],models=len(records),epochs=50,test_evaluated=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--slot',type=int,required=True);a=ap.parse_args();worker(a.plan,a.slot)
