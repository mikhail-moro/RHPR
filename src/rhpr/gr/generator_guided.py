"""Frozen-generator likelihood search over whole SID paths; TRAIN data only."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rhpr.gr.data import build_examples,file_sha256
from rhpr.gr.experiment import _event,_write


def validate_guided(config):
    spec=config['generator_guided'];trials=config['trials']
    if (config['quantizer']['kind']!='frozen_full_sids' or len(trials)!=2
            or trials[0]['id']!=spec['control_trial'] or trials[1]['id']!=spec['candidate_trial']
            or config.get('index_gate') or config.get('sid_corruption')
            or any('generator_seed' in t for t in trials)
            or config['evaluation'].get('test_after_selection',True)):
        raise ValueError('invalid original/generator-guided comparison')
    for key in ['batch_size','mining_rows','max_proposals','max_accepted','max_scored_rows','seed','minimum_target_count']:
        if type(spec[key]) is not int or spec[key]<1:raise ValueError('positive integer guided budget required: '+key)
    for key in ['max_seconds','min_total_gain','min_gain_per_affected','confidence_log_margin','gap_clip','identity_tolerance']:
        if not np.isfinite(spec[key]) or spec[key]<=0:raise ValueError('positive finite guided threshold required: '+key)
    if spec['precision']!='fp32' or spec['semantic_lambda']!=0:
        raise ValueError('first guided screen uses raw FP32likelihood with no semantic penalty')


def prepare_originals(config,source_ids,output):
    from rhpr.gr.full_sid_index import prepare_full_sid_indices
    base={**config,'trials':[{**t,'use_original_paths':True,'role':'original',
        'semantic_lambda':0,'expected_variant':'original'} for t in config['trials']]}
    return prepare_full_sid_indices(base,source_ids,output)


def incidence(examples,items):
    contexts,users,targets=examples
    rows=[[] for _ in range(items)]
    if len(contexts)!=len(users) or len(targets)!=len(users):raise ValueError('misaligned TRAINcases')
    for row,(history,target) in enumerate(zip(contexts,targets)):
        ids=np.unique(np.r_[history[history>=0],target])
        if np.any((ids<0)|(ids>=items)):raise ValueError('item outside incidence catalogue')
        for i in ids:rows[int(i)].append(row)
    return [np.asarray(v,dtype=np.int64) for v in rows]


class LikelihoodSwapSearch:
    """Fixed original minibatch layout makes cached before/after scores comparable."""
    def __init__(self,sids,examples,score_batch,batch_size,*,min_total_gain=.01,min_gain_per_affected=1e-5):
        self.sids=np.asarray(sids).copy();self.initial=self.sids.copy();self.owner=np.arange(len(sids))
        self.examples=examples;self.score_batch=score_batch;self.batch_size=batch_size
        self.min_total_gain=min_total_gain;self.min_gain_per_affected=min_gain_per_affected
        self.incidence=incidence(examples,len(sids));self.rows=len(examples[2]);self.accepted=[]
        self.cache=np.empty(self.rows,dtype=np.float64)
        self.scored_rows=0;self.proposals=0
        for begin in range(0,self.rows,batch_size):
            ids=np.arange(begin,min(begin+batch_size,self.rows));self.cache[ids]=self.score(ids,self.sids)
        self.initial_cache=self.cache.copy()

    def score(self,ids,sids):
        score=np.asarray(self.score_batch(ids,sids),dtype=np.float64)
        if score.shape!=(len(ids),) or not np.isfinite(score).all():raise FloatingPointError('invalid frozen likelihood scores')
        return score

    def plan(self,a,b):
        if any(type(i) not in [int,np.int64] or not 0<=i<len(self.sids) for i in [a,b]):
            raise ValueError('swap item IDs outside catalogue')
        affected=np.union1d(self.incidence[a],self.incidence[b]) if a!=b else np.array([],dtype=np.int64)
        groups=np.unique(affected//self.batch_size)
        batches=[np.arange(g*self.batch_size,min((g+1)*self.batch_size,self.rows)) for g in groups]
        return affected,batches

    def try_swap(self,a,b):
        affected,batches=self.plan(a,b);self.proposals+=1
        count=sum(map(len,batches));self.scored_rows+=count
        candidate=self.sids.copy();candidate[[a,b]]=candidate[[b,a]]
        ids=np.concatenate(batches) if batches else np.array([],dtype=np.int64)
        after=np.concatenate([self.score(batch,candidate) for batch in batches]) if batches else np.array([])
        before=self.cache[ids];delta=float(np.sum(after-before,dtype=np.float64))
        threshold=max(self.min_total_gain,self.min_gain_per_affected*len(affected))
        accepted=bool(len(affected) and delta>threshold)
        record=dict(a=int(a),b=int(b),affected_rows=len(affected),scored_rows=count,
            delta_log_likelihood=delta,threshold=threshold,accepted=accepted)
        if accepted:
            self.sids=candidate;self.owner[[a,b]]=self.owner[[b,a]];self.cache[ids]=after
            self.accepted.append(record)
        return record

    def audit(self,tolerance):
        np.testing.assert_array_equal(np.sort(self.owner),np.arange(len(self.owner)))
        np.testing.assert_array_equal(self.sids,self.initial[self.owner])
        fresh=np.concatenate([self.score(np.arange(i,min(i+self.batch_size,self.rows)),self.sids)
                              for i in range(0,self.rows,self.batch_size)])
        drift=float(np.max(np.abs(fresh-self.cache)))
        if drift>tolerance or abs(float(fresh.sum()-self.cache.sum()))>.02:
            raise RuntimeError('local likelihood cache differs from complete fixed-batch rescore')
        return dict(max_row_drift=drift,full_rescore_delta=float(fresh.sum()-self.initial_cache.sum()))


def model_scores(model,examples,sids,ids,device,future_override=None):
    import torch
    from rhpr.gr.experiment import _model_batch
    contexts,users,targets=examples
    batch=_model_batch(contexts[ids],users[ids],targets[ids],sids,device)
    if future_override is not None:batch['future_ids']=torch.as_tensor(future_override,dtype=torch.long,device=device)
    with torch.inference_mode():
        hidden=model(**batch)[:,:-1]
        terms=[-torch.nn.functional.cross_entropy(head(hidden[:,level]).float(),
            batch['future_ids'][:,level],reduction='none').double()
            for level,head in enumerate(model.decoder.decoder_mlp)]
        values=torch.stack(terms,dim=1).sum(1).cpu().numpy()
    if not np.isfinite(values).all():raise FloatingPointError('nonfinite teacher-forced path likelihood')
    return values


def mine_pairs(model,examples,sids,truth_cache,spec,device,width,output):
    import torch
    from rhpr.gr.experiment import _model_batch
    rng=np.random.default_rng(spec['seed']);rows=np.sort(rng.choice(len(examples[2]),min(spec['mining_rows'],len(examples[2])),replace=False))
    frequency=np.bincount(examples[2],minlength=len(sids));powers=width**np.arange(sids.shape[1]-1,-1,-1,dtype=np.int64)
    decode={int(key):i for i,key in enumerate(sids@powers)};pairs={};invalid=0;confident=0
    old_width=model.top_k_for_generation;model.top_k_for_generation=1
    targets=[];predicted=[];gaps=[]
    try:
        with torch.inference_mode():
            for begin in range(0,len(rows),spec['batch_size']):
                ids=rows[begin:begin+spec['batch_size']]
                batch=_model_batch(examples[0][ids],examples[1][ids],examples[2][ids],sids,device)
                paths,_=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
                paths=paths[:,0].cpu().numpy()
                # Native generate returns probability products. They are ignored:
                # proposal confidence uses full-path teacher-forced logP instead.
                proposed_scores=model_scores(model,examples,sids,ids,device,paths)
                for row,path,score in zip(ids,paths,proposed_scores):
                    a=int(examples[2][row]);b=decode.get(int(path@powers),-1)
                    gap=float(score-truth_cache[row]);targets.append(a);predicted.append(b);gaps.append(gap)
                    if b<0:invalid+=1;continue
                    if a==b or min(frequency[a],frequency[b])<spec['minimum_target_count'] or gap<spec['confidence_log_margin']:continue
                    confident+=1;pair=tuple(sorted((a,b)))
                    value=pairs.setdefault(pair,dict(a=pair[0],b=pair[1],mistakes=0,priority=0.,maximum_log_margin=gap))
                    value['mistakes']+=1;value['priority']+=min(gap,spec['gap_clip'])
                    value['maximum_log_margin']=max(value['maximum_log_margin'],gap)
                if begin%(spec['batch_size']*32)==0:_event('guided_mining_progress',rows=min(begin+len(ids),len(rows)),total=len(rows))
    finally:model.top_k_for_generation=old_width
    ordered=sorted(pairs.values(),key=lambda r:(-r['priority'],-r['mistakes'],r['a'],r['b']))[:spec['max_proposals']]
    np.savez_compressed(output/'mining-probes.npz',train_row_ids=rows,target_item_ids=np.asarray(targets),
                        proposed_item_ids=np.asarray(predicted),log_margin=np.asarray(gaps))
    _write(output/'candidate-pairs.json',ordered)
    return ordered,dict(rows=len(rows),invalid_greedy_paths=invalid,confident_mistakes=confident,unique_pairs=len(pairs),selected_pairs=len(ordered),
        method='Frozen original-map free greedy autoregressive proposal;exact teacher-forced confidence. '
               f"Invalid paths discarded;minimum{spec['minimum_target_count']}TRAINtargets per item. "
               'Not exact catalogue argmax;no beam score in acceptance.')


def optimize_generator_index(config,users,output,source,records):
    import torch
    from rhpr.gr.generator_suite import verify_file,load_trial_index,build_trial_generator
    from rhpr.gr.resources import verify_parent_artifacts
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.sid import SidIndex
    validate_guided(config);spec=config['generator_guided'];root=Path(output)
    directory=root/'generator-guided';directory.mkdir(parents=True)
    parent=verify_parent_artifacts(spec['parent'])
    files={k:verify_file(parent,spec[k]) for k in ['checkpoint','sids','source_result']}
    parent_manifest=json.loads((parent/'run-manifest.json').read_text())
    checkpoint=torch.load(files['checkpoint'],map_location='cpu',weights_only=True)
    for key in ['data','generator','seed']:
        if checkpoint['config'][key]!=config[key] or parent_manifest['config'][key]!=config[key]:raise ValueError('guided source protocol drift: '+key)
    if checkpoint['epoch']!=spec['checkpoint_epoch'] or checkpoint['updates']!=spec['checkpoint_updates']:
        raise ValueError('guided source is not the pinned20epoch checkpoint')
    source_result=json.loads(files['source_result'].read_text())
    if source_result['selection']['selected_epoch']!=checkpoint['epoch'] or source_result['trial']!=spec['source_trial']:
        raise ValueError('guided source trial/selection mismatch')
    if checkpoint['config']['training']!=config['training']:
        raise ValueError('guided source training budget differs from the fresh fits')
    control=load_trial_index(spec['control_trial'],config,root)
    if not np.array_equal(np.load(files['sids']),control.sids):raise ValueError('guided source SIDtable differs')
    device='cuda:0' if config['training']['device']=='cuda' else 'cpu'
    if device!='cpu':torch.cuda.set_device(0)
    model=build_trial_generator(control,config,device,source);model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint;gc.collect();model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=state_digest(model.state_dict());rng_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state_all() if device!='cpu' else []
    examples=build_examples(users,config['data']['history_items'],'train')
    _event('guided_baseline_start',examples=len(examples[2]),batch_size=spec['batch_size'])
    started=time.monotonic()
    calls={'rows':0}
    def score(ids,sids):
        result=model_scores(model,examples,sids,ids,device);calls['rows']+=len(ids)
        if calls['rows']%8192<len(ids):_event('guided_scoring_progress',scored_rows=calls['rows'])
        return result
    search=LikelihoodSwapSearch(control.sids,examples,score,spec['batch_size'],
        min_total_gain=spec['min_total_gain'],min_gain_per_affected=spec['min_gain_per_affected'])
    baseline_seconds=time.monotonic()-started
    probe=np.arange(min(spec['batch_size'],search.rows));identity_drift=float(np.max(np.abs(score(probe,search.sids)-search.cache[probe])))
    if identity_drift>spec['identity_tolerance']:raise RuntimeError('identity likelihood drift exceeds numerical threshold')
    _event('guided_baseline_ready',seconds=baseline_seconds,mean_train_nll=float(-search.cache.mean()),identity_drift=identity_drift)
    mining_started=time.monotonic()
    pairs,mining=mine_pairs(model,examples,search.sids,search.cache,spec,device,config['quantizer']['codebook_size'],directory)
    mining['seconds']=time.monotonic()-mining_started
    search_started=time.monotonic();reason='candidate_list_exhausted';last_cost=0.
    with (directory/'search-trace.jsonl').open('w') as trace:
        for candidate in pairs:
            if len(search.accepted)>=spec['max_accepted']:reason='accepted_cap';break
            _,batches=search.plan(candidate['a'],candidate['b']);cost=sum(map(len,batches))
            if search.scored_rows+cost>spec['max_scored_rows']:reason='scored_row_budget';break
            if time.monotonic()-search_started+last_cost>spec['max_seconds']:reason='search_time_budget';break
            tick=time.monotonic();record=search.try_swap(candidate['a'],candidate['b']);last_cost=time.monotonic()-tick
            record.update(proposal=search.proposals,seconds=last_cost,elapsed=time.monotonic()-search_started)
            trace.write(json.dumps(record,allow_nan=False)+'\n');trace.flush()
            if record['accepted'] or search.proposals%25==0:
                _event('guided_search_progress',proposals=search.proposals,accepted=len(search.accepted),scored_rows=search.scored_rows,
                       train_nll=float(-search.cache.mean()),elapsed=record['elapsed'])
    search_seconds=time.monotonic()-search_started
    audit=search.audit(spec['identity_tolerance'])
    if state_digest(model.state_dict())!=before or not torch.equal(rng_before,torch.get_rng_state()):
        raise RuntimeError('frozen teacher weights or CPU RNGchanged')
    if device!='cpu' and any(not torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all())):
        raise RuntimeError('frozen teacher CUDA RNGchanged')
    if search.accepted and audit['full_rescore_delta']<=0:raise RuntimeError('accepted search did not improve complete TRAINlikelihood')
    name=spec['candidate_trial'];d=root/'indices'/name
    index=SidIndex.from_full_sids(search.sids,config['quantizer']['codebook_size'])
    for depth in range(1,search.sids.shape[1]+1):
        a,n=np.unique(control.sids[:,:depth],axis=0,return_counts=True);b,m=np.unique(search.sids[:,:depth],axis=0,return_counts=True)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(n,m)
    np.save(d/'semantic-ids.npy',search.sids,allow_pickle=False);np.save(d/'semantic-codes.npy',index.semantic_codes,allow_pickle=False)
    np.save(d/'owner.npy',search.owner,allow_pickle=False)
    np.savez_compressed(directory/'train-likelihood-cache.npz',user_ids=examples[1],target_item_ids=examples[2],
        initial_log_probability=search.initial_cache,final_log_probability=search.cache)
    moved=search.owner!=np.arange(len(search.owner));lcp=np.cumprod(search.sids==control.sids,axis=1).sum(1)
    report=dict(source_checkpoint_sha256=file_sha256(files['checkpoint']),source_epoch=spec['checkpoint_epoch'],
        source_state_sha256=before,frozen_weights_and_rng_verified=True,baseline_seconds=baseline_seconds,mining=mining,
        search_seconds=search_seconds,proposals=search.proposals,accepted=len(search.accepted),stop_reason=reason,
        scored_rows=search.scored_rows,changed_items=int(moved.sum()),lcp_counts=np.bincount(lcp[moved],minlength=4).tolist(),
        initial_mean_train_nll=float(-search.initial_cache.mean()),final_mean_train_nll=float(-search.cache.mean()),
        total_train_log_likelihood_gain=audit['full_rescore_delta'],cache_audit=audit,identity_drift=identity_drift,
        training_examples=search.rows,validation_or_test_used_for_search=False,semantic_lambda=0,
        acceptance='Exact four-token likelihood on all affected TRAINrows,including changed histories;fixed original batch groups.',
        note='Frozen-teacher NLLimprovement is a search check,not proof of fresh-generator NDCGgain.')
    _write(directory/'optimization.json',report);_write(directory/'accepted-swaps.json',search.accepted)
    record=records[name];record['original_source']=record['source'];record['source']=dict(role='generator_guided',variant='frozen_teacher_likelihood',semantic_lambda=0)
    record.update(semantic_ids_sha256=file_sha256(d/'semantic-ids.npy'),diagnostics=index.diagnostics(),generator_guided=report)
    _write(d/'index-provenance.json',record);_write(d/'sid-diagnostics.json',index.diagnostics())
    catalogue=json.loads((root/'sid-catalogue.json').read_text());catalogue['indices']=records;_write(root/'sid-catalogue.json',catalogue)
    _event('guided_search_complete',**{k:report[k] for k in ['accepted','proposals','changed_items','initial_mean_train_nll','final_mean_train_nll','stop_reason']})
    del model,search;gc.collect()
    if device!='cpu':torch.cuda.empty_cache()
    return records,report
