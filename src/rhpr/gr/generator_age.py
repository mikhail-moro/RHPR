"""Matched TRAIN-only static and sequential SID searches at teacher ages1/5/20."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

from rhpr.gr.data import build_examples,file_sha256,load_beauty
from rhpr.gr.experiment import _event,_write
from rhpr.gr.generator_guided import LikelihoodSwapSearch,incidence,mine_pairs,model_scores
from rhpr.gr.generator_suite import build_trial_generator,load_trial_index,training_manifest,verify_file


def validate_age(config):
    s=config['age_screen']
    if (s['epochs']!=[1,5,20] or config['training']!=s['source_training']
            or s['new_training_epochs']!=5 or s['pool_size']!=2000
            or s['search']['semantic_lambda']!=0 or s['search']['precision']!='fp32'
            or config['evaluation']['test_after_selection']
            or len(config['trials'])!=3 or config.get('generator_guided')
            or config.get('index_gate') or config.get('sid_corruption')):
        raise ValueError('invalid one-shot age screen protocol')
    if [t['epoch'] for t in config['trials']]!=s['epochs']:
        raise ValueError('teacher trial order differs from ages')
    if config['training']['sampling']!='all_train_prefixes' or len(config['execution']['worker_devices'])!=2:
        raise ValueError('age screen requires native prefix sampling and two workers')
    for k in ['max_scored_rows','max_proposals','batch_size','mining_rows']:
        if type(s['search'][k]) is not int or s['search'][k]<1:raise ValueError('invalid search budget')
    for k in ['max_seconds','min_total_gain','min_gain_per_affected','confidence_log_margin','gap_clip','identity_tolerance']:
        if not np.isfinite(s['search'][k]) or s['search'][k]<=0:raise ValueError('invalid threshold')


def common_pool(by_epoch,limit,seed):
    """Round-robin source ranks; no comparison of uncalibrated margins across ages."""
    ages=sorted(by_epoch);chosen=[];seen=set();ranks={}
    for age in ages:
        for rank,p in enumerate(by_epoch[age],1):
            key=tuple(sorted((int(p['a']),int(p['b']))))
            ranks.setdefault(key,{})[str(age)]=rank
    for offset in range(max((len(x) for x in by_epoch.values()),default=0)):
        for age in ages:
            if offset>=len(by_epoch[age]):continue
            p=by_epoch[age][offset];key=tuple(sorted((int(p['a']),int(p['b']))))
            if key in seen:continue
            seen.add(key);chosen.append(dict(a=key[0],b=key[1],source_ranks=ranks[key]))
            if len(chosen)==limit:break
        if len(chosen)==limit:break
    order=np.random.default_rng(seed).permutation(len(chosen))
    return [chosen[i] for i in order]


def budget_prefix(pairs,examples,items,batch_size,row_budget):
    """Reserve both original-map and current-map rescoring for every proposal."""
    rows=incidence(examples,items);n=len(examples[2]);selected=[];spent=0
    for p in pairs:
        affected=np.union1d(rows[p['a']],rows[p['b']]);groups=np.unique(affected//batch_size)
        cost=sum(min(batch_size,n-int(g)*batch_size) for g in groups)
        if spent+2*cost>row_budget:break
        selected.append({**p,'affected_rows':len(affected),'scored_rows_per_evaluation':cost})
        spent+=2*cost
    return selected,spent


def freeze_model(model):
    import torch
    from rhpr.gr.sid_study import state_digest
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return (state_digest(model.state_dict()),torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])


def verify_frozen(model,guard):
    import torch
    from rhpr.gr.sid_study import state_digest
    if state_digest(model.state_dict())!=guard[0] or not torch.equal(torch.get_rng_state(),guard[1]):
        raise RuntimeError('teacher weights/CPU RNG changed')
    if guard[2] and any(not torch.equal(a,b) for a,b in zip(guard[2],torch.cuda.get_rng_state_all())):
        raise RuntimeError('teacher CUDA RNG changed')


def compare_teacher(model,examples,original,pairs,spec,device,directory):
    """Same proposals scored on original mapping AND each age's greedy trajectory."""
    directory.mkdir(parents=True,exist_ok=True);guard=freeze_model(model)
    score=lambda ids,sids:model_scores(model,examples,sids,ids,device)
    search=LikelihoodSwapSearch(original,examples,score,spec['batch_size'],
        min_total_gain=spec['min_total_gain'],min_gain_per_affected=spec['min_gain_per_affected'])
    probe=np.arange(min(spec['batch_size'],search.rows))
    drift=float(np.max(np.abs(score(probe,original)-search.cache[probe])))
    if drift>spec['identity_tolerance']:raise RuntimeError('identity likelihood drift')
    started=time.monotonic();static_records=[];trace=[]
    for p in pairs:
        if time.monotonic()-started>spec['max_seconds']:
            raise RuntimeError('age search timed out before shared prefix completed; no matched result')
        # Copy only cached arrays; never recompute baseline or retain changes from a static probe.
        static=copy.copy(search);static.sids=original.copy();static.owner=np.arange(len(original))
        static.cache=search.initial_cache.copy();static.accepted=[];static.proposals=0;static.scored_rows=0
        a=static.try_swap(p['a'],p['b']);static_records.append(a)
        b=search.try_swap(p['a'],p['b'])
        assert a['scored_rows']==b['scored_rows']==p['scored_rows_per_evaluation']
        b={**b,'proposal':search.proposals,'static_delta':a['delta_log_likelihood'],
           'static_accepted':a['accepted'],'original_lcp':int(np.cumprod(original[p['a']]==original[p['b']]).sum())}
        trace.append(b)
        with (directory/'search-trace.jsonl').open('a') as f:f.write(json.dumps(b)+'\n')
        if b['accepted'] or search.proposals%25==0:
            _event('age_search_progress',trial=directory.name,proposals=search.proposals,accepted=len(search.accepted),elapsed=time.monotonic()-started)
    seconds=time.monotonic()-started;audit=search.audit(spec['identity_tolerance']);verify_frozen(model,guard)
    moved=search.owner!=np.arange(len(original));lcp=np.cumprod(search.sids==original,axis=1).sum(1)
    np.save(directory/'semantic-ids.npy',search.sids);np.save(directory/'owner.npy',search.owner)
    np.savez_compressed(directory/'train-likelihood-cache.npz',user_ids=examples[1],target_item_ids=examples[2],
        initial_log_probability=search.initial_cache,final_log_probability=search.cache)
    for depth in range(1,original.shape[1]+1):
        for a,b in zip(np.unique(original[:,:depth],axis=0,return_counts=True),np.unique(search.sids[:,:depth],axis=0,return_counts=True)):
            np.testing.assert_array_equal(a,b)
    report=dict(proposals=len(pairs),static_accepted=sum(r['accepted'] for r in static_records),
        accepted=len(search.accepted),changed_items=int(moved.sum()),changed_lcp_counts=np.bincount(lcp[moved],minlength=4).tolist(),
        initial_mean_train_nll=float(-search.initial_cache.mean()),final_mean_train_nll=float(-search.cache.mean()),
        total_train_log_likelihood_gain=audit['full_rescore_delta'],cache_audit=audit,identity_drift=drift,
        teacher_state_sha256=guard[0],frozen_weights_and_rng_verified=True,search_seconds=seconds,
        scored_rows=2*search.scored_rows,train_affected_rows=int((np.isin(examples[0],np.flatnonzero(moved)).any(1)|np.isin(examples[2],np.flatnonzero(moved))).sum()),
        static_by_lcp=[dict(lcp=d,proposed=sum(r['original_lcp']==d for r in trace),
            accepted=sum(r['original_lcp']==d and r['static_accepted'] for r in trace)) for d in range(original.shape[1])],
        stop_reason='completed_shared_budget_prefix',validation_or_test_used_for_search=False)
    _write(directory/'optimization.json',report);_write(directory/'accepted-swaps.json',search.accepted)
    return report


def run_worker(plan_path,slot):
    import torch
    plan=json.loads(Path(plan_path).read_text());c=plan['config'];out=Path(plan['output']);job=plan['assignments'][slot]
    torch.set_num_threads(c['execution']['cpu_threads_per_worker']);device='cpu'
    if job['device']!='cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('worker requires assigned GPU')
        torch.cuda.set_device(0);device='cuda:0'
    _,users,_,_=load_beauty(Path(plan['resources'])/'data',c['data'])
    examples=build_examples(users,c['data']['history_items'],'train');index=load_trial_index(c['trials'][0]['id'],c,out)
    pairs=json.loads((out/'shared-search-pairs.json').read_text());status=dict(results={},failures=[],status='running')
    if file_sha256(out/'shared-search-pairs.json')!=plan['proposal_sha256']:
        raise ValueError('shared proposal list changed')
    path=out/'workers'/f'worker-{slot}.json'
    for trial in job['trials']:
        name=trial['id'];epoch=trial['epoch'];_event('age_trial_start',trial=name)
        try:
            checkpoint=plan['checkpoints'][str(epoch)]
            if file_sha256(Path(checkpoint['path']))!=checkpoint['sha256']:raise ValueError('checkpoint changed')
            saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
            if saved['epoch']!=epoch or saved['updates']!=epoch*plan['updates_per_epoch']:raise ValueError('teacher age/update mismatch')
            model=build_trial_generator(index,c,device,Path(plan['source']));model.load_state_dict(saved['model'],strict=True);del saved
            report=compare_teacher(model,examples,index.sids,pairs,c['age_screen']['search'],device,out/'searches'/name)
            report.update(epoch=epoch,checkpoint_sha256=checkpoint['sha256'],proposal_sha256=plan['proposal_sha256']);status['results'][name]=report
            del model;gc.collect()
            if device!='cpu':torch.cuda.empty_cache()
        except Exception as e:
            status['failures'].append(dict(trial=name,error_type=type(e).__name__,message=str(e)))
            _event('age_trial_failed',**status['failures'][-1])
            (out/'workers'/f'failure-{name}.log').write_text(traceback.format_exc())
        _write(path,status)
    status['status']='complete';_write(path,status)


def run_age_screen(config,output):
    import torch
    from rhpr.gr.resources import verify_resources,verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.full_sid_index import prepare_full_sid_indices
    from rhpr.gr.rqvae_experiment import _train_rqvae_generator
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.offline_runtime import resolve_dataset_directory
    validate_age(config);s=config['age_screen'];output=Path(output);started=time.monotonic()
    device=config['training']['device']
    if device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('age screen requires2GPUs')
        torch.cuda.set_device(0);device='cuda:0'
    torch.set_num_threads(config['training']['cpu_threads'])
    resources=resolve_dataset_directory(config['resources']['path']);verify_resources(resources,config['resources']['manifest_sha256'])
    _,users,ids,counts=load_beauty(resources/'data',config['data']);prepare_full_sid_indices(config,ids,output)
    stream=training_manifest(users,config);assert stream['train_cases_sha256']==config['expected_train_cases_sha256']
    _write(output/'shared-training-stream.json',stream)
    original=unpack_grid(resources/'grid-source.bin',Path(config['resources']['scratch_path'])/'original')
    source=prepare_grid_source(original,Path(config['resources']['scratch_path'])/'runtime')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    parent=verify_parent_artifacts(s['parent']);late=verify_file(parent,s['checkpoint']);late_sids=verify_file(parent,s['sids'])
    index=load_trial_index(config['trials'][0]['id'],config,output)
    np.testing.assert_array_equal(index.sids,np.load(late_sids))
    saved=torch.load(late,map_location='cpu',weights_only=True)
    if saved['epoch']!=20 or saved['updates']!=s['checkpoint_updates']:raise ValueError('late teacher age mismatch')
    for key in ['data','generator','training','seed']:
        if saved['config'][key]!=config[key]:raise ValueError('late teacher protocol drift: '+key)
    del saved
    warm=copy.deepcopy(config);warm['training'].update(max_epochs=5,min_epochs=5,save_epoch_checkpoints=[1,5])
    warm['evaluation'].update(save_validation_predictions=False,save_teacher_forced_diagnostics=False)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=build_trial_generator(index,warm,device,source)
    init=state_digest(dict(model.named_parameters()))
    if init!=config['expected_initial_state_sha256']:raise ValueError('warmup initial weights differ')
    directory=output/'early-teacher';directory.mkdir()
    _event('age_warmup_start',epochs=5,initial_state_sha256=init)
    trained=_train_rqvae_generator(model,users,index,warm,directory,device)
    if trained['epochs']!=5:raise RuntimeError('early teacher stopped before5epochs')
    del model;gc.collect()
    if device!='cpu':torch.cuda.empty_cache()
    paths={1:directory/'generator-epoch-001.pt',5:directory/'generator-epoch-005.pt',20:late}
    checkpoints={str(e):dict(path=str(p),sha256=file_sha256(p)) for e,p in paths.items()}
    examples=build_examples(users,config['data']['history_items'],'train');mined={};mining=[]
    for e in s['epochs']:
        d=output/'mining'/f'epoch-{e:03d}';d.mkdir(parents=True)
        model=build_trial_generator(index,config,device,source)
        saved=torch.load(paths[e],map_location='cpu',weights_only=True);model.load_state_dict(saved['model'],strict=True);del saved
        guard=freeze_model(model);spec=s['search'];cache=np.empty(len(examples[2]),dtype=np.float64)
        for i in range(0,len(cache),spec['batch_size']):
            rows=np.arange(i,min(i+spec['batch_size'],len(cache)));cache[rows]=model_scores(model,examples,index.sids,rows,device)
        mined[e],report=mine_pairs(model,examples,index.sids,cache,spec,device,config['quantizer']['codebook_size'],d)
        verify_frozen(model,guard);mining.append(dict(epoch=e,**report));_event('age_mining_complete',epoch=e,**report)
        del model,cache;gc.collect()
        if device!='cpu':torch.cuda.empty_cache()
    pool=common_pool(mined,s['pool_size'],s['pool_order_seed']);_write(output/'common-candidate-pool.json',pool)
    selected,reserved=budget_prefix(pool,examples,len(index.sids),s['search']['batch_size'],s['search']['max_scored_rows'])
    _write(output/'shared-search-pairs.json',selected);_write(output/'mining.json',mining)
    _write(output/'input-provenance.json',dict(counts=counts,source_initial_state_sha256=init,checkpoints=checkpoints,
        pool_pairs=len(pool),shared_proposals=len(selected),reserved_scored_rows_per_teacher=reserved,new_training_epochs=5,
        static_and_greedy=True,semantic_lambda=0,test_evaluations=0))
    (output/'workers').mkdir()
    devices=config['execution']['worker_devices'];trials=config['trials']
    jobs=[dict(slot=0,device=devices[0],trials=[trials[0],trials[1]]),dict(slot=1,device=devices[1],trials=[trials[2]])]
    plan=dict(config=config,assignments=jobs,output=str(output.resolve()),resources=str(resources),source=str(source),
        checkpoints=checkpoints,proposal_sha256=file_sha256(output/'shared-search-pairs.json'),
        updates_per_epoch=(len(examples[2])+config['training']['batch_size']-1)//config['training']['batch_size'])
    path=output/'workers/plan.json';_write(path,plan)
    summary=launch_workers(path,worker_module='rhpr.gr.generator_age',event_prefix='age')
    return dict(teachers=3,new_training_epochs=5,downstream_fits=0,test_evaluations=0,shared_proposals=len(selected),
        common_pool_pairs=len(pool),reserved_scored_rows_per_teacher=reserved,elapsed_seconds=time.monotonic()-started,
        **{f"epoch{r['epoch']}_accepted":r['accepted'] for r in summary['results'].values()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--slot',type=int,required=True)
    a=p.parse_args();run_worker(a.plan,a.slot)
