"""Fresh TIGER predictors with longer contexts and frozen historical indices."""
import argparse,copy,gc,json,random,shutil,time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_write


def validate(config):
    seeds=config['replicate_seeds'];windows=config['context_windows'];trials=config['trials']
    if (tuple(seeds),tuple(windows))not in {((42,),(30,40)),((101,202),(40,))}or config['seed']!=seeds[0]:
        raise ValueError('context design must be the seed42pilot or frozen40item101/202replication')
    expected={(c,h,s)for c in ['none','actor_w01']for h in windows for s in seeds}
    if (len(trials)!=len(expected)or {(t['candidate'],t['history_items'],t['seed'])for t in trials}!=expected
            or len({t['id']for t in trials})!=len(trials)):
        raise ValueError('context matrix must contain each control/graph/window/seed exactly once')
    for t in trials:
        if t['id']!=f"{t['candidate']}_L{t['history_items']}_seed{t['seed']}"or t['source_trial']!=f"{t['candidate']}_seed{t['seed']}":raise ValueError('invalid context trial lineage')
        if set(t['files'])!={'semantic-codes.npy','semantic-ids.npy','sid-catalogue.json','trial-result.json','training-curves.json'}:raise ValueError('frozen index lineage must be pinned')
    t=config['training']
    if (t['max_epochs']!=30 or t['min_epochs']!=30 or t['patience_epochs']!=20
            or t['validation_interval']!=1 or t['validate_first_epoch']or t['batch_size']!=256
            or t['physical_batch_size']!=128 or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
            or t['scheduler']!={'kind':'reduce_on_plateau','factor':.5,'patience_epochs':5,'min_lr':1e-5,'threshold':1e-4,'start_epoch':21}
            or t['max_seconds']!=7200 or config['generator']['beam_width']!=20
            or config['generator']['should_check_prefix']or config['generator'].get('attention_backend','eager')!='eager'
            or config.get('known_item_filter',False)):
        raise ValueError('context pilot must retain full original eager/FP16/unmasked training protocol')
    if config['data']['items']!=1273 or config['data']['users']!=943:raise ValueError('context pilot catalogue differs')


def verify_schedules(users,history_items,expected_hashes=None):
    """Only visible context changes; targets/user order and old suffix must match."""
    from rhpr.gr.movie_data import temporal_examples,case_digest
    result={}
    for split in ['train','validation','test']:
        base=temporal_examples(users,20,split);long=temporal_examples(users,history_items,split)
        if expected_hashes and case_digest(base)!=expected_hashes[split]:raise ValueError('reference temporal schedule differs')
        np.testing.assert_array_equal(long[1],base[1]);np.testing.assert_array_equal(long[2],base[2]);np.testing.assert_array_equal(long[0][:,-20:],base[0])
        result[split]=dict(examples=len(base[2]),reference_sha256=case_digest(base),context_sha256=case_digest(long),targets_and_users_unchanged=True,last20_suffix_unchanged=True)
    return result


def run_trial(trial,config,plan,users,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.movie_data import temporal_examples
    from rhpr.gr.movie_graph_loss import train_generator
    from rhpr.gr.experiment import _evaluate
    source=Path(plan['parent'])/'trials'/trial['source_trial'];out=Path(plan['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    c=copy.deepcopy(config);seed=trial['seed'];c['seed']=seed;c['data']['history_items']=trial['history_items']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    index=SidIndex(np.load(source/'semantic-codes.npy'),width=c['generator']['codebook_size']);np.testing.assert_array_equal(index.sids,np.load(source/'semantic-ids.npy'))
    for n in ['semantic-codes.npy','semantic-ids.npy','sid-catalogue.json']:shutil.copy2(source/n,out/n)
    original=json.loads((source/'trial-result.json').read_text());expected=original['initial_state_sha256']
    model=build_grid_generator(index.sids,c['generator'],seed,device,plan['source']);initial=state_digest(dict(model.named_parameters()))
    if initial!=expected or initial!=c['expected_initial_state_sha256_by_seed'][str(seed)]:raise ValueError('fresh predictor initialization differs from original seed')
    _write(out/'generator.json',dict(config=c['generator'],initial_state_sha256=initial,parameters=sum(p.numel()for p in model.parameters())))
    _write(out/'frozen-index-source.json',dict(source_kernel=config['parent']['kernel'],source_trial=trial['source_trial'],files=trial['files'],learned_generator_weights_loaded=False,quantizer_retrained=False,reference20_selection=original['selection'],reference20_test=original['test']))
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    selection=train_generator(model,users,index,c,out,device)
    peak=torch.cuda.max_memory_allocated()if device.startswith('cuda')else 0;test=temporal_examples(users,trial['history_items'],'test')
    metrics,pred,scores=_evaluate(model,test,index,c['training']['evaluation_batch_size'],device,'fp32')
    np.savez_compressed(out/'test-predictions.npz',semantic_ids=pred,scores=scores,user_ids=test[1],target_item_ids=test[2])
    # Repeat occupancy remains a diagnostic; it never filters or changes ranks.
    from rhpr.gr.movie_seen_serving import decode_paths,known_sets
    decoded=decode_paths(pred,index.sids)[:,:10];known=known_sets(users,test[1],'test');repeat=float(np.mean([int(i)in seen for row,seen in zip(decoded,known)for i in row]))
    result=dict(trial=trial['id'],seed=seed,candidate=trial['candidate'],history_items=trial['history_items'],initial_state_sha256=initial,
        selection=selection,test=metrics,test_known_item_top10_fraction=repeat,peak_training_memory_allocated_bytes=peak,wall_seconds=time.monotonic()-started,
        mean_epoch_training_seconds=selection['optimizer_training_seconds']/selection['training_epochs'],index_frozen=True,generator_fresh=True,
        budget_censored=True,fixed_budget_epochs=30,known_item_filter=False,attention_backend='original_T5_eager')
    curves=json.loads((source/'training-curves.json').read_text());reference=[r for r in curves if r['epoch']<=30 and 'validation'in r]
    result['reference20_matched_budget_validation']=dict(best_ndcg10=max(r['validation']['ndcg@10']for r in reference),epoch30=next(r['validation']for r in reference if r['epoch']==30),checks=[r['epoch']for r in reference],test_budget_matched=False)
    _write(out/'trial-result.json',result);return result


def run_worker(plan_path,slot):
    import torch
    from rhpr.gr.movie_data import load_temporal_users
    plan=json.loads(Path(plan_path).read_text());torch.set_num_threads(1)
    if torch.cuda.device_count()!=1:raise RuntimeError('context worker requires its assigned GPU; no fallback')
    users=load_temporal_users(plan['sequences'],plan['source_ids']);status=dict(results={},failures=[])
    for trial in plan['assignments'][slot]['trials']:
        _event('movie_context_trial_start',worker=slot,trial=trial['id'],history_items=trial['history_items'])
        try:
            status['results'][trial['id']]=run_trial(trial,plan['config'],plan,users,'cuda:0')
            _event('movie_context_trial_complete',trial=trial['id'],result=status['results'][trial['id']])
        except Exception as e:
            import traceback
            out=Path(plan['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True);(out/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)))
            _event('movie_context_trial_failed',**status['failures'][-1])
        _write(Path(plan['output'])/'workers'/f'worker-{slot}.json',status);gc.collect();torch.cuda.empty_cache()
    _write(Path(plan['output'])/'workers'/f'worker-{slot}.json',status)


def run_context_training(config,output):
    import torch
    from rhpr.gr.resources import verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.data import file_sha256
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.movie_data import load_temporal_users
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    parent=verify_parent_artifacts(config['parent']);pc=json.loads((parent/'run-manifest.json').read_text())['config']
    if config['generator']!=pc['generator']:raise ValueError('context pilot must preserve predictor architecture')
    a=copy.deepcopy(config['training']);b=copy.deepcopy(pc['training']);a.pop('max_epochs');b.pop('max_epochs')
    if a!=b:raise ValueError('training settings changed beyond the30epoch budget and context')
    for trial in config['trials']:
        p=parent/'trials'/trial['source_trial']
        for n,sha in trial['files'].items():
            if Path(n).name!=n or file_sha256(p/n)!=sha:raise ValueError('parent index/lineage bytes changed')
    ids=json.loads((parent/'trials'/config['trials'][0]['source_trial']/'sid-catalogue.json').read_text())['original_item_ids']
    for t in config['trials']:
        if json.loads((parent/'trials'/t['source_trial']/'sid-catalogue.json').read_text())['original_item_ids']!=ids:raise ValueError('catalogue order changed')
    data=resolve_dataset_directory(config['movie_input']['path']);seq=data/'sequences.json'
    if file_sha256(seq)!=config['movie_input']['sequences_sha256']:raise ValueError('sequence bytes changed')
    users=load_temporal_users(seq,ids);hashes=json.loads((parent/'split-report.json').read_text())['case_sha256']
    schedules={str(h):verify_schedules(users,h,hashes)for h in config['context_windows']};_write(output/'context-schedule-audit.json',schedules)
    resources=resolve_dataset_directory(config['resources']['path']);manifest=resources/'resources-manifest.json'
    if file_sha256(manifest)!=pc['resources']['manifest_sha256']:raise ValueError('resources changed')
    entry=next(r for r in json.loads(manifest.read_text())['files']if r['path']=='grid-source.bin')
    if file_sha256(resources/'grid-source.bin')!=entry['sha256']:raise ValueError('GRID source changed')
    scratch=Path('/kaggle/temp/rhpr-movie-context');original=unpack_grid(resources/'grid-source.bin',scratch/'original');source=prepare_grid_source(original,scratch/'runtime')
    plan=dict(config=config,parent=str(parent),source=str(source),sequences=str(seq),source_ids=ids,output=str(output),assignments=assignments(config,torch.cuda.device_count()))
    _write(output/'worker-plan.json',plan);_event('movie_context_ready',trials=len(config['trials']),windows=config['context_windows'],seeds=config['replicate_seeds'],quantizer_retrained=False)
    summary=launch_workers(output/'worker-plan.json',worker_module=__name__,event_prefix='movie_context');metrics={}
    for h in config['context_windows']:
        for candidate in ['none','actor_w01']:
            rows=[r for r in summary['results'].values()if r['history_items']==h and r['candidate']==candidate]
            for name,value in [('validation_ndcg10',np.mean([r['selection']['validation_ndcg10']for r in rows])),('test_ndcg10',np.mean([r['test']['ndcg@10']for r in rows])),('epoch_training_seconds',np.mean([r['mean_epoch_training_seconds']for r in rows])),('peak_training_memory_bytes',max(r['peak_training_memory_allocated_bytes']for r in rows))]:metrics[f'L{h}_{candidate}_{name}']=float(value)
    metrics.update(completed_models=float(len(summary['results'])),quantizer_training_updates=0.);return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();run_worker(a.plan,a.slot)
