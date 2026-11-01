"""Continue four verified seed42 predictors with their frozen historical indices."""
import argparse,copy,gc,json,os,shutil,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _write,_event


def validate(config):
    if (config['seed']!=42 or len(config['trials'])!=4 or config['training']['max_epochs']!=100
            or config['training']['min_epochs']!=30 or config['training']['validation_interval']!=1
            or config['training']['patience_epochs']!=20
            or config['training']['scheduler']!={'kind':'reduce_on_plateau','factor':.5,'patience_epochs':5,'min_lr':1e-5,'threshold':1e-4,'start_epoch':21}
            or config['data']['history_items']!=20 or not config['evaluation']['save_validation_losses']
            or {t['id'] for t in config['trials']}!={'tiger_no_people_seed42','tiger_graph_seed42','tiger_names_seed42','sasrec_seed42'}
            or any(t['seed']!=42 for t in config['trials']) or config['compare_initialization_by_shape'] is not True):
        raise ValueError('continuation differs from approved four-model seed42 pilot')
    for trial in config['trials']:
        required={'generator-last.pt','generator-best.pt','training-curves.json','training-selection.json','generator.json'}
        if trial['predictor']=='tiger':required.update({'semantic-codes.npy','semantic-ids.npy','sid-catalogue.json'})
        if not required<=trial['files'].keys() or any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in trial['files'].values()):
            raise ValueError('continuation source artifacts must be hash-pinned')


def verify_parents(config,source_ids,split):
    parents={}
    for name,spec in config['parents'].items():
        parent=next((Path(p) for p in [spec['path'],spec['legacy_path']] if Path(p).is_dir()),None)
        if parent is None:raise FileNotFoundError('declared continuation notebook output missing')
        if file_sha256(parent/'run-manifest.json')!=spec['manifest_sha256']:raise ValueError('continuation parent manifest changed')
        m=json.loads((parent/'run-manifest.json').read_text())
        if m['git_sha']!=spec['git_sha'] or json.loads((parent/'split-report.json').read_text())!=split:
            raise ValueError('continuation parent executable/split differs')
        parents[name]=parent
    directories={}
    for trial in config['trials']:
        directory=parents[trial['parent']]/'trials'/trial['source_trial']
        for name,digest in trial['files'].items():
            if Path(name).name!=name or file_sha256(directory/name)!=digest:raise ValueError('continuation artifact changed: '+name)
        if trial['predictor']=='tiger' and json.loads((directory/'sid-catalogue.json').read_text())['original_item_ids']!=source_ids:
            raise ValueError('continuation SID catalogue order differs')
        directories[trial['id']]=str(directory)
    return directories


def run_trial(trial,config,plan,users,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.movie_sasrec import build_movie_sasrec,ItemIDIndex
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.movie_data import temporal_examples
    from rhpr.gr.experiment import _train_generator,_evaluate
    source=Path(plan['directories'][trial['id']]);root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(source/'generator-last.pt',map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=20 or checkpoint['updates']!=6680 or checkpoint['config']['seed']!=42:
        raise ValueError('pilot must continue an intact20epoch seed42 fit')
    c=copy.deepcopy(checkpoint['config']);c['training'].update(config['training']);c['evaluation'].update(config['evaluation'])
    if c['data']!=config['data'] or c['generator']!=config['generator']:raise ValueError('source predictor/data differs')
    for name in ['generator.json','quantizer.json','representation.json','sid-diagnostics.json','sid-catalogue.json','semantic-codes.npy','semantic-ids.npy']:
        if (source/name).exists():shutil.copy2(source/name,root/name)
    if trial['predictor']=='tiger':
        index=SidIndex(np.load(source/'semantic-codes.npy'),width=c['quantizer']['codebook_size'])
        np.testing.assert_array_equal(index.sids,np.load(source/'semantic-ids.npy'))
        model=build_grid_generator(index.sids,c['generator'],42,device,plan['source'])
    else:
        index=ItemIDIndex(c['data']['items']);model=build_movie_sasrec(c,42,device)
    initial=state_digest(dict(model.named_parameters()));original=json.loads((source/'generator.json').read_text())
    if original['initial_state_sha256']!=initial:raise ValueError('predictor constructor differs from saved architecture')
    del checkpoint;gc.collect()
    _write(root/'continuation-source.json',dict(parent=config['parents'][trial['parent']],source_trial=trial['source_trial'],files=trial['files'],
        frozen_index=True,offline_encoder_and_quantizer_retrained=False,prefix_validation_epochs=[5,10,15,20],continuation_validation_interval=1))
    selection=_train_generator(model,users,index,c,root,device,example_builder=temporal_examples,resume_from=source)
    examples=temporal_examples(users,c['data']['history_items'],'test')
    metrics,predictions,scores=_evaluate(model,examples,index,c['training']['evaluation_batch_size'],device,'fp32')
    np.savez_compressed(root/'test-predictions.npz',**{getattr(index,'prediction_array_name','semantic_ids'):predictions},scores=scores,user_ids=examples[1],target_item_ids=examples[2])
    result=dict(trial=trial['id'],seed=42,predictor=trial['predictor'],initial_state_sha256=initial,
        initialization_group=trial['predictor']+'_seed42',selection=selection,test=metrics,
        index_frozen=True,budget_censored=selection['training_epochs']==100 and selection['stopping_reason']=='epoch_limit')
    _write(root/'trial-result.json',result)
    return result


def run_worker(plan_path,slot):
    import torch
    from rhpr.gr.movie_data import load_temporal_users
    plan=json.loads(Path(plan_path).read_text());config=plan['config'];job=plan['assignments'][slot]
    torch.set_num_threads(config['execution']['cpu_threads_per_worker'])
    device='cpu'
    if job['device']!='cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('continuation worker needs its assigned GPU; no fallback')
        torch.cuda.set_device(0);device='cuda:0'
    users=load_temporal_users(Path(plan['data_root'])/'sequences.json',plan['source_ids'])
    status=dict(slot=slot,status='running',results={},failures=[]);path=Path(plan['output'])/'workers'/f'worker-{slot}.json';_write(path,status)
    for trial in job['trials']:
        _event('movie_continuation_trial_start',trial=trial['id'],worker=slot)
        try:
            result=run_trial(trial,config,plan,users,device);status['results'][trial['id']]=result
            _event('movie_continuation_trial_complete',trial=trial['id'],selection=result['selection'],test=result['test'])
        except Exception as e:
            root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True);(root/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)))
            _event('movie_continuation_trial_failed',**status['failures'][-1])
        _write(path,status);gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    status['status']='complete';_write(path,status)


def run_continuation(config,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.resources import verify_resources,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=assignments(config,torch.cuda.device_count())
    resources=resolve_dataset_directory(config['resources']['path']);verify_resources(resources,config['resources']['manifest_sha256'])
    data=resolve_dataset_directory(config['movie_input']['path']);verify_resources(data,config['movie_input']['manifest_sha256'])
    ids=[r['item_id'] for r in json.loads((data/'catalog-stage-01.json').read_text())];users=load_temporal_users(data/'sequences.json',ids);split=json.loads((data/'split-report.json').read_text())
    if len(ids)!=config['data']['items'] or len(users)!=config['data']['users']:raise ValueError('continuation input counts differ')
    for name in ['train','validation','test']:
        if case_digest(temporal_examples(users,config['data']['history_items'],name))!=split['case_sha256'][name]:raise ValueError('continuation cases differ')
    directories=verify_parents(config,ids,split)
    original=unpack_grid(resources/'grid-source.bin',Path(config['resources']['scratch_path'])/'original');source=prepare_grid_source(original,Path(config['resources']['scratch_path'])/'runtime')
    (output/'workers').mkdir(exist_ok=True);_write(output/'split-report.json',split)
    _write(output/'input-provenance.json',dict(source_ids=ids,case_sha256=split['case_sha256'],parents=config['parents'],frozen_indices=True,config=config))
    plan=dict(config=config,output=str(output),source=str(source),data_root=str(data),source_ids=ids,directories=directories,assignments=jobs)
    path=output/'workers/plan.json';_write(path,plan)
    result=launch_workers(path,worker_module='rhpr.gr.movie_continuation',event_prefix='movie_continuation')
    metrics=dict(experiment=config['name'],seed=42,trials=len(result['results']),items=len(ids),users=len(users),resume_epoch=20,max_epochs=100)
    for name,r in result['results'].items():
        metrics.update({name+'_test_'+k.replace('@',''):v for k,v in r['test'].items()})
        for key in ['training_epochs','best_epoch','continuation_epochs','optimizer_updates']:metrics[name+'_'+key]=r['selection'][key]
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',required=True,type=int);a=p.parse_args();run_worker(a.plan,a.slot)
