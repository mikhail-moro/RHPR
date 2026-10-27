"""Inference-only replay of byte-pinned selected TIGER checkpoints on validation."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.experiment import _evaluate, _event, _write
from rhpr.gr.study_parallel import atomic_json


def replay_trial(trial, config, resources, parents, source, output, device):
    import numpy as np
    import torch
    from rhpr.gr.generator_suite import verify_file
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    parent=Path(parents[trial['parent']])
    paths={k:verify_file(parent,trial[k]) for k in ['checkpoint','sids','selection']}
    parent_manifest=json.loads((parent/'run-manifest.json').read_text())
    for key in ['data','generator','seed']:
        if parent_manifest['config'][key]!=config[key]:
            raise ValueError('parent validation protocol mismatch: '+key)
    checkpoint=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False)
    checkpoint_epoch=trial.get('checkpoint_epoch',config['validation']['checkpoint_epoch'])
    if type(checkpoint_epoch) is not int or checkpoint_epoch < 1 or checkpoint['epoch']!=checkpoint_epoch:
        raise ValueError('selected checkpoint epoch mismatch')
    for key in ['data','generator','seed']:
        if checkpoint['config'][key]!=config[key]:
            raise ValueError('checkpoint protocol mismatch: '+key)
    if checkpoint['config']['training']!=trial.get('expected_training',config['expected_training']):
        raise ValueError('checkpoint training budget differs')
    sids=np.load(paths['sids'],allow_pickle=False);index=SidIndex.from_full_sids(sids,config['generator']['codebook_size'])
    model=build_grid_generator(sids,config['generator'],config['seed'],device,source)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;gc.collect()
    before=state_digest(model.state_dict())
    _,users,_,_=load_beauty(Path(resources)/'data',config['data'])
    examples=build_examples(users,config['data']['history_items'],'validation')
    if config.get('context_ablation'):
        from rhpr.gr.context_ablation import evaluate_context_ablation
        reference_parent=Path(parents[trial.get('reference_predictions_parent',trial['parent'])])
        paths['reference_predictions']=verify_file(reference_parent,trial['reference_predictions'])
        return evaluate_context_ablation(model,index,examples,users,trial,config,paths,before,output,device)
    score,predictions,probabilities=_evaluate(model,examples,index,config['validation']['batch_size'],device,
                                            precision=config['validation']['precision'])
    if state_digest(model.state_dict())!=before:
        raise RuntimeError('model state changed during read-only validation')
    directory=Path(output)/'trials'/trial['id'];directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'validation-predictions.npz',semantic_ids=predictions,scores=probabilities,
        user_ids=examples[1],target_item_ids=examples[2])
    hit=np.all(predictions==sids[examples[2],None,:],axis=-1)
    rank=np.where(hit.any(1),hit.argmax(1)+1,999)
    per_user={f'ndcg@{k}':np.where(rank<=k,1/np.log2(rank+1),0) for k in [5,10]}
    per_user.update({f'recall@{k}':(rank<=k).astype(float) for k in [5,10]})
    for key,value in per_user.items():
        if abs(float(value.mean())-score[key])>1e-12:
            raise RuntimeError('user-level metric disagrees with aggregate')
    np.savez_compressed(directory/'validation-user-metrics.npz',user_ids=examples[1],target_item_ids=examples[2],
                        rank=rank,**per_user)
    reference=json.loads(paths['selection'].read_text())['selected_validation']
    result=dict(trial=trial['id'],seed=config['seed'],epoch=checkpoint_epoch,
        validation=score,original_validation=reference,
        replay_delta={k:score[k]-reference[k] for k in per_user},
        checkpoint_sha256=file_sha256(paths['checkpoint']),sids_sha256=file_sha256(paths['sids']),
        loaded_state_sha256=before,model_unchanged=True,optimizer_steps=0,training_performed=False,
        test_evaluations=0,validation_evaluations=1)
    _write(directory/'result.json',result)
    return result


def run_worker(plan_path, slot):
    import torch
    plan=json.loads(Path(plan_path).read_text());config=plan['config'];assignment=plan['assignments'][slot]
    torch.set_num_threads(config['execution']['cpu_threads_per_worker']);device='cpu'
    if assignment['device']!='cpu':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('validation worker requires exactly its assigned GPU')
        torch.cuda.set_device(0);device='cuda:0'
    output=Path(plan['output']);path=output/'workers'/f'worker-{slot}.json'
    status=dict(status='running',results={},failures=[],device=assignment['device'])
    for trial in assignment['trials']:
        status['current_trial']=trial['id'];atomic_json(path,status)
        _event('validation_replay_trial_start',trial=trial['id'],worker=slot)
        try:
            result=replay_trial(trial,config,plan['resources'],plan['parents'],plan['source'],output,device)
            status['results'][trial['id']]=result
            _event('validation_replay_complete',trial=trial['id'],validation=result['validation'],replay_delta=result['replay_delta'])
        except Exception as error:
            directory=output/'trials'/trial['id'];directory.mkdir(parents=True,exist_ok=True)
            (directory/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
        atomic_json(path,status);gc.collect()
        if device!='cpu':torch.cuda.empty_cache()
    status['status']='complete';atomic_json(path,status)


def run_validation_replay(config, output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.resources import verify_resources,verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.offline_runtime import resolve_dataset_directory

    if (config['validation']['precision']!='fp32' or config['validation']['batch_size']!=32
            or set(config['training'])!={'device'}):
        raise ValueError('validation replay is FP32 batch32 and has no training settings')
    started=time.monotonic();output=Path(output).resolve();jobs=assignments(config,torch.cuda.device_count())
    resources=resolve_dataset_directory(config['resources']['path'])
    manifest=verify_resources(resources,config['resources']['manifest_sha256'])
    parents={key:str(verify_parent_artifacts(spec)) for key,spec in config['parents'].items()}
    _,_,_,counts=load_beauty(resources/'data',config['data'])
    scratch=Path(config['resources']['scratch_path']);original=unpack_grid(resources/'grid-source.bin',scratch/'original')
    source=prepare_grid_source(original,scratch/'runtime')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    _write(output/'input-provenance.json',dict(resources=manifest,counts=counts,parents=config['parents'],training_performed=False))
    (output/'workers').mkdir()
    plan=dict(config=config,assignments=jobs,output=str(output),resources=str(resources),parents=parents,source=str(source))
    path=output/'workers/plan.json';atomic_json(path,plan)
    summary=launch_workers(path,worker_module='rhpr.gr.validation_replay',event_prefix='validation_replay')
    metrics=dict(models=len(summary['results']),training_performed=False,optimizer_steps=0,test_evaluations=0,
                 validation_evaluations=sum(r['validation_evaluations'] for r in summary['results'].values()),
                 elapsed_seconds=time.monotonic()-started)
    for name,result in summary['results'].items():
        for key in ['ndcg@5','ndcg@10','recall@5','recall@10']:
            metrics[name+'_validation_'+key.replace('@','')]=result['validation'][key]
            metrics[name+'_replay_delta_'+key.replace('@','')]=result['replay_delta'][key]
        for cohort in result.get('context_cohorts',[]):
            for length, score in cohort['metrics_by_length'].items():
                metrics[f'{name}_{cohort["id"]}_L{length}_ndcg10']=score['ndcg@10']
                metrics[f'{name}_{cohort["id"]}_L{length}_recall10']=score['recall@10']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',required=True,type=int);a=p.parse_args();run_worker(a.plan,a.slot)
