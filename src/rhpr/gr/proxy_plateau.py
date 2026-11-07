"""Bounded exact continuation of six full TIGERs; no fresh models or tokenizer fits."""
import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.proxy_target import INDEX_IDS, TARGET_SEED, _stats, validate_target


def validate_plateau(c):
    t=c['training']
    recovery=c.get('recovery_parent')
    if recovery and (recovery['kernel']!='carasik/rhpr-ml1m-proxy-plateau'
            or recovery['source_code_sha']!='64500ca852e7bc478df3127e3421d661f36bfa12'
            or set(recovery['completed_trials'])!={f'full_{m}_seed{s}_gen161803' for s in [42,101] for m in ['sum','gradnorm']}):
        raise ValueError('recovery must reuse all four finished42/101 models; only202 may train')
    if recovery:
        required={'experiment.json','plan.json'}|{f'trials/{i}/{n}' for i in recovery['completed_trials']
            for n in ['generator-last.pt','generator-best.pt','generator.json','training-curves.json','training-selection.json','resume-verification.json']}
        if not required<=set(recovery['files']):raise ValueError('all completed checkpoints and provenance must be pinned')
    if (t['max_epochs']!=100 or t['min_epochs']!=60 or t['patience_epochs']!=20
            or t['max_seconds']!=5000 or 'scheduler' in t or c['suite_deadline_seconds']!=(5500 if recovery else 15500)
            or c['continuation_parent']['kernel']!='carasik/rhpr-ml1m-proxy-target'
            or c['continuation_parent']['source_code_sha']!='5ed498804545ea41686eee9eaa8d9f67747be0db'
            or set(c['continuation_parent']['artifacts'])!={x['id'] for x in c['trials']}):
        raise ValueError('plateau continuation is six epoch30 resumes, min60/max100/patience20, unchanged LR')
    old=copy.deepcopy(c);old['calibration_phase']='full_seed_replication'
    old['training'].update(max_epochs=30,min_epochs=30,patience_epochs=30,max_seconds=2300)
    old['suite_deadline_seconds']=7500
    extra={c['continuation_parent']['kernel']}|({recovery['kernel']} if recovery and not recovery.get('external_records') else set())
    old['kaggle']['kernel_sources']=[k for k in old['kaggle']['kernel_sources'] if k not in extra]
    validate_target(old)
    if set(c['kaggle']['kernel_sources'])!={p['kernel'] for p in c['parents'].values()}|extra:
        raise ValueError('only original frozen-index parents and completed continuation parent may be mounted')
    for row in c['continuation_parent']['artifacts'].values():
        if set(row)!={'generator-last.pt','generator-best.pt','training-curves.json','training-selection.json','generator.json'}:
            raise ValueError('complete optimizer/RNG checkpoint and provenance must be pinned')
    if recovery and recovery.get('external_records'):
        records=recovery['external_records']
        if set(records)!=set(recovery['completed_trials']):raise ValueError('all four external results are required')
        for tid,r in records.items():
            validate_completion(r['selection'],c)
            i=r['index_id'];seed=i.split('_seed')[1]
            if (r['trial']!=tid or r['initial_state_sha256']!=c['initialization_pins']['full'][str(TARGET_SEED)]
                    or r['generator']['sids_sha256']!=c['parents'][seed]['artifacts'][i]['semantic-ids.npy']):
                raise ValueError('external result initializer/index changed')
            if max(x['validation']['ndcg@10'] for x in r['training_curves'] if 'validation' in x)!=r['selection']['validation_ndcg10']:
                raise ValueError('external result selection changed')


def verify_continuation(c):
    from rhpr.gr.resources import verify_parent_artifacts
    source=verify_parent_artifacts(c['continuation_parent'])
    original=json.loads((source/'experiment.json').read_text());validate_target(original)
    for k in ['seed','data','generator','proxy_reference','index_ids','trials']:
        if original[k]!=c[k]:raise ValueError('continuation changes '+k)
    for trial,files in c['continuation_parent']['artifacts'].items():
        for name,expected in files.items():
            p=source/'trials'/trial/name
            if p.stat().st_size!=expected['size_bytes'] or file_sha256(p)!=expected['sha256']:
                raise ValueError('continuation artifact changed: '+trial+'/'+name)
    return source


def verify_recovery(c, input_root=Path('/kaggle/input')):
    """A failed parent has no run-manifest; pin its exact plan/config and all reused bytes."""
    parent=c['recovery_parent'];owner,slug=parent['kernel'].split('/')
    if parent.get('external_records'):
        validate_plateau(c)
        return 'external-verified-results-checkpoints-remain-in-original-kernel'
    roots=[Path(parent['path'])] if parent.get('path') else []
    roots += [input_root/slug,input_root/owner/slug,input_root/'notebooks'/owner/slug,input_root/'kernels'/owner/slug]
    source=next((p for p in roots if (p/'experiment.json').is_file()),None)
    if source is None and input_root.exists():
        source=next((p.parent for p in input_root.rglob('experiment.json') if slug in p.parts),None)
    if source is None:raise FileNotFoundError('failed plateau recovery source is not mounted')
    for name,expected in parent['files'].items():
        p=source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not p.resolve().is_relative_to(source.resolve()):raise ValueError('unsafe recovery path')
        if p.stat().st_size!=expected['size_bytes'] or file_sha256(p)!=expected['sha256']:raise ValueError('recovery bytes changed: '+name)
    old=json.loads((source/'experiment.json').read_text());validate_plateau(old)
    for k in ['seed','data','generator','training','proxy_reference','index_ids','trials']:
        if old[k]!=c[k]:raise ValueError('recovery changes '+k)
    return source


def validate_completion(selection,c):
    expected=((selection['training_examples_per_epoch']+c['training']['batch_size']-1)
              //c['training']['batch_size'])*selection['training_epochs']
    # The engine counts successful Adam steps separately from GradScaler skips.
    skipped=selection['amp_skipped_updates']
    if (selection['resumed_from_epoch']!=30 or not 60<=selection['training_epochs']<=100
            or selection['stopping_reason'] not in ['early_stopping','epoch_limit']
            or not 0<=skipped<=16 or selection['optimizer_updates']+skipped!=expected):
        raise RuntimeError('continuation incomplete; do not report a budget exit as plateau')


def record_trial(trial,c,out,initial,selection,**extra):
    validate_completion(selection,c)
    curves=json.loads((out/'training-curves.json').read_text())
    row=dict(trial=trial['id'],architecture='full',index_id=trial['index_id'],seed=trial['seed'],
        initialization_group=f"full_{trial['seed']}",initial_state_sha256=initial,selection=selection,
        plateau_reached=selection['stopping_reason']=='early_stopping',budget_censored=selection['stopping_reason']=='epoch_limit',
        epochs={str(x['epoch']):x['validation']['ndcg@10'] for x in curves if 'validation' in x},**extra)
    _write(out/'trial-result.json',row);return row


def run_trial(trial,c,plan,device):
    out=Path(plan['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    if trial['id'] in c.get('recovery_parent',{}).get('completed_trials',[]):
        external=c['recovery_parent'].get('external_records',{}).get(trial['id'])
        if external:
            for name,key in [('generator.json','generator'),('training-curves.json','training_curves'),('training-selection.json','selection'),('resume-verification.json','resume_verification')]:_write(out/name,external[key])
            _write(out/'external-checkpoint-reference.json',dict(kernel=c['recovery_parent']['kernel'],version=1,trial=trial['id'],source_code_sha=c['recovery_parent']['source_code_sha'],files={n:v for n,v in c['recovery_parent']['files'].items() if n.startswith('trials/'+trial['id']+'/')},weights_and_predictions_in_original_parent=True))
            return record_trial(trial,c,out,external['initial_state_sha256'],external['selection'],reused_completed=True,recovery_kernel=c['recovery_parent']['kernel'],new_training_updates=0,external_artifacts=True)
        source=Path(plan['recovery_root'])/'trials'/trial['id']
        for name in c['recovery_parent']['files']:
            if Path(name).parent==Path('trials')/trial['id']:shutil.copyfile(Path(plan['recovery_root'])/name,out/Path(name).name)
        selection=json.loads((out/'training-selection.json').read_text());g=json.loads((out/'generator.json').read_text())
        if g['initial_state_sha256']!=c['initialization_pins']['full'][str(trial['seed'])] or g['sids_sha256']!=plan['references'][trial['index_id']]['sids_sha256']:raise ValueError('recovered generator differs')
        return record_trial(trial,c,out,g['initial_state_sha256'],selection,reused_completed=True,recovery_kernel=c['recovery_parent']['kernel'],new_training_updates=0)
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    out=Path(plan['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    sid=Path(plan['index_root'])/(trial['index_id']+'-semantic-ids.npy')
    if file_sha256(sid)!=plan['references'][trial['index_id']]['sids_sha256']:raise ValueError('SID changed')
    index=SidIndex.from_full_sids(np.load(sid,allow_pickle=False),width=256)
    seed=trial['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_grid_generator(index.sids,c['generator'],seed,device,plan['source'])
    initial=state_digest(dict(model.named_parameters()))
    if initial!=c['initialization_pins']['full'][str(seed)]:raise ValueError('initialization changed')
    _write(out/'generator.json',dict(config=c['generator'],seed=seed,parameters=sum(p.numel() for p in model.parameters()),
        initial_state_sha256=initial,sids_sha256=file_sha256(sid),frozen_index=True,resumed=True,test_evaluated=False))
    source=Path(plan['resume_root'])/'trials'/trial['id']
    selection=_train_generator(model,plan['data_root'],index,c,out,device,example_builder=case_examples,resume_from=source)
    return record_trial(trial,c,out,initial,selection,reused_completed=False)


def summarize(c,records,refs,output):
    if len(records)!=6 or set(records)!={t['id'] for t in c['trials']}:raise ValueError('missing continuation')
    rows={r['index_id']:r for r in records.values()};ids=c['index_ids']
    target=[rows[i]['selection']['validation_ndcg10'] for i in ids]
    scores={f'mini_gen{s}_10':[c['proxy_reference']['indexes'][i]['scores_by_seed'][str(s)] for i in ids] for s in [31415,27182]}
    scores['mini_mean10']=np.mean(list(scores.values()),axis=0).tolist()
    scores['same_full30']=[rows[i]['epochs']['30'] for i in ids]
    scores['historical_full30']=[refs[i]['epochs']['30'] for i in ids]
    result=dict(target='best VAL NDCG@10 under min60/max100/patience20 continuation rule',index_ids=ids,
        target_scores=dict(zip(ids,target)),test_evaluated=False,
        plateau_reached=sum(r['plateau_reached'] for r in rows.values()),budget_censored=sum(r['budget_censored'] for r in rows.values()),
        estimators={k:_stats(v,target,ids) for k,v in scores.items()},estimator_scores=scores,
        caveats=['Same previously examined six indexes/VAL; one common generator seed.',
                 'Epoch100 cap is censoring, not proof of plateau; ES rule is operational, not mathematical convergence.',
                 'This selected-checkpoint target differs from the previous fixed30 primary; report both.'])
    _write(Path(output)/'correlations.json',result)
    metrics=dict(experiment=c['name'],completed=6,test_evaluated=False,plateau_reached=result['plateau_reached'],budget_censored=result['budget_censored'])
    for k,v in result['estimators'].items():
        if v['spearman'] is not None:metrics[k+'_spearman']=v['spearman']
    return metrics
