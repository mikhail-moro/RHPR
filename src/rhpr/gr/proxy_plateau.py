"""Bounded exact continuation of six full TIGERs; no fresh models or tokenizer fits."""
import copy
import json
import random
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _train_generator, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.proxy_target import INDEX_IDS, TARGET_SEED, _stats, validate_target


def validate_plateau(c):
    t=c['training']
    if (t['max_epochs']!=100 or t['min_epochs']!=60 or t['patience_epochs']!=20
            or t['max_seconds']!=5000 or 'scheduler' in t or c['suite_deadline_seconds']!=15500
            or c['continuation_parent']['kernel']!='carasik/rhpr-ml1m-proxy-target'
            or c['continuation_parent']['source_code_sha']!='5ed498804545ea41686eee9eaa8d9f67747be0db'
            or set(c['continuation_parent']['artifacts'])!={x['id'] for x in c['trials']}):
        raise ValueError('plateau continuation is six epoch30 resumes, min60/max100/patience20, unchanged LR')
    old=copy.deepcopy(c);old['calibration_phase']='full_seed_replication'
    old['training'].update(max_epochs=30,min_epochs=30,patience_epochs=30,max_seconds=2300)
    old['suite_deadline_seconds']=7500
    old['kaggle']['kernel_sources']=[k for k in old['kaggle']['kernel_sources'] if k!=c['continuation_parent']['kernel']]
    validate_target(old)
    if set(c['kaggle']['kernel_sources'])!={p['kernel'] for p in c['parents'].values()}|{c['continuation_parent']['kernel']}:
        raise ValueError('only original frozen-index parents and completed continuation parent may be mounted')
    for row in c['continuation_parent']['artifacts'].values():
        if set(row)!={'generator-last.pt','generator-best.pt','training-curves.json','training-selection.json','generator.json'}:
            raise ValueError('complete optimizer/RNG checkpoint and provenance must be pinned')


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


def run_trial(trial,c,plan,device):
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
    expected_updates=((selection['training_examples_per_epoch']+c['training']['batch_size']-1)
                      //c['training']['batch_size'])*selection['training_epochs']
    if (selection['resumed_from_epoch']!=30 or selection['training_epochs']<60
            or selection['stopping_reason'] not in ['early_stopping','epoch_limit']
            or selection['optimizer_updates']!=expected_updates):
        raise RuntimeError('continuation incomplete; do not report a budget exit as plateau')
    curves=json.loads((out/'training-curves.json').read_text())
    row=dict(trial=trial['id'],architecture='full',index_id=trial['index_id'],seed=seed,
        initialization_group=f'full_{seed}',initial_state_sha256=initial,selection=selection,
        plateau_reached=selection['stopping_reason']=='early_stopping',budget_censored=selection['stopping_reason']=='epoch_limit',
        epochs={str(x['epoch']):x['validation']['ndcg@10'] for x in curves if 'validation' in x})
    _write(out/'trial-result.json',row);return row


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
