"""Two bounded intermediate mini-TIGER designs against frozen plateau references."""
import copy
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _write
from rhpr.gr.mini_tiger import MIDDLE_ARCHITECTURES
from rhpr.gr.proxy_target import INDEX_IDS, TARGET_SEED, _stats, reference_digest, validate_target


def validate_middle(c):
    expected=[dict(id=f'{a}_{i}_gen{s}',architecture=a,index_id=i,seed=s)
              for a in MIDDLE_ARCHITECTURES for i in INDEX_IDS for s in [31415,27182]]
    t=c['training']
    if (c['calibration_phase']!='middle_architectures' or c['seed']!=31415
            or c['generator_seeds']!=[31415,27182] or c['trials']!=expected
            or c['index_ids']!=INDEX_IDS or c['suite_deadline_seconds']!=8400
            or t['max_epochs']!=10 or t['min_epochs']!=10 or t['patience_epochs']!=10
            or t['max_seconds']!=1200 or c['primary_proxy_epoch']!=10
            or set(c['initialization_pins'])!=set(MIDDLE_ARCHITECTURES)
            or any(set(v)!={'31415','27182'} or any(len(p)!=64 for p in v.values()) for v in c['initialization_pins'].values())):
        raise ValueError('middle proxy screen is exactly24fits,2shapes/6indexes/2seeds,10epochs')
    # Preserve the exact full baseline data, model envelope, optimizer and VAL-only policy.
    base=copy.deepcopy(c);base.update(seed=TARGET_SEED,calibration_phase='full_seed_replication',
        generator_seeds=[TARGET_SEED],suite_deadline_seconds=7500)
    base['trials']=[dict(id=f'full_{i}_gen{TARGET_SEED}',architecture='full',index_id=i,seed=TARGET_SEED) for i in INDEX_IDS]
    base['training'].update(max_epochs=30,min_epochs=30,patience_epochs=30,max_seconds=2300)
    base['initialization_pins']={'full':{str(TARGET_SEED):c['reference_full_initialization_pin']}}
    validate_target(base)
    ref=c['plateau_reference']
    if (reference_digest(ref)!=c['plateau_reference_sha256'] or ref['target_seed']!=TARGET_SEED
            or ref['policy']!=dict(min_epochs=60,max_epochs=100,patience_epochs=20,validation_interval=5)
            or set(ref['indexes'])!=set(INDEX_IDS)):
        raise ValueError('fixed plateau reference changed')
    for i,r in ref['indexes'].items():
        if (r['sids_sha256']!=c['parents'][i.split('_seed')[1]]['artifacts'][i]['semantic-ids.npy']
                or not np.isfinite(r['selected_ndcg10']) or not 0<=r['selected_ndcg10']<=1
                or r['stop_epoch']-r['best_epoch']<20 or not 60<=r['stop_epoch']<=100):
            raise ValueError('plateau reference does not match frozen index or ES policy')


def summarize_middle(c,records,output):
    if len(records)!=24 or set(records)!={r['id'] for r in c['trials']}:
        raise ValueError('all24 proxy fits required; no partial calibration')
    for t in c['trials']:
        r=records[t['id']]
        if (r['trial']!=t['id'] or any(r[k]!=t[k] for k in ['architecture','index_id','seed'])
                or set(r['epochs'])!={'5','10'}):
            raise ValueError('proxy result identity or epoch schedule changed')
    ids=c['index_ids'];ref=c['plateau_reference']['indexes'];target=[ref[i]['selected_ndcg10'] for i in ids]
    controls={f'narrow_gen{s}_10':[c['proxy_reference']['indexes'][i]['scores_by_seed'][str(s)] for i in ids] for s in [31415,27182]}
    controls['narrow_mean10']=np.mean(list(controls.values()),axis=0).tolist()
    for e in [5,10,15,20,25,30]:controls[f'early_full_best{e}']=[ref[i]['early_selected'][str(e)] for i in ids]
    groups={'all':ids,**{f'method_{m}':[i for i in ids if i.startswith(m+'_')] for m in ['sum','gradnorm']},
            **{f'index_seed{s}':[i for i in ids if i.endswith(str(s))] for s in [42,101,202]}}
    result=dict(target='frozen full TIGER bestVAL under min60/max100/patience20',primary_proxy_epoch=10,
        plateau_reference_sha256=c['plateau_reference_sha256'],index_ids=ids,target_scores=dict(zip(ids,target)),test_evaluated=False,groups={},
        limitations=['Architecture development on six related previously examined indexes,not independent confirmation.',
                    'One full seed/two mini seeds;same VAL cohort;no new tokenizer family or dataset.',
                    'Within-seed groups have only two candidates;conditional descriptive rankings only.',
                    'Early full shares its reference trajectory;posthoc epoch selection is not a validated recipe.'])
    for name,members in groups.items():
        ix=[ids.index(i) for i in members];t=[target[j] for j in ix]
        group=dict(index_ids=members,controls={k:_stats([v[j] for j in ix],t,members) for k,v in controls.items()},mini={})
        for a in MIDDLE_ARCHITECTURES:
            for epoch in [5,10]:
                scores={str(s):[records[f'{a}_{i}_gen{s}']['epochs'][str(epoch)] for i in members] for s in [31415,27182]}
                mean=np.mean(list(scores.values()),axis=0).tolist()
                group['mini'][f'{a}_epoch{epoch}']=dict(mean=_stats(mean,t,members),mean_scores=mean,
                    scores_by_seed=scores,seeds={s:_stats(v,t,members) for s,v in scores.items()})
        result['groups'][name]=group
    _write(Path(output)/'correlations.json',result)
    metrics=dict(experiment=c['name'],completed=24,indexes=6,architectures=2,generator_seeds=2,test_evaluated=False)
    for a in MIDDLE_ARCHITECTURES:
        rho=result['groups']['all']['mini'][f'{a}_epoch10']['mean']['spearman']
        if rho is not None:metrics[a+'_epoch10_spearman']=rho
    return metrics
