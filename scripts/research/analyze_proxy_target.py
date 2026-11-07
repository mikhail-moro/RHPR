"""CPU-only replay of the six-index independent full-generator-seed diagnostic."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.research.analyze_mini_transfer import decode, load, sha
from rhpr.gr.mini_tiger import validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.proxy_target import correlate_target


def rho(x,y):
    return float(spearmanr(x,y).statistic)


def replay(root, data, pilot, transfer, output):
    output.mkdir(parents=True, exist_ok=True)
    c=load(root/'experiment.json');validate(c)
    manifest=load(root/'run-manifest.json');plan=load(root/'plan.json');suite=load(root/'suite-results.json');metrics=load(root/'metrics.json')
    assert manifest['config']==plan['config']==c
    assert manifest['git_sha']=='5ed498804545ea41686eee9eaa8d9f67747be0db'
    assert metrics['completed']==6 and metrics['generator_seeds']==1 and metrics['indexes']==6 and metrics['test_evaluated'] is False
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
    assert (root/'summary.md').is_file() and len(suite['results'])==6 and not suite['failures']
    workers=[load(p) for p in sorted((root/'workers').glob('worker-*.json'))]
    assert len(workers)==2 and all(len(w['results'])==3 and not w['failures'] for w in workers)
    assert {k:v for w in workers for k,v in w['results'].items()}==suite['results']
    for split in ['train','validation']:
        with np.load(data/(split+'-cases.npz'),allow_pickle=False) as z:cases=tuple(z[k] for k in ['contexts','user_ids','targets'])
        assert case_digest(cases)==c['data']['case_sha256'][split]
    refs=load(root/'frozen-indexes/references.json');saved=load(root/'correlations.json')
    assert saved==correlate_target(c,suite['results'],refs)
    ids=c['index_ids'];checks=[];table=[];new_users=[];old_users=[];mini_users=[];epoch_users={e:[] for e in [5,10,15,20,25,30]};curves={};timing=[]
    for i in ids:
        seed=i.split('_seed')[1];r=root/'trials'/f'full_{i}_gen161803'
        for name,digest in c['parents'][seed]['artifacts'][i].items():assert sha(root/'frozen-indexes'/(i+'-'+name))==digest
        sids=np.load(root/'frozen-indexes'/(i+'-semantic-ids.npy'),allow_pickle=False)
        g=load(r/'generator.json');sel=load(r/'training-selection.json');row=load(r/'trial-result.json');curve=load(r/'training-curves.json')
        assert g['config']==c['generator'] and g['seed']==161803 and g['parameters']==4977920
        assert g['initial_state_sha256']==c['initialization_pins']['full']['161803'] and g['sids_sha256']==refs[i]['sids_sha256']
        assert row==suite['results'][row['trial']] and row['seed']==161803
        assert sel['training_epochs']==30 and sel['optimizer_updates']==8880 and sel['validation_checks']==6 and sel['amp_skipped_updates']==0
        assert sel['training_examples_per_epoch']==151166 and sel['training_precision']=='fp16' and sel['evaluation_precision']=='fp32'
        assert len(curve)==30 and all(x['updates']==x['epoch']*296 and x['train_examples']==151166 for x in curve)
        assert [x['epoch'] for x in curve if 'validation' in x]==[5,10,15,20,25,30]
        vals={}
        for x in curve:
            if 'validation' not in x:continue
            e=x['epoch'];v,ch=decode(r/f'validation-epoch-{e:03d}.npz',sids,cases,x['validation']);checks.append(dict(path=str(r/f'validation-epoch-{e:03d}.npz'),**ch))
            np.testing.assert_allclose(v.mean(),row['epochs'][str(e)],atol=1e-12,rtol=0);epoch_users[e].append(v);vals[e]=float(v.mean())
        curves[i]=vals;new_users.append(epoch_users[30][-1])
        oldcurve=load(root/'frozen-indexes'/(i+'-training-curves.json'));old30=next(x['validation'] for x in oldcurve if x['epoch']==30)
        parent=transfer if seed=='202' else pilot
        v,ch=decode(parent/'full-validation'/(i+'.npz'),sids,cases,old30);checks.append(dict(path=str(parent/'full-validation'/(i+'.npz')),**ch));old_users.append(v)
        mm=[];mini_cost=[]
        for s in [31415,27182]:
            p=parent/'trials'/f'narrow_{i}_gen{s}';reference=c['proxy_reference']['indexes'][i]
            for name,digest in reference['evidence'][str(s)].items():assert sha(p/name)==digest
            mc=load(p/'training-curves.json');m10=next(x['validation'] for x in mc if x['epoch']==10)
            v,ch=decode(p/'validation-epoch-010.npz',sids,cases,m10);checks.append(dict(path=str(p/'validation-epoch-010.npz'),**ch));mm.append(v)
            np.testing.assert_allclose(v.mean(),reference['scores_by_seed'][str(s)],atol=1e-12,rtol=0)
            mini_cost.append(sum(x['train_seconds']+x.get('validation',{}).get('seconds',0) for x in mc if x['epoch']<=10))
        mini_users.append(np.mean(mm,axis=0))
        timing.append(dict(index_id=i,full_fit_seconds=sel['training_seconds'],full_training_seconds=sel['optimizer_training_seconds'],mini_two_seed_training_and_ranking_seconds=sum(mini_cost),full10_training_and_ranking_seconds=sum(x['train_seconds']+x.get('validation',{}).get('seconds',0) for x in curve if x['epoch']<=10)))
        table.append(dict(index_id=i,old_full30=float(old_users[-1].mean()),new_full30=vals[30],new_full_best=max(vals.values()),new_best_epoch=sel['best_epoch'],mini_mean10=float(mini_users[-1].mean())))
    old_users,new_users,mini_users=map(np.asarray,[old_users,new_users,mini_users]);old=old_users.mean(1);new=new_users.mean(1);mini=mini_users.mean(1)
    np.testing.assert_allclose(rho(mini,new),metrics['mini_mean_spearman'],atol=1e-12)
    np.testing.assert_allclose(rho(old,new),metrics['historical_full_spearman'],atol=1e-12)
    chosen=int(np.argmax(mini));best=int(np.argmax(new));rng=np.random.default_rng(20261107)
    boot={'mini_new':[],'old_new':[],'best_minus_proxy_choice':[],'gradnorm_minus_sum_by_index_seed':[]}
    for _ in range(50):
        draws=rng.integers(0,2786,size=(100,2786));pp=mini_users[:,draws].mean(-1);nn=new_users[:,draws].mean(-1);oo=old_users[:,draws].mean(-1)
        for name,a,b in [('mini_new',pp,nn),('old_new',oo,nn)]:
            aa=rankdata(a,axis=0);bb=rankdata(b,axis=0);aa-=aa.mean(0);bb-=bb.mean(0)
            den=np.sqrt((aa**2).sum(0)*(bb**2).sum(0));boot[name].extend(np.divide((aa*bb).sum(0),den,out=np.full(100,np.nan),where=den>0))
        boot['best_minus_proxy_choice'].extend(nn[best]-nn[chosen]);boot['gradnorm_minus_sum_by_index_seed'].extend((nn[[1,3,5]]-nn[[0,2,4]]).T)
    uncertainty={k:np.nanquantile(v,[.025,.975],axis=0).tolist() for k,v in boot.items()}
    best_scores=np.asarray([r['new_full_best'] for r in table]);late=np.asarray([np.mean([curves[i][e] for e in [20,25,30]]) for i in ids])
    old_late=np.asarray([np.mean([x['validation']['ndcg@10'] for x in load(root/'frozen-indexes'/(i+'-training-curves.json')) if x['epoch'] in [20,25,30]]) for i in ids])
    result=dict(complete=True,source_code_sha=manifest['git_sha'],run_minutes=manifest['duration_seconds']/60,finished_at=manifest['finished_at'],fits=6,test_evaluated=False,verified_new_predictions=36,verified_old_full_predictions=6,verified_mini_predictions=12,verified_prediction_files=len(checks),all_full_budget=True,amp_skips=0,table=table,primary=saved,
        mini_to_old_full30=rho(mini,old),mini_to_new_full30=rho(mini,new),old_to_new_full30=rho(old,new),
        posthoc_sensitivity=dict(mini_to_new_best_of30=rho(mini,best_scores),mini_to_new_late_mean=rho(mini,late),old_to_new_late_mean=rho(old_late,late),mini_to_two_full_seed_mean=rho(mini,(old+new)/2)),
        new_full_rank_of_proxy_choice=int(1+np.sum(new>new[chosen])),random_uniform_expected_regret=float(new.max()-new.mean()),
        fixed_pair_new_best_minus_proxy_choice=float(new[best]-new[chosen]),conditional_user_bootstrap_ci95=uncertainty,
        uncertainty_scope='5000 shared-user bootstrap draws, fixed trained models/indexes; excludes training-seed and tokenizer uncertainty; best-versus-choice pair selected on these data, no selection correction.',
        conditional_new_method_delta_by_seed={str(s):float(new[j+1]-new[j]) for s,j in [(42,0),(101,2),(202,4)]},
        timing=timing,mean_epoch_training_seconds=float(np.mean([t['full_training_seconds']/30 for t in timing])),
        mean_full_fit_seconds=float(np.mean([t['full_fit_seconds'] for t in timing])),
        scope='Exploratory independent generator-seed check on six related previously examined indexes and same VAL users. Historical full seeds differ by index seed, new full seed common across indexes. Fixed30 primary is not convergence; no new tokenizer-family or dataset transfer. No GPU/TEST used in this replay.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(output/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (output/'indexes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    plot(result,curves,output)
    return result


def plot(r,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,9));rows=r['table'];old=[x['old_full30'] for x in rows];new=[x['new_full30'] for x in rows]
    ax[0,0].scatter(old,new)
    for row in rows:ax[0,0].annotate(row['index_id'].replace('gradnorm','GN'),(row['old_full30'],row['new_full30']),fontsize=8,xytext=(3,3),textcoords='offset points')
    ax[0,0].set(xlabel='Historical full TIGER, epoch30',ylabel='New full TIGER, epoch30',title=f"Full-to-full Spearman = {r['old_to_new_full30']:.2f}");ax[0,0].grid(alpha=.2)
    for i,vals in curves.items():ax[0,1].plot(list(vals),list(vals.values()),marker='o',label=i.replace('gradnorm','GN'),linewidth=1)
    ax[0,1].set(xlabel='Epoch',ylabel='VAL NDCG@10',title='New seed161803: full-model trajectories');ax[0,1].legend(fontsize=7)
    ax[1,0].bar(['Mini → old full','Mini → new full','Old full → new full'],[r['mini_to_old_full30'],r['mini_to_new_full30'],r['old_to_new_full30']],color=['#4c78a8','#e45756','#72b7b2']);ax[1,0].axhline(0,color='black',linewidth=.7);ax[1,0].set(ylabel='Spearman across six indexes',ylim=(-1,1),title='Primary fixed-epoch30 comparison');ax[1,0].tick_params(axis='x',rotation=12)
    ordered=sorted(rows,key=lambda x:x['new_full30'],reverse=True);chosen=r['primary']['groups']['all']['estimators']['mini_mean10']['selected_index'];colors=['#e45756' if x['index_id']==chosen else '#4c78a8' for x in ordered]
    ax[1,1].bar([x['index_id'].replace('gradnorm','GN') for x in ordered],[x['new_full30'] for x in ordered],color=colors);ax[1,1].set(ylabel='New full TIGER VAL NDCG@10',title='Red = index chosen by mini');ax[1,1].tick_params(axis='x',rotation=40)
    fig.suptitle('Independent target-seed diagnostic — six fixed indexes, no TEST');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('proxy-target.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','data','pilot','transfer','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=replay(a.root,a.data,a.pilot,a.transfer,a.output)
    print(json.dumps({k:r[k] for k in ['complete','fits','run_minutes','verified_prediction_files','mini_to_old_full30','mini_to_new_full30','old_to_new_full30','posthoc_sensitivity','conditional_user_bootstrap_ci95']},indent=2))
