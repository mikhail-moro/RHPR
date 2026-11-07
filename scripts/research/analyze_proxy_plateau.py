"""CPU-only join/replay of four recovered and two newly completed plateau fits."""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.research.analyze_mini_transfer import decode, load, sha
from rhpr.gr.movie_data import case_digest
from rhpr.gr.proxy_plateau import validate_plateau, validate_completion, summarize
from rhpr.gr.proxy_target import _stats


def rho(x,y):return float(spearmanr(x,y).statistic)


def replay(root,recovered,data,pilot,transfer,out):
    out.mkdir(parents=True,exist_ok=True)
    c=load(root/'experiment.json');validate_plateau(c)
    manifest=load(root/'run-manifest.json');plan=load(root/'plan.json');suite=load(root/'suite-results.json');metrics=load(root/'metrics.json')
    assert manifest['git_sha']=='ac6b5de04dc492fce22af3e885bdc190c1d7bd7b' and manifest['config']==plan['config']==c
    assert metrics['completed']==6 and metrics['plateau_reached']==6 and metrics['budget_censored']==0 and not metrics['test_evaluated']
    assert (root/'summary.md').is_file() and len(suite['results'])==6 and not suite['failures']
    workers=[load(p) for p in sorted((root/'workers').glob('worker-*.json'))]
    assert len(workers)==2 and all(len(w['results'])==3 and not w['failures'] for w in workers)
    assert {k:v for w in workers for k,v in w['results'].items()}==suite['results']
    for name,expected in c['recovery_parent']['files'].items():
        if name.endswith('.pt'):continue # All eight weights separately inspected/hashed before the finish submission.
        p=recovered/name;assert p.stat().st_size==expected['size_bytes'] and sha(p)==expected['sha256']
    for split in ['train','validation']:
        with np.load(data/(split+'-cases.npz'),allow_pickle=False) as z:cases=tuple(z[k] for k in ['contexts','user_ids','targets'])
        assert case_digest(cases)==c['data']['case_sha256'][split]
    refs=load(root/'frozen-indexes/references.json');saved=load(root/'correlations.json')
    with tempfile.TemporaryDirectory() as d:
        assert summarize(c,suite['results'],refs,Path(d))==metrics
        assert load(Path(d)/'correlations.json')==saved
    checks=[];table=[];users=[];mini_users=[];historical_users=[];early_users={e:[] for e in [5,10,15,20,25,30]};best30_users=[];curves={};timing=[];records=suite['results'];ids=c['index_ids']
    new_predictions=old_predictions=0
    for i in ids:
        tid=f'full_{i}_gen161803';row=records[tid];path=root/'trials'/tid;sel=load(path/'training-selection.json');g=load(path/'generator.json');curve=load(path/'training-curves.json');validate_completion(sel,c)
        assert load(path/'trial-result.json')==row and all(sel[k]==v for k,v in row['selection'].items())
        assert sel['criterion']=='validation_item_ndcg@10' and sel['selected_validation']['ndcg@10']==sel['validation_ndcg10']
        reused=i.endswith('42') or i.endswith('101');source=recovered/'trials'/tid if reused else path
        assert row['reused_completed']==reused
        if reused:
            assert row['new_training_updates']==0 and row['external_artifacts']
            assert sel==load(source/'training-selection.json') and curve==load(source/'training-curves.json') and g==load(source/'generator.json')
            assert (path/'external-checkpoint-reference.json').is_file() and not list(path.glob('validation-epoch-*.npz'))
        audit=load(path/'resume-verification.json')
        assert audit['source_epoch']==30 and audit['source_updates']==8880 and audit['optimizer_restored'] and audit['scaler_restored'] and audit['rng_restored_and_verified']
        assert audit['train_case_sha256']==c['data']['case_sha256']['train'] and audit['validation_case_sha256']==c['data']['case_sha256']['validation']
        assert g['initial_state_sha256']==c['initialization_pins']['full']['161803'] and g['config']==c['generator'] and g['parameters']==4977920
        assert sel['training_precision']=='fp16' and sel['evaluation_precision']=='fp32' and sel['training_examples_per_epoch']==151166
        assert len(curve)==sel['training_epochs'] and sel['training_epochs']-sel['best_epoch']>=20 and sel['stopping_reason']=='early_stopping'
        assert [x['epoch'] for x in curve]==list(range(1,sel['training_epochs']+1))
        assert all(x['updates']+x['amp_skipped_updates']==296*x['epoch'] and x['train_examples']==151166 and np.isfinite(x['train_loss_sum_over_codes']) for x in curve)
        for name,digest in c['parents'][i.split('_seed')[1]]['artifacts'][i].items():assert sha(root/'frozen-indexes'/(i+'-'+name))==digest
        sids=np.load(root/'frozen-indexes'/(i+'-semantic-ids.npy'),allow_pickle=False);assert g['sids_sha256']==refs[i]['sids_sha256']
        vals={};peruser={}
        for x in curve:
            if 'validation' not in x:continue
            e=x['epoch'];p=source/f'validation-epoch-{e:03d}.npz';v,ch=decode(p,sids,cases,x['validation']);checks.append(dict(path=str(p),**ch));peruser[e]=v;vals[e]=float(v.mean())
            assert np.isfinite(x['validation_loss']['total_ce'])
            np.testing.assert_allclose(v.mean(),row['epochs'][str(e)],atol=1e-12,rtol=0)
            if reused:old_predictions+=1
            else:new_predictions+=1
        assert len(vals)==sel['validation_checks']==sel['training_epochs']//5
        np.testing.assert_allclose([max(vals.values()),vals[sel['best_epoch']]],sel['validation_ndcg10'],atol=1e-12,rtol=0)
        curves[i]=vals;users.append(peruser[sel['best_epoch']])
        for e in early_users:early_users[e].append(peruser[e])
        e30=max((e for e in vals if e<=30),key=lambda e:vals[e]);best30=vals[e30];best30_users.append(peruser[e30])
        parent=transfer if i.endswith('202') else pilot;mm=[];mini_seconds=[]
        for s in [31415,27182]:
            p=parent/'trials'/f'narrow_{i}_gen{s}';ref=c['proxy_reference']['indexes'][i]
            for f,digest in ref['evidence'][str(s)].items():assert sha(p/f)==digest
            mc=load(p/'training-curves.json');expected=next(x['validation'] for x in mc if x['epoch']==10);v,ch=decode(p/'validation-epoch-010.npz',sids,cases,expected);checks.append(dict(path=str(p/'validation-epoch-010.npz'),**ch));mm.append(v)
            np.testing.assert_allclose(v.mean(),ref['scores_by_seed'][str(s)],atol=1e-12,rtol=0)
            mini_seconds.append(sum(x['train_seconds']+x.get('validation',{}).get('seconds',0) for x in mc if x['epoch']<=10))
        mini_users.append(np.mean(mm,axis=0))
        oldcurve=load(root/'frozen-indexes'/(i+'-training-curves.json'));old30=next(x['validation'] for x in oldcurve if x['epoch']==30)
        v,ch=decode(parent/'full-validation'/(i+'.npz'),sids,cases,old30);checks.append(dict(path=str(parent/'full-validation'/(i+'.npz')),**ch));historical_users.append(v)
        table.append(dict(index_id=i,full30=vals[30],best_first30=best30,plateau_selected=sel['validation_ndcg10'],best_epoch=sel['best_epoch'],stop_epoch=sel['training_epochs'],amp_skips=sel['amp_skipped_updates'],gain_vs_best_first30=float(sel['validation_ndcg10']/best30-1),mini_mean10=float(mini_users[-1].mean()),reused=reused))
        timing.append(dict(index_id=i,full_total_fit_seconds=sel['training_seconds'],full_total_training_seconds=sel['optimizer_training_seconds'],continuation_training_seconds=sel['continuation_optimizer_training_seconds'],mini_two_seed_training_and_ranking_seconds=sum(mini_seconds),early_training_ranking_seconds={str(e):sum(x['train_seconds']+x.get('validation',{}).get('seconds',0) for x in curve if x['epoch']<=e) for e in early_users}))
    users,mini_users,historical_users,best30_users=map(np.asarray,[users,mini_users,historical_users,best30_users]);target=users.mean(1);mini=mini_users.mean(1)
    comparisons={}
    for e in early_users:
        fixed=np.asarray(early_users[e]).mean(1)
        best=[max(v for ep,v in curves[i].items() if ep<=e) for i in ids]
        comparisons[str(e)]={'fixed':_stats(fixed,target,ids),'selected':_stats(best,target,ids)}
    groups={}
    for m in ['sum','gradnorm']:
        ix=[j for j,i in enumerate(ids) if i.startswith(m+'_')];groups[m]=dict(n=len(ix),mini_to_plateau_rho=rho(mini[ix],target[ix]))
    rng=np.random.default_rng(20261107);chosen=int(np.argmax(mini));best=int(np.argmax(target));boot={'mini_to_plateau':[],'selected_full30_to_plateau':[],'best_minus_mini_choice':[]}
    for _ in range(50):
        draws=rng.integers(0,2786,size=(100,2786));p=mini_users[:,draws].mean(-1);t=users[:,draws].mean(-1);b=best30_users[:,draws].mean(-1)
        for name,a in [('mini_to_plateau',p),('selected_full30_to_plateau',b)]:
            aa=rankdata(a,axis=0);bb=rankdata(t,axis=0);aa-=aa.mean(0);bb-=bb.mean(0);den=np.sqrt((aa**2).sum(0)*(bb**2).sum(0));boot[name].extend(np.divide((aa*bb).sum(0),den,out=np.full(100,np.nan),where=den>0))
        boot['best_minus_mini_choice'].extend(t[best]-t[chosen])
    result=dict(complete=True,source_code_sha=manifest['git_sha'],finished_at=manifest['finished_at'],finish_run_minutes=manifest['duration_seconds']/60,finished_models=6,newly_trained_models=2,reused_models=4,plateau_reached=6,budget_censored=0,test_evaluated=False,
        verified_prediction_files=len(checks),verified_plateau_predictions=old_predictions+new_predictions,verified_reused_predictions=old_predictions,verified_new_model_predictions=new_predictions,verified_mini_predictions=12,verified_historical_full_predictions=6,
        primary=saved,table=table,mini_to_fixed_full30=rho(mini,np.asarray(early_users[30]).mean(1)),mini_to_best_first30=rho(mini,best30_users.mean(1)),mini_to_plateau=rho(mini,target),fixed30_to_plateau=rho(np.asarray(early_users[30]).mean(1),target),selected_full30_to_plateau=rho(best30_users.mean(1),target),early_full=comparisons,method_groups=groups,
        conditional_user_bootstrap_ci95={k:np.nanquantile(v,[.025,.975]).tolist() for k,v in boot.items()},
        bootstrap_scope='5000 shared-user draws, fixed selected checkpoints/indexes; excludes training/tokenizer randomness, checkpoint/index selection correction and target-item dependence.',
        mean_relative_gain_vs_selected_first30=float(np.mean([r['gain_vs_best_first30'] for r in table])),timing=timing,
        scope='Six related pre-examined indexes,one common full generator seed;two old mini seeds;VAL only.Operational ES is not mathematical convergence. All six use the same min60/max100/20epoch patience rule. Four reused from failed job, two newly completed; original weights/predictions stay in their respective private kernels.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (out/'indexes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    plot(result,curves,out);return result


def plot(r,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(13,9))
    for row in r['table']:
        i=row['index_id'];v=curves[i];p=ax[0,0].plot(list(v),list(v.values()),label=i.replace('gradnorm','GN'),linewidth=1)[0];ax[0,0].scatter([row['best_epoch']],[row['plateau_selected']],marker='*',s=75,color=p.get_color())
    ax[0,0].axvline(30,color='gray',linestyle='--');ax[0,0].set(xlabel='Total epoch',ylabel='VAL NDCG@10',title='Full TIGER: six operational ES runs');ax[0,0].legend(fontsize=7)
    for row in r['table']:
        ax[0,1].scatter(row['mini_mean10'],row['plateau_selected']);ax[0,1].annotate(row['index_id'].replace('gradnorm','GN'),(row['mini_mean10'],row['plateau_selected']),fontsize=8,xytext=(3,3),textcoords='offset points')
    ax[0,1].margins(x=.22,y=.12);ax[0,1].set(xlabel='Mini epoch10 mean of two seeds',ylabel='Selected full-TIGER VAL NDCG@10',title=f"Mini → selected full: rho={r['mini_to_plateau']:.2f}")
    labels=['Mini10 → fixed30','Mini10 → best≤30','Mini10 → plateau','Best≤30 → plateau'];values=[r['mini_to_fixed_full30'],r['mini_to_best_first30'],r['mini_to_plateau'],r['selected_full30_to_plateau']]
    ax[1,0].bar(labels,values,color=['#e45756','#4c78a8','#72b7b2','#54a24b']);ax[1,0].axhline(0,color='black',linewidth=.6);ax[1,0].set(ylim=(-1,1),ylabel='Spearman across six indexes',title='Target definition matters');ax[1,0].tick_params(axis='x',rotation=20)
    ordered=sorted(r['table'],key=lambda x:x['plateau_selected'],reverse=True);xx=np.arange(6);ax[1,1].bar(xx-.18,[x['best_first30'] for x in ordered],width=.36,label='Best≤30');ax[1,1].bar(xx+.18,[x['plateau_selected'] for x in ordered],width=.36,label='Selected after continuation');ax[1,1].set_xticks(xx,[x['index_id'].replace('gradnorm','GN') for x in ordered],rotation=35);ax[1,1].set(ylabel='VAL NDCG@10',title='Longer training vs standard checkpoint selection');ax[1,1].legend(fontsize=8)
    fig.suptitle('Proxy calibration after longer full-TIGER training — no TEST');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('proxy-plateau.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','recovered','data','pilot','transfer','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=replay(a.root,a.recovered,a.data,a.pilot,a.transfer,a.output)
    print(json.dumps({k:r[k] for k in ['complete','finish_run_minutes','verified_prediction_files','mini_to_fixed_full30','mini_to_best_first30','mini_to_plateau','selected_full30_to_plateau','conditional_user_bootstrap_ci95','mean_relative_gain_vs_selected_first30']},indent=2))
