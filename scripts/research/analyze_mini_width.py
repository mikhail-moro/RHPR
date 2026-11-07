"""CPU replay of the frozen-index mini-TIGER width calibration, VAL only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.research.analyze_mini_transfer import decode, load, sha
from rhpr.gr.mini_tiger import ALL_ARCHITECTURES, correlate, validate
from rhpr.gr.movie_data import case_digest


def replay(root, data, pilot, transfer, output):
    output.mkdir(parents=True, exist_ok=True)
    c = load(root/'experiment.json'); validate(c)
    assert c['calibration_phase'] == 'width_reduction'
    manifest = load(root/'run-manifest.json'); plan = load(root/'plan.json')
    assert manifest['config'] == plan['config'] == c
    assert manifest['git_sha'] == 'd80787c82e1cb93c016359df7e069ea0b8753207'
    metrics = load(root/'metrics.json'); suite = load(root/'suite-results.json')
    assert metrics['completed'] == 48 and metrics['indexes'] == 12 and metrics['architectures'] == 2
    assert metrics['test_evaluated'] is False and not suite['failures']
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
    assert (root/'summary.md').is_file() and len(suite['results']) == 48
    workers = [load(p) for p in sorted((root/'workers').glob('worker-*.json'))]
    assert len(workers) == 2 and all(len(w['results']) == 24 and not w['failures'] for w in workers)
    assert {k:v for w in workers for k,v in w['results'].items()} == suite['results']
    for split in ['train','validation']:
        with np.load(data/(split+'-cases.npz'), allow_pickle=False) as z:
            cases = tuple(z[k] for k in ['contexts','user_ids','targets'])
        assert case_digest(cases) == c['data']['case_sha256'][split]
    refs=load(root/'frozen-indexes/references.json'); checks=[]; rows=[]; full=[]; users={}; records={}
    counts={'narrow':1375040,'ffn_small':1112896,'attention_small':719552}
    for i in c['index_ids']:
        seed=i.split('_seed')[1]
        for f,digest in c['parents'][seed]['artifacts'][i].items():
            assert sha(root/'frozen-indexes'/(i+'-'+f)) == digest
        sid=np.load(root/'frozen-indexes'/(i+'-semantic-ids.npy'),allow_pickle=False)
        fc=load(root/'frozen-indexes'/(i+'-training-curves.json'))
        full_metrics=next(x['validation'] for x in fc if x['epoch']==30)
        parent=transfer if seed=='202' else pilot
        p=parent/'full-validation'/(i+'.npz')
        v,ch=decode(p,sid,cases,full_metrics);checks.append(dict(path=str(p),**ch));full.append(v)
        np.testing.assert_allclose(refs[i]['epochs']['30'],float(v.mean()),atol=1e-12,rtol=0)
        for a in counts:
            for s in c['generator_seeds']:
                name=f'{a}_{i}_gen{s}'; base=parent if a=='narrow' else root; p=base/'trials'/name
                row=load(p/'trial-result.json'); g=load(p/'generator.json'); sel=load(p/'training-selection.json')
                assert (row['architecture'],row['index_id'],row['seed']) == (a,i,s)
                config=load(base/'experiment.json')
                assert g['initial_state_sha256']==config['initialization_pins'][a][str(s)]
                assert g['config']=={**c['generator'],**ALL_ARCHITECTURES[a]} and g['parameters']==counts[a]
                assert g['sids_sha256']==refs[i]['sids_sha256']
                assert sel['training_epochs']==(20 if a=='narrow' and seed!='202' else 10)
                assert sel['optimizer_updates']==sel['training_epochs']*296
                assert sel['training_examples_per_epoch']==151166 and sel['amp_skipped_updates']==0
                assert sel['training_precision']=='fp16' and sel['evaluation_precision']=='fp32'
                curve=load(p/'training-curves.json'); selected=[x for x in curve if x['epoch']<=10]
                assert len(selected)==10 and all(x['updates']==x['epoch']*296 and x['train_examples']==151166 for x in selected)
                assert [x['epoch'] for x in selected if 'validation' in x]==[5,10]
                for x in selected:
                    if 'validation' not in x:continue
                    pred=p/f"validation-epoch-{x['epoch']:03d}.npz"
                    v,ch=decode(pred,sid,cases,x['validation']);checks.append(dict(path=str(pred),**ch))
                    np.testing.assert_allclose(row['epochs'][str(x['epoch'])],float(v.mean()),atol=1e-12,rtol=0)
                    users[a,i,s,x['epoch']]=v
                if a!='narrow':
                    assert row==suite['results'][name];records[name]=row
                rows.append(dict(architecture=a,index_id=i,generator_seed=s,ndcg10=float(users[a,i,s,10].mean()),
                    train_epoch_seconds=float(np.mean([x['train_seconds'] for x in selected])),
                    training_seconds=sel['optimizer_training_seconds'],fit_seconds=sel['training_seconds'],fit_epochs=sel['training_epochs']))
    saved=load(root/'correlations.json');recomputed=correlate(c,records,refs)
    assert saved==recomputed
    full=np.asarray(full);target=full.mean(1); tables=[]; summary={}
    groups={'all':list(range(12)),**{f'index_seed{s}':[j for j,i in enumerate(c['index_ids']) if i.endswith(f'seed{s}')] for s in [42,101,202]},
        **{f'method_{m}':[j for j,i in enumerate(c['index_ids']) if i.startswith(m+'_')] for m in ['sum','uw','gradnorm','pcgrad']}}
    for a in counts:
        arr=np.asarray([np.mean([users[a,i,s,10] for s in c['generator_seeds']],axis=0) for i in c['index_ids']])
        summary[a]={}
        for group,ix in groups.items():
            scores=arr[ix].mean(1);t=target[ix];winner=int(np.argmax(scores));best=int(np.argmax(t))
            rho=float(spearmanr(scores,t).statistic)
            summary[a][group]=dict(spearman=rho,selected_index=c['index_ids'][ix[winner]],best_full_index=c['index_ids'][ix[best]],
                target_regret=float(t.max()-t[winner]),relative_target_regret=float((t.max()-t[winner])/t.max()),
                seed_spearman={str(s):float(spearmanr([users[a,c['index_ids'][j],s,10].mean() for j in ix],t).statistic) for s in c['generator_seeds']})
            if a!='narrow':np.testing.assert_allclose(rho,saved['groups'][group]['mini'][a+'_epoch10']['mean']['spearman'],atol=1e-12)
        # Shared-user bootstrap, conditional on fixed fitted models/indexes.
        rng=np.random.default_rng(20261106);boot={k:[] for k in ['all','method_sum']}
        for _ in range(50):
            draws=rng.integers(0,2786,size=(100,2786));pp=arr[:,draws].mean(-1);ff=full[:,draws].mean(-1)
            for group in boot:
                ix=groups[group];pr=rankdata(pp[ix],axis=0);fr=rankdata(ff[ix],axis=0)
                pr-=pr.mean(0);fr-=fr.mean(0);den=np.sqrt((pr**2).sum(0)*(fr**2).sum(0))
                boot[group].extend(np.divide((pr*fr).sum(0),den,out=np.full(100,np.nan),where=den>0).tolist())
        for group,vals in boot.items():summary[a][group]['conditional_user_rho_ci95']=np.nanquantile(vals,[.025,.975]).tolist()
        rr=[r for r in rows if r['architecture']==a]
        summary[a]['timing']=dict(parameters=counts[a],mean_training_epoch_seconds=float(np.mean([r['train_epoch_seconds'] for r in rr])),
            mean_fit_seconds=float(np.mean([r['fit_seconds'] for r in rr])),fit_epochs=sorted({r['fit_epochs'] for r in rr}))
        for j,i in enumerate(c['index_ids']):
            tables.append(dict(architecture=a,index_id=i,full30_ndcg=float(target[j]),mini10_mean_ndcg=float(arr[j].mean()),
                gen31415=float(users[a,i,31415,10].mean()),gen27182=float(users[a,i,27182,10].mean())))
    for a in counts:
        summary[a]['timing']['speedup_vs_full_historical']=57.9124/summary[a]['timing']['mean_training_epoch_seconds']
        summary[a]['timing']['speedup_vs_narrow']=summary['narrow']['timing']['mean_training_epoch_seconds']/summary[a]['timing']['mean_training_epoch_seconds']
    result=dict(complete=True,completed_fits=48,run_minutes=manifest['duration_seconds']/60,finished_at=manifest['finished_at'],
        source_code_sha=manifest['git_sha'],test_evaluated=False,verified_prediction_files=len(checks),
        verified_new_prediction_files=96,verified_narrow_prediction_files=48,verified_full_prediction_files=12,
        comparisons=summary,table=tables,bootstrap_scope='5000 shared-user draws;fixed models/indexes;excludes training randomness,index selection,target-item dependence',
        limitations=['12 related indexes and one full generator seed/index; no universal proxy claim',
                     '3 SUM indexes only; width screen follows inspection of earlier narrow results',
                     'narrow historical fits include20 epochs; epoch10 comparisons use first10 epochs only'])
    for name,value in [('summary',result),('verification',checks)]: (output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    with (output/'fits.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot(result,output)
    return result


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    colors={'narrow':'#1b9e77','ffn_small':'#d95f02','attention_small':'#7570b3'}
    for a,color in colors.items():
        rows=[r for r in result['table'] if r['architecture']==a]
        axes[0].scatter([r['full30_ndcg'] for r in rows],[r['mini10_mean_ndcg'] for r in rows],label=a,color=color,s=28)
        rows=[r for r in rows if r['index_id'].startswith('sum_')]
        rows.sort(key=lambda r:r['index_id'])
        axes[1].plot([r['index_id'].replace('sum_','') for r in rows],[r['mini10_mean_ndcg'] for r in rows],marker='o',label=a,color=color)
    axes[0].set(xlabel='Full TIGER epoch30 VAL NDCG@10',ylabel='Mini epoch10 mean VAL NDCG@10')
    axes[1].set(xlabel='SUM index seed',ylabel='Mini epoch10 mean VAL NDCG@10')
    names=list(colors);axes[2].bar(names,[result['comparisons'][a]['timing']['mean_training_epoch_seconds'] for a in names],color=list(colors.values()))
    axes[2].set(ylabel='Mean training epoch seconds');axes[2].tick_params(axis='x',rotation=20)
    for ax in axes[:2]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Mini-TIGER width calibration: frozen indexes, fixed epoch10, two generator seeds')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('width-calibration.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','data','pilot','transfer','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=replay(a.root,a.data,a.pilot,a.transfer,a.output)
    print(json.dumps({k:r[k] for k in ['complete','completed_fits','run_minutes','comparisons']},indent=2))
