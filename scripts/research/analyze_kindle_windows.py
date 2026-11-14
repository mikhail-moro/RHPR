"""Replay all saved Kindle window predictions and paired VAL comparisons, CPU only."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE='55350fd1a3013a2673b1f9795b13c6e9d1efa0c7'
SEEDS=[42,101,202]
WINDOWS=[20,32,64]


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def analyze(root,data,out):
    root,data,out=map(Path,(root,data,out));out.mkdir(parents=True,exist_ok=True)
    manifest=read(root/'run-manifest.json');c=read(root/'experiment.json');suite=read(root/'suite-results.json');study=read(root/'study-summary.json')
    assert manifest['git_sha']==SOURCE and manifest['config']==c
    assert not suite['failures'] and suite['results']==study['results']
    assert read(root/'metrics.json')['models']==9 and (root/'summary.md').is_file()
    assert set(suite['results'])=={f'recent{w}_{s}'for w in WINDOWS for s in SEEDS}
    for name,sha in c['movie_input']['sha256'].items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha
    raw={}
    for split in ('train','validation'):
        with np.load(data/(split+'-cases.npz'))as z:raw[split]=tuple(z[k]for k in ('contexts','user_ids','targets'))
    targets=raw['validation'][2];users=raw['validation'][1];n=len(users);steps=math.ceil(len(raw['train'][2])/512)
    records=[];credits={};curves={};replayed=0;total_invalid=0;total_duplicates=0;total_predictions=0
    for seed in SEEDS:
        path=root/'frozen-indexes'/f'items-{seed}.npy';sids=np.load(path);idx=SidIndex.from_full_sids(sids,width=256)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();matched=[]
        for w in WINDOWS:
            name=f'recent{w}_{seed}';trial=suite['results'][name];selection=trial['selection'];d=root/'trials'/name
            assert read(d/'trial-result.json')==trial and trial['sid_file_sha256']==digest and trial['test_evaluated']is False
            assert trial['seed']==seed and trial['history_items']==w
            assert trial['raw_case_sha256']=={s:case_digest(a)for s,a in raw.items()}
            assert trial['case_targets_sha256']=={s:case_digest(a[1:])for s,a in raw.items()}
            matched.append(trial['initial_state_sha256'])
            curve=read(d/'training-curves.json');curves[name]=curve
            assert [r['epoch']for r in curve]==list(range(1,selection['training_epochs']+1))
            assert selection['optimizer_updates']+selection['amp_skipped_updates']==steps*len(curve)
            ema=reference=None;last=None;previous_updates=0;eval_rows=[]
            def lr(step):
                warm=2*steps;total=60*steps
                if step<warm:return .003*(step+1)/warm
                return .001+.002*.5*(1+math.cos(math.pi*min(1.,(step-warm+1)/(total-warm))))
            for r in curve:
                ep=r['epoch'];assert r['updates']+r['amp_skipped_updates']==ep*steps and r['train_examples']==len(raw['train'][2])
                assert abs(r['learning_rate']-lr(previous_updates))<1e-14
                assert abs(r['next_learning_rate']-lr(r['updates']))<1e-14
                assert min(abs(r['learning_rate_last_batch']-lr(k))for k in [r['updates']-1,r['updates']])<1e-14
                previous_updates=r['updates']
                if 'validation'not in r:continue
                score=r['validation']['ndcg@10'];ema=score if ema is None else .5*score+.5*ema
                if reference is None or ema>reference+.0005:reference=ema;last=ep
                assert r['validation_ema_ndcg10']==ema and r['ema_reference_ndcg10']==reference and r['ema_last_improvement_epoch']==last
                if ep>=40 and ep<len(curve):assert ep-last<15
                with np.load(d/f'validation-epoch-{ep:03d}.npz')as z:
                    pred=z['semantic_ids'];np.testing.assert_array_equal(z['user_ids'],users);np.testing.assert_array_equal(z['target_item_ids'],targets)
                    assert np.isfinite(z['scores']).all()
                metric=evaluate_ranked_sids(idx,pred,targets,cutoffs=(5,10))
                for key,value in metric.items():assert abs(value-r['validation'][key])<1e-12,(name,ep,key)
                for k in (5,10):
                    hit=(pred[:,:k]==sids[targets,None,:]).all(-1)
                    credit=np.where(hit.any(1),1/np.log2(hit.argmax(1)+2),0.)
                    assert abs(float(credit.mean())-metric[f'ndcg@{k}'])<1e-12
                    if k==10:credits[(w,seed,ep)]=credit
                total_invalid+=metric['invalid_predictions'];total_duplicates+=metric['duplicate_predictions'];total_predictions+=metric['prediction_count'];replayed+=1;eval_rows.append(r)
            best=max(eval_rows,key=lambda r:r['validation']['ndcg@10'])
            assert selection['best_epoch']==best['epoch'] and selection['validation_ndcg10']==best['validation']['ndcg@10']
            assert selection['final_early_stopping_state']==dict(ema=ema,reference=reference,last_improvement_epoch=last)
            if selection['stopping_reason']=='early_stopping':assert len(curve)>=40 and len(curve)-last>=15
            else:assert selection['stopping_reason']=='epoch_limit'and len(curve)==60
            assert selection['validation_checks']==len(eval_rows)==len(list(d.glob('validation-epoch-*.npz')))
            assert abs(selection['optimizer_training_seconds']-sum(r['train_seconds']for r in curve))<1e-6
            records.append(dict(trial=name,window=w,seed=seed,best_epoch=best['epoch'],stop_epoch=len(curve),stop_reason=selection['stopping_reason'],
                ndcg10=best['validation']['ndcg@10'],recall10=best['validation']['recall@10'],hits10=round(best['validation']['recall@10']*n),
                fixed40=next(r['validation']['ndcg@10']for r in curve if r['epoch']==40),fixed45=next(r['validation']['ndcg@10']for r in curve if r['epoch']==45),
                best_within40=max(r['validation']['ndcg@10']for r in eval_rows if r['epoch']<=40),
                forward_gflops=trial['forward']['supported_forward_flops_batch1']/1e9,parameters=trial['forward']['parameters'],
                warm_epoch_seconds=trial['timing']['warm_epoch_training_median_seconds'],first_epoch_seconds=curve[0]['train_seconds'],
                fit_seconds=selection['training_seconds'],train_seconds=selection['optimizer_training_seconds'],validation_seconds=selection['validation_seconds'],
                peak_cuda_allocated_gib=trial['timing']['cuda_peak_allocated_bytes']/2**30,peak_cuda_reserved_gib=trial['timing']['cuda_peak_reserved_bytes']/2**30,
                amp_skips=selection['amp_skipped_updates'],last_train_loss=curve[-1]['train_loss_sum_over_codes'],last_validation_ce=eval_rows[-1]['validation_loss']['total_ce'],
                best_invalid_fraction=best['validation']['invalid_prediction_fraction']))
        assert len(set(matched))==1
    assert not list(root.glob('trials/**/test*predictions*'))
    aggregate=[]
    for w in WINDOWS:
        rows=[r for r in records if r['window']==w]
        aggregate.append(dict(window=w,mean_selected_ndcg10=float(np.mean([r['ndcg10']for r in rows])),sd_selected_ndcg10=float(np.std([r['ndcg10']for r in rows],ddof=1)),
            mean_recall10=float(np.mean([r['recall10']for r in rows])),mean_fixed40=float(np.mean([r['fixed40']for r in rows])),mean_fixed45=float(np.mean([r['fixed45']for r in rows])),
            mean_best_within40=float(np.mean([r['best_within40']for r in rows])),mean_warm_epoch_seconds=float(np.mean([r['warm_epoch_seconds']for r in rows])),
            mean_fit_seconds=float(np.mean([r['fit_seconds']for r in rows])),forward_gflops=rows[0]['forward_gflops'],early_stopped=sum(r['stop_reason']=='early_stopping'for r in rows),
            capped=sum(r['stop_reason']=='epoch_limit'for r in rows),peak_cuda_allocated_gib=max(r['peak_cuda_allocated_gib']for r in rows),peak_cuda_reserved_gib=max(r['peak_cuda_reserved_gib']for r in rows)))
    bootstrap=np.random.default_rng(4242).integers(0,n,size=(5000,n));comparisons=[]
    for left,right in [(32,20),(64,20),(64,32)]:
        for criterion in ('selected','fixed40','fixed45'):
            a=[];b=[]
            for seed in SEEDS:
                def at(w):return next(r['best_epoch']for r in records if r['window']==w and r['seed']==seed)if criterion=='selected'else int(criterion[5:])
                a.append(credits[(left,seed,at(left))]);b.append(credits[(right,seed,at(right))])
            aa=np.array(a);bb=np.array(b);delta=(aa-bb).mean(0);ci=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            comparisons.append(dict(left=left,right=right,criterion=criterion,absolute_delta=float(delta.mean()),relative_delta=float(aa.mean()/bb.mean()-1),
                seed_deltas=(aa.mean(1)-bb.mean(1)).tolist(),conditional_shared_user_ci95=ci.tolist(),
                improved_user_seed_cases=int((aa>bb).sum()),worsened_user_seed_cases=int((aa<bb).sum()),tied_user_seed_cases=int((aa==bb).sum())))
    # Descriptive user-level export and cohort breakdown; not additional hypothesis tests.
    frequency=np.bincount(raw['train'][2],minlength=c['data']['items']);audit=read(data/'user-audit.json');length=np.array([r['events']-2 for r in audit]);groups=[]
    masks=[('available_history_64_95',(length>=64)&(length<96)),('available_history_96_127',(length>=96)&(length<128)),('available_history_128plus',length>=128),
           ('target_train_positions_0',frequency[targets]==0),('target_train_positions_1_5',(frequency[targets]>=1)&(frequency[targets]<=5)),('target_train_positions_6_20',(frequency[targets]>=6)&(frequency[targets]<=20)),('target_train_positions_21plus',frequency[targets]>=21)]
    user_rows=[]
    selected={w:np.array([credits[(w,s,next(r['best_epoch']for r in records if r['window']==w and r['seed']==s))]for s in SEEDS])for w in WINDOWS}
    for label,mask in masks:
        groups.append(dict(group=label,users=int(mask.sum()),mean_selected_ndcg10={str(w):float(selected[w][:,mask].mean())if mask.any()else None for w in WINDOWS}))
    for at,uid in enumerate(users):
        user_rows.append(dict(user_id=int(uid),target_item_id=int(targets[at]),available_history=int(length[at]),target_train_positions=int(frequency[targets[at]]),**{f'ndcg10_window{w}_mean3':float(selected[w][:,at].mean())for w in WINDOWS}))
    def csvwrite(path,rows):
        with path.open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvwrite(out/'trial-metrics.csv',records);csvwrite(out/'user-metrics.csv',user_rows)
    result=dict(source=SOURCE,session_seconds=manifest['duration_seconds'],models=9,validation_users=n,train_examples=len(raw['train'][2]),
        validation_files_replayed=replayed,total_epochs=sum(r['stop_epoch']for r in records),amp_skipped_updates=sum(r['amp_skips']for r in records),
        all_metrics_cases_pins_lr_ema_accounting_verified=True,invalid_predictions=total_invalid,duplicate_predictions=total_duplicates,total_predictions=total_predictions,
        test_evaluated=False,aggregate=aggregate,comparisons=comparisons,trials=records,groups=groups,
        limits=['VAL only; selected maxima use different epoch budgets and validation selection.',
            'Three coupled tokenizer/generator seeds on one fixed selected cohort; no variance decomposition.',
            'User bootstrap conditions on the nine trained fits and their selected epochs; excludes seed/index/item-clustering/selection uncertainty.',
            'Six fits censored at60; EMA operational stopping is not a convergence proof.',
            'Only13-24 correct top10 targets per selected model among958 users; relative deltas are noisy.',
            'Window comparison does not isolate a capacity bottleneck or prove absence of useful older history.'])
    write(out/'analysis.json',result)
    return result,curves


def plot(result,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    colors={20:'#2463a8',32:'#df912d',64:'#278455'}
    for j,seed in enumerate(SEEDS):
        for w in WINDOWS:
            rows=[r for r in curves[f'recent{w}_{seed}']if 'validation'in r]
            axes[0,j].plot([r['epoch']for r in rows],[r['validation']['ndcg@10']for r in rows],label=f'Recent{w}',color=colors[w],marker='.',ms=4)
            axes[1,j].plot([r['epoch']for r in rows],[r['validation_loss']['total_ce']for r in rows],label=f'Recent{w}',color=colors[w])
        axes[0,j].set_title(f'Pipeline seed {seed}');axes[0,j].set_ylabel('VAL NDCG@10');axes[1,j].set_ylabel('Teacher-forced VAL CE');axes[1,j].set_xlabel('Epoch')
        for ax in axes[:,j]:ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'learning-curves.png',dpi=180);fig.savefig(out/'learning-curves.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    r,c=analyze(a.results,a.data,a.output)
    if a.plot:plot(r,c,a.output)
    print(json.dumps({k:r[k]for k in ('models','session_seconds','validation_files_replayed','total_epochs','amp_skipped_updates','aggregate','comparisons')},indent=2))
