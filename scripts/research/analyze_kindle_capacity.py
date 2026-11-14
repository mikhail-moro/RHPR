"""Independent CPU replay of the four fixed-index capacity/window cells."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex,evaluate_ranked_sids

SOURCE='6d2919a155893e4c6bbb859a1fbcc0c800573c6c'

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze(root,parent,data,out):
    root,parent,data,out=map(Path,(root,parent,data,out));out.mkdir(parents=True,exist_ok=True)
    c=read(root/'experiment.json');manifest=read(root/'run-manifest.json');study=read(root/'study-summary.json');suite=read(root/'suite-results.json')
    assert manifest['git_sha']==SOURCE and manifest['config']==c and not suite['failures']
    assert suite['results']==study['results'] and study['tokenizer_refitted']is False and study['test_evaluated']is False
    assert read(root/'metrics.json')==dict(experiment=c['name'],models=2,reused_baselines=2,seed=42,test_evaluated=False)
    assert (root/'summary.md').is_file()
    for f,expected in c['parent']['files'].items():
        assert (parent/f).stat().st_size==expected['size_bytes']and digest(parent/f)==expected['sha256']
        if f.startswith('trials/'):
            _,trial,name=f.split('/');assert digest(root/'baselines'/trial/name)==expected['sha256']
    assert read(parent/'run-manifest.json')['git_sha']==c['parent']['git_sha']
    assert digest(root/'frozen-indexes/items-42.npy')==digest(parent/'frozen-indexes/items-42.npy')
    for f,sha in c['movie_input']['sha256'].items():assert digest(data/f)==sha
    raw={}
    for s in ('train','validation'):
        with np.load(data/(s+'-cases.npz'))as z:raw[s]=tuple(z[k]for k in ('contexts','user_ids','targets'))
    users=raw['validation'][1];targets=raw['validation'][2];n=len(users);steps=math.ceil(len(raw['train'][2])/512)
    idx=SidIndex.from_full_sids(np.load(root/'frozen-indexes/items-42.npy'),width=256)
    records=[];credits={};curves={};replayed=0;invalid=duplicates=prediction_count=0;initial={}
    for size,rows,directory in [('mini',study['baselines'],'baselines'),('wide',study['results'],'trials')]:
        for trial,r in rows.items():
            width=r['history_items'];key=f'{size}{width}';d=root/directory/trial;s=r['selection'];curve=read(d/'training-curves.json');curves[key]=curve
            assert read(d/'trial-result.json')==r and r['seed']==42 and r['test_evaluated']is False
            assert r['sid_file_sha256']==digest(root/'frozen-indexes/items-42.npy')
            assert r['raw_case_sha256']=={s:case_digest(a)for s,a in raw.items()}
            assert r['case_targets_sha256']=={s:case_digest(a[1:])for s,a in raw.items()}
            initial.setdefault(size,set()).add(r['initial_state_sha256'])
            assert [x['epoch']for x in curve]==list(range(1,s['training_epochs']+1))
            ema=reference=None;last=None;previous=0;evals=[]
            def lr(update):
                if update<2*steps:return .003*(update+1)/(2*steps)
                return .001+.002*.5*(1+math.cos(math.pi*min(1.,(update-2*steps+1)/(58*steps))))
            for row in curve:
                ep=row['epoch'];assert row['updates']+row['amp_skipped_updates']==ep*steps and row['train_examples']==len(raw['train'][2])
                assert abs(row['learning_rate']-lr(previous))<1e-14 and abs(row['next_learning_rate']-lr(row['updates']))<1e-14
                assert min(abs(row['learning_rate_last_batch']-lr(k))for k in [row['updates']-1,row['updates']])<1e-14;previous=row['updates']
                if 'validation'not in row:continue
                score=row['validation']['ndcg@10'];ema=score if ema is None else .5*score+.5*ema
                if reference is None or ema>reference+.0005:reference=ema;last=ep
                assert (ema,reference,last)==(row['validation_ema_ndcg10'],row['ema_reference_ndcg10'],row['ema_last_improvement_epoch'])
                if ep>=40 and ep<len(curve):assert ep-last<15
                with np.load(d/f'validation-epoch-{ep:03d}.npz')as z:
                    pred=z['semantic_ids'];np.testing.assert_array_equal(z['user_ids'],users);np.testing.assert_array_equal(z['target_item_ids'],targets);assert np.isfinite(z['scores']).all()
                metrics=evaluate_ranked_sids(idx,pred,targets,cutoffs=(5,10))
                for k,value in metrics.items():assert abs(value-row['validation'][k])<1e-12,(key,ep,k)
                hit=(pred[:,:10]==idx.sids[targets,None,:]).all(-1);credit=np.where(hit.any(1),1/np.log2(hit.argmax(1)+2),0.)
                assert abs(float(credit.mean())-score)<1e-12;credits[(key,ep)]=credit
                invalid+=metrics['invalid_predictions'];duplicates+=metrics['duplicate_predictions'];prediction_count+=metrics['prediction_count'];replayed+=1;evals.append(row)
            best=max(evals,key=lambda x:x['validation']['ndcg@10']);assert s['best_epoch']==best['epoch']and s['validation_ndcg10']==best['validation']['ndcg@10']
            assert len(evals)==s['validation_checks']==len(list(d.glob('validation-epoch-*.npz')))
            assert s['final_early_stopping_state']==dict(ema=ema,reference=reference,last_improvement_epoch=last)
            assert s['optimizer_updates']==curve[-1]['updates']and s['amp_skipped_updates']==curve[-1]['amp_skipped_updates']
            if s['stopping_reason']=='early_stopping':assert len(curve)>=40 and len(curve)-last>=15
            else:assert s['stopping_reason']=='epoch_limit'and len(curve)==60
            best40=max((x for x in evals if x['epoch']<=40),key=lambda x:x['validation']['ndcg@10'])
            best45=max((x for x in evals if x['epoch']<=45),key=lambda x:x['validation']['ndcg@10'])
            records.append(dict(cell=key,trial=trial,size=size,window=width,parameters=r['forward']['parameters'],
                selected_ndcg10=s['validation_ndcg10'],selected_recall10=best['validation']['recall@10'],selected_hits10=round(best['validation']['recall@10']*n),
                best_epoch=best['epoch'],last_epoch=len(curve),stop_reason=s['stopping_reason'],
                fixed40=next(x['validation']['ndcg@10']for x in evals if x['epoch']==40),fixed45=next(x['validation']['ndcg@10']for x in evals if x['epoch']==45),
                best_within40=best40['validation']['ndcg@10'],best_within40_epoch=best40['epoch'],best_within45=best45['validation']['ndcg@10'],best_within45_epoch=best45['epoch'],
                forward_gflops=r['forward']['supported_forward_flops_batch1']/1e9,warm_epoch_seconds=r['timing']['warm_epoch_training_median_seconds'],
                fit_seconds=s['training_seconds'],peak_cuda_allocated_gib=r['timing']['cuda_peak_allocated_bytes']/2**30,amp_skips=s['amp_skipped_updates'],
                last_train_loss=curve[-1]['train_loss_sum_over_codes'],last_val_ce=evals[-1]['validation_loss']['total_ce'],
                minimum_val_ce=min(x['validation_loss']['total_ce']for x in evals),minimum_val_ce_epoch=min(evals,key=lambda x:x['validation_loss']['total_ce'])['epoch']))
    assert all(len(s)==1 for s in initial.values())and initial['mini']!=initial['wide']
    assert not list(root.glob('**/test*predictions*'))
    bootstrap=np.random.default_rng(4242).integers(0,n,size=(5000,n));by={r['cell']:r for r in records};comparisons=[];interactions=[]
    def vector(key,criterion):
        r=by[key];epoch={'selected':r['best_epoch'],'fixed40':40,'fixed45':45,'best_within40':r['best_within40_epoch'],'best_within45':r['best_within45_epoch']}[criterion]
        return credits[(key,epoch)]
    for criterion in ('selected','fixed40','fixed45','best_within40','best_within45'):
        for left,right in [('wide20','mini20'),('wide32','mini32'),('wide32','wide20'),('mini32','mini20')]:
            a,b=vector(left,criterion),vector(right,criterion);delta=a-b;ci=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            comparisons.append(dict(left=left,right=right,criterion=criterion,absolute_delta=float(delta.mean()),relative_delta=float(a.mean()/b.mean()-1),conditional_user_ci95=ci.tolist(),improved_users=int((delta>0).sum()),worsened_users=int((delta<0).sum()),tied_users=int((delta==0).sum())))
        delta=(vector('wide32',criterion)-vector('wide20',criterion))-(vector('mini32',criterion)-vector('mini20',criterion))
        interactions.append(dict(criterion=criterion,absolute_change_in_context_gap=float(delta.mean()),conditional_user_ci95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist()))
    result=dict(source=SOURCE,models=2,reused_baselines=2,seed=42,validation_users=n,session_seconds=manifest['duration_seconds'],validation_files_replayed=replayed,
        parent_files_verified=len(c['parent']['files']),test_evaluated=False,tokenizer_refitted=False,all_metrics_pins_cases_lr_ema_updates_verified=True,
        invalid_predictions=invalid,duplicate_predictions=duplicates,prediction_count=prediction_count,cells=records,comparisons=comparisons,context_interactions=interactions,
        limits=['One fixed examined tokenizer+generator seed; user bootstrap excludes seed/index/item-clustering/selection uncertainty.',
                'Selected wide fits ran60epochs, narrow45/55; fixed40/45 and common-budget selected comparisons are included.',
                'Same LR policy is not a tuned scaling law; widening changes optimization as well as capacity.',
                'No TEST results or confirmed long-context improvement.'])
    write(out/'analysis.json',result);return result,curves

def plot(curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),constrained_layout=True)
    for key,color,style in [('mini20','#3969ac','--'),('mini32','#d78b24','--'),('wide20','#3969ac','-'),('wide32','#d78b24','-')]:
        rows=[r for r in curves[key]if 'validation'in r];x=[r['epoch']for r in rows]
        axes[0].plot(x,[r['validation']['ndcg@10']for r in rows],label=key,color=color,ls=style,marker='.',ms=4)
        axes[1].plot(x,[r['validation_loss']['total_ce']for r in rows],label=key,color=color,ls=style)
    axes[0].set_ylabel('VAL NDCG@10');axes[1].set_ylabel('Teacher-forced VAL CE');axes[0].legend()
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.savefig(out/'capacity-curves.png',dpi=180);fig.savefig(out/'capacity-curves.pdf');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args()
    result,curves=analyze(a.results,a.parent,a.data,a.output)
    if a.plot:plot(curves,a.output)
    print(json.dumps(result,indent=2))
