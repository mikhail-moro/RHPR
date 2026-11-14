"""CPU replay of the seed42 long-pair pilot and its frozen control TESTs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_pairs import check_parent, collect_keys, validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import unique_pair_sids
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = '33754d70acc0e4cf826f139202632bf0ffe6ac5b'


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def analyze(root, parent, data, out):
    root,parent,data,out=map(Path,(root,parent,data,out));out.mkdir(parents=True,exist_ok=True)
    c=read(root/'experiment.json');validate(c)
    manifest=read(root/'run-manifest.json');study=read(root/'study-summary.json');suite=read(root/'suite-results.json')
    assert manifest['git_sha']==SOURCE and manifest['config']==c
    assert suite['results']==study['results'] and not suite['failures']
    assert study['new_generator_fits']==2 and study['test_evaluated'] and not study['control64_trained'] and not study['item_tokenizer_refitted']
    assert read(root/'metrics.json')==dict(experiment=c['name'],models=2,read_only_control_tests=2,test_evaluated=True,control64_trained=False)
    assert (root/'summary.md').is_file()
    check_parent(c,read(parent/'run-manifest.json')['config'])
    for name,spec in c['parent']['files'].items():
        assert (parent/name).stat().st_size==spec['size_bytes'] and digest(parent/name)==spec['sha256']
        if name.startswith('trials/') and not name.endswith('.pt'):
            _,trial,file=name.split('/');assert digest(root/'baselines'/trial/file)==spec['sha256']
    for name,sha in c['movie_input']['sha256'].items():assert digest(data/name)==sha
    raw={}
    for split in ('train','validation','test'):
        with np.load(data/f'{split}-cases.npz') as z:raw[split]=tuple(z[k] for k in ('contexts','user_ids','targets'))
    steps=math.ceil(len(raw['train'][2])/512)
    idx=SidIndex.from_full_sids(np.load(root/'frozen-indexes/items-42.npy'),width=256)
    assert digest(root/'frozen-indexes/items-42.npy')==digest(parent/'frozen-indexes/items-42.npy')
    # Reconstruct split membership and collision ordinals without any encoder fit.
    sets={s:np.union1d(*[collect_keys(a[0],w,len(idx.sids)) for w in (32,64)]) for s,a in raw.items()}
    keys=np.load(root/'frozen-indexes/pair-keys.npy');pairs=np.load(root/'frozen-indexes/pairs-42.npy')
    np.testing.assert_array_equal(keys,np.unique(np.concatenate(list(sets.values()))))
    np.testing.assert_array_equal(np.load(root/'pair-cache/train-mask.npy'),np.isin(keys,sets['train']))
    np.testing.assert_array_equal(pairs,unique_pair_sids(keys,pairs[:,:3],sets['train'],np.union1d(sets['train'],sets['validation'])))
    index_report=read(root/'frozen-indexes/index-42.json')
    assert index_report['unique_pair_sids']==len(keys)==len(np.unique(pairs,axis=0))
    assert index_report['counts']=={s:len(k) for s,k in sets.items()} and index_report['fit_on_train_pairs_only']
    assert index_report['max_suffix']==int(pairs[:,3].max())<256
    records=[];curves={};credits={};initial=set();files=invalid=duplicates=predictions=0

    def replay(path,split,expected):
        nonlocal files,invalid,duplicates,predictions
        with np.load(path) as z:
            pred=z['semantic_ids'];np.testing.assert_array_equal(z['user_ids'],raw[split][1])
            np.testing.assert_array_equal(z['target_item_ids'],raw[split][2]);assert np.isfinite(z['scores']).all()
        metric=evaluate_ranked_sids(idx,pred,raw[split][2],cutoffs=(5,10))
        for key,value in metric.items():assert abs(value-expected[key])<1e-12,(path,key)
        hit=(pred[:,:10]==idx.sids[raw[split][2],None,:]).all(-1)
        credit=np.where(hit.any(1),1/np.log2(hit.argmax(1)+2),0.)
        assert abs(float(credit.mean())-metric['ndcg@10'])<1e-12
        files+=1;invalid+=metric['invalid_predictions'];duplicates+=metric['duplicate_predictions'];predictions+=metric['prediction_count']
        return credit,metric

    cells=[('recent20','baselines','recent20_wide_42'),('recent32','baselines','recent32_wide_42'),
           ('pair32','trials','pair32_wide_42'),('pair64','trials','pair64_wide_42')]
    for key,directory,trial in cells:
        d=root/directory/trial;r=read(d/'trial-result.json');selection=r['selection'];curve=read(d/'training-curves.json');curves[key]=curve
        initial.add(r['initial_state_sha256']);assert r['sid_file_sha256']==digest(root/'frozen-indexes/items-42.npy')
        assert r['raw_case_sha256']=={s:case_digest(raw[s]) for s in ('train','validation')}
        assert r['case_targets_sha256']=={s:case_digest(raw[s][1:]) for s in ('train','validation')}
        if directory=='trials':
            assert r==study['results'][trial] and r['test_evaluated']
            assert r['pair_sid_file_sha256']==digest(root/'frozen-indexes/pairs-42.npy')
            assert r['test_case_sha256']==case_digest(raw['test'])
        assert [r['epoch'] for r in curve]==list(range(1,selection['training_epochs']+1))
        ema=reference=last=None;previous=0;evals=[]
        def lr(update):
            if update<2*steps:return .003*(update+1)/(2*steps)
            return .001+.002*.5*(1+math.cos(math.pi*min(1.,(update-2*steps+1)/(58*steps))))
        for row in curve:
            ep=row['epoch'];assert row['updates']+row['amp_skipped_updates']==ep*steps and row['train_examples']==len(raw['train'][2])
            assert abs(row['learning_rate']-lr(previous))<1e-14 and abs(row['next_learning_rate']-lr(row['updates']))<1e-14
            assert min(abs(row['learning_rate_last_batch']-lr(k)) for k in (row['updates']-1,row['updates']))<1e-14
            previous=row['updates']
            if 'validation' not in row:continue
            score=row['validation']['ndcg@10'];ema=score if ema is None else .5*score+.5*ema
            if reference is None or ema>reference+.0005:reference=ema;last=ep
            assert (ema,reference,last)==(row['validation_ema_ndcg10'],row['ema_reference_ndcg10'],row['ema_last_improvement_epoch'])
            if ep>=40 and ep<len(curve):assert ep-last<15
            credits[(key,ep)],_=replay(d/f'validation-epoch-{ep:03d}.npz','validation',row['validation']);evals.append(row)
        best=max(evals,key=lambda x:x['validation']['ndcg@10'])
        assert selection['best_epoch']==best['epoch'] and selection['validation_ndcg10']==best['validation']['ndcg@10']
        assert len(evals)==selection['validation_checks']==len(list(d.glob('validation-epoch-*.npz')))
        assert selection['final_early_stopping_state']==dict(ema=ema,reference=reference,last_improvement_epoch=last)
        assert selection['optimizer_updates']==curve[-1]['updates'] and selection['amp_skipped_updates']==curve[-1]['amp_skipped_updates']
        assert abs(selection['optimizer_training_seconds']-sum(x['train_seconds'] for x in curve))<1e-6
        if selection['stopping_reason']=='epoch_limit':assert len(curve)==60
        else:assert selection['stopping_reason']=='early_stopping' and len(curve)>=40 and len(curve)-last>=15
        if directory=='trials':test=r['test'];testpath=d/'test-best-predictions.npz'
        else:
            td=root/'control-tests'/trial;control=read(td/'control-result.json');assert control==study['control_tests'][trial]
            assert control['parent_result']==r and control['training_performed'] is False and control['test_case_sha256']==case_digest(raw['test'])
            test=control['test'];testpath=td/'test-best-predictions.npz'
            assert test['checkpoint_sha256']==c['parent']['files'][f'trials/{trial}/generator-best.pt']['sha256']
        assert test['epoch']==best['epoch'] and test['selection']=='raw validation NDCG@10 only' and not test['seen_items_filtered']
        credits[(key,'test')],metric=replay(testpath,'test',test['metrics'])
        best40=max((r for r in evals if r['epoch']<=40),key=lambda x:x['validation']['ndcg@10'])
        records.append(dict(model=key,trial=trial,window=r['history_items'],selected_val_ndcg10=best['validation']['ndcg@10'],
            selected_val_hits10=round(best['validation']['recall@10']*len(raw['validation'][2])),test_ndcg10=metric['ndcg@10'],
            test_recall10=metric['recall@10'],test_hits10=round(metric['recall@10']*len(raw['test'][2])),
            best_epoch=best['epoch'],last_epoch=len(curve),stop_reason=selection['stopping_reason'],
            fixed40=next(x['validation']['ndcg@10'] for x in evals if x['epoch']==40),
            fixed45=next(x['validation']['ndcg@10'] for x in evals if x['epoch']==45),
            best_within40=best40['validation']['ndcg@10'],best_within40_epoch=best40['epoch'],
            parameters=r['forward']['parameters'],encoder_positions=r['forward']['encoder_positions'],
            forward_gflops=r['forward']['supported_forward_flops_batch1']/1e9,warm_epoch_seconds=r['timing']['warm_epoch_training_median_seconds'],
            first_epoch_seconds=curve[0]['train_seconds'],fit_seconds=selection['training_seconds'],amp_skips=selection['amp_skipped_updates'],
            peak_cuda_allocated_gib=r['timing']['cuda_peak_allocated_bytes']/2**30,
            last_train_loss=curve[-1]['train_loss_sum_over_codes'],last_val_ce=evals[-1]['validation_loss']['total_ce'],
            minimum_val_ce=min(x['validation_loss']['total_ce'] for x in evals),minimum_val_ce_epoch=min(evals,key=lambda x:x['validation_loss']['total_ce'])['epoch']))
    assert len(initial)==1
    by={r['model']:r for r in records};comparisons=[]
    boot=np.random.default_rng(4242).integers(0,len(raw['validation'][2]),size=(5000,len(raw['validation'][2])))
    for criterion in ('selected_val','test','fixed40','fixed45','best_within40'):
        def vector(key):
            epoch={'selected_val':by[key]['best_epoch'],'test':'test','fixed40':40,'fixed45':45,'best_within40':by[key]['best_within40_epoch']}[criterion]
            return credits[(key,epoch)]
        for left,right in [('pair32','recent32'),('pair32','recent20'),('pair64','recent32'),('pair64','recent20'),('pair64','pair32'),('recent32','recent20')]:
            a,b=vector(left),vector(right);delta=a-b
            comparisons.append(dict(left=left,right=right,criterion=criterion,absolute_delta=float(delta.mean()),relative_delta=float(a.mean()/b.mean()-1),
                conditional_user_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),improved_users=int((delta>0).sum()),worsened_users=int((delta<0).sum()),tied_users=int((delta==0).sum())))
    result=dict(source=SOURCE,new_models=2,reused_control_tests=2,seed=42,session_seconds=manifest['duration_seconds'],
        validation_users=len(raw['validation'][2]),test_users=len(raw['test'][2]),parent_files_verified=len(c['parent']['files']),
        prediction_files_replayed=files,prediction_count=predictions,invalid_predictions=invalid,duplicate_predictions=duplicates,
        all_metrics_pins_cases_lr_ema_updates_verified=True,records=records,comparisons=comparisons,index=index_report,
        limits=['One previously examined generator/index seed; shared-user intervals exclude seed/index/item-clustering/checkpoint-selection uncertainty.',
                'TEST is final reporting for four VAL-selected checkpoints, not a tuning signal; later experiments on this cohort are exploratory.',
                'All four selected55/capped60; no demonstrated convergence or equal-wall-time budget.',
                'Pair64 retains32 raw recent items, Pair32 retains16; their gap does not isolate usefulness of compressed old history.',
                'No matched wide Recent64 was trained. FLOPs exclude beam/backward/offline pair encoding; historical timings are not a simultaneous serving benchmark.'])
    write(out/'analysis.json',result)
    with (out/'model-metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result,curves


def plot(result,curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(recent20='#2463a8',recent32='#df912d',pair32='#278455',pair64='#a35caa')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for name,color in colors.items():
        rows=[r for r in curves[name] if 'validation' in r];x=[r['epoch'] for r in rows]
        axes[0,0].plot(x,[r['validation']['ndcg@10'] for r in rows],color=color,label=name,marker='.',ms=4)
        axes[0,1].plot(x,[r['validation_loss']['total_ce'] for r in rows],color=color,label=name)
        axes[1,0].plot([r['epoch'] for r in curves[name]],[r['train_loss_sum_over_codes'] for r in curves[name]],color=color,label=name)
    for r in result['records']:
        axes[1,1].scatter(r['forward_gflops'],r['test_ndcg10'],color=colors[r['model']],s=45)
        axes[1,1].annotate(r['model'],(r['forward_gflops'],r['test_ndcg10']),xytext=(5,5),textcoords='offset points')
    axes[0,0].set_ylabel('VAL NDCG@10');axes[0,0].legend()
    axes[0,1].set_ylabel('Teacher-forced VAL CE');axes[1,0].set_ylabel('Train CE');axes[1,0].set_ylim(7,16)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Epoch')
    axes[1,1].set_xlabel('B1 supported forward GFLOPs');axes[1,1].set_ylabel('TEST NDCG@10 (VAL-selected)')
    axes[1,1].set_xlim(.3,1.32);axes[1,1].set_ylim(.0095,.016)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(out/'pair-curves.png',dpi=180);fig.savefig(out/'pair-curves.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();result,curves=analyze(args.results,args.parent,args.data,args.output)
    if args.plot:plot(result,curves,args.output)
    print(json.dumps({k:result[k] for k in ('session_seconds','prediction_files_replayed','records','comparisons')},indent=2))
