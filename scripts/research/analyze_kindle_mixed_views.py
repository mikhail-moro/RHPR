"""CPU replay of the completed Full32/Pair32 mixture and fixed-index controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_mixed_views import MixedViews, validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = '88722e9081aae120e9af261f550d50fe74864da6'
CUTOFF = 2.300820444821813


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def analyze(root,parent,partial,data,archive):
    root,parent,partial,data,archive=map(Path,(root,parent,partial,data,archive))
    c=read(root/'experiment.json');validate(c)
    manifest=read(root/'run-manifest.json');study=read(root/'study-summary.json');suite=read(root/'suite-results.json')
    assert manifest['git_sha']==SOURCE and manifest['config']==c
    assert suite['results']==study['results'] and not suite['failures']
    assert study['new_generator_fits']==0 and study['resumed_generator_fits']==1 and not study['tokenizer_refitted']
    assert study['inference_format']=='pair32' and (root/'summary.md').is_file()
    assert read(parent/'run-manifest.json')['git_sha']==c['parent']['git_sha']
    for rel,expected in c['parent']['files'].items():
        assert (parent/rel).stat().st_size==expected['size_bytes'] and digest(parent/rel)==expected['sha256']
        if rel.startswith(('trials/','baselines/','control-tests/')):
            assert digest(root/'baselines'/rel)==expected['sha256']
    for rel,expected in c['resume_parent']['files'].items():
        assert (partial/rel).stat().st_size==expected['size_bytes'] and digest(partial/rel)==expected['sha256']
    assert archive.stat().st_size==c['resume_parent']['archive_size_bytes'] and digest(archive)==c['resume_parent']['archive_sha256']
    for rel,sha in c['movie_input']['sha256'].items():assert digest(data/rel)==sha
    for name in ('items-42.npy','pairs-42.npy','pair-keys.npy'):
        assert digest(root/'frozen-indexes'/name)==digest(parent/'frozen-indexes'/name)==digest(partial/'frozen-indexes'/name)
    raw={}
    for split in ('train','validation','test'):
        with np.load(data/f'{split}-cases.npz')as z:raw[split]=tuple(z[k]for k in ('contexts','user_ids','targets'))
    n=len(raw['train'][2]);steps=math.ceil(n/512)
    idx=SidIndex.from_full_sids(np.load(root/'frozen-indexes/items-42.npy'),width=256)
    # Only the shape and epoch RNG are needed to reconstruct batch choices.
    views=MixedViews(np.broadcast_to(0,(n,128)),np.broadcast_to(0,(n,96)))
    records=[];curves={};credits={};ranks={};predictions={};initial=set()
    files=invalid=duplicates=prediction_count=0

    def replay(path,split,expected):
        nonlocal files,invalid,duplicates,prediction_count
        with np.load(path)as z:
            pred=z['semantic_ids'];np.testing.assert_array_equal(z['user_ids'],raw[split][1])
            np.testing.assert_array_equal(z['target_item_ids'],raw[split][2]);assert np.isfinite(z['scores']).all()
        metric=evaluate_ranked_sids(idx,pred,raw[split][2],cutoffs=(5,10))
        for k,v in metric.items():assert abs(v-expected[k])<1e-12,(path,k)
        hit=(pred==idx.sids[raw[split][2],None,:]).all(-1)
        rank=np.where(hit.any(1),hit.argmax(1)+1,21)
        credit=np.where(rank<=10,1/np.log2(rank+1),0.)
        assert abs(float(credit.mean())-metric['ndcg@10'])<1e-12
        files+=1;invalid+=metric['invalid_predictions'];duplicates+=metric['duplicate_predictions'];prediction_count+=metric['prediction_count']
        return credit,rank,pred

    cells=[('mixed32',root/'trials/mixed32_wide_42'),('pair32',parent/'trials/pair32_wide_42'),
           ('recent32',parent/'baselines/recent32_wide_42'),('recent20',parent/'baselines/recent20_wide_42')]
    for name,d in cells:
        r=read(d/'trial-result.json');s=r['selection'];curve=read(d/'training-curves.json');curves[name]=curve
        assert [x['epoch']for x in curve]==list(range(1,s['training_epochs']+1))
        assert r['raw_case_sha256']=={k:case_digest(raw[k])for k in ('train','validation')}
        assert r['case_targets_sha256']=={k:case_digest(raw[k][1:])for k in ('train','validation')}
        assert r['sid_file_sha256']==digest(root/'frozen-indexes/items-42.npy');initial.add(r['initial_state_sha256'])
        previous=0;ema=reference=last=None;evals=[]
        def lr(update):
            if update<2*steps:return .003*(update+1)/(2*steps)
            return .001+.002*.5*(1+math.cos(math.pi*min(1.,(update-2*steps+1)/(58*steps))))
        for row in curve:
            ep=row['epoch'];assert row['updates']+row['amp_skipped_updates']==steps*ep and row['train_examples']==n
            assert abs(row['learning_rate']-lr(previous))<1e-14 and abs(row['next_learning_rate']-lr(row['updates']))<1e-14
            assert min(abs(row['learning_rate_last_batch']-lr(k))for k in (row['updates']-1,row['updates']))<1e-14
            previous=row['updates']
            if name=='mixed32':
                order=np.random.default_rng(42+ep).permutation(n);choice=views.schedule(ep)
                expected_examples=dict(full32=0,pair32=0)
                for b,begin in enumerate(range(0,n,512)):
                    expected_examples['full32'if choice[b]else'pair32']+=len(order[begin:begin+512])
                v=row['training_views']
                assert v['examples']==expected_examples and v['batches']==dict(full32=int(choice.sum()),pair32=int((~choice).sum()))
                assert v['schedule_sha256']==hashlib.sha256(choice.tobytes()).hexdigest() and v['examples_consumed_once']
                assert v['native_sid_widths']==dict(full32=128,pair32=96)
            if 'validation'not in row:continue
            score=row['validation']['ndcg@10'];ema=score if ema is None else .5*score+.5*ema
            if reference is None or ema>reference+.0005:reference=ema;last=ep
            assert (ema,reference,last)==(row['validation_ema_ndcg10'],row['ema_reference_ndcg10'],row['ema_last_improvement_epoch'])
            if ep>=40 and ep<len(curve):assert ep-last<15
            credits[name,ep],ranks[name,ep],predictions[name,ep]=replay(d/f'validation-epoch-{ep:03d}.npz','validation',row['validation'])
            evals.append(row)
        best=max(evals,key=lambda x:x['validation']['ndcg@10'])
        assert s['best_epoch']==best['epoch'] and s['validation_ndcg10']==best['validation']['ndcg@10']
        assert s['validation_checks']==len(evals)==len(list(d.glob('validation-epoch-*.npz')))
        assert s['optimizer_updates']==curve[-1]['updates'] and s['amp_skipped_updates']==curve[-1]['amp_skipped_updates']
        assert s['final_early_stopping_state']==dict(ema=ema,reference=reference,last_improvement_epoch=last)
        assert abs(s['optimizer_training_seconds']-sum(x['train_seconds']for x in curve))<1e-6
        if s['stopping_reason']=='epoch_limit':assert len(curve)==60
        else:assert s['stopping_reason']=='early_stopping'and len(curve)>=40 and len(curve)-last>=15
        if name.startswith('recent'):
            td=parent/'control-tests'/d.name;control=read(td/'control-result.json');assert control['parent_result']==r
            test=control['test'];testfile=td/'test-best-predictions.npz'
        else:test=r['test'];testfile=d/'test-best-predictions.npz'
        assert test['epoch']==s['best_epoch']and not test['seen_items_filtered']
        credits[name,'test'],ranks[name,'test'],predictions[name,'test']=replay(testfile,'test',test['metrics'])
        records.append(dict(model=name,best_epoch=s['best_epoch'],last_epoch=s['training_epochs'],stop_reason=s['stopping_reason'],
            validation_ndcg10=s['validation_ndcg10'],validation_hits10=int((ranks[name,s['best_epoch']]<=10).sum()),
            test_ndcg10=test['metrics']['ndcg@10'],test_recall10=test['metrics']['recall@10'],test_hits10=int((ranks[name,'test']<=10).sum()),
            forward_gflops=r['forward']['supported_forward_flops_batch1']/1e9,parameters=r['forward']['parameters'],
            encoder_positions=r['forward']['encoder_positions'],warm_epoch_seconds=r['timing']['warm_epoch_training_median_seconds'],
            optimizer_training_seconds=s['optimizer_training_seconds'],amp_skips=s['amp_skipped_updates'],
            minimum_val_ce=min(x['validation_loss']['total_ce']for x in evals),
            minimum_val_ce_epoch=min(evals,key=lambda x:x['validation_loss']['total_ce'])['epoch']))
    assert len(initial)==1
    mixed=read(cells[0][1]/'trial-result.json');assert study['results']=={mixed['trial']:mixed}
    assert read(root/'metrics.json')==dict(experiment=c['name'],models=1,seed=42,test_evaluated=True,
        validation_ndcg10=mixed['selection']['validation_ndcg10'],test_ndcg10=mixed['test']['metrics']['ndcg@10'])
    md=cells[0][1];old=partial/'trials/mixed32_wide_42'
    assert curves['mixed32'][:30]==read(old/'training-curves.json')
    assert read(md/'training-view-provenance.json')==read(old/'training-view-provenance.json')
    for path in old.glob('validation-epoch-*.npz'):assert digest(md/path.name)==digest(path)
    verification=read(md/'resume-verification.json')
    assert verification['source_epoch']==30 and verification['source_updates']==8280 and verification['rng_restored_and_verified']
    assert verification['optimizer_restored']and verification['scaler_restored']
    selection=mixed['selection'];assert selection['resumed_from_epoch']==30
    assert selection['continuation_epochs']==len(curves['mixed32'])-30
    assert selection['continuation_optimizer_updates']==selection['optimizer_updates']-8280
    assert abs(selection['continuation_optimizer_training_seconds']-sum(r['train_seconds']for r in curves['mixed32'][30:]))<1e-6
    # Inspect actual saved weights/optimizer rather than inferring resume from logs.
    import torch
    from rhpr.gr.sid_study import state_digest
    source_state=torch.load(old/'generator-last.pt',map_location='cpu',weights_only=False)
    assert state_digest(source_state['model'])==verification['model_state_sha256'];del source_state
    last=torch.load(md/'generator-last.pt',map_location='cpu',weights_only=False)
    best=torch.load(md/'generator-best.pt',map_location='cpu',weights_only=False)
    assert last['epoch']==selection['training_epochs']and last['updates']==selection['optimizer_updates']
    assert last['scheduler']['last_epoch']==last['updates'] and best['epoch']==selection['best_epoch']
    assert digest(md/'generator-best.pt')==mixed['test']['checkpoint_sha256']
    assert best['validation']['ndcg@10']==selection['validation_ndcg10']
    for checkpoint in (last,best):
        assert checkpoint['config']['mixed_views']==c['mixed_views']
        assert all(bool(torch.isfinite(v).all())for v in checkpoint['model'].values()if v.is_floating_point())
    del last,best
    by={r['model']:r for r in records};rng=np.random.default_rng(4242)
    comparisons=[];groups=[];recovery=[];saved={}
    freq=np.bincount(raw['train'][2],minlength=len(idx.sids))
    def compare(a,b,ar,br):
        delta=a-b;boot=rng.integers(0,len(a),size=(5000,len(a)))
        return dict(absolute_delta=float(delta.mean()),relative_delta=float(a.mean()/b.mean()-1)if b.mean()>0 else None,
            conditional_user_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),
            improved_users=int((delta>0).sum()),worsened_users=int((delta<0).sum()),tied_users=int((delta==0).sum()),
            gained_hits=int(((ar<=10)&(br>10)).sum()),lost_hits=int(((ar>10)&(br<=10)).sum()))
    for split in ('validation','test'):
        h,u,t=raw[split];popularity=np.log1p(freq[h[:,-16:]]).mean(1)
        masks=dict(all=np.ones(len(u),dtype=bool),less_popular_recent=popularity<=CUTOFF,more_popular_recent=popularity>CUTOFF,
            target_cold=freq[t]==0,target_frequency_1_5=(freq[t]>=1)&(freq[t]<=5),target_frequency_6_20=(freq[t]>=6)&(freq[t]<=20),target_frequency_over20=freq[t]>20)
        def key(name):return by[name]['best_epoch']if split=='validation'else'test'
        for name in by:
            k=key(name);v=credits[name,k];rank=ranks[name,k]
            saved[f'{split}_{name}_ranks']=rank
            for group,mask in masks.items():
                if not mask.any():continue
                groups.append(dict(split=split,model=name,group=group,users=int(mask.sum()),ndcg10=float(v[mask].mean()),hits10=int((rank[mask]<=10).sum())))
        for right in ('pair32','recent32','recent20'):
            a,b=credits['mixed32',key('mixed32')],credits[right,key(right)]
            ar,br=ranks['mixed32',key('mixed32')],ranks[right,key(right)]
            comparisons.append(dict(split=split,left='mixed32',right=right,**compare(a,b,ar,br)))
        original_lost=(ranks['pair32',key('pair32')]>10)&(ranks['recent32',key('recent32')]<=10)
        recovery.append(dict(split=split,original_recent32_hits_lost_by_pair32=int(original_lost.sum()),
            recovered_by_mixture=int((original_lost&(ranks['mixed32',key('mixed32')]<=10)).sum())))
    late=[]
    for ep in sorted({r['epoch']for r in curves['mixed32']if 'validation'in r}):
        row=dict(epoch=ep)
        for name in by:row[name]=float(credits[name,ep].mean())
        late.append(row)
    exposure={k:sum(r['training_views']['examples'][k]for r in curves['mixed32'])for k in ('full32','pair32')}
    result=dict(source=SOURCE,completed_fit=True,seed=42,validation_users=len(raw['validation'][2]),test_users=len(raw['test'][2]),
        resumed_from_epoch=30,completed_epochs=len(curves['mixed32']),continuation_session_seconds=manifest['duration_seconds'],
        original_parent_pins_verified=len(c['parent']['files']),resume_parent_pins_verified=len(c['resume_parent']['files']),
        checkpoint_archive_verified=True,original_30_epochs_preserved_exactly=True,weights_optimizer_rng_resume_verified=True,
        all_case_sid_initializer_lr_ema_selection_metrics_verified=True,prediction_files_replayed=files,
        prediction_count=prediction_count,invalid_predictions=invalid,duplicate_predictions=duplicates,
        records=records,comparisons=comparisons,validation_by_epoch=late,groups=groups,recovery=recovery,
        training_view_examples=exposure,full_view_fraction=exposure['full32']/sum(exposure.values()),
        recent_popularity_cutoff_from_prior_probe=CUTOFF,
        limits=['One previously examined seed/index/cohort; no seed-robust improvement or harm established.',
            'Conditional paired-user intervals omit seed/index/item clustering and checkpoint selection uncertainty; no multiplicity correction.',
            'TEST was examined before this new experiment; follow-up results are exploratory.',
            'Mixing whole batches is not random per-region mixing and does not isolate embedding supervision from regularization or reduced Pair exposure.',
            'Resume restores epoch-boundary state, but CUDA/compilation need not reproduce an uninterrupted GPU trajectory bitwise.',
            'Historical warm timings are not simultaneous benchmarks; extra cold resume compilation is included in total training time.',
            'B1 supported forward FLOPs exclude beam search, backward and offline text/pair processing.'])
    write(root/'mixed-analysis.json',result);np.savez_compressed(root/'mixed-user-ranks.npz',**saved)
    with (root/'mixed-model-metrics.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return result,curves


def plot(root,result,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(mixed32='#a145a7',pair32='#278455',recent32='#dd912d',recent20='#3268a8')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for name,color in colors.items():
        val=[r for r in curves[name]if 'validation'in r]
        axes[0,0].plot([r['epoch']for r in val],[r['validation']['ndcg@10']for r in val],color=color,label=name,marker='.',ms=4)
        axes[0,1].plot([r['epoch']for r in val],[r['validation_loss']['total_ce']for r in val],color=color,label=name)
        axes[1,0].plot([r['epoch']for r in curves[name]],[r['train_seconds']for r in curves[name]],color=color,label=name)
    for row in result['records']:
        axes[1,1].scatter(row['forward_gflops'],row['test_ndcg10'],color=colors[row['model']],s=45)
        offset=(-60,-15)if row['model']=='mixed32'else(5,5)
        axes[1,1].annotate(row['model'],(row['forward_gflops'],row['test_ndcg10']),xytext=offset,textcoords='offset points')
    axes[0,0].set_ylabel('VAL NDCG@10');axes[0,0].legend();axes[0,1].set_ylabel('Teacher-forced VAL CE')
    axes[1,0].set_ylabel('Train seconds / epoch (incl. cold compile)');axes[1,1].set_ylabel('VAL-selected TEST NDCG@10')
    axes[1,1].set_xlabel('B1 supported forward GFLOPs')
    axes[1,1].margins(x=.18,y=.2)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Epoch')
    axes[1,0].axvline(31,color='#777777',linestyle=':',alpha=.7)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(root/'mixed-curves.png',dpi=180);fig.savefig(root/'mixed-curves.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--partial',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--plot',action='store_true');args=p.parse_args()
    result,curves=analyze(args.results,args.parent,args.partial,args.data,args.archive)
    if args.plot:plot(args.results,result,curves)
    print(json.dumps({k:result[k]for k in ('records','comparisons','recovery','training_view_examples')},indent=2))
