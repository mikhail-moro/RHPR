"""Verify saved frozen-generator SID search and matched fresh fits; no inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import build_examples,file_sha256,load_beauty
from rhpr.gr.generator_suite import training_manifest
from rhpr.gr.generator_guided import incidence

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--completed',type=Path,default=Path('/tmp/rhpr-generator-guided-completed'))
p.add_argument('--output',type=Path,default=Path('/tmp/rhpr-generator-guided-analysis'))
p.add_argument('--data',type=Path,default=Path('/workspace/scratch/gr-beauty-input/data'))
p.add_argument('--plot',action='store_true')
args=p.parse_args();root,out=args.completed,args.output;out.mkdir(parents=True,exist_ok=True)
load=lambda path:json.loads(path.read_text())
def write(name,value):
    (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

repo=Path(__file__).resolve().parents[1]
manifest=load(root/'run-manifest.json');c=manifest['config']
assert manifest['git_sha']=='c3459221dfae7163b46de0d9ec28c4003a679735'
assert c==load(repo/'experiments/amazon_beauty_sid_generator_guided_2t4.json')
suite=load(root/'suite-results.json');assert not suite['failures']
metrics=load(root/'metrics.json');assert metrics['runs']==2
_,users,source_ids,counts=load_beauty(args.data,c['data'])
val=build_examples(users,20,'validation');train=build_examples(users,20,'train')
contexts,uids,targets=val
assert training_manifest(users,c)==load(root/'shared-training-stream.json')
assert len(train[2])==131413 and len(targets)==22363
models=['rqkmeans_original','generator_guided'];epochs=[5,10,15,20]
original=np.load(root/'indices/rqkmeans_original/semantic-ids.npy')
updated=np.load(root/'indices/generator_guided/semantic-ids.npy')
owner=np.load(root/'indices/generator_guided/owner.npy')
np.testing.assert_array_equal(np.sort(owner),np.arange(len(original)))
np.testing.assert_array_equal(updated,original[owner])
assert file_sha256(root/'indices/rqkmeans_original/semantic-ids.npy')==c['generator_guided']['sids']['sha256']
for depth in range(1,5):
    for a,b in zip(np.unique(original[:,:depth],axis=0,return_counts=True),np.unique(updated[:,:depth],axis=0,return_counts=True)):
        np.testing.assert_array_equal(a,b)
changed=np.flatnonzero(owner!=np.arange(len(owner)));changed_set=set(changed.tolist())
lcp=lambda a,b:int(np.cumprod(a==b).sum())
report=load(root/'generator-guided/optimization.json')
accepted=load(root/'generator-guided/accepted-swaps.json')
trace=[json.loads(s) for s in (root/'generator-guided/search-trace.jsonl').read_text().splitlines()]
pairs=load(root/'generator-guided/candidate-pairs.json')
assert report['frozen_weights_and_rng_verified'] and not report['validation_or_test_used_for_search']
assert report['cache_audit']['max_row_drift']==report['identity_drift']==0
assert len(trace)==report['proposals']==214 and len(accepted)==report['accepted']==2
assert len(changed)==report['changed_items']==4
assert report['scored_rows']==sum(r['scored_rows'] for r in trace)
np.testing.assert_allclose(sum(r['delta_log_likelihood'] for r in accepted),report['total_train_log_likelihood_gain'],rtol=0,atol=1e-8)
replay=np.arange(len(original));incs=incidence(train,len(original))
for step,row in enumerate(trace):
    candidate=pairs[step];assert (row['a'],row['b'])==(candidate['a'],candidate['b'])
    affected=np.union1d(incs[row['a']],incs[row['b']])
    assert len(affected)==row['affected_rows']
    assert row['accepted']==(row['delta_log_likelihood']>row['threshold'])
    if row['accepted']:replay[[row['a'],row['b']]]=replay[[row['b'],row['a']]]
np.testing.assert_array_equal(replay,owner)
with np.load(root/'generator-guided/train-likelihood-cache.npz') as f:
    np.testing.assert_array_equal(f['user_ids'],train[1]);np.testing.assert_array_equal(f['target_item_ids'],train[2])
    gain=f['final_log_probability']-f['initial_log_probability']
    assert abs(gain.sum()-report['total_train_log_likelihood_gain'])<1e-8
    for key,field in [('initial_log_probability','initial_mean_train_nll'),('final_log_probability','final_mean_train_nll')]:
        assert abs(-f[key].mean()-report[field])<1e-12
train_affected=np.isin(train[2],changed)|np.isin(train[0],changed).any(1)
assert np.max(np.abs(gain[~train_affected]))==0

frequency=np.bincount(train[2],minlength=len(original))
catalog=load(args.data/'catalog.json')
items=[]
for i in changed:
    items.append(dict(item_id=int(i),source_id=source_ids[i],old_sid=original[i].tolist(),new_sid=updated[i].tolist(),
        old_owner_of_new_sid=int(owner[i]),lcp=lcp(original[i],updated[i]),train_targets=int(frequency[i]),
        validation_targets=int((targets==i).sum()),visible_validation_histories=int((contexts==i).any(1).sum()),
        text=catalog[i]['text']))
val_hist=np.isin(contexts,changed).any(1);val_target=np.isin(targets,changed);val_affected=val_hist|val_target
rank={};teacher={};rows=[];training=[];verified=[]
def ndcg(r,k=10):return np.where(r<=k,1/np.log2(r.astype(float)+1),0)
powers=256**np.arange(3,-1,-1,dtype=np.int64);valid_paths=original@powers
for model,paths in zip(models,[original,updated]):
    trial=suite['results'][model];sel=trial['selection'];training.append(dict(model=model,**sel))
    assert trial['parameters']==4977920 and trial['test_evaluations']==0 and not trial['test'] and not trial['quantizer_retrained']
    assert trial['initial_state_sha256']==c['expected_initial_state_sha256']
    assert trial['seed']==trial['training_order_seed']==trial['generator_seed']==42
    assert sel['epochs']==20 and sel['model_examples_consumed']==2628260 and sel['optimizer_updates']==10280
    assert sel['amp_skipped_updates']==0 and sel['validation_checks']==4
    assert load(root/'trials'/model/'training-stream.json')==load(root/'trials'/models[0]/'training-stream.json')
    curves=load(root/'trials'/model/'training-curves.json')
    for epoch in epochs:
        beam=root/'trials'/model/f'validation-epoch-{epoch:03d}.npz'
        with np.load(beam) as f:
            np.testing.assert_array_equal(f['user_ids'],uids);np.testing.assert_array_equal(f['target_item_ids'],targets)
            pred=f['semantic_ids'];assert pred.shape==(len(uids),20,4) and np.isfinite(f['scores']).all()
            packed=pred@powers;hits=packed==(paths[targets]@powers)[:,None]
            r=np.where(hits.any(1),hits.argmax(1)+1,999).astype(np.int16)
            invalid=int((~np.isin(packed,valid_paths)).sum())
        logged=curves[epoch-1]['validation'];assert invalid==logged['invalid_predictions']
        for k in [5,10]:
            assert abs(ndcg(r,k).mean()-logged[f'ndcg@{k}'])<1e-12
            assert abs((r<=k).mean()-logged[f'recall@{k}'])<1e-12
        tf=root/'trials'/model/f'teacher-forced-epoch-{epoch:03d}.npz'
        stats=load(tf.with_suffix('.json'))
        with np.load(tf) as f:
            np.testing.assert_array_equal(f['user_ids'],uids);np.testing.assert_array_equal(f['target_item_ids'],targets)
            nll=f['nll_by_level'];token_rank=f['true_token_rank_by_level']
            assert nll.shape==token_rank.shape==(len(uids),4) and np.isfinite(nll).all()
            np.testing.assert_allclose(nll.mean(0,dtype=float),stats['nll_per_level'],atol=1e-12,rtol=0)
            np.testing.assert_allclose((token_rank==1).mean(0),stats['true_token_top1_per_level'],atol=1e-12,rtol=0)
            teacher[model,epoch]=nll.copy()
        rank[model,epoch]=r
        rows.append(dict(model=model,epoch=epoch,ndcg10=float(ndcg(r).mean()),recall10=float((r<=10).mean()),hits10=int((r<=10).sum()),
            invalid=invalid,train_loss=curves[epoch-1]['train_loss_sum_over_codes'],teacher_nll=stats['nll_sum_over_codes'],
            teacher_nll_per_level=stats['nll_per_level'],teacher_top1_per_level=stats['true_token_top1_per_level']))
        verified.append(dict(beam=beam.relative_to(root).as_posix(),beam_sha256=file_sha256(beam),teacher_sha256=file_sha256(tf)))

matrix=np.stack([ndcg(rank[models[1],e])-ndcg(rank[models[0],e]) for e in epochs],axis=1)
rng=np.random.default_rng(2760);draws=[];cluster_draws=[]
_,groups=np.unique(targets,return_inverse=True);sizes=np.bincount(groups)
sums=np.stack([np.bincount(groups,weights=matrix[:,j]) for j in range(4)],axis=1)
for _ in range(50):
    ix=rng.integers(len(uids),size=(40,len(uids)));draws.append(matrix[ix].mean(1))
    ix=rng.integers(len(sizes),size=(40,len(sizes)));cluster_draws.append(sums[ix].sum(1)/sizes[ix].sum(1)[:,None])
uci=np.quantile(np.concatenate(draws),[.025,.975],axis=0);tci=np.quantile(np.concatenate(cluster_draws),[.025,.975],axis=0)
comparison=[]
for j,e in enumerate(epochs):
    a=ndcg(rank[models[0],e]);b=ndcg(rank[models[1],e])
    comparison.append(dict(epoch=e,original=float(a.mean()),guided=float(b.mean()),delta=float((b-a).mean()),
        relative_percent=100*(b.mean()/a.mean()-1),user_ci95=uci[:,j].tolist(),target_cluster_ci95=tci[:,j].tolist(),
        users_better=int((b>a).sum()),users_worse=int((b<a).sum()),users_equal=int((b==a).sum())))

def group(name,mask):
    a=ndcg(rank[models[0],20])[mask];b=ndcg(rank[models[1],20])[mask]
    return dict(group=name,users=int(mask.sum()),original=float(a.mean()) if len(a) else None,guided=float(b.mean()) if len(b) else None,
        delta=float((b-a).mean()) if len(a) else None,delta_contribution_to_global=float((b-a).sum()/len(uids)),
        original_hits=int((rank[models[0],20][mask]<=10).sum()),guided_hits=int((rank[models[1],20][mask]<=10).sum()))
groups_report=[group('all',np.ones(len(uids),dtype=bool)),group('changed history or target',val_affected),
    group('unchanged visible history and target',~val_affected),group('changed target',val_target),group('changed history',val_hist)]
length=np.array([len(u.train) for u in users])
for lo,hi in [(1,4),(5,9),(10,19),(20,1000000)]:groups_report.append(group(f'TRAIN history {lo}..{hi}',(length>=lo)&(length<=hi)))

search=[]
for depth in range(4):
    subset=[r for r in trace if lcp(original[r['a']],original[r['b']])==depth]
    if subset:
        search.append(dict(pair_original_lcp=depth,proposed=len(subset),accepted=sum(r['accepted'] for r in subset),
            delta_quantiles=np.quantile([r['delta_log_likelihood'] for r in subset],[0,.25,.5,.75,1]).tolist()))
summary=dict(source_sha=manifest['git_sha'],elapsed_minutes=metrics['elapsed_seconds']/60,search=report,
    train_nll_relative_improvement_percent=100*(1-report['final_mean_train_nll']/report['initial_mean_train_nll']),
    moved_fraction_percent=100*len(changed)/len(original),train_affected_rows=int(train_affected.sum()),
    train_affected_users=int(len(np.unique(train[1][train_affected]))),val_affected_users=int(val_affected.sum()),
    val_history_affected_users=int(val_hist.sum()),val_target_affected_users=int(val_target.sum()),
    search_depths=search,comparisons=comparison,endpoint_groups=groups_report,changed_items=items)
write('summary',summary);write('epochs',rows);write('training',training);write('items',items)
write('verification',dict(exact_config=True,shared_cases_orders_initialization_and_training_budgets=True,
    accepted_move_replay_and_full_path_multiset=True,cache_gain_and_affected_rows=True,all_8_beams_rescored=True,
    all_8_teacher_arrays_checked=True,files=verified,bootstrap_repeats=2000,bootstrap_seed=2760,
    interval_scope='Paired users / target clusters conditional on these fitted weights; no seed variance, no multiplicity correction.',
    test_evaluations=0,new_training=False,new_inference=False))
np.savez_compressed(out/'user-metrics.npz',user_id=uids,target_item_id=targets,train_history_length=length,
    changed_history=val_hist,changed_target=val_target,**{f'{m}_rank_e{e}':r for (m,e),r in rank.items()})
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for model in models:
        a=[r for r in rows if r['model']==model]
        axes[0].plot(epochs,[r['ndcg10'] for r in a],'o-',label=model)
        axes[1].plot(epochs,[r['teacher_nll'] for r in a],'o-',label=model)
    axes[0].set_title('Fresh generator validation NDCG@10');axes[0].legend(fontsize=8)
    axes[1].set_title('Validation teacher-forced path NLL')
    axes[2].plot([0]+[r['proposal'] for r in accepted],
        [0]+np.cumsum([r['delta_log_likelihood'] for r in accepted]).tolist(),'o-')
    axes[2].axvline(len(trace),color='gray',linestyle=':');axes[2].set_title('Frozen TRAIN log-likelihood gain')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].set_xlabel('Epoch');axes[1].set_xlabel('Epoch');axes[2].set_xlabel('Evaluated proposal')
    fig.savefig(out/'overview.png',dpi=150);fig.savefig(out/'overview.svg');plt.close(fig)
print(json.dumps(summary,indent=2))
