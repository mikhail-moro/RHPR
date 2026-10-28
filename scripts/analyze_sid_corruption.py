"""Independently verify completed corruption fits and report the fixed-budget curve."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from rhpr.gr.data import build_examples,file_sha256,load_beauty
from rhpr.gr.generator_suite import training_manifest
from rhpr.gr.sid_corruption import corruption_owner,corruption_diagnostics
from rhpr.gr.sid_graph_swap import NeighborGraph
from rhpr.metrics import validate_metrics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--completed',type=Path,default=Path('/tmp/rhpr-sid-corruption-completed'))
p.add_argument('--output',type=Path,default=Path('/workspace/artifacts/sid-corruption/analysis'))
p.add_argument('--data',type=Path,default=Path('/workspace/scratch/gr-beauty-input/data'))
p.add_argument('--assets',type=Path,default=Path('/workspace/artifacts/sid-context-screen-input'))
p.add_argument('--graph-assets',type=Path,default=Path('/workspace/artifacts/sid-als-screen/input'))
p.add_argument('--plot',action='store_true')
args=p.parse_args();root,out=args.completed,args.output;out.mkdir(parents=True,exist_ok=True)
load=lambda path:json.loads(path.read_text())
write=lambda name,value:(out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
repo=Path(__file__).resolve().parents[1]
manifest=load(root/'run-manifest.json');c=manifest['config']
assert manifest['git_sha']=='1e87396aa65fd8b2dc9259a650e501cec25eec35'
assert c==load(repo/'experiments/amazon_beauty_sid_corruption_2t4.json')
metrics=validate_metrics(load(root/'metrics.json'));assert metrics['runs']==5
suite=load(root/'suite-results.json');assert not suite['failures']
_,users,source_ids,counts=load_beauty(args.data,c['data'])
contexts,uids,targets=build_examples(users,20,'validation')
stream=training_manifest(users,c);assert stream==load(root/'shared-training-stream.json')
assert stream['train_cases_sha256']==c['expected_train_cases_sha256']
assert stream['examples_per_epoch']==131413 and len(uids)==22363
original=np.load(args.assets/'original.npy')
assert file_sha256(args.assets/'original.npy')==c['sid_corruption']['original_sids_sha256']
assert file_sha256(args.graph_assets/'semantic-graph.npz')==c['sid_corruption']['semantic_graph']['file_sha256']
with np.load(args.graph_assets/'semantic-graph.npz') as f:
    graph=NeighborGraph(f['source'],f['target'],f['weight'],len(original))
frequency=np.bincount(np.concatenate([u.train[1:] for u in users]),minlength=len(original))
np.testing.assert_array_equal(frequency,np.load(root/'train-target-item-frequency.npy'))
diagnostics=load(root/'corruption-diagnostics.json');assert not diagnostics['validation_and_test_used']
weights=256**np.arange(3,-1,-1,dtype=np.int64);valid_paths=original@weights
epochs=[5,10,15,20];rows=[];ranks={};values={};verified=[];training=[];previous=set()
def ndcg(r,k=10):return np.where(r<=k,1/np.log2(r.astype(float)+1),0)
for trial in c['trials']:
    name=trial['id'];d=root/'trials'/name;ix=root/'indices'/name
    assert file_sha256(ix/'semantic-ids.npy')==trial['expected_sid_sha256']
    assert file_sha256(ix/'owner.npy')==trial['expected_owner_sha256']
    paths=np.load(ix/'semantic-ids.npy');owner=np.load(ix/'owner.npy');chosen=np.load(ix/'selected-items.npy')
    expected_owner,expected_chosen=corruption_owner(len(original),trial['corruption_fraction'],2753)
    np.testing.assert_array_equal(owner,expected_owner);np.testing.assert_array_equal(chosen,expected_chosen)
    np.testing.assert_array_equal(paths,original[owner])
    assert previous.issubset(set(chosen));previous=set(chosen)
    diag=corruption_diagnostics(original,owner,frequency,graph)
    saved=next(r for r in diagnostics['trials'] if r['trial']==name)
    for key in ['changed_items','semantic_neighbor_normalized_lcp','changed_item_train_target_mass']:
        assert abs(saved[key]-diag[key])<1e-12
    np.testing.assert_allclose(saved['semantic_neighbor_shared_prefix_fractions'],diag['semantic_neighbor_shared_prefix_fractions'],atol=1e-12,rtol=0)
    for a,b in zip(saved['levels'],diag['levels']):
        assert a.keys()==b.keys()
        for key in a:assert abs(a[key]-b[key])<1e-12
    result=suite['results'][name];selection=result['selection']
    assert result['seed']==42 and result['parameters']==4977920 and result['test_evaluations']==0 and not result['test']
    assert not result['quantizer_retrained'] and result['initial_state_sha256']==c['expected_initial_state_sha256']
    assert selection['epochs']==20 and selection['model_examples_consumed']==2628260
    assert selection['optimizer_updates']==10280 and selection['amp_skipped_updates']==0 and selection['validation_checks']==4
    training.append(dict(model=name,**selection))
    curve=load(d/'training-curves.json');assert [r['epoch'] for r in curve if 'validation'in r]==epochs
    for epoch in epochs:
        file=d/f'validation-epoch-{epoch:03d}.npz'
        with np.load(file) as f:
            np.testing.assert_array_equal(f['user_ids'],uids);np.testing.assert_array_equal(f['target_item_ids'],targets)
            pred=f['semantic_ids'];assert pred.shape==(len(uids),20,4) and np.isfinite(f['scores']).all()
            assert np.all((pred>=0)&(pred<256))
            packed=pred@weights;hit=packed==(paths[targets]@weights)[:,None]
            rank=np.where(hit.any(1),hit.argmax(1)+1,999).astype(np.int16)
            valid=np.isin(packed,valid_paths);ordered=np.sort(np.where(valid,packed,-1),axis=1)
            duplicate=(ordered[:,1:]==ordered[:,:-1])&(ordered[:,1:]>=0)
        logged=curve[epoch-1]['validation']
        assert int((~valid).sum())==logged['invalid_predictions'] and int(duplicate.sum())==logged['duplicate_predictions']
        for k in [5,10]:
            assert abs(float(ndcg(rank,k).mean())-logged[f'ndcg@{k}'])<1e-12
            assert abs(float((rank<=k).mean())-logged[f'recall@{k}'])<1e-12
        teacher_file=d/f'teacher-forced-epoch-{epoch:03d}.npz'
        teacher=load(d/f'teacher-forced-epoch-{epoch:03d}.json')
        with np.load(teacher_file) as f:
            np.testing.assert_array_equal(f['user_ids'],uids);np.testing.assert_array_equal(f['target_item_ids'],targets)
            nll=f['nll_by_level'];token_rank=f['true_token_rank_by_level']
            assert nll.shape==token_rank.shape==(len(uids),4) and np.isfinite(nll).all() and (nll>=0).all()
            assert np.all((token_rank>=1)&(token_rank<=256))
            np.testing.assert_allclose(nll.mean(0,dtype=float),teacher['nll_per_level'],atol=1e-12,rtol=0)
            np.testing.assert_allclose((token_rank==1).mean(0),teacher['true_token_top1_per_level'],atol=1e-12,rtol=0)
            assert abs(float(nll.sum(1,dtype=float).mean())-teacher['nll_sum_over_codes'])<1e-12
        assert curve[epoch-1]['teacher_forced']==teacher
        ranks[name,epoch]=rank;values[name,epoch]=ndcg(rank)
        rows.append(dict(model=name,fraction=trial['corruption_fraction'],epoch=epoch,ndcg10=float(ndcg(rank).mean()),
            recall10=float((rank<=10).mean()),hits10=int((rank<=10).sum()),invalid=int((~valid).sum()),
            train_nll=curve[epoch-1]['train_loss_sum_over_codes'],teacher_nll=teacher['nll_sum_over_codes'],
            teacher_nll_per_level=teacher['nll_per_level'],teacher_top1_per_level=teacher['true_token_top1_per_level']))
        verified.append(dict(beam=file.relative_to(root).as_posix(),beam_sha256=file_sha256(file),teacher_sha256=file_sha256(teacher_file)))

base=values[c['trials'][0]['id'],20]
matrix=np.stack([values[t['id'],20]-base for t in c['trials'][1:]],axis=1)
rng=np.random.default_rng(2754);draws=[];cluster_draws=[]
_,groups=np.unique(targets,return_inverse=True);sizes=np.bincount(groups)
sums=np.stack([np.bincount(groups,weights=matrix[:,j]) for j in range(4)],axis=1)
for _ in range(50):
    ix=rng.integers(len(uids),size=(40,len(uids)));draws.append(matrix[ix].mean(1))
    ix=rng.integers(len(sizes),size=(40,len(sizes)));cluster_draws.append(sums[ix].sum(1)/sizes[ix].sum(1)[:,None])
uc=np.quantile(np.concatenate(draws),[.025,.975],axis=0);tc=np.quantile(np.concatenate(cluster_draws),[.025,.975],axis=0)
endpoint=[]
for j,t in enumerate(c['trials']):
    row=dict(next(r for r in rows if r['model']==t['id'] and r['epoch']==20))
    row.update(delta_ndcg10=row['ndcg10']-float(base.mean()),relative_percent=100*(row['ndcg10']/base.mean()-1))
    if j:row.update(user_ci95=uc[:,j-1].tolist(),target_cluster_ci95=tc[:,j-1].tolist())
    endpoint.append(row)
write('epochs',rows);write('endpoint',endpoint);write('training',training);write('diagnostics',diagnostics)
write('verification',dict(executable_commit=manifest['git_sha'],exact_config=True,all5_initial_hashes_and_budgets_match=True,
    shared_case_order_hashes=True,all_owners_and_paths_match_pinned_derangements=True,all_prefix_occupancies_verified=True,
    all20_raw_beams_metrics_verified=True,all20_teacher_sidecar_aggregates_verified=True,predictions=verified,
    bootstrap_repeats=2000,bootstrap_seed=2754,test_evaluations=0,new_training=False,new_inference=False))
np.savez_compressed(out/'user-metrics.npz',user_id=uids,target_item_id=targets,
    train_history_length=np.array([len(u.train) for u in users]),target_train_positions=frequency[targets],
    **{f'{m}_rank_e{e}':v for (m,e),v in ranks.items()})
for name in ['run-manifest.json','suite-results.json','metrics.json','corruption-diagnostics.json','download-manifest.json']:
    shutil.copyfile(root/name,out.parent/name)
lines=['# Completed random whole-SID corruption screen','',
    'All five fits:20epochs,2628260examples,10280updates,zeroAMPskips,4977920params,same neural/data/order seed42. '
    'Catalogue prefix occupancies preserved exactly. One permutation seed2753,nested moved sets;TRAINfrequency distributions vary. '
    f"Execution body{metrics['elapsed_seconds']/60:.1f}minutes. No test or new inference.",'',
    '|Permuted|NDCG10|Relative|Recall10|Teacher NLL|First-token top1|','|---|---:|---:|---:|---:|---:|']
for r in endpoint:
    lines.append(f"|{100*r['fraction']:g}%|{r['ndcg10']:.6f}|{r['relative_percent']:+.2f}%|{r['recall10']:.6f}|{r['teacher_nll']:.4f}|{100*r['teacher_top1_per_level'][0]:.2f}%|")
lines+=['','The original organization improves quality at this fixed training budget;randomizing item grouping causes a monotone endpoint decline. '
    'Teacher-forced likelihood also worsens,especially the first token,so the effect is not explained solely by beam-search pruning. '
    'A larger beam may still change retrieval gaps. TRAINweighted prefix frequencies and associations also change,so this does not isolate '
    'pure textual semantics from all statistical consequences of assignment. All fits continue improving;random indices might recover with more training. '
    'One fitted/permutation seed and shared users prevent universal or independent-replication claims.','',
    'User/target-cluster intervals are2000paired resamples,seed2754,conditional on fitted weights and unadjusted across four comparisons. '
    'They exclude training/permutation-seed variance. Rank999means absent from saved20beams,not exact full-catalogue rank.','',
    'All20beam files rescored independently;all20teacher sidecar arrays/aggregates checked. '
    'Raw135numerical/report files70.17MB:/tmp/rhpr-sid-corruption-completed. Ten checkpoint filenames retained on private Kaggle;'
    'checkpoint contents not downloaded/verified. epochs.json,endpoint.json,training.json,diagnostics.json,verification.json,user-metrics.npz retained here.','',
    'Reproduce:PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_sid_corruption.py --plot']
(out/'README.md').write_text('\n'.join(lines)+'\n')
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors=plt.cm.viridis(np.linspace(.1,.9,5))
    x=np.array([100*r['fraction'] for r in endpoint]);y=np.array([r['relative_percent'] for r in endpoint])
    ax[0,0].plot(x,y,'o-',color='tab:blue')
    for r in endpoint[1:]:
        lo,hi=100*np.array(r['user_ci95'])/base.mean();yy=r['relative_percent']
        ax[0,0].errorbar(100*r['fraction'],yy,yerr=[[yy-lo],[hi-yy]],capsize=4,color='tab:blue')
    ax[0,0].axhline(0,color='gray',lw=.8);ax[0,0].set_title('Epoch20 NDCG@10 change; paired-user 95% CI');ax[0,0].set_ylabel('Change vs original (%)')
    for t,color in zip(c['trials'],colors):
        a=[r for r in rows if r['model']==t['id']]
        ax[0,1].plot(epochs,[r['ndcg10'] for r in a],'o-',color=color,label=f"{100*t['corruption_fraction']:g}%")
    ax[0,1].set_title('Validation through training');ax[0,1].set_xlabel('Epoch');ax[0,1].set_ylabel('NDCG@10');ax[0,1].legend(title='Permuted items')
    for level in range(4):
        ax[1,0].plot(x,[r['teacher_nll_per_level'][level] for r in endpoint],'o-',label=f'Level{level+1}')
    ax[1,0].set_title('Teacher-forced NLL with true preceding tokens');ax[1,0].set_ylabel('Mean NLL (nats)');ax[1,0].legend()
    ax[1,1].plot(x,[100*r['semantic_neighbor_shared_prefix_fractions'][0] for r in diagnostics['trials']],'o-',label='T5 neighbors sharing first prefix')
    ax[1,1].plot(x,[100*r['teacher_top1_per_level'][0] for r in endpoint],'o-',label='Generator first-token top1')
    ax[1,1].set_title('Measured structure and first-token prediction');ax[1,1].set_ylabel('Percent');ax[1,1].legend(fontsize=8)
    for a in [ax[0,0],ax[1,0],ax[1,1]]:a.set_xlabel('Permuted catalogue items (%)')
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Beauty: fixed SID trie, increasingly random item assignments')
    fig.savefig(out/'corruption-results.png',dpi=170);fig.savefig(out/'corruption-results.svg');plt.close(fig)
print(json.dumps(endpoint,indent=2))
