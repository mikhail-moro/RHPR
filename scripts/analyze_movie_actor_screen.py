"""Replay fixed30 actor-graph screen; no inference/training/model reselection."""
import csv,gzip,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from rhpr.gr.sid import SidIndex
from rhpr.gr.movie_actor_screen import validate
from rhpr.gr.multimodal_rqvae import normalize_modalities
ROOT=Path('/tmp/movie-actor-completed');OUT=Path('/tmp/movie-actor-analysis');OUT.mkdir(exist_ok=True)
REPO=Path('/workspace/RHPR');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=load(ROOT/'experiment.json');validate(c);m=load(ROOT/'run-manifest.json');suite=load(ROOT/'suite-results.json');metrics=load(ROOT/'metrics.json');split=load(ROOT/'split-report.json')
assert load(ROOT/'get_notebook_session_status.json')['status']=='COMPLETE'
assert m['git_sha']=='5858df9f4682a15fdc740d3717469c8d4c6e22c5' and m['config']==c==load(REPO/'experiments/movielens100k_actor_screen_2t4.json')
assert len(suite['results'])==metrics['trials']==10 and not suite['failures'] and suite['parallel_workers']==2
assert all(type(v) in (str,int,float,bool) and (not isinstance(v,float) or math.isfinite(v)) for v in metrics.values())
for r in load(ROOT/'download-manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
for slot in [0,1]:
 w=load(ROOT/f'workers/worker-{slot}.json');assert w['status']=='complete' and len(w['results'])==5 and not w['failures'];assert {v['seed'] for v in w['results'].values()}==[{42},{101}][slot]
base=Path('/tmp/movie-graph-channel-audit');actor=Path('/tmp/movie-actor-graph-input');ids=load(base/'source-ids.json')
assert sha(base/'features.npz')==c['modality_input']['features_sha256']
for name,key in [('people.npy','people_sha256'),('source-ids.json','source_ids_sha256'),('people-provenance.json','provenance_sha256')]:assert sha(actor/name)==c['actor_graph_input'][key]
assert load(actor/'source-ids.json')==ids and load(ROOT/'input-provenance.json')['source_ids']==ids
with np.load(base/'features.npz') as z:arrays={k:z[k] for k in ['text','poster','plot','people']};masks={k:z[k+'_observed'] for k in arrays}
actorarray=np.load(actor/'people.npy');assert actorarray.shape==(1273,128) and not actorarray[:,64:].any()
cases={};vectors={};userrows=[];rows=[];curves=[];heads=[];qcurves=[];nfiles=0

def rescore(path,index,expected,ident,epoch,s):
 global nfiles
 with np.load(path) as z:
  pred=z['semantic_ids'];uu=z['user_ids'];tt=z['target_item_ids'];scores=z['scores']
 assert pred.shape==(943,20,4) and scores.shape==(943,20) and len(np.unique(uu))==943 and np.isfinite(scores).all()
 if s in cases:
  np.testing.assert_array_equal(cases[s][0],uu);np.testing.assert_array_equal(cases[s][1],tt)
 else:cases[s]=(uu.copy(),tt.copy())
 lookup={tuple(v):i for i,v in enumerate(index)};decoded=np.array([[lookup.get(tuple(v),-1) for v in beam] for beam in pred]);hits=decoded==tt[:,None];ranks=np.where(hits.any(1),hits.argmax(1)+1,0)
 for k in [5,10]:
  valid=(ranks>0)&(ranks<=k);v=np.where(valid,1/np.log2(np.maximum(ranks,1)+1),0)
  assert abs(v.mean()-expected[f'ndcg@{k}'])<1e-12 and abs(valid.mean()-expected[f'recall@{k}'])<1e-12
 dup=np.any((decoded[:,:,None]==decoded[:,None,:])&np.tril(np.ones((20,20),bool),-1)[None,:,:],axis=2)
 assert int((decoded<0).sum())==expected['invalid_predictions'] and int(dup[decoded>=0].sum())==expected['duplicate_predictions']
 for u,t,r in zip(uu,tt,ranks):userrows.append(dict(trial=ident,split=s,epoch=epoch,user_id=int(u),target=int(t),rank=int(r)))
 nfiles+=1
 return np.where((ranks>0)&(ranks<=10),1/np.log2(np.maximum(ranks,1)+1),0.)

for trial in c['trials']:
 ident=trial['id'];p=ROOT/'trials'/ident;r=load(p/'trial-result.json');s=load(p/'training-selection.json');cv=load(p/'training-curves.json');q=load(p/'quantizer.json');ae=load(p/'ae-warmup/autoencoder.json');norm=load(p/'modality-normalization.json')
 assert r==suite['results'][ident] and r['seed']==trial['seed'] and r['candidate']==trial['candidate'] and r['people_source']==trial['people_source'] and r['fixed_epoch_screen'] and not r['budget_censored']
 assert all(s[k]==v for k,v in r['selection'].items())
 assert s['training_epochs']==30 and s['validation_checks']==6 and s['stopping_reason']=='epoch_limit'
 assert [v['epoch'] for v in cv]==list(range(1,31)) and [v['epoch'] for v in cv if 'validation'in v]==[5,10,15,20,25,30]
 assert all(v['train_examples']==85470 and v['updates']+v['amp_skipped_updates']==334*v['epoch'] for v in cv)
 assert s['optimizer_updates']+s['amp_skipped_updates']==10020
 assert 'scheduler' not in c['training'] and all(v.get('learning_rate',.001)==.001 and v.get('next_learning_rate',.001)==.001 for v in cv)
 codes=np.load(p/'semantic-codes.npy');index=np.load(p/'semantic-ids.npy');np.testing.assert_array_equal(SidIndex(codes,width=256).sids,index)
 assert load(p/'sid-catalogue.json')['original_item_ids']==ids
 assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed'][str(trial['seed'])]
 assert load(p/'generator.json')['initial_state_sha256']==r['initial_state_sha256']==c['expected_initial_state_sha256_by_seed'][str(trial['seed'])]
 assert load(p/'representation.json')['source_features_sha256']==c['modality_input']['features_sha256']
 aa=dict(arrays)
 if trial['people_source']=='actor_graph':
  prov=load(p/'actor-input-provenance.json');assert prov['input']==c['actor_graph_input'] and prov['provenance']==load(actor/'people-provenance.json');aa['people']=actorarray
  assert prov['provenance']['graph_role']=='cast' and prov['provenance']['no_interactions_used']
 else:assert not (p/'actor-input-provenance.json').exists()
 columns=None if trial['people_role']=='none' else {'people':list(range(64))}
 _,expectednorm=normalize_modalities(aa,masks,columns);assert norm==expectednorm
 assert q['active_modalities']==trial['modalities']
 if columns:assert q['config']['reconstruction_columns']==columns
 else:assert 'reconstruction_columns'not in q['config']
 weights=trial['reconstruction_weights'];assert r['reconstruction_weights']==q['reconstruction_weights']==ae['reconstruction_weights']==weights
 assert q['steps']==q['reconstruction_batches']==3000 and q['batches_without_reconstruction']==1
 assert ae['steps']==r['ae_steps']==3500 and ae['stopping']['reason']=='fixed_steps'
 assert ae['codebooks_unchanged'] and q['warmup']['neural_transfer_verified'] and q['warmup']['fresh_rq_optimizer_verified'] and q['initialization_latent_audit']['max_absolute_difference']<1e-5
 for v in load(p/'ae-warmup/learning-curves.json'):assert abs(v['weighted_reconstruction_mse']-sum(weights[k]*x for k,x in v['head_mse'].items()))<1e-10
 for v in load(p/'codebook-learning-curves.json'):
  assert abs(v['weighted_reconstruction_mse']-sum(weights[k]*x for k,x in v['reconstruction_mse_by_head'].items()))<1e-10
  qcurves.append(dict(trial=ident,step=v['step'],weighted_mse=v['weighted_reconstruction_mse'],**{k+'_mse':x for k,x in v['reconstruction_mse_by_head'].items()}))
 for v in cv:
  if 'validation'in v:
   vec=rescore(p/f"validation-epoch-{v['epoch']:03d}.npz",index,v['validation'],ident,v['epoch'],'validation')
   if v['epoch']==s['best_epoch']:vectors[ident,'validation']=vec
   loss=v['validation_loss'];assert loss['examples']==943 and loss['precision']=='fp32' and abs(sum(loss['ce_by_level'])-loss['total_ce'])<1e-10
  curves.append(dict(trial=ident,epoch=v['epoch'],train_ce=v['train_loss_sum_over_codes'],val_ndcg10=v.get('validation',{}).get('ndcg@10',''),val_ce=v.get('validation_loss',{}).get('total_ce',''),train_seconds=v['train_seconds']))
 assert s['validation_ndcg10']==max(v['validation']['ndcg@10'] for v in cv if 'validation'in v)==cv[s['best_epoch']-1]['validation']['ndcg@10']
 vectors[ident,'test']=rescore(p/'test-predictions.npz',index,r['test'],ident,s['best_epoch'],'test')
 for k,v in q['head_reconstruction_mse'].items():heads.append(dict(trial=ident,head=k,rq_mse=v,ae_mse=ae['final_head_mse'][k],weight=weights[k]))
 rows.append(dict(trial=ident,candidate=trial['candidate'],seed=trial['seed'],val_ndcg10=s['validation_ndcg10'],test_ndcg10=r['test']['ndcg@10'],test_recall10=r['test']['recall@10'],best_epoch=s['best_epoch'],epochs=30,amp_skips=s['amp_skipped_updates'],unique_semantic_ids=q['semantic_code_statistics']['semantic_unique_ids'],invalid_test=r['test']['invalid_predictions'],duplicate_test=r['test']['duplicate_predictions']))

# Verify exact targets against the independently reconstructed historical split.
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
users=load_temporal_users(base/'sequences.json',ids)
for name in ['train','validation','test']:
 examples=temporal_examples(users,20,name);assert case_digest(examples)==split['case_sha256'][name]
 if name!='train':np.testing.assert_array_equal(examples[1],cases[name][0]);np.testing.assert_array_equal(examples[2],cases[name][1])

candidates=['none','shared_cast','actor_w1','actor_w03','actor_w01'];agg=[];comparisons=[];rng=np.random.default_rng(42)
for name in candidates:
 rr=[r for r in rows if r['candidate']==name];a=np.array([r['test_ndcg10'] for r in rr])
 agg.append(dict(candidate=name,mean_val_ndcg10=float(np.mean([r['val_ndcg10']for r in rr])),mean_test_ndcg10=float(a.mean()),sd_test_ndcg10=float(a.std(ddof=1)),mean_test_recall10=float(np.mean([r['test_recall10']for r in rr]))))
for splitname in ['validation','test']:
 for a,b in [('shared_cast','none'),('actor_w1','none'),('actor_w03','none'),('actor_w01','none'),('actor_w1','shared_cast'),('actor_w01','actor_w1'),('actor_w01','shared_cast')]:
  va=np.stack([vectors[f'{a}_seed{s}',splitname]for s in [42,101]]);vb=np.stack([vectors[f'{b}_seed{s}',splitname]for s in [42,101]]);d=va-vb;dm=d.mean(0)
  boot=np.concatenate([dm[rng.integers(0,943,(500,943))].mean(1)for _ in range(10)])
  comparisons.append(dict(split=splitname,a=a,b=b,delta=float(d.mean()),relative_percent=float(100*d.mean()/vb.mean()),delta_by_seed=d.mean(1).tolist(),positive_seeds=int((d.mean(1)>0).sum()),conditional_user_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),better_user_seed_pairs=int((d>0).sum()),worse_user_seed_pairs=int((d<0).sum())))
winner=max(agg,key=lambda r:r['mean_val_ndcg10'])['candidate']
verification=dict(executable_sha=m['git_sha'],fits=10,seeds=[42,101],body_minutes=m['duration_seconds']/60,prediction_files_rescored=nfiles,all30epochs_six_checks_constant_lr=True,all943cases_hashes_verified=True,initializations_normalization_and_actor_source_verified=True,ae3500_rq3000_transfer_verified=True,weighted_and_raw_losses_verified=True,validation_selected_candidate=winner,amp_skips={r['trial']:r['amp_skips']for r in rows},invalid_test={r['trial']:r['invalid_test']for r in rows},scope='Exploratory selection on previously examined validation, two fitted pipeline seeds, fixed graph seed. Conditional shared-user intervals exclude seed/runtime variation and multiple testing; test is descriptive.')
for name,obj in [('metrics',rows),('aggregates',agg),('comparisons',comparisons),('verification',verification)]:
 (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
for name,rs in [('metrics',rows),('aggregates',agg),('comparisons',comparisons),('head-mse',heads),('learning-curves',curves),('quantizer-curves',qcurves)]:
 with (OUT/(name+'.csv')).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
with gzip.open(OUT/'per-user-ranks.csv.gz','wt',newline='')as f:w=csv.DictWriter(f,fieldnames=list(userrows[0]));w.writeheader();w.writerows(userrows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels=['No people','Old graph, cast','Actor graph: 1','Actor graph: 0.3','Actor graph: 0.1'];colors=['#32688e','#7c8a9c','#e76f51','#e9c46a','#2a9d8f']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for ax,field,title in zip(axes,['val_ndcg10','test_ndcg10'],['Validation (selection)','Test (descriptive)']):
 for j,(name,label,color)in enumerate(zip(candidates,labels,colors)):
  vals=[next(r[field]for r in rows if r['candidate']==name and r['seed']==s)for s in [42,101]]
  ax.scatter([j-.08,j+.08],vals,color=color,s=40);ax.plot([j-.08,j+.08],vals,color=color,alpha=.4);ax.scatter(j,np.mean(vals),color=color,marker='_',s=180)
 ax.set(xticks=range(5),xticklabels=labels,title=title,ylabel='Selected NDCG@10');ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.2)
fig.suptitle('Two pipeline seeds, fixed30epochs; actor weight0.1 selected by mean validation',fontsize=11)
fig.savefig(OUT/'quality.png',dpi=150);fig.savefig(OUT/'quality.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for ax,seed in zip(axes,[42,101]):
 for name,label,color in zip(candidates,labels,colors):
  rr=[r for r in curves if r['trial']==f'{name}_seed{seed}'and r['val_ndcg10']!=''];ax.plot([r['epoch']for r in rr],[r['val_ndcg10']for r in rr],marker='.',label=label,color=color)
 ax.set(title=f'Seed{seed}',xlabel='Epoch',ylabel='Validation NDCG@10');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(OUT/'curves.png',dpi=150);fig.savefig(OUT/'curves.pdf');plt.close(fig)
shutil.copyfile(__file__,OUT/'analyze.py');print(json.dumps(dict(aggregates=agg,comparisons=comparisons,metrics=rows,verification=verification),indent=2))
