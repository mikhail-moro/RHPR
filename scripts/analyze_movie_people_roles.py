"""Audit the graph-weight paired pilot without new neural training."""
import csv,gzip,hashlib,json,math,shutil,zipfile
from pathlib import Path
import numpy as np
from rhpr.gr.sid import SidIndex
ROOT=Path('/tmp/movie-roles-completed');OUT=Path('/tmp/movie-roles-analysis');OUT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
config=load(ROOT/'experiment.json');manifest=load(ROOT/'run-manifest.json');suite=load(ROOT/'suite-results.json');metrics=load(ROOT/'metrics.json');split=load(ROOT/'split-report.json')
assert load(ROOT/'get_notebook_session_status.json')['status']=='COMPLETE'
assert manifest['git_sha']=='682a23dabed6803d3340607f6ecd7924483ce3ba' and manifest['config']==config
assert config==load(Path('/workspace/RHPR/experiments/movielens100k_people_roles_2t4.json'))
assert len(suite['results'])==metrics['trials']==9 and not suite['failures'] and suite['parallel_workers']==2
assert all(type(v) in (str,int,float,bool) and (not isinstance(v,float) or math.isfinite(v)) for v in metrics.values())
for r in load(ROOT/'download-manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
for slot in [0,1]:
 w=load(ROOT/f'workers/worker-{slot}.json');assert w['status']=='complete' and len(w['results'])==([5,4][slot]) and not w['failures']
with np.load('/tmp/movie-graph-channel-audit/features.npz') as z:features={k:z[k].astype(float) for k in ['text','people','plot']}
assert sha(Path('/tmp/movie-graph-channel-audit/features.npz'))==config['modality_input']['features_sha256']
neighbors={}
for key,x in features.items():
 x-=x.mean(0);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);sim=x@x.T;np.fill_diagonal(sim,-np.inf);neighbors[key]=np.argsort(-sim,axis=1,kind='stable')[:,:10]
cases={};vectors={};userrows=[];rows=[];curverows=[];qrows=[];headrows=[];neighborrows=[];nfiles=0

def rescore(path,index,expected,trial,epoch,which):
 global nfiles
 with np.load(path,allow_pickle=False) as f:
  pred=f['semantic_ids'];users=f['user_ids'];targets=f['target_item_ids'];scores=f['scores']
  assert pred.shape==(943,20,4) and scores.shape==(943,20) and np.isfinite(scores).all() and len(np.unique(users))==943
  if which not in cases:cases[which]=(users.copy(),targets.copy())
  else:np.testing.assert_array_equal(users,cases[which][0]);np.testing.assert_array_equal(targets,cases[which][1])
  lookup={tuple(v):i for i,v in enumerate(index)};decoded=np.array([[lookup.get(tuple(v),-1) for v in beam] for beam in pred]);hits=decoded==targets[:,None];ranks=np.where(hits.any(1),hits.argmax(1)+1,0)
  for k in [5,10]:
   valid=(ranks>0)&(ranks<=k);v=np.where(valid,1/np.log2(np.maximum(ranks,1)+1),0.)
   assert abs(v.mean()-expected[f'ndcg@{k}'])<1e-12 and abs(valid.mean()-expected[f'recall@{k}'])<1e-12
  assert int((decoded<0).sum())==expected['invalid_predictions']
  dups=int(np.any((decoded[:,:,None]==decoded[:,None,:]) & np.tril(np.ones((20,20),dtype=bool),k=-1)[None,:,:],axis=2)[decoded>=0].sum());assert dups==expected['duplicate_predictions']
  for uid,target,rank in zip(users,targets,ranks):userrows.append(dict(trial=trial,split=which,epoch=epoch,user_id=int(uid),target=int(target),rank=int(rank)))
  nfiles+=1
 return np.where((ranks>0)&(ranks<=10),1/np.log2(np.maximum(ranks,1)+1),0.)

for trial in config['trials']:
 ident=trial['id'];p=ROOT/'trials'/ident;t=load(p/'trial-result.json');sel=load(p/'training-selection.json');cv=load(p/'training-curves.json');prefix=load(p/'prefix-training-selection.json');pc=load(p/'prefix-training-curves.json');resume=load(p/'resume-verification.json');q=load(p/'quantizer.json');ae=load(p/'ae-warmup/autoencoder.json')
 assert t==suite['results'][ident] and t['seed']==trial['seed'] and all(sel[k]==v for k,v in t['selection'].items())
 assert sel['stopping_reason']=='early_stopping' and not t['budget_censored'] and sel['training_epochs']==max(30,sel['best_epoch']+20)
 assert sel['resumed_from_epoch']==20 and sel['continuation_epochs']==sel['training_epochs']-20
 assert cv[:20]==pc and prefix['training_epochs']==20 and prefix['validation_checks']==4
 assert sha(p/'prefix-training-curves.json')==hashlib.sha256((json.dumps(cv[:20],indent=2,sort_keys=True,allow_nan=False)+'\n').encode()).hexdigest()
 assert [r['epoch'] for r in cv]==list(range(1,sel['training_epochs']+1))
 assert [r['epoch'] for r in cv if 'validation' in r]==[5,10,15,20]+list(range(21,sel['training_epochs']+1))
 assert all(r['train_examples']==85470 and r['updates']+r['amp_skipped_updates']==334*r['epoch'] for r in cv)
 assert sel['optimizer_updates']+sel['amp_skipped_updates']==334*sel['training_epochs'] and sel['continuation_optimizer_updates']==sel['optimizer_updates']-prefix['optimizer_updates']
 assert resume['source_epoch']==20 and resume['source_updates']==6680 and all(resume[k] for k in ['optimizer_restored','scaler_restored','rng_restored_and_verified'])
 assert resume['train_case_sha256']==split['case_sha256']['train'] and resume['validation_case_sha256']==split['case_sha256']['validation']
 codes=np.load(p/'semantic-codes.npy');index=np.load(p/'semantic-ids.npy');np.testing.assert_array_equal(SidIndex(codes,width=256).sids,index)
 assert hashlib.sha256(index.tobytes()).hexdigest()==resume['sid_matrix_sha256']
 assert load(p/'sid-catalogue.json')['original_item_ids']==load(Path('/tmp/movie-graph-channel-audit/source-ids.json'))==load(ROOT/'input-provenance.json')['source_ids']
 role=trial['people_role'];norm=load(p/'modality-normalization.json')
 if role!='none':
  cols=list(range(64)) if role=='cast' else list(range(64,128))
  assert q['config']['reconstruction_columns']=={'people':cols}
  assert norm['people']['selected_columns']==cols and norm['people']['inactive_input_zero']
  assert abs(norm['people']['zero_predictor_mse']-1.)<1e-6
 else:assert 'reconstruction_columns' not in q['config'] and q['active_modalities']==['text','poster','plot']
 assert t['people_role']==role
 weights=trial['reconstruction_weights'];assert t['reconstruction_weights']==q['reconstruction_weights']==ae['reconstruction_weights']==weights
 assert q['steps']==q['reconstruction_batches']==3000 and q['batches_without_reconstruction']==1
 assert ae['steps']==t['ae_steps']==3500 and ae['stopping']['reason']=='fixed_steps'
 assert ae['codebooks_unchanged'] and q['warmup']['neural_transfer_verified'] and q['warmup']['fresh_rq_optimizer_verified'] and q['initialization_latent_audit']['max_absolute_difference']<1e-5
 assert q['quantizer_initial_state_sha256']==config['quantizer']['expected_initial_state_sha256_by_seed'][str(trial['seed'])]
 assert load(p/'generator.json')['initial_state_sha256']==t['initial_state_sha256']==config['expected_initial_state_sha256_by_seed'][str(trial['seed'])]
 assert load(p/'representation.json')['source_features_sha256']==config['modality_input']['features_sha256']
 ac=load(p/'ae-warmup/learning-curves.json');assert [r['step'] for r in ac]==list(range(0,3501,500))
 for r in ac:assert not r['spectral_plateau'] and abs(r['weighted_reconstruction_mse']-sum(weights[k]*v for k,v in r['head_mse'].items()))<1e-10
 qc=load(p/'codebook-learning-curves.json');assert [r['step'] for r in qc]==list(range(500,3001,500))
 for r in qc:
  assert abs(r['weighted_reconstruction_mse']-sum(weights[k]*v for k,v in r['reconstruction_mse_by_head'].items()))<1e-10
  qrows.append(dict(trial=ident,step=r['step'],weighted_reconstruction_mse=r['weighted_reconstruction_mse'],semantic_unique_ids=r['semantic_unique_ids'],**{k+'_mse':v for k,v in r['reconstruction_mse_by_head'].items()}))
 # Replay phase-two ReduceLR and raw-metric epoch-based stopping.
 lr=.001;sbest=prefix['validation_ndcg10'];rawbest=sbest;raw_epoch=prefix['best_epoch'];bad=0;reductions=[]
 for r in cv[20:]:
  assert r['learning_rate']==lr
  score=r['validation']['ndcg@10']
  if score>sbest+1e-4:sbest=score;bad=0
  else:bad+=1
  if bad>=5:
   nlr=max(lr*.5,1e-5)
   if lr-nlr>1e-8:reductions.append(r['epoch']);lr=nlr
   bad=0
  assert r['next_learning_rate']==lr
  if score>rawbest:rawbest=score;raw_epoch=r['epoch']
  assert (r['epoch']>=30 and r['epoch']-raw_epoch>=20)==(r['epoch']==sel['training_epochs'])
 for r in cv:
  if 'validation' in r:
   valvec=rescore(p/f"validation-epoch-{r['epoch']:03d}.npz",index,r['validation'],ident,r['epoch'],'validation')
   if r['epoch']==sel['best_epoch']:vectors[ident,'validation']=valvec
  if 'validation_loss' in r:
   loss=r['validation_loss'];assert loss['examples']==943 and loss['precision']=='fp32' and len(loss['ce_by_level'])==4 and abs(sum(loss['ce_by_level'])-loss['total_ce'])<1e-10
  curverows.append(dict(trial=ident,epoch=r['epoch'],train_ce=r['train_loss_sum_over_codes'],val_ndcg10=r.get('validation',{}).get('ndcg@10',''),val_ce=r.get('validation_loss',{}).get('total_ce',''),lr=r.get('learning_rate',.001),next_lr=r.get('next_learning_rate',.001)))
 assert sel['validation_ndcg10']==max(r['validation']['ndcg@10'] for r in cv if 'validation' in r)==cv[sel['best_epoch']-1]['validation']['ndcg@10']
 vectors[ident]=rescore(p/'test-predictions.npz',index,t['test'],ident,sel['best_epoch'],'test')
 counts=np.bincount(codes[:,0],minlength=256);base=float(np.sum(counts*(counts-1))/(1273*1272))
 neighborrow=dict(trial=ident,random_pair_same_first_token=base)
 for key,nn in neighbors.items():
  lcp=np.logical_and.accumulate(codes[:,None,:]==codes[nn],axis=-1).sum(-1);neighborrow[key+'_same_first_token']=float((lcp>0).mean());neighborrow[key+'_mean_lcp']=float(lcp.mean());neighborrow[key+'_same_first_token_lift']=float((lcp>0).mean()/base)
 neighborrows.append(neighborrow)
 for head in trial['modalities']:headrows.append(dict(trial=ident,head=head,weight=weights[head],ae_mse=ae['final_head_mse'][head],rq_mse=q['head_reconstruction_mse'][head]))
 rows.append(dict(trial=ident,role=trial['people_role'],seed=trial['seed'],graph_weight=weights['people'],val_ndcg10=sel['validation_ndcg10'],test_ndcg10=t['test']['ndcg@10'],test_recall10=t['test']['recall@10'],best_epoch=sel['best_epoch'],stop_epoch=sel['training_epochs'],ae_steps=ae['steps'],rq_steps=q['steps'],semantic_unique_ids=q['semantic_code_statistics']['semantic_unique_ids'],used_codes_per_level=[int(len(np.unique(codes[:,level]))) for level in range(3)],lr_reductions=reductions,selected_lr=cv[sel['best_epoch']-1].get('learning_rate',.001),amp_skips=sel['amp_skipped_updates']))

# Paired comparisons and the pre-existing director/cast audit groups.
import collections
meta={}
with gzip.open('/tmp/graph-helped-audit/cases.csv.gz','rt') as f:
 for row in csv.DictReader(f):
  if row['phase']=='full' and row['seed']=='42' and row['variant']=='tiger_no_people':meta[row['split'],int(row['user_id'])]=row
rng=np.random.default_rng(42);comparisons=[];subgroups=[];aggregates=[]
def ci(d):
 n=len(d);boot=np.concatenate([d[rng.integers(0,n,(500,n))].mean(1) for _ in range(10)])
 return np.quantile(boot,[.025,.975]).tolist()
for role in ['none','cast','directors']:
 rr=[r for r in rows if r['role']==role];v=np.array([r['test_ndcg10'] for r in rr])
 aggregates.append(dict(role=role,mean_test_ndcg10=float(v.mean()),sd_test_ndcg10=float(v.std(ddof=1)),mean_test_recall10=float(np.mean([r['test_recall10'] for r in rr])),mean_val_ndcg10=float(np.mean([r['val_ndcg10'] for r in rr]))))
for s in ['validation','test']:
 uu,tt=cases[s]
 masks={'all':np.ones(943,dtype=bool)}
 for field in ['directors','cast']:
  m=np.array([meta[s,int(u)]['shared_'+field]=='True' for u in uu]);masks['shared_'+field]=m;masks['no_shared_'+field]=~m
 for a,b in [('cast','none'),('directors','none'),('directors','cast')]:
  va=np.stack([vectors[f'{a}_seed{seed}',s] if s=='validation' else vectors[f'{a}_seed{seed}'] for seed in [42,101,202]])
  vb=np.stack([vectors[f'{b}_seed{seed}',s] if s=='validation' else vectors[f'{b}_seed{seed}'] for seed in [42,101,202]])
  d=va-vb
  comparisons.append(dict(split=s,a=a,b=b,delta=float(d.mean()),relative_percent=float(100*d.mean()/vb.mean()),delta_by_seed=d.mean(1).tolist(),conditional_shared_user_ci=ci(d.mean(0)),positive_seeds=int((d.mean(1)>0).sum())))
  for group,m in masks.items():
   assert len(va[:,m].shape)==2
   subgroups.append(dict(split=s,a=a,b=b,group=group,n=int(m.sum()),baseline_ndcg10=float(vb[:,m].mean()),treatment_ndcg10=float(va[:,m].mean()),delta=float(d[:,m].mean()),relative_percent=float(100*d[:,m].mean()/vb[:,m].mean()),delta_by_seed=d[:,m].mean(1).tolist(),conditional_shared_user_ci=ci(d[:,m].mean(0)),baseline_hits=int((vb[:,m]>0).sum()),treatment_hits=int((va[:,m]>0).sum())))
# Relate input metadata to exact saved cases; the prior audit verified hashes.
for s in ['validation','test']:
 assert all(int(meta[s,int(u)]['target'])==int(t) for u,t in zip(*cases[s]))
verify=dict(executable_sha=manifest['git_sha'],fits=9,seeds=[42,101,202],body_minutes=manifest['duration_seconds']/60,
 files_rescored=nfiles,all_cases_same943_users_targets=True,all_early_stopped=True,no_budget_censoring=True,
 initial_neural_pins_verified=True,role_normalization_and_loss_selection_verified=True,ae3500_rq3000_transfer_verified=True,
 prefix_resume_hashes_verified=True,lr_es_replayed=True,amp_skips={r['trial']:r['amp_skips'] for r in rows},
 test_invalid_predictions={r['id']:suite['results'][r['id']]['test']['invalid_predictions'] for r in config['trials']},
 note='Exploratory, same holdout. Shared-user CIs condition on fitted models, excluding seed/runtime variation and multiplicity.')
for name,value in [('metrics.json',rows),('aggregates.json',aggregates),('comparisons.json',comparisons),('subgroups.json',subgroups),('verification.json',verify),('sid-neighborhoods.json',neighborrows)]:write(name,value)
for name,data in [('metrics.csv',rows),('aggregates.csv',aggregates),('comparisons.csv',comparisons),('subgroups.csv',subgroups),('learning-curves.csv',curverows),('quantizer-curves.csv',qrows),('head-mse.csv',headrows)]:
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)));w.writeheader();w.writerows(data)
with gzip.open(OUT/'per-user-ranks.csv.gz','wt',newline='') as f:w=csv.DictWriter(f,fieldnames=list(userrows[0]));w.writeheader();w.writerows(userrows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors={'none':'#32688e','cast':'#2a9d8f','directors':'#e76f51'}
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for role in colors:
 rr=[next(r for r in rows if r['role']==role and r['seed']==seed) for seed in [42,101,202]]
 axes[0].plot([0,1,2],[r['test_ndcg10'] for r in rr],marker='o',color=colors[role],label=role)
 x=list(colors).index(role);a=next(r for r in aggregates if r['role']==role)
 axes[1].errorbar(x,a['mean_test_ndcg10'],yerr=a['sd_test_ndcg10'],fmt='o',color=colors[role],capsize=4)
 axes[1].scatter(np.full(3,x),[r['test_ndcg10'] for r in rr],color=colors[role],alpha=.4,s=20)
axes[0].set(xticks=range(3),xticklabels=[42,101,202],title='Selected test quality by pipeline seed',xlabel='Seed',ylabel='NDCG@10');axes[0].legend()
axes[1].set(xticks=range(3),xticklabels=['No people','Cast only','Directors only'],title='Mean and sample SD; three fitted seeds',ylabel='NDCG@10')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(OUT/'quality.png',dpi=150);fig.savefig(OUT/'quality.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,seed in zip(axes,[42,101,202]):
 for role in colors:
  rr=[r for r in curverows if r['trial']==f'{role}_seed{seed}' and r['val_ndcg10']!='']
  ax.plot([r['epoch'] for r in rr],[r['val_ndcg10'] for r in rr],label=role,color=colors[role])
 ax.set(title=f'Seed {seed}',xlabel='Epoch',ylabel='Validation NDCG@10');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(OUT/'curves.png',dpi=150);fig.savefig(OUT/'curves.pdf');plt.close(fig)
shutil.copyfile(__file__,OUT/'analyze.py')
print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons,subgroups=subgroups,metrics=rows,verification=verify),indent=2))
