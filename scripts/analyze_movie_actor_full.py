"""Audit the graph-weight paired pilot without new neural training."""
import csv,gzip,hashlib,json,math,shutil,zipfile
from pathlib import Path
import numpy as np
from rhpr.gr.sid import SidIndex
ROOT=Path('/tmp/movie-actor-full-completed');OUT=Path('/tmp/movie-actor-full-analysis');OUT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
config=load(ROOT/'experiment.json');manifest=load(ROOT/'run-manifest.json');suite=load(ROOT/'suite-results.json');metrics=load(ROOT/'metrics.json');split=load(ROOT/'split-report.json')
assert load(ROOT/'get_notebook_session_status.json')['status']=='COMPLETE'
assert manifest['git_sha']=='70a0c16d65fab69df3d64742fee527975e12df70' and manifest['config']==config
assert config==load(Path('/workspace/RHPR/experiments/movielens100k_actor_full_2t4.json'))
assert len(suite['results'])==metrics['trials']==6 and not suite['failures'] and suite['parallel_workers']==2
assert all(type(v) in (str,int,float,bool) and (not isinstance(v,float) or math.isfinite(v)) for v in metrics.values())
for r in load(ROOT/'download-manifest.json'):assert sha(ROOT/r['path'])==r['sha256']
for slot in [0,1]:
 w=load(ROOT/f'workers/worker-{slot}.json');assert w['status']=='complete' and len(w['results'])==3 and not w['failures']
base=Path('/tmp/movie-graph-channel-audit');actor=Path('/tmp/movie-actor-graph-input')
assert sha(base/'features.npz')==config['modality_input']['features_sha256']
for name,key in [('people.npy','people_sha256'),('source-ids.json','source_ids_sha256'),('people-provenance.json','provenance_sha256')]:assert sha(actor/name)==config['actor_graph_input'][key]
with np.load(base/'features.npz')as z:arrays={k:z[k]for k in ['text','poster','plot','people']};masks={k:z[k+'_observed']for k in arrays}
actorarray=np.load(actor/'people.npy');assert actorarray.shape==(1273,128)and not actorarray[:,64:].any()
from rhpr.gr.multimodal_rqvae import normalize_modalities
neighbors={}
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
 assert resume['source_epoch']==20 and resume['source_updates']==prefix['optimizer_updates'] and all(resume[k] for k in ['optimizer_restored','scaler_restored','rng_restored_and_verified'])
 assert resume['train_case_sha256']==split['case_sha256']['train'] and resume['validation_case_sha256']==split['case_sha256']['validation']
 codes=np.load(p/'semantic-codes.npy');index=np.load(p/'semantic-ids.npy');np.testing.assert_array_equal(SidIndex(codes,width=256).sids,index)
 assert hashlib.sha256(index.tobytes()).hexdigest()==resume['sid_matrix_sha256']
 assert load(p/'sid-catalogue.json')['original_item_ids']==load(Path('/tmp/movie-graph-channel-audit/source-ids.json'))==load(ROOT/'input-provenance.json')['source_ids']
 role=trial['people_role'];norm=load(p/'modality-normalization.json')
 raw=dict(arrays)
 if trial['candidate']=='actor_w01':
  prov=load(p/'actor-input-provenance.json');assert prov['input']==config['actor_graph_input']and prov['provenance']==load(actor/'people-provenance.json');raw['people']=actorarray
 else:assert not (p/'actor-input-provenance.json').exists()
 _,norm_expected=normalize_modalities(raw,masks,None if role=='none'else {'people':list(range(64))})
 assert norm==norm_expected
 assert t['candidate']==trial['candidate']and not t['fixed_epoch_screen']
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
 rows.append(dict(trial=ident,candidate=trial['candidate'],role=trial['people_role'],seed=trial['seed'],graph_weight=weights['people'],val_ndcg10=sel['validation_ndcg10'],test_ndcg10=t['test']['ndcg@10'],test_recall10=t['test']['recall@10'],best_epoch=sel['best_epoch'],stop_epoch=sel['training_epochs'],ae_steps=ae['steps'],rq_steps=q['steps'],semantic_unique_ids=q['semantic_code_statistics']['semantic_unique_ids'],used_codes_per_level=[int(len(np.unique(codes[:,level]))) for level in range(3)],lr_reductions=reductions,selected_lr=cv[sel['best_epoch']-1].get('learning_rate',.001),amp_skips=sel['amp_skipped_updates']))

from sklearn.metrics import adjusted_rand_score
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
users=load_temporal_users(Path('/tmp/movie-graph-channel-audit/sequences.json'),load(Path('/tmp/movie-graph-channel-audit/source-ids.json')))
for s in ['train','validation','test']:
 x=temporal_examples(users,20,s);assert case_digest(x)==split['case_sha256'][s]
 if s!='train':np.testing.assert_array_equal(x[1],cases[s][0]);np.testing.assert_array_equal(x[2],cases[s][1])
aggregates=[];comparisons=[];seedpairs=[];historical=[];rng=np.random.default_rng(42)
for candidate in ['none','actor_w01']:
 rr=[r for r in rows if r['candidate']==candidate];test=np.array([r['test_ndcg10']for r in rr])
 aggregates.append(dict(candidate=candidate,mean_val_ndcg10=float(np.mean([r['val_ndcg10']for r in rr])),mean_test_ndcg10=float(test.mean()),sd_test_ndcg10=float(test.std(ddof=1)),mean_test_recall10=float(np.mean([r['test_recall10']for r in rr]))))
 for seed in [42,101]:
  p=ROOT/'trials'/f'{candidate}_seed{seed}';old=Path('/tmp/movie-actor-completed/trials')/p.name
  a=load(p/'quantizer.json');b=load(old/'quantizer.json');cv=load(p/'training-curves.json');ocv=load(old/'training-curves.json')
  aa=dict(a['config']);bb=dict(b['config']);aa.pop('expected_initial_state_sha256_by_seed');bb.pop('expected_initial_state_sha256_by_seed');assert aa==bb
  assert a['resolved_model']==b['resolved_model']and load(p/'modality-normalization.json')==load(old/'modality-normalization.json')
  historical.append(dict(candidate=candidate,seed=seed,quantizer_training_config_same_except_pin_map=True,initial_neural_state_same=a['warmup']['initial_neural_state_sha256']==b['warmup']['initial_neural_state_sha256'],ae_final_state_same=a['warmup']['neural_state_sha256']==b['warmup']['neural_state_sha256'],codes_same=bool(np.array_equal(np.load(p/'semantic-codes.npy'),np.load(old/'semantic-codes.npy'))),first_level_ari=float(adjusted_rand_score(np.load(p/'semantic-codes.npy')[:,0],np.load(old/'semantic-codes.npy')[:,0])),generator_initial_same=load(p/'generator.json')['initial_state_sha256']==load(old/'generator.json')['initial_state_sha256'],first_generator_train_ce_difference=next((a['epoch']for a,b in zip(cv,ocv)if a['train_loss_sum_over_codes']!=b['train_loss_sum_over_codes']),None),old_test_ndcg10=load(old/'trial-result.json')['test']['ndcg@10'],new_test_ndcg10=load(p/'trial-result.json')['test']['ndcg@10'],note='Fresh end-to-end fits: AE final states and codes differ despite same initial states/config/inputs. Exact numerical/runtime cause unidentified; not an isolated longer-training ablation.'))
for seed in [42,101,202]:
 a=next(r for r in rows if r['candidate']=='actor_w01'and r['seed']==seed);b=next(r for r in rows if r['candidate']=='none'and r['seed']==seed)
 seedpairs.append(dict(seed=seed,baseline_test=b['test_ndcg10'],graph_test=a['test_ndcg10'],relative_test_percent=100*(a['test_ndcg10']/b['test_ndcg10']-1),baseline_val=b['val_ndcg10'],graph_val=a['val_ndcg10'],relative_val_percent=100*(a['val_ndcg10']/b['val_ndcg10']-1)))
for s in ['validation','test']:
 a=np.stack([vectors[(f'actor_w01_seed{seed}','validation')]if s=='validation'else vectors[f'actor_w01_seed{seed}']for seed in [42,101,202]])
 b=np.stack([vectors[(f'none_seed{seed}','validation')]if s=='validation'else vectors[f'none_seed{seed}']for seed in [42,101,202]])
 d=a-b;dm=d.mean(0);boot=np.concatenate([dm[rng.integers(0,943,(500,943))].mean(1)for _ in range(10)])
 comparisons.append(dict(split=s,delta=float(d.mean()),relative_percent=float(100*d.mean()/b.mean()),delta_by_seed=d.mean(1).tolist(),positive_seeds=int((d.mean(1)>0).sum()),conditional_shared_user_ci=np.quantile(boot,[.025,.975]).tolist(),better_user_seed_pairs=int((d>0).sum()),worse_user_seed_pairs=int((d<0).sum())))
verification=dict(executable_sha=manifest['git_sha'],fits=6,seeds=[42,101,202],body_minutes=manifest['duration_seconds']/60,prediction_files_rescored=nfiles,all_early_stopped=True,no_budget_censoring=True,exact_cases_and_hashes_verified=True,actor_inputs_and_normalization_verified=True,initial_neural_pins_verified=True,ae3500_rq3000_transfer_verified=True,full_scheduler_es_and_resume_replayed=True,amp_skips={r['trial']:r['amp_skips']for r in rows},invalid_test={t['id']:suite['results'][t['id']]['test']['invalid_predictions']for t in config['trials']},scope='Exploratory previously examined holdout; two seeds used for earlier screening. Shared-user CI conditions on three fitted models and excludes training/runtime uncertainty and multiplicity.')
for name,value in [('metrics',rows),('aggregates',aggregates),('comparisons',comparisons),('seed-pairs',seedpairs),('verification',verification),('screen-repeat',historical)]:
 (OUT/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
for name,rs in [('metrics',rows),('aggregates',aggregates),('comparisons',comparisons),('seed-pairs',seedpairs),('head-mse',headrows),('learning-curves',curverows),('quantizer-curves',qrows)]:
 with(OUT/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
with gzip.open(OUT/'per-user-ranks.csv.gz','wt',newline='')as f:w=csv.DictWriter(f,fieldnames=list(userrows[0]));w.writeheader();w.writerows(userrows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors={'none':'#32688e','actor_w01':'#2a9d8f'};labels={'none':'No people','actor_w01':'Actor graph, weight0.1'}
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for ax,field,title in zip(axes,['val_ndcg10','test_ndcg10'],['Selected validation','Selected test']):
 for candidate in colors:
  rr=[next(r for r in rows if r['candidate']==candidate and r['seed']==seed)for seed in [42,101,202]]
  ax.plot([0,1,2],[r[field]for r in rr],marker='o',label=labels[candidate],color=colors[candidate])
 ax.set(xticks=[0,1,2],xticklabels=[42,101,202],title=title,xlabel='Pipeline seed',ylabel='NDCG@10');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Full actor-graph comparison: fresh pipelines, three seeds',fontsize=11)
fig.savefig(OUT/'quality.png',dpi=150);fig.savefig(OUT/'quality.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,seed in zip(axes,[42,101,202]):
 for candidate in colors:
  rr=[r for r in curverows if r['trial']==f'{candidate}_seed{seed}'and r['val_ndcg10']!=''];ax.plot([r['epoch']for r in rr],[r['val_ndcg10']for r in rr],label=labels[candidate],color=colors[candidate])
 ax.set(title=f'Seed{seed}',xlabel='Epoch',ylabel='Validation NDCG@10');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(OUT/'curves.png',dpi=150);fig.savefig(OUT/'curves.pdf');plt.close(fig)
shutil.copyfile(__file__,OUT/'analyze.py');print(json.dumps(dict(aggregates=aggregates,comparisons=comparisons,seeds=seedpairs,metrics=rows,verification=verification,historical=historical),indent=2))
