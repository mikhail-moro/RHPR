"""CPU-only exploratory attribution of fixed context50 validation/test gap."""
import csv,gzip,hashlib,json,collections
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,rankdata
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
O=Path('/tmp/context50-split-gap');O.mkdir(exist_ok=True)
I=Path('/tmp/context50-split-gap-inputs');P=Path('/tmp/movie-actor-full-completed');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ids=load(P/'trials/none_seed42/sid-catalogue.json')['original_item_ids'];users=load_temporal_users(I/'sequences.json',ids);uids=np.array([u.user_id for u in users]);N=len(users);seeds=[42,101,202];splits=['validation','test'];cands=['none','actor_w01'];checks=[5,10,15,20,*range(21,31)]
catp=I/'catalogue-fields.json.gz';pp=I/'movie-people.json.gz';vecp=I/'people.npy'
cat=json.loads(gzip.decompress(catp.read_bytes()));people=json.loads(gzip.decompress(pp.read_bytes()));assert ids==[r['item_id']for r in cat]==[r['item_id']for r in people]
assert ids==load(I/'source-ids.json')
fields=[r['fields']for r in cat];titles=[r['title']for r in fields];actors=[{a['person_id']for a in r['cast']}for r in people];dirs=[{a['person_id']for a in r['directors']}for r in people]
degree=collections.Counter(a for aa in actors for a in aa);hub=np.array([np.mean([np.log1p(degree[a])for a in aa])for aa in actors]);castn=np.array(list(map(len,actors)));assert all(castn>0)
oldmovie=load(I/'actor-movie-features.json');assert ids==[r['item_id']for r in oldmovie]
coacting=np.array([r['repeated_coacting_weight']for r in oldmovie]);norm=np.array([r['actor_embedding_norm']for r in oldmovie]);np.testing.assert_allclose(hub,[r['actor_catalogue_frequency']for r in oldmovie])
vec=np.load(vecp).astype(float);assert vec.shape==(len(ids),128);assert np.count_nonzero(vec[:,64:])==0;vec=vec[:,:64];vec=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-12)
train=temporal_examples(users,50,'train');pop=np.bincount(train[2],minlength=len(ids));np.testing.assert_array_equal(pop,[r['train_popularity']for r in oldmovie])
ex={s:temporal_examples(users,50,s)for s in splits};features={};targets={};targetcounts={};mean_scores={};mat={};predchecks=[];allv={}
for s in splits:
 h,uu,t=ex[s];np.testing.assert_array_equal(uu,uids);targets[s]=t;targetcounts[s]=np.bincount(t,minlength=len(ids));feat=[]
 for j,u in enumerate(users):
  hh=h[j][h[j]>=0];a=actors[t[j]];seen=set().union(*(actors[i]for i in hh));dd=set().union(*(dirs[i]for i in hh));coverage=len(a&seen)/len(a);pooled=vec[hh].mean(0);cos=float(vec[t[j]]@pooled/max(np.linalg.norm(pooled),1e-12));gap=(u.validation_time-u.train_times[-1])if s=='validation'else(u.test_time-u.validation_time)
  feat.append(dict(user_id=u.user_id,target=int(t[j]),movie_id=ids[t[j]],title=titles[t[j]],train_history_length=len(u.train),visible_history_length=len(hh),target_train_popularity=int(pop[t[j]]),target_cast_size=int(castn[t[j]]),target_actor_hub=float(hub[t[j]]),target_coacting=float(coacting[t[j]]),target_actor_vector_norm=float(norm[t[j]]),actor_coverage=coverage,actor_overlap=bool(coverage>0),director_overlap=bool(dirs[t[j]]&dd),graph_cosine=cos,gap_seconds=gap,target_year=fields[t[j]]['year']))
 features[s]=feat
for seed in seeds:
 root=Path('/tmp/context50-completed')if seed==42 else Path('/tmp/context50-replication-completed')
 for cand in cands:
  p=root/'trials'/f'{cand}_L50_seed{seed}';res=load(p/'trial-result.json');sid=np.load(p/'semantic-ids.npy');epoch=res['selection']['best_epoch'];v=[]
  def values(path,split):
   with np.load(path)as z:
    np.testing.assert_array_equal(z['user_ids'],uids);np.testing.assert_array_equal(z['target_item_ids'],targets[split]);hit=(z['semantic_ids']==sid[targets[split],None,:]).all(-1)
   rank=np.where(hit.any(1),hit.argmax(1)+1,0);score=np.where((rank>0)&(rank<=10),1/np.log2(np.maximum(rank,1)+1),0.)
   predchecks.append(dict(path=str(path),sha256=sha(path)));return score
  for e in checks:
   val=values(p/f'validation-epoch-{e:03d}.npz','validation');v.append(val);expected=next(r for r in load(p/'training-curves.json')if r['epoch']==e)['validation']['ndcg@10'];assert abs(val.mean()-expected)<1e-12
  mat[cand,seed]=np.stack(v);assert checks[int(np.argmax(mat[cand,seed].mean(1)))]==epoch
  allv[cand,seed,'validation']=mat[cand,seed][checks.index(epoch)];allv[cand,seed,'test']=values(p/'test-predictions.npz','test')
  assert abs(allv[cand,seed,'test'].mean()-res['test']['ndcg@10'])<1e-12
D={s:np.stack([allv['actor_w01',seed,s]-allv['none',seed,s]for seed in seeds])for s in splits};dm={s:D[s].mean(0)for s in splits};gap=dm['test']-dm['validation']
# Independently verify against the committed collection summary.
ref=load(I/'context50-three-seed-reference.json')
for s in splits:
 for cand in cands:
  for seed in seeds:
   expected=next(r['ndcg10']for r in ref['metrics']if r['history_items']==50 and r['split']==s and r['candidate']==cand and r['seed']==seed);assert abs(allv[cand,seed,s].mean()-expected)<1e-12
rng=np.random.default_rng(5040301);W=rng.multinomial(N,np.ones(N)/N,size=4000).astype(float)
def ci(x):return np.quantile(W@x/N,[.025,.975]).tolist()
def cohort(s,mask):
 base=np.stack([allv['none',seed,s]for seed in seeds]);graph=np.stack([allv['actor_w01',seed,s]for seed in seeds]);n=int(mask.sum());d=dm[s];nc=int((~mask).sum());bs=(W[:,mask]@d[mask])/np.maximum(W[:,mask].sum(1),1);bc=(W[:,~mask]@d[~mask])/np.maximum(W[:,~mask].sum(1),1)
 return dict(n=n,control=float(base[:,mask].mean()),graph=float(graph[:,mask].mean()),delta=float(d[mask].mean()),delta_by_seed=D[s][:,mask].mean(1).tolist(),conditional_user95=np.quantile(bs,[.025,.975]).tolist(),complement_delta=float(d[~mask].mean())if nc else None,interaction_vs_complement=float(d[mask].mean()-d[~mask].mean())if nc else None,interaction95=np.quantile(bs-bc,[.025,.975]).tolist()if nc else None,contribution_to_overall_delta=float(d[mask].sum()/N))
# Metadata-only bin definitions, shared across splits; no quality values set cuts.
pool=lambda k:np.array([r[k]for s in splits for r in features[s]],dtype=float)
thresholds={k:np.quantile(pool(k),[1/3,2/3]).tolist()for k in ['target_train_popularity','target_actor_hub','graph_cosine','target_coacting']}
bins={};groups=[]
for s in splits:
 f=features[s];get=lambda k:np.array([r[k]for r in f]);b={}
 b['history_length']=np.digitize(get('train_history_length'),[20.5,50.5,100.5]);b['actor_overlap']=get('actor_overlap').astype(int);b['director_overlap']=get('director_overlap').astype(int);b['time_gap']=np.digitize(get('gap_seconds'),[60.5,3600.5])
 for key,cuts in thresholds.items():b[key]=np.searchsorted(cuts,get(key),side='right')
 bins[s]=b
 for key,bb in b.items():
  for level in range(int(max(bb))+1):
   mask=bb==level
   if mask.any():groups.append(dict(split=s,feature=key,level=level,**cohort(s,mask)))
 for genre in sorted(set(g for x in fields for g in x['genres'])):
  mask=np.array([genre in fields[t]['genres']for t in targets[s]])
  if mask.sum()>=30:groups.append(dict(split=s,feature='genre',level=genre,overlapping=True,**cohort(s,mask)))
# Additive contribution to the validation/test gap for mutually exclusive facets.
contributions=[]
for key in bins['validation']:
 for level in sorted(set(bins['validation'][key])|set(bins['test'][key])):
  mv=bins['validation'][key]==level;mt=bins['test'][key]==level;x=dm['test']*mt-dm['validation']*mv
  contributions.append(dict(feature=key,level=int(level),n_validation=int(mv.sum()),n_test=int(mt.sum()),gap_contribution=float(x.mean()),gap_contribution95=ci(x),gap_by_seed=(D['test']*mt-D['validation']*mv).mean(1).tolist()))
 assert abs(sum(r['gap_contribution']for r in contributions if r['feature']==key)-gap.mean())<1e-12
# Exact Kitagawa decomposition: broad popularity/actor-overlap composition vs within strata.
def decomposition(bv,bt):
 levels=sorted(set(bv)|set(bt));pieces=[]
 for level in levels:
  av=bv==level;at=bt==level;nv=av.sum();nt=at.sum()
  if not nv or not nt:return dict(supported=False,missing_level=str(level))
  pv=nv/N;pt=nt/N;mv=dm['validation'][av].mean();mt=dm['test'][at].mean();pieces.append(dict(level=int(level),n_validation=int(nv),n_test=int(nt),validation_delta=float(mv),test_delta=float(mt),composition=float((pt-pv)*(mt+mv)/2),within=float((mt-mv)*(pt+pv)/2)))
 comp=sum(r['composition']for r in pieces);within=sum(r['within']for r in pieces);assert abs(comp+within-gap.mean())<1e-12
 return dict(supported=True,composition=comp,within=within,total_gap=float(gap.mean()),pooled_standardized_validation_delta=float(sum((r['n_validation']+r['n_test'])/(2*N)*r['validation_delta']for r in pieces)),pooled_standardized_test_delta=float(sum((r['n_validation']+r['n_test'])/(2*N)*r['test_delta']for r in pieces)),strata=pieces)
decomp={key:decomposition(bins['validation'][key],bins['test'][key])for key in ['target_train_popularity','target_actor_hub','actor_overlap','history_length']}
decomp['popularity_x_actor_overlap']=decomposition(2*bins['validation']['target_train_popularity']+bins['validation']['actor_overlap'],2*bins['test']['target_train_popularity']+bins['test']['actor_overlap'])
# Movie-level exact additive attribution and common-target standardization.
movies=[]
for i in range(len(ids)):
 mv=targets['validation']==i;mt=targets['test']==i
 if not mv.any()and not mt.any():continue
 vv=D['validation'][:,mv].sum(1);tt=D['test'][:,mt].sum(1)
 movies.append(dict(item_id=ids[i],target_index=i,title=titles[i],genres=fields[i]['genres'],year=fields[i]['year'],train_popularity=int(pop[i]),actor_hub=float(hub[i]),cast_size=int(castn[i]),actors=[a['name']for a in people[i]['cast'][:5]],n_validation=int(mv.sum()),n_test=int(mt.sum()),val_delta_sum=float(vv.mean()),test_delta_sum=float(tt.mean()),gap_contribution=float((tt-vv).mean()/N),gap_by_seed=((tt-vv)/N).tolist()))
assert abs(sum(r['gap_contribution']for r in movies)-gap.mean())<1e-12
common=[]
for min_count in [1,3,5]:
 eligible=[r for r in movies if r['n_validation']>=min_count and r['n_test']>=min_count];weight=np.array([r['n_validation']+r['n_test']for r in eligible],dtype=float);weight/=weight.sum();dv=np.array([r['val_delta_sum']/r['n_validation']for r in eligible]);dt=np.array([r['test_delta_sum']/r['n_test']for r in eligible]);common.append(dict(min_count_per_split=min_count,items=len(eligible),n_validation=sum(r['n_validation']for r in eligible),n_test=sum(r['n_test']for r in eligible),standardized_val_delta=float(weight@dv),standardized_test_delta=float(weight@dt),standardized_gap=float(weight@(dt-dv))))
# Same-user support and per-user attribution, not a selection rule.
userrows=[]
for j,u in enumerate(users):
 userrows.append(dict(user_id=u.user_id,history_length=len(u.train),validation_movie=ids[u.validation],validation_title=titles[u.validation],test_movie=ids[u.test],test_title=titles[u.test],validation_delta=float(dm['validation'][j]),test_delta=float(dm['test'][j]),gap=float(gap[j]),gap_by_seed=(D['test'][:,j]-D['validation'][:,j]).tolist()))
vv=dm['validation'];tt=dm['test'];masks={'both_zero':(vv==0)&(tt==0),'only_validation_differs':(vv!=0)&(tt==0),'only_test_differs':(vv==0)&(tt!=0),'both_positive':(vv>0)&(tt>0),'both_negative':(vv<0)&(tt<0),'val_positive_test_negative':(vv>0)&(tt<0),'val_negative_test_positive':(vv<0)&(tt>0)}
patterns={key:dict(n=int(mask.sum()),gap_contribution=float(gap[mask].sum()/N))for key,mask in masks.items()};assert sum(r['n']for r in patterns.values())==N
# Selection diagnostic: split validation users, select checkpoints on opposite halves.
selections=[];partitions=[];rg=np.random.default_rng(5040302)
for run in range(200):
 perm=rg.permutation(N);a=perm[:N//2];b=perm[N//2:];score={};chosen={}
 for seed in seeds:
  for cand in cands:
   aepoch=int(np.argmax(mat[cand,seed][:,a].mean(1)));bepoch=int(np.argmax(mat[cand,seed][:,b].mean(1)));out=np.empty(N);out[b]=mat[cand,seed][aepoch,b];out[a]=mat[cand,seed][bepoch,a];score[cand,seed]=out;chosen[cand,seed]=(checks[aepoch],checks[bepoch])
 base=np.mean([score['none',s].mean()for s in seeds]);graph=np.mean([score['actor_w01',s].mean()for s in seeds]);partitions.append(dict(partition=run,control=base,graph=graph,delta=graph-base,relative=graph/base-1))
for e in checks:
 base=np.mean([mat['none',s][checks.index(e)].mean()for s in seeds]);graph=np.mean([mat['actor_w01',s][checks.index(e)].mean()for s in seeds]);selections.append(dict(epoch=e,control=float(base),graph=float(graph),delta=float(graph-base),relative=float(graph/base-1)))
# Partial Spearman probes with explicit popularity, cast and history controls.
correlations=[]
for split in splits:
 f=features[split]
 for key in ['target_train_popularity','train_history_length','target_actor_hub','target_coacting','target_actor_vector_norm','actor_coverage','graph_cosine','gap_seconds']:
  x=np.array([r[key]for r in f],dtype=float);y=dm[split];cols=['target_train_popularity','train_history_length','target_cast_size'];cols=[k for k in cols if k!=key];controls=np.column_stack([np.ones(N),*[rankdata([r[k]for r in f])for k in cols]]);rx=rankdata(x);ry=rankdata(y);rx-=controls@np.linalg.lstsq(controls,rx,rcond=None)[0];ry-=controls@np.linalg.lstsq(controls,ry,rcond=None)[0]
  correlations.append(dict(split=split,feature=key,spearman=float(spearmanr(x,y).statistic),partial_spearman=float(np.corrcoef(rx,ry)[0,1]),controls=cols))
summary=dict(users=N,training_targets=len(train[2]),catalogue_items=len(ids),delta_validation=float(vv.mean()),delta_test=float(tt.mean()),delta_gap=float(gap.mean()),gap_conditional_user95=ci(gap),user_delta_pearson=float(np.corrcoef(vv,tt)[0,1]),user_patterns=patterns,thresholds=thresholds,split_profiles={s:dict(unique_targets=len(np.unique(targets[s])),target_pop_mean=float(pop[targets[s]].mean()),target_pop_median=float(np.median(pop[targets[s]])),actor_overlap_fraction=float(np.mean([r['actor_overlap']for r in features[s]])),actor_hub_mean=float(hub[targets[s]].mean()),gap_seconds_median=float(np.median([r['gap_seconds']for r in features[s]])),graph_cosine_mean=float(np.mean([r['graph_cosine']for r in features[s]])))for s in splits},common_target_items=len(set(targets['validation'])&set(targets['test'])),composition=decomp,common_item_standardization=common,checkpoint_cross_selection=dict(partitions=200,control_mean=float(np.mean([r['control']for r in partitions])),graph_mean=float(np.mean([r['graph']for r in partitions])),delta_mean=float(np.mean([r['delta']for r in partitions])),partition_delta_quantiles=np.quantile([r['delta']for r in partitions],[.025,.5,.975]).tolist(),scope='Partition sensitivity only, not a confidence interval or independent holdout: LR scheduling used full validation, models already fitted and previously inspected.'),largest_abs_movie_gap_fraction=float(max(abs(r['gap_contribution'])for r in movies)/abs(gap.mean())),largest_abs_user_gap_fraction=float(np.max(abs(gap))/N/abs(gap.mean())),bootstrap_scope='4000 paired-user multinomial resamples conditional on fitted models; no training seed uncertainty, target/movie clustering or multiplicity correction. Cohorts and feature probes exploratory, not causal.')
for name,data in [('summary',summary),('groups',groups),('gap-contributions',contributions),('movies',movies),('users',userrows),('common-targets',common),('fixed-epochs',selections),('cross-selection',partitions),('correlations',correlations)]:
 (O/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 if isinstance(data,list)and data:
  with(O/(name+'.csv')).open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
with gzip.open(O/'cases.csv.gz','wt',newline='')as f:
 rr=[]
 for split in splits:
  for j,feat in enumerate(features[split]):rr.append(dict(split=split,**feat,control_ndcg10=float(np.mean([allv['none',s,split][j]for s in seeds])),graph_ndcg10=float(np.mean([allv['actor_w01',s,split][j]for s in seeds])),delta=float(dm[split][j]),delta_by_seed=D[split][:,j].tolist()))
 w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(O/'verification.json').write_text(json.dumps(dict(prediction_files=len({r['path']for r in predchecks}),files=predchecks,input_sha256={str(p):sha(p)for p in [catp,pp,vecp,Path(I/'sequences.json')]},all_case_orders_and_targets_verified=True,all_curve_metrics_and_selected_epochs_verified=True,official_predictions_and_checkpoints_unchanged=True,new_gpu_work=False),indent=2)+'\n')
print(json.dumps(summary,indent=2));print('MOST NEGATIVE MOVIES',json.dumps(sorted(movies,key=lambda r:r['gap_contribution'])[:12],ensure_ascii=False,indent=2));print('MOST POSITIVE MOVIES',json.dumps(sorted(movies,key=lambda r:r['gap_contribution'],reverse=True)[:8],ensure_ascii=False,indent=2))
