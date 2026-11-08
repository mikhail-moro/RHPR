"""Descriptive graph/recommendation correlations, with movie-frequency controls."""
import collections,csv,gzip,json
from pathlib import Path
import numpy as np
from scipy import sparse,stats
from scipy.stats import rankdata,spearmanr
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions',type=Path,required=True)
parser.add_argument('--sequences',type=Path,required=True)
parser.add_argument('--people',type=Path,required=True,help='movie-people.json.gz')
parser.add_argument('--catalogue',type=Path,required=True,help='catalogue-fields.json.gz')
parser.add_argument('--vectors',type=Path,required=True,help='people-vectors.npz from frozen actor dataset')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();R=args.predictions;O=args.output;O.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text());people=json.loads(gzip.decompress(args.people.read_bytes()));cat=json.loads(gzip.decompress(args.catalogue.read_bytes()));ids=[r['item_id']for r in people];assert ids==[r['item_id']for r in cat]
users=load_temporal_users(args.sequences,ids);lookup={u.user_id:u for u in users};train=temporal_examples(users,20,'train');pop=np.bincount(train[2],minlength=len(ids));actors=[sorted({p['person_id']for p in r['cast']})for r in people];names={p['person_id']:p['name']for r in people for p in r['cast']};pids=sorted(names);pi={p:j for j,p in enumerate(pids)};aa=[[pi[p]for p in a]for a in actors];sizes=np.array(list(map(len,aa)));rr=[];cc=[]
for j,a in enumerate(aa):rr.extend(a);cc.extend([j]*len(a))
B=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(pids),len(ids)));W=(B@sparse.diags(1/np.maximum(sizes-1,1))@B.T).tocsr();W.setdiag(0);W.eliminate_zeros();degree=np.asarray(B.sum(1)).ravel();strength=np.asarray(W.sum(1)).ravel();neighbors=W.getnnz(1);assert np.allclose(np.asarray(B@np.asarray(sizes>1,dtype=float)).ravel(),strength)
weighted_pop=np.asarray(B@pop).ravel();features=dict(cast_size=sizes.astype(float),actor_catalogue_frequency=np.array([np.mean(np.log1p(degree[a]))for a in aa]),actor_unique_neighbors=np.array([np.mean(np.log1p(neighbors[a]))for a in aa]),actor_interaction_weighted_popularity=np.array([np.mean(np.log1p(weighted_pop[a]))for a in aa]),actor_max_catalogue_frequency=np.array([np.log1p(degree[a].max())for a in aa]))
excess=[];strong=[]
for j,a in enumerate(aa):
 sub=W[a][:,a].toarray();mask=~np.eye(len(a),dtype=bool);delta=sub[mask]-1/max(len(a)-1,1);excess.append(delta.mean()if len(delta)else 0.);strong.append((delta>1e-8).mean()if len(delta)else 0.)
features['actor_weighted_strength']=np.array([np.mean(np.log1p(strength[a]))for a in aa]);features.update(repeated_coacting_weight=np.array(excess),repeated_coacting_pair_fraction=np.array(strong))
vp=args.vectors
if vp.exists():
 with np.load(vp)as z:
  assert np.array_equal(z['person_ids'],pids);norm=np.linalg.norm(z['vectors'],axis=1);features['actor_embedding_norm']=np.array([norm[a].mean()for a in aa]);embedding_degree_rho=float(stats.spearmanr(norm,degree).statistic)
else:embedding_degree_rho=None
rows=[];filmrows=[];exposure={};profiles=[]
# Partial Spearman via rank residualization on log movie training popularity and cast size.
controls=np.column_stack([np.ones(len(ids)),stats.rankdata(np.log1p(pop)),stats.rankdata(sizes)])
def rho(a,b):return float(stats.spearmanr(a,b).statistic)if np.std(a)>0 and np.std(b)>0 else None
def partial(a,b):
 ra=stats.rankdata(a);rb=stats.rankdata(b);ra-=controls@np.linalg.lstsq(controls,ra,rcond=None)[0];rb-=controls@np.linalg.lstsq(controls,rb,rcond=None)[0]
 return float(np.corrcoef(ra,rb)[0,1])if np.std(ra)>1e-9 and np.std(rb)>1e-9 else None
for split in ['validation','test']:
 h,uu,t=temporal_examples(users,20,split);assert case_digest((h,uu,t))==load(R/'split-report.json')['case_sha256'][split]
 past=[set(lookup[int(u)].train)|({lookup[int(u)].validation}if split=='test'else set())for u in uu];counts={};repeats={};movieknown=np.zeros(len(ids))
 for p in past:movieknown[list(p)]+=1
 for cand in ['none','actor_w01']:
  for seed in [42,101,202]:
   p=R/'trials'/f'{cand}_seed{seed}';ep=load(p/'trial-result.json')['selection']['best_epoch'];sids=np.load(p/'semantic-ids.npy');lu={tuple(x):i for i,x in enumerate(sids)}
   with np.load(p/('test-predictions.npz'if split=='test'else f'validation-epoch-{ep:03d}.npz'))as z:pred=z['semantic_ids'];assert np.array_equal(uu,z['user_ids'])
   count=np.zeros(len(ids));repeat=np.zeros(len(ids));recommended=[]
   for j,row in enumerate(pred[:,:10]):
    for sid in row:
     i=lu.get(tuple(sid),-1)
     if i>=0:count[i]+=1;repeat[i]+=int(i in past[j]);recommended.append(i)
   counts[cand,seed]=count;repeats[cand,seed]=repeat
   profiles.append(dict(split=split,candidate=cand,seed=seed,mean_log_movie_train_popularity=float(np.log1p(pop[recommended]).mean()),**{k:float(v[recommended].mean())for k,v in features.items()}))
 for seed in [42,101,202,'mean']:
  seq=[seed]if seed!='mean'else[42,101,202];base=np.mean([counts['none',s]for s in seq],0);graph=np.mean([counts['actor_w01',s]for s in seq],0);b_repeat=np.mean([repeats['none',s]for s in seq],0);g_repeat=np.mean([repeats['actor_w01',s]for s in seq],0)
  for feature,x in features.items():
   for name,y in [('recommendation_delta',graph-base),('repeat_delta',g_repeat-b_repeat),('log_exposure_ratio',np.log((graph+.5)/(base+.5)))]:
    rows.append(dict(split=split,seed=seed,feature=feature,outcome=name,spearman=rho(x,y),partial_spearman_movie_pop_cast=partial(x,y)))
  if seed=='mean':
   exposure[split]=(base,graph,b_repeat,g_repeat)
   for i in range(len(ids)):filmrows.append(dict(split=split,item_id=ids[i],title=cat[i]['fields']['title'],train_popularity=int(pop[i]),known_users=int(movieknown[i]),baseline_recommendations=float(base[i]),graph_recommendations=float(graph[i]),recommendation_delta=float(graph[i]-base[i]),baseline_repeats=float(b_repeat[i]),graph_repeats=float(g_repeat[i]),**{k:float(v[i])for k,v in features.items()}))
for n,data in [('correlations',rows),('movies',filmrows),('recommendation-profiles',profiles)]:
 (O/(n+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 with(O/(n+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary=dict(persons=len(pids),movies=len(ids),undirected_edges=int(W.nnz//2),weighted_degree_equals_nonsingleton_film_frequency=True,singleton_cast_movies=int((sizes==1).sum()),actor_embedding_norm_catalogue_degree_spearman=embedding_degree_rho,scope='Exploratory correlations, no causal or external-popularity claim. Mean seed exposures; partial Spearman controls movie train popularity and cast size; shared-film graph dependence and retraining uncertainty not accounted for.')
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
for r in rows:
 if r['seed']=='mean'and r['outcome']=='recommendation_delta':print(r)
for split in ['validation','test']:
 print('PROFILE',split)
 for c in ['none','actor_w01']:
  pp=[p for p in profiles if p['split']==split and p['candidate']==c];print(c,{k:round(float(np.mean([p[k]for p in pp])),4)for k in pp[0]if k not in ['split','candidate','seed']})

# Context-conditioned probe
V=B@sparse.diags(1/sizes);D=sparse.diags(1/np.sqrt(np.maximum(strength,1)));K=(V.T@D@W@D@V).toarray()
result=[]
for split in ['validation','test']:
 h,uu,t=temporal_examples(users,20,split);past=[set(lookup[int(u)].train)|({lookup[int(u)].validation}if split=='test'else set())for u in uu]
 affinity=np.stack([K[:,row[row>=0]].mean(1)for row in h]);poplog=np.log1p(pop);castlog=np.log1p(sizes)
 for seed in [42,101,202]:
  pred={}
  for cand in ['none','actor_w01']:
   p=R/'trials'/f'{cand}_seed{seed}';ep=load(p/'trial-result.json')['selection']['best_epoch'];sids=np.load(p/'semantic-ids.npy');lu={tuple(x):i for i,x in enumerate(sids)}
   with np.load(p/('test-predictions.npz'if split=='test'else f'validation-epoch-{ep:03d}.npz'))as z:pred[cand]=np.array([[lu.get(tuple(x),-1)for x in r[:10]]for r in z['semantic_ids']])
  diffs=[];prom=[];dem=[];allaff=[];labels=[];allpop=[];allcast=[];group=[]
  for j in range(len(uu)):
   b={i for i in pred['none'][j]if i>=0 and i not in past[j]and i!=t[j]};g={i for i in pred['actor_w01'][j]if i>=0 and i not in past[j]and i!=t[j]};gp=sorted(g-b);bp=sorted(b-g)
   if gp and bp:diffs.append(affinity[j,gp].mean()-affinity[j,bp].mean());prom.append(affinity[j,gp].mean());dem.append(affinity[j,bp].mean())
   for ii,label in [(gp,1),(bp,0)]:
    allaff.extend(affinity[j,ii]);labels.extend([label]*len(ii));allpop.extend(poplog[ii]);allcast.extend(castlog[ii]);group.extend([j]*len(ii))
  # Descriptive pooled residual correlation, user fixed effects plus item popularity/cast.
  x=np.asarray(allaff);y=np.asarray(labels,dtype=float);q=np.asarray(allpop);c=np.asarray(allcast);group=np.asarray(group)
  for a in [x,y,q,c]:
   sums=np.bincount(group,weights=a,minlength=len(uu));nn=np.bincount(group,minlength=len(uu));a-=np.divide(sums,np.maximum(nn,1))[group]
  z=np.column_stack([q,c]);xr=x-z@np.linalg.lstsq(z,x,rcond=None)[0];yr=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
  result.append(dict(split=split,seed=seed,paired_cases=len(diffs),promoted_unseen_wrong_affinity=float(np.mean(prom)),demoted_unseen_wrong_affinity=float(np.mean(dem)),mean_paired_delta=float(np.mean(diffs)),fraction_positive_paired_delta=float(np.mean(np.array(diffs)>0)),pooled_partial_pearson_with_user_effect_movie_pop_cast=float(np.corrcoef(xr,yr)[0,1]),scope='Descriptive; outcome-selected candidates, overlapping histories/actors, no causal claim. Direct normalized actor graph affinity, excluding known and target films.'))
(O/'context-affinity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

# Target-quality probe
P=O;movie=json.loads((P/'movies.json').read_text());out=[];groups=[]
def value(p,ids,t):
 hit=(p==ids[t,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0);return np.where((rank>0)&(rank<=10),1/np.log2(np.maximum(rank,1)+1),0.)
for split in ['validation','test']:
 mm=[m for m in movie if m['split']==split];ids=[m['item_id']for m in mm];allv={}
 for seed in [42,101,202]:
  for c in ['none','actor_w01']:
   p=R/'trials'/f'{c}_seed{seed}';epoch=json.loads((p/'trial-result.json').read_text())['selection']['best_epoch'];sids=np.load(p/'semantic-ids.npy')
   with np.load(p/('test-predictions.npz'if split=='test'else f'validation-epoch-{epoch:03d}.npz'))as z:t=z['target_item_ids'];allv[c,seed]=value(z['semantic_ids'],sids,t)
 control=np.column_stack([np.ones(len(t)),rankdata([mm[i]['train_popularity']for i in t]),rankdata([mm[i]['cast_size']for i in t])])
 for key in ['train_popularity','actor_catalogue_frequency','actor_unique_neighbors','actor_interaction_weighted_popularity','repeated_coacting_weight','repeated_coacting_pair_fraction','actor_embedding_norm']:
  x=np.array([mm[i][key]for i in t]);rx=rankdata(x);rx-=control@np.linalg.lstsq(control,rx,rcond=None)[0]
  for seed in [42,101,202,'mean']:
   vv=[seed]if seed!='mean'else[42,101,202];base=np.mean([allv['none',s]for s in vv],0);graph=np.mean([allv['actor_w01',s]for s in vv],0);y=graph-base;ry=rankdata(y);ry-=control@np.linalg.lstsq(control,ry,rcond=None)[0]
   out.append(dict(split=split,feature=key,seed=seed,delta_ndcg_spearman=float(spearmanr(x,y).statistic),partial_spearman_movie_pop_cast=float(np.corrcoef(rx,ry)[0,1])if np.std(rx)>1e-8 else None))
  edges=np.quantile([m[key]for m in mm],[1/3,2/3]);bins=np.searchsorted(edges,x,side='right')
  for j in range(3):
   mask=bins==j;bv=np.array([allv['none',s][mask].mean()for s in [42,101,202]]);gv=np.array([allv['actor_w01',s][mask].mean()for s in [42,101,202]])
   groups.append(dict(split=split,feature=key,tercile=j+1,n=int(mask.sum()),baseline_ndcg10=float(bv.mean()),graph_ndcg10=float(gv.mean()),delta_by_seed=(gv-bv).tolist()))
(P/'quality-correlations.json').write_text(json.dumps(out,indent=2)+'\n');(P/'quality-groups.json').write_text(json.dumps(groups,indent=2)+'\n')
print(json.dumps([r for r in out if r['seed']=='mean'],indent=2))
