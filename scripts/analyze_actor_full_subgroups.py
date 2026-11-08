"""Exploratory metadata-defined groups; existing fitted predictions only.

Feature cuts are computed before loading predictions; no threshold optimization.
All groups, null results, complement contrasts and screen repeats are retained.
"""
import collections,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
OUT=Path('/tmp/actor-full-subgroups');OUT.mkdir(exist_ok=True)
DATA=Path('/tmp/movie-graph-channel-audit');ART=Path('/workspace/artifacts')
load=lambda p:json.loads(p.read_text())
config=load(Path('/tmp/movie-actor-full-completed/experiment.json'))
assert hashlib.sha256((DATA/'features.npz').read_bytes()).hexdigest()==config['modality_input']['features_sha256']
assert hashlib.sha256(Path('/tmp/movie-actor-graph-input/people.npy').read_bytes()).hexdigest()==config['actor_graph_input']['people_sha256']
ids=load(DATA/'source-ids.json');users=load_temporal_users(DATA/'sequences.json',ids)
people=json.loads(gzip.decompress((ART/'movielens100k-people-graph-audit/movie-people.json.gz').read_bytes()))
catalog=json.loads(gzip.decompress((ART/'movielens100k-text-ladder/catalogue-fields.json.gz').read_bytes()))
assert ids==[r['item_id']for r in people]==[r['item_id']for r in catalog]
actor=[{p['person_id']for p in r['cast']}for r in people]
top5=[{p['person_id']for p in r['cast'][:5]}for r in people]
director=[{p['person_id']for p in r['directors']}for r in people]
degree=collections.Counter(p for a in actor for p in a)
support=np.array([sum(degree[p]>1 for p in a)/max(len(a),1)for a in actor]);support_cut=float(np.median(support))
examples={s:temporal_examples(users,20,s)for s in ['train','validation','test']}
split=load(Path('/tmp/movie-actor-full-completed/split-report.json'))
for s,e in examples.items():assert case_digest(e)==split['case_sha256'][s]
pop=np.bincount(examples['train'][2],minlength=len(ids));pastlen={u.user_id:len(u.train)for u in users}
graph=np.load('/tmp/movie-actor-graph-input/people.npy')[:,:64].astype(float)
with np.load(DATA/'features.npz')as z:text=z['text'].astype(float)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
G=unit(graph-graph.mean(0));T=unit(text-text.mean(0))
F={};cuts={};groups={};input_only={}
for s in ['validation','test']:
 hist,uids,targets=examples[s];rr=[]
 hG=np.stack([G[h[h>=0]].mean(0)for h in hist]);hT=np.stack([T[h[h>=0]].mean(0)for h in hist])
 gs=hG@G.T;ts=hT@T.T;grank=np.argsort(-gs,axis=1,kind='stable');trank=np.argsort(-ts,axis=1,kind='stable')
 for j,(h,u,t)in enumerate(zip(hist,uids,targets)):
  h=h[h>=0].tolist();t=int(t);a=set().union(*(actor[i]for i in h));aa=set().union(*(top5[i]for i in h));dd=set().union(*(director[i]for i in h))
  pairs=[(i,k)for i in range(len(h))for k in range(i)]
  coh=float(np.mean([np.dot(G[h[i]],G[h[k]])for i,k in pairs]))if pairs else 0.
  repeat=float(np.mean([bool(actor[h[i]]&actor[h[k]])for i,k in pairs]))if pairs else 0.
  row=dict(user_id=int(u),target=t,title=catalog[t]['fields']['title'],visible_history=len(h),past_events=pastlen[int(u)]+(s=='test'),target_train_examples=int(pop[t]),target_cast_size=len(actor[t]),target_actor_support=float(support[t]),shared_actor=bool(a&actor[t]),shared_top5_actor=bool(aa&top5[t]),shared_director=bool(dd&director[t]),history_actor_repetition=repeat,history_graph_coherence=coh,history_graph_text_top20_overlap=len(set(grank[j,:20])&set(trank[j,:20]))/20,graph_target_rank=int(np.flatnonzero(grank[j]==t)[0])+1,text_target_rank=int(np.flatnonzero(trank[j]==t)[0])+1)
  rr.append(row)
 F[s]=rr
 if s=='validation':
  cuts={k:float(np.median([r[k]for r in rr]))for k in ['history_actor_repetition','history_graph_coherence','history_graph_text_top20_overlap']}
 masks={'all':np.ones(943,dtype=bool)};available={'all':True}
 for k in ['shared_actor','shared_top5_actor','shared_director']:
  m=np.array([r[k]for r in rr]);masks[k]=m;masks['no_'+k]=~m;available[k]=False;available['no_'+k]=False
 for k,cut in cuts.items():
  m=np.array([r[k]for r in rr])>cut;masks[k+'_high']=m;masks[k+'_low']=~m;available[k+'_high']=True;available[k+'_low']=True
 sup=np.array([r['target_actor_support']for r in rr])>=support_cut;masks['target_actor_support_high']=sup;masks['target_actor_support_low']=~sup
 for k in ['target_actor_support_high','target_actor_support_low']:available[k]=False
 vals=np.array([r['target_train_examples']for r in rr]);masks.update(target_train_le20=vals<=20,target_train_21to100=(vals>20)&(vals<=100),target_train_gt100=vals>100)
 sizes=np.array([r['target_cast_size']for r in rr]);masks.update(cast_le10=sizes<=10,cast_11to30=(sizes>10)&(sizes<=30),cast_gt30=sizes>30)
 lens=np.array([r['past_events']for r in rr]);masks.update(past_lt50=lens<50,past_50to99=(lens>=50)&(lens<100),past_ge100=lens>=100)
 for k in ['past_lt50','past_50to99','past_ge100']:available[k]=True
 m=np.array([r['graph_target_rank']<=100 for r in rr]);masks['graph_target_top100']=m;masks['graph_target_outside100']=~m
 for k in masks:available.setdefault(k,False)
 groups[s]=masks;input_only.update(available)
(OUT/'definitions-before-predictions.json').write_text(json.dumps(dict(cuts_from_validation_metadata=cuts,catalogue_support_cut=support_cut,input_only=input_only,n_by_group={s:{k:int(v.sum())for k,v in g.items()}for s,g in groups.items()},scope='Predefined feature families for this exploratory analysis. Feature cuts use metadata only, not model outcomes. Target-dependent groups are diagnostics. Full and screen share users/targets and are not independent holdouts.'),indent=2)+'\n')

vectors={};pred_prov=[];case_rows=[]
for phase,seeds,rootname in [('full',[42,101,202],'movie-actor-full-completed'),('screen',[42,101],'movie-actor-completed')]:
 root=Path('/tmp')/rootname
 for seed in seeds:
  for cand in ['none','actor_w01']:
   p=root/'trials'/f'{cand}_seed{seed}';assert load(p/'sid-catalogue.json')['original_item_ids']==ids;result=load(p/'trial-result.json');sel=result['selection'];idx=np.load(p/'semantic-ids.npy');lookup={tuple(v):i for i,v in enumerate(idx)}
   for s in ['validation','test']:
    path=p/('test-predictions.npz'if s=='test'else f"validation-epoch-{sel['best_epoch']:03d}.npz")
    with np.load(path)as z:pred=z['semantic_ids'];uu=z['user_ids'];tt=z['target_item_ids']
    np.testing.assert_array_equal(uu,examples[s][1]);np.testing.assert_array_equal(tt,examples[s][2]);assert len(set(uu))==943
    dec=np.array([[lookup.get(tuple(v),-1)for v in b]for b in pred]);hit=dec==tt[:,None];rank=np.where(hit.any(1),hit.argmax(1)+1,0)
    ndcg=np.where((rank>0)&(rank<=10),1/np.log2(np.maximum(rank,1)+1),0.);expected=result['test']['ndcg@10']if s=='test'else sel['validation_ndcg10'];assert abs(ndcg.mean()-expected)<1e-12
    vectors[phase,seed,cand,s]=ndcg
    pred_prov.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for meta,r,n in zip(F[s],rank,ndcg):case_rows.append(dict(phase=phase,seed=seed,candidate=cand,split=s,rank=int(r),ndcg10=float(n),**meta))

rng=np.random.default_rng(777);rows=[];contrasts=[];stability=[]
def interval(x):
 if len(x)==0:return [None,None]
 boot=np.concatenate([x[rng.integers(0,len(x),(250,len(x)))].mean(1)for _ in range(12)])
 return np.quantile(boot,[.025,.975]).tolist()
axes=[('shared_actor','no_shared_actor'),('shared_top5_actor','no_shared_top5_actor'),('shared_director','no_shared_director'),('history_actor_repetition_high','history_actor_repetition_low'),('history_graph_coherence_high','history_graph_coherence_low'),('history_graph_text_top20_overlap_high','history_graph_text_top20_overlap_low'),('target_actor_support_high','target_actor_support_low'),('graph_target_top100','graph_target_outside100')]
for phase,seeds in [('full',[42,101,202]),('screen',[42,101])]:
 for s in ['validation','test']:
  a=np.stack([vectors[phase,seed,'actor_w01',s]for seed in seeds]);b=np.stack([vectors[phase,seed,'none',s]for seed in seeds]);d=a-b;dm=d.mean(0)
  for group,mask in groups[s].items():
   n=int(mask.sum())
   if n==0:continue
   lo,hi=interval(dm[mask]);row=dict(phase=phase,split=s,group=group,input_only=input_only[group],n=n,baseline_ndcg10=float(b[:,mask].mean()),graph_ndcg10=float(a[:,mask].mean()),delta=float(d[:,mask].mean()),relative_percent=float(100*d[:,mask].mean()/b[:,mask].mean())if b[:,mask].mean()>0 else None,positive_seeds=int((d[:,mask].mean(1)>0).sum()),ci_low=lo,ci_high=hi,baseline_hits=int((b[:,mask]>0).sum()),graph_hits=int((a[:,mask]>0).sum()))
   for j,seed in enumerate(seeds):row[f'delta_seed{seed}']=float(d[j,mask].mean())
   rows.append(row)
  for ga,gb in axes:
   ma=groups[s][ga];mb=groups[s][gb];na=int(ma.sum());nb=int(mb.sum())
   if not na or not nb:continue
   boot=np.concatenate([dm[ma][rng.integers(0,na,(250,na))].mean(1)-dm[mb][rng.integers(0,nb,(250,nb))].mean(1)for _ in range(12)])
   contrasts.append(dict(phase=phase,split=s,a=ga,b=gb,delta_difference=float(dm[ma].mean()-dm[mb].mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
  stability.append(dict(phase=phase,split=s,all_seeds_better=int((d>0).all(0).sum()),all_seeds_worse=int((d<0).all(0).sum()),both_signs=int(((d>0).any(0)&(d<0).any(0)).sum())))
for name,obj in [('groups',rows),('interactions',contrasts),('stability',stability)]:
 (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
 with(OUT/(name+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in obj for k in r)));w.writeheader();w.writerows(obj)
with gzip.open(OUT/'cases.csv.gz','wt',newline='')as f:w=csv.DictWriter(f,fieldnames=list(case_rows[0]));w.writeheader();w.writerows(case_rows)
(OUT/'verification.json').write_text(json.dumps(dict(files_rescored=pred_prov,exact_case_hashes=split['case_sha256'],feature_cuts_before_model_reads=True,all_groups_reported=True,conditional_user_CI_note='Intervals condition on these fitted models, exclude seed/runtime variation and multiple subgroup comparisons. Full/screen reuse holdouts; these are exploratory associations, not causal effects or a validated router.'),indent=2)+'\n')
for s in ['validation','test']:
 print('\nFULL',s)
 for r in rows:
  if r['phase']=='full'and r['split']==s:print(r['group'],r['n'],round(r['baseline_ndcg10'],5),round(r['graph_ndcg10'],5),'deltas',[round(r[f'delta_seed{k}'],5)for k in [42,101,202]],'CI',round(r['ci_low'],5),round(r['ci_high'],5))
print('STABILITY',stability)
# Examples and leave-one-target-out sensitivity: descriptive post-hoc checks.
for s in ['validation','test']:
 uu=examples[s][1]
 for label in ['better','worse']:
  out=[]
  for j,u in enumerate(uu):
   b=np.array([vectors['full',seed,'none',s][j]for seed in [42,101,202]])
   a=np.array([vectors['full',seed,'actor_w01',s][j]for seed in [42,101,202]]);d=a-b
   if ((d>0).all()if label=='better'else(d<0).all()):
    out.append(dict(user=int(u),**{k:F[s][j][k]for k in ['title','target_train_examples','shared_actor','shared_director']},baseline_ndcg10=b.tolist(),graph_ndcg10=a.tolist(),mean_delta=float(d.mean())))
  out.sort(key=lambda r:-abs(r['mean_delta']));(OUT/f'consistent-{s}-{label}.json').write_text(json.dumps(out,indent=2)+'\n')
 for group,mask in [('shared_director',groups[s]['shared_director']),('rare',groups[s]['target_train_le20'])]:
  a=np.stack([vectors['full',seed,'actor_w01',s]for seed in [42,101,202]])
  b=np.stack([vectors['full',seed,'none',s]for seed in [42,101,202]]);d=a-b;total=d[:,mask].sum(1);n=int(mask.sum());result=[]
  titles=sorted({F[s][j]['title']for j in np.flatnonzero(mask)})
  for title in titles:
   remove=mask&np.array([r['title']==title for r in F[s]]);nr=int(remove.sum());remaining=(total-d[:,remove].sum(1))/(n-nr)
   result.append(dict(title=title,cases=nr,contribution=float(d[:,remove].sum(1).mean()),remaining_mean_delta=float(remaining.mean()),remaining_seed_deltas=remaining.tolist()))
  result.sort(key=lambda r:-r['contribution']);(OUT/f'influence-{s}-{group}.json').write_text(json.dumps(result,indent=2)+'\n')
# Scientific plot of the descriptive leads; all other groups stay in CSV.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pg=['all','shared_director','target_train_le20','past_ge100','shared_actor'];pn=['All cases','Target shares a director','Target train count ≤20','At least100past events','Target shares an actor']
fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,layout='constrained')
for ax,s in zip(axes,['validation','test']):
 for j,(g,name)in enumerate(zip(pg,pn)):
  r=next(r for r in rows if r['phase']=='full'and r['split']==s and r['group']==g)
  ax.errorbar(r['delta'],j,xerr=np.array([[r['delta']-r['ci_low']],[r['ci_high']-r['delta']]]),fmt='o',color='#264653',capsize=3)
  for k,seed in enumerate([42,101,202]):ax.scatter(r['delta_seed'+str(seed)],j+(k-1)*.1,color=['#e76f51','#2a9d8f','#e9c46a'][k],marker='x',s=26,label=f'Seed{seed}'if j==0 else None)
 ax.set_yticks(range(5),[f'{name} (n={next(r["n"]for r in rows if r["phase"]=="full"and r["split"]==s and r["group"]==g)})'for g,name in zip(pg,pn)]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1);ax.set(title=s.capitalize(),xlabel='Graph minus control: NDCG@10');ax.grid(axis='x',alpha=.2)
axes[1].legend(fontsize=8,loc='lower right');fig.suptitle('Exploratory subgroup audit: full training, three fitted pairs\nIntervals condition on fitted models; exclude training variance and multiple-comparison adjustment',fontsize=11)
fig.savefig(OUT/'groups.png',dpi=150);fig.savefig(OUT/'groups.pdf');plt.close(fig)
