"""Supplemental saved-prediction checks; no fitted model calls."""
import csv,gzip,json
from pathlib import Path
import numpy as np
O=Path('/tmp/context50-split-gap');users=json.loads(Path('/tmp/context50-split-gap-inputs/sequences.json').read_text());uids=[u['user_id']for u in users];idx={u:i for i,u in enumerate(uids)};N=len(users);scores={};features={}
with gzip.open('/tmp/context50-three-seed-analysis/cases.csv.gz','rt')as f:
 for r in csv.DictReader(f):
  h=int(r['history_items']);s=r['split'];seed=int(r['seed']);c=r['candidate'];key=h,s,seed,c
  if h not in [40,50]:continue
  scores.setdefault(key,np.zeros(N))[idx[int(r['user_id'])]]=float(r['ndcg10'])
with gzip.open(O/'cases.csv.gz','rt')as f:
 for r in csv.DictReader(f):features.setdefault(r['split'],{})[int(r['user_id'])]=r
result=[]
for h in [40,50]:
 for split in ['validation','test']:
  for group in ['<=1h','>1h']:
   mask=np.array([(float(features[split][u]['gap_seconds'])>3600)==(group=='>1h')for u in uids]);b=np.stack([scores[h,split,s,'none']for s in [42,101,202]]);g=np.stack([scores[h,split,s,'actor_w01']for s in [42,101,202]]);d=g-b;dv=d.mean(0)
   result.append(dict(history_items=h,split=split,gap_group=group,n=int(mask.sum()),control=float(b[:,mask].mean()),graph=float(g[:,mask].mean()),delta=float(d[:,mask].mean()),delta_by_seed=d[:,mask].mean(1).tolist(),control_hits_by_seed=(b[:,mask]>0).sum(1).tolist(),graph_hits_by_seed=(g[:,mask]>0).sum(1).tolist(),improved_users=int(((dv>0)&mask).sum()),worsened_users=int(((dv<0)&mask).sum()),tied_users=int(((dv==0)&mask).sum()),positive_all3_users=int(((d>0).all(0)&mask).sum())))
# Target, actor and popularity support of the >1h lead.
case=[]
b=np.stack([scores[50,'test',s,'none']for s in [42,101,202]]);g=np.stack([scores[50,'test',s,'actor_w01']for s in [42,101,202]]);d=g-b
for j,u in enumerate(uids):
 r=features['test'][u]
 if float(r['gap_seconds'])>3600:case.append(dict(user_id=u,title=r['title'],gap_hours=float(r['gap_seconds'])/3600,popularity=int(r['target_train_popularity']),history_length=int(r['train_history_length']),actor_overlap=r['actor_overlap'],control=float(b[:,j].mean()),graph=float(g[:,j].mean()),delta=float(d[:,j].mean()),delta_by_seed=d[:,j].tolist()))
# Predefined >=1h bin, plus simple threshold sensitivity without selecting a model.
thresholds=[]
for t in [1800,3600,7200,86400]:
 m=np.array([float(features['test'][u]['gap_seconds'])>t for u in uids]);thresholds.append(dict(threshold_seconds=t,n=int(m.sum()),control=float(b[:,m].mean()),graph=float(g[:,m].mean()),delta_by_seed=d[:,m].mean(1).tolist()))
neg=sorted(json.loads((O/'movies.json').read_text()),key=lambda r:r['gap_contribution']);net=json.loads((O/'summary.json').read_text())['delta_gap'];mass={str(k):dict(signed_sum=sum(r['gap_contribution']for r in neg[:k]),fraction_of_net=sum(r['gap_contribution']for r in neg[:k])/net)for k in [1,2,5,10]}
output=dict(groups=result,long_gap_cases=sorted(case,key=lambda r:r['delta'],reverse=True),threshold_sensitivity=thresholds,negative_movie_concentration=mass,scope='Exploratory subgroup and outcome-selected example attribution; context40 shares users/targets/indices and is not independent replication. Threshold sensitivity is not a tuned policy.')
(O/'time-gap-robustness.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in output.items()if k!='long_gap_cases'},indent=2));print('TOPLONG',json.dumps(output['long_gap_cases'][:12],indent=2,ensure_ascii=False))
