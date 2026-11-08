"""Secondary exploratory confidence probe; no fitted/tuned routing rule."""
from pathlib import Path
import json,gzip,csv,numpy as np
OUT=Path('/tmp/actor-full-subgroups');raw=Path('/tmp/movie-actor-full-completed')
load=lambda p:json.loads(p.read_text())
rows=list(csv.DictReader(gzip.open(OUT/'cases.csv.gz','rt')));vec={}
for s in ['validation','test']:
 for seed in [42,101,202]:
  for cand in ['none','actor_w01']:
   rr=[r for r in rows if r['phase']=='full'and r['split']==s and int(r['seed'])==seed and r['candidate']==cand];assert len(rr)==943
   vec[s,seed,cand]=np.array([float(r['ndcg10'])for r in rr])
gaps={};cuts={};masks={}
for seed in [42,101,202]:
 p=raw/'trials'/f'none_seed{seed}';epoch=load(p/'training-selection.json')['best_epoch']
 for s in ['validation','test']:
  with np.load(p/('test-predictions.npz'if s=='test'else f'validation-epoch-{epoch:03d}.npz'))as z:
   scores=z['scores'];assert np.all(np.diff(scores,axis=1)<=1e-6);gaps[s,seed]=scores[:,0]-scores[:,1]
 cuts[seed]=float(np.median(gaps['validation',seed]))
 for s in ['validation','test']:masks[s,seed]=gaps[s,seed]<=cuts[seed]

rng=np.random.default_rng(814);rs=[];gates=[]
for s in ['validation','test']:
 b=np.stack([vec[s,seed,'none']for seed in [42,101,202]])
 a=np.stack([vec[s,seed,'actor_w01']for seed in [42,101,202]])
 other=np.stack([np.mean([vec[s,j,'none']for j in [42,101,202]if j!=seed],axis=0)for seed in [42,101,202]])
 low=np.stack([masks[s,seed]for seed in [42,101,202]])
 for group,m in [('low_gap',low),('high_gap',~low)]:
  for alternative,alt in [('graph',a),('other_baseline_seed_mean',other)]:
   d=alt-b;weighted=(d*m).sum(0);counts=m.sum(0);boot=[]
   for _ in range(12):
    ii=rng.integers(0,943,(250,943));boot.extend((weighted[ii].sum(1)/counts[ii].sum(1)).tolist())
   row=dict(split=s,group=group,alternative=alternative,n_user_seed_cases=int(m.sum()),unique_users=int(m.any(0).sum()),baseline_ndcg10=float(b[m].mean()),alternative_ndcg10=float(alt[m].mean()),delta=float(d[m].mean()),relative_percent=float(100*d[m].mean()/b[m].mean()),delta_by_seed=[float(d[j,m[j]].mean())for j in range(3)],n_by_seed=m.sum(1).tolist(),ci=np.quantile(boot,[.025,.975]).tolist())
   rs.append(row);print(row)
 # This is a hypothetical metadata-threshold gate, with thresholds fixed from
 # baseline scores only. It is analyzed post hoc on an already used holdout.
 chosen=np.where(low,a,b);d=chosen-b;dm=d.mean(0);boot=[]
 for _ in range(12):boot.extend(dm[rng.integers(0,943,(250,943))].mean(1).tolist())
 gates.append(dict(split=s,baseline_ndcg10=float(b.mean()),gated_ndcg10=float(chosen.mean()),delta_by_seed=d.mean(1).tolist(),ci=np.quantile(boot,[.025,.975]).tolist()))
 print('GATE',s,float(chosen.mean()),float(b.mean()),'seed deltas',d.mean(1).tolist(),'CI',np.quantile(boot,[.025,.975]).tolist())
(OUT/'confidence-probe.json').write_text(json.dumps(dict(cuts=cuts,rows=rs,gates=gates,note='Secondary exploratory check chosen after metadata audit. Per-seed gap threshold = validation score median, without using target labels. Scores describe a finite beam, not calibrated catalogue uncertainty. Other-baseline result averages two separately substituted seeds, not an actual averaged-score ensemble. No gate was fitted or deployed; all CIs conditional on the fixed models, unadjusted for comparisons.'),indent=2)+'\n')
