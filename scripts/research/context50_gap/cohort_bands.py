"""Simultaneous conditional bootstrap bands for the inspected subgroup family."""
import csv,gzip,json
from pathlib import Path
import numpy as np
O=Path('/tmp/context50-split-gap');groups=json.loads((O/'groups.json').read_text());summary=json.loads((O/'summary.json').read_text());cat=json.loads(gzip.decompress(Path('/tmp/context50-split-gap-inputs/catalogue-fields.json.gz').read_bytes()));genres={x['item_id']:x['fields']['genres']for x in cat};f={}
with gzip.open(O/'cases.csv.gz','rt')as stream:
 for r in csv.DictReader(stream):f.setdefault(r['split'],[]).append(r)
N=943;rng=np.random.default_rng(5040301);W=rng.multinomial(N,np.ones(N)/N,size=4000).astype(float);boots=[];points=[]
for g in groups:
 rows=f[g['split']];delta=np.array([float(r['delta'])for r in rows]);feature=g['feature'];level=g['level']
 if feature=='genre':mask=np.array([level in genres[int(r['movie_id'])]for r in rows])
 elif feature=='history_length':mask=np.digitize([float(r['train_history_length'])for r in rows],[20.5,50.5,100.5])==level
 elif feature in ['actor_overlap','director_overlap']:mask=np.array([r[feature]=='True'for r in rows])==bool(level)
 elif feature=='time_gap':mask=np.digitize([float(r['gap_seconds'])for r in rows],[60.5,3600.5])==level
 else:mask=np.searchsorted(summary['thresholds'][feature],[float(r[feature])for r in rows],side='right')==level
 assert mask.sum()==g['n'];bs=(W[:,mask]@delta[mask])/W[:,mask].sum(1);bc=(W[:,~mask]@delta[~mask])/W[:,~mask].sum(1);boots.append(bs-bc);points.append(g['interaction_vs_complement'])
B=np.stack(boots,axis=1);point=np.array(points);se=B.std(0,ddof=1);t=(B-point)/np.maximum(se,1e-12);critical=float(np.quantile(np.max(abs(t),axis=1),.95));rows=[]
for j,g in enumerate(groups):
 lo=float(point[j]-critical*se[j]);hi=float(point[j]+critical*se[j]);rows.append(dict(split=g['split'],feature=g['feature'],level=g['level'],n=g['n'],interaction=float(point[j]),conditional_bootstrap_se=float(se[j]),simultaneous95=[lo,hi],excludes_zero=bool(lo>0 or hi<0)))
r=dict(groups=len(groups),resamples=4000,max_standardized_deviation95=critical,nonzero_bands=sum(x['excludes_zero']for x in rows),rows=rows,scope='Approximate simultaneous paired-user bootstrap bands for these 72 inspected subgroup interactions only. Conditional on fitted models; excludes seed uncertainty, earlier research and subsequent probes. Not independent confirmation or a causal test.')
(O/'familywise-bands.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items()if k!='rows'});print([x for x in rows if x['excludes_zero']])
