"""Recompute matched-user stability summaries without training or TEST.

The bootstrap conditions on fixed fitted models and selected checkpoints.
"""
import argparse
import json, pathlib,numpy as np
from scipy.stats import spearmanr
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=pathlib.Path,required=True)
parser.add_argument('--data',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
args=parser.parse_args()
p=args.root;o=args.output;o.mkdir(parents=True,exist_ok=True)
c=json.loads((p/'experiment.json').read_text());suite=json.loads((p/'suite-results.json').read_text());records=suite['results'];
with np.load(args.data/'validation-cases.npz',allow_pickle=False) as z:ids=z['user_ids'];targets=z['targets']
assert len(ids)==2786
per={};raw={};curves={};preds=0
for i in c['index_ids']:
 sid=np.load(p/'frozen-indexes'/(i+'-semantic-ids.npy'),allow_pickle=False)
 for gen in c['generator_seeds']:
  tid=f'narrow_{i}_gen{gen}';row=records[tid];trial=p/'trials'/tid
  cc=json.loads((trial/'training-curves.json').read_text());curves[i,gen]=cc
  for label,epoch in [('fixed30',30),('fixed50',50),('selected',row['selection']['best_epoch'])]:
   if (i,gen,epoch) not in raw:
    with np.load(trial/f'validation-epoch-{epoch:03}.npz',allow_pickle=False) as z:
     np.testing.assert_array_equal(ids,z['user_ids']);np.testing.assert_array_equal(targets,z['target_item_ids'])
     pred=z['semantic_ids'];assert pred.shape==(2786,20,4)
     hit=(pred==sid[targets,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0)
     arr=np.where((rank>0)&(rank<=10),1/np.log2(np.maximum(rank,1)+1),0.)
     raw[i,gen,epoch]=arr;preds+=1
     expected=next(x['validation']['ndcg@10'] for x in cc if x['epoch']==epoch)
     np.testing.assert_allclose(arr.mean(),expected,rtol=0,atol=1e-12)
   per[i,gen,label]=raw[i,gen,epoch]

def method(m,t): return f'{m}_seed{t}'
methods=['meta_kmeans','meta_rqvae','multimodal_rqvae'];token_seeds=[42,101];gens=c['generator_seeds']
rng=np.random.default_rng(20261108)
res={'comparison':{},'method_summaries':{},'index_diagnostics':{},'curves':{},'saved_prediction_files_rescored':preds,
     'scope':'Post-hoc shared-user bootstrap on fixed trained models/selected checkpoints, 5000 jointly resampled users; excludes index and training seed variation, selection bias, multiple comparisons and target-item clustering. Six fits per method are 2 tokenizer indices x 3 paired generator seeds, not six independent tokenizer replicates. Scores are VAL only.',
     'stopping':{'censored':sum(x['budget_censored'] for x in records.values()),'reached':sum(x['plateau_reached'] for x in records.values())}}
for m in methods:
 r={}
 for e in [10,20,30,40,50,60]:
  values=[records[f'narrow_{method(m,t)}_gen{g}']['epochs'][str(e)] for t in token_seeds for g in gens if str(e) in records[f'narrow_{method(m,t)}_gen{g}']['epochs']]
  r[str(e)]={'n':len(values),'mean':float(np.mean(values))}
 res['curves'][m]=r
 for label in ['fixed30','fixed50','selected']:
  vals=np.array([[per[method(m,t),g,label].mean() for g in gens] for t in token_seeds]);sd=vals.std(axis=1,ddof=1)
  r=res['method_summaries'].setdefault(m,{})
  r[label]={'mean':float(vals.mean()),'index_seed_means':dict(zip(map(str,token_seeds),vals.mean(axis=1).tolist())),
   'within_index_sample_sd':dict(zip(map(str,token_seeds),sd.tolist())),
   'within_index_range':dict(zip(map(str,token_seeds),np.ptp(vals,axis=1).tolist())),
   'scores':dict(zip(map(str,token_seeds),vals.tolist())),
   'selected_epochs':[[records[f'narrow_{method(m,t)}_gen{g}']['selection']['best_epoch'] for g in gens] for t in token_seeds]}
 for t in token_seeds:
  rs=[records[f'narrow_{method(m,t)}_gen{g}'] for g in gens];idx=method(m,t)
  res['index_diagnostics'][idx]={'stop_epochs':[x['selection']['training_epochs'] for x in rs],
   'best_epochs':[x['selection']['best_epoch'] for x in rs],
   'stopping_reason':[x['selection']['stopping_reason'] for x in rs],
   'epoch30_to50_ce':[[next(x['validation_loss']['total_ce'] for x in curves[idx,g] if x['epoch']==e) for e in [30,50]] for g in gens],
   'amp_skips':[x['selection']['amp_skipped_updates'] for x in rs]}
for label in ['fixed30','fixed50','selected']:
 for a,b in [('meta_kmeans','meta_rqvae'),('meta_rqvae','multimodal_rqvae'),('meta_kmeans','multimodal_rqvae')]:
  pairs=np.array([per[method(a,t),g,label]-per[method(b,t),g,label] for t in token_seeds for g in gens]);users=pairs.mean(axis=0);draws=[]
  for _ in range(50):
   ix=rng.integers(0,len(users),size=(100,len(users)));draws.extend(users[ix].mean(axis=1))
  delta=pairs.mean(axis=1);key=f'{label}:{a}_minus_{b}'
  res['comparison'][key]={'delta':float(delta.mean()),'relative_delta':float(delta.mean()/np.mean([per[method(b,t),g,label].mean() for t in token_seeds for g in gens])),
   'paired_runs_positive':int((delta>0).sum()),'paired_runs_total':6,'per_pair':delta.tolist(),
   'user_bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist()}
(o/'statistics.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'completed':True,'saved_prediction_files_rescored':preds,'stopping':res['stopping'],'output':str(o/'statistics.json')},indent=2))
