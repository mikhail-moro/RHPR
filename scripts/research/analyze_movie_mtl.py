"""CPU replay of fixed seed42 MTL screen; no training or inference."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('inputs',type=Path);p.add_argument('output',type=Path);p.add_argument('--sequences',type=Path);args=p.parse_args();root=args.inputs;out=args.output;out.mkdir(parents=True,exist_ok=True)
 load=lambda p:json.loads(p.read_text());cfg=load(root/'experiment.json');manifest=load(root/'run-manifest.json')
 assert manifest['git_sha']=='5af82b106c3d2687459dcb9ebc4cc2e4e1547415' and cfg==manifest['config']
 expected_cases={}
 if args.sequences:
  from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
  ids=load(root/'trials/sum_seed42/sid-catalogue.json')['original_item_ids']
  users=load_temporal_users(args.sequences,ids)
  split=load(root/'split-report.json')
  for part in ['train','validation','test']:
   ex=temporal_examples(users,20,part);assert case_digest(ex)==split['case_sha256'][part]
   if part!='train':expected_cases[part]=(ex[1],ex[2])
 records=[];scores={};checks=[];curves={};diagnostics=[];mtl=[];case_ref={};rng=np.random.default_rng(703)
 for log in sorted((root/'workers').glob('*.log')):
  for line in log.read_text().splitlines():
   if line.startswith('{'):
    ev=json.loads(line)
    if ev.get('phase')=='modality_mtl':mtl.append(ev)
 for mode in ['sum','uw','gradnorm','pcgrad']:
  d=root/'trials'/f'{mode}_seed42';r=load(d/'trial-result.json');q=load(d/'quantizer.json');c=load(d/'training-curves.json');curves[mode]=c
  assert r['seed']==42 and r['active_modalities']==['text','poster','plot'];sel=r['selection']
  assert sel['training_epochs']==30 and sel['validation_checks']==6 and sel['optimizer_updates']==10020 and sel['amp_skipped_updates']==0
  assert len(c)==30 and all(row['updates']==334*row['epoch'] and row['train_examples']==85470 for row in c)
  assert [row['epoch'] for row in c if 'validation' in row]==[5,10,15,20,25,30]
  best=max((row for row in c if 'validation' in row),key=lambda row:row['validation']['ndcg@10'])
  assert sel['best_epoch']==best['epoch'];np.testing.assert_allclose(best['validation']['ndcg@10'],sel['validation_ndcg10'],atol=1e-12,rtol=0)
  assert q['quantizer_initial_state_sha256']==cfg['quantizer']['expected_initial_state_sha256_by_seed']['42']
  assert load(d/'generator.json')['initial_state_sha256']==cfg['expected_initial_state_sha256']
  assert q['steps']==3000 and q['warmup']['steps']==3500 and q['warmup']['codebooks_unchanged'] and q['warmup']['fresh_rq_optimizer_verified'] and q['warmup']['neural_transfer_verified']
  sid=np.load(d/'semantic-ids.npy');codes=np.load(d/'semantic-codes.npy');assert sid.shape==(1273,4) and codes.shape==(1273,3)
  assert len(np.unique(sid,axis=0))==1273
  from rhpr.gr.sid import SidIndex
  np.testing.assert_array_equal(SidIndex(codes,width=256).sids,sid)
  with np.load(d/'codebooks-final.npz') as books:
   np.testing.assert_array_equal(books['counts'],np.stack([np.bincount(codes[:,i],minlength=256) for i in range(3)]))
  lookup={tuple(s):i for i,s in enumerate(sid)}
  def score(path,split,expected):
   with np.load(path) as z:
    u=z['user_ids'];t=z['target_item_ids'];pred=z['semantic_ids'];ranked=np.array([[lookup.get(tuple(s),-1) for s in row] for row in pred])
   assert len(u)==943
   if split in expected_cases:
    np.testing.assert_array_equal(u,expected_cases[split][0]);np.testing.assert_array_equal(t,expected_cases[split][1])
   if split in case_ref:
    np.testing.assert_array_equal(u,case_ref[split][0]);np.testing.assert_array_equal(t,case_ref[split][1])
   else:case_ref[split]=(u,t)
   hit=(pred==sid[t,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0)
   vals={}
   for k in [5,10]:
    ok=(rank>0)&(rank<=k);vals[f'ndcg@{k}']=np.where(ok,1/np.log2(np.maximum(rank,1)+1),0);vals[f'recall@{k}']=ok.astype(float)
   for k,v in vals.items():np.testing.assert_allclose(v.mean(),expected[k],atol=1e-12,rtol=0)
   invalid=int((ranked<0).sum());duplicates=sum(len(row[row>=0])-len(set(row[row>=0])) for row in ranked)
   assert invalid==expected['invalid_predictions'] and duplicates==expected['duplicate_predictions']
   checks.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),invalid=invalid,duplicates=duplicates))
   return vals['ndcg@10']
  for row in c:
   if 'validation' in row:
    val=score(d/f"validation-epoch-{row['epoch']:03d}.npz",'validation',row['validation'])
    if row['epoch']==sel['best_epoch']:scores[mode,'validation']=val
  scores[mode,'test']=score(d/'test-predictions.npz','test',r['test'])
  ev=[e for e in mtl if e['mode']==mode];assert len(ev)==13
  cos=np.array([e['shared_encoder_gradient_cosines'] for e in ev]);upper=cos[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]
  diag=dict(mode=mode,probe_count=len(ev),encoder_cos_min=float(upper.min()),encoder_cos_median=float(np.median(upper)),encoder_negative_pairs=int((upper<0).sum()),pair_probes=int(upper.size),pcgrad_probe_projection_counts=[e['negative_projection_count'] for e in ev] if mode=='pcgrad' else None,
      weights=q['mtl_final_weights'],weight_sum=sum(q['mtl_final_weights']),reconstruction_mse=q['head_reconstruction_mse'],unique_triples=len(np.unique(codes,axis=0)),used_codes=[len(np.unique(codes[:,i])) for i in range(3)],ae_seconds=q['warmup']['seconds'],rq_seconds=q['fit_seconds'])
  diagnostics.append(diag)
  records.append(dict(mode=mode,val=sel['validation_ndcg10'],test=r['test']['ndcg@10'],recall10=r['test']['recall@10'],best_epoch=sel['best_epoch'],mean_epoch_seconds=float(np.mean([row['train_seconds'] for row in c])),tokenizer_seconds=diag['ae_seconds']+diag['rq_seconds']))
 for row in records:
  row['relative_val']=row['val']/records[0]['val']-1;row['relative_test']=row['test']/records[0]['test']-1
 comparisons=[]
 for mode in ['uw','gradnorm','pcgrad']:
  for split in ['validation','test']:
   delta=scores[mode,split]-scores['sum',split];boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
   comparisons.append(dict(mode=mode,split=split,delta=float(delta.mean()),user95=np.quantile(boot,[.025,.975]).tolist(),better=int((delta>0).sum()),worse=int((delta<0).sum()),ties=int((delta==0).sum())))
 result=dict(temporal_cases_reconstructed=bool(expected_cases),executable=manifest['git_sha'],run_minutes=manifest['duration_seconds']/60,models=records,diagnostics=diagnostics,comparisons=comparisons,prediction_files_checked=len(checks),bootstrap_scope='5000 paired-user draws conditional on each fitted pair; excludes pipeline seed uncertainty, target clustering, multiplicity and repeated holdout inspection.',diagnostic_scope='13 sampled steps per method, 39 pair cosines on last shared encoder weight; not all updates or full network. PCGrad counts are instantaneous at logged probes only.',logging_note='Dedicated mtl-learning-curves.json was not exported; all 52 MTL events recovered from preserved worker stdout.')
 for name,data in [('summary',result),('verification',checks),('mtl-learning-curves',mtl),('training-curves',curves)]:
  (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 with (out/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,4))
 for mode in ['sum','uw','gradnorm','pcgrad']:
  c=curves[mode];v=[r for r in c if 'validation'in r];axes[0].plot([r['epoch']for r in v],[r['validation']['ndcg@10']for r in v],marker='o',label=mode)
  ev=[e for e in mtl if e['mode']==mode];cs=[np.array(e['shared_encoder_gradient_cosines'])[np.triu_indices(3,1)].mean()for e in ev];axes[1].plot([e['step']for e in ev],cs,label=mode)
 for mode in ['uw','gradnorm']:
  ev=[e for e in mtl if e['mode']==mode]
  for i,task in enumerate(['text','poster','plot']):axes[2].plot([e['step']for e in ev],[e['weights'][i]for e in ev],label=f'{mode}:{task}')
 axes[0].set(xlabel='TIGER epoch',ylabel='Validation NDCG@10');axes[1].set(xlabel='AE + RQ optimizer call',ylabel='Mean sampled encoder gradient cosine');axes[2].set(xlabel='AE + RQ optimizer call',ylabel='Task weight')
 for a in axes:a.legend(fontsize=7);a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'mtl-results.png',dpi=170);fig.savefig(out/'mtl-results.pdf')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
