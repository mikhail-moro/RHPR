"""Audit 12 fixed-budget singleton MTL fits on CPU; no model execution."""
import argparse,csv,hashlib,json,collections
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import load_temporal_users,temporal_examples,case_digest
from rhpr.gr.sid import SidIndex

def main():
 p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);p.add_argument('--original',type=Path,required=True);a=p.parse_args();root=a.raw;out=a.output;out.mkdir(parents=True,exist_ok=True);load=lambda p:json.loads(p.read_text())
 c=load(root/'experiment.json');man=load(root/'run-manifest.json');assert man['git_sha']=='35ce22dd1f07aaf02251af7847eb242f91187413' and c==man['config']
 report=load(root/'split-report.json');assert report==load(root/'singleton-input/split-report.json');assert report['case_sha256']==c['data']['case_sha256']
 ids=load(root/'trials/sum_seed42/sid-catalogue.json')['original_item_ids'];users=load_temporal_users(root/'singleton-input/sequences.json',ids)
 cases={s:temporal_examples(users,20,s)for s in ['train','validation','test']};assert len(users)==903 and len(cases['train'][2])==80268
 for s,ex in cases.items():assert case_digest(ex)==report['case_sha256'][s]
 audit=load(root/'singleton-input/singleton-audit.json');assert all(r['validation_group_size']==r['test_group_size']==1 for r in audit['users'])
 assert sum(len(u.test_between)for u in users)==3173
 original=load(a.original/'sequences.json');events={u['user_id']:[*u['train'],u['validation'],u['test']]for u in original}
 for e in load(a.original/'embargoed-events.json'):events[e['user_id']].append([e['movie_id'],e['timestamp']])
 for name,digest in audit['source_files'].items():assert hashlib.sha256((a.original/name).read_bytes()).hexdigest()==digest
 derived=load(root/'singleton-input/sequences.json');independent_kept=[]
 for uid,ev in sorted(events.items()):
  counts=collections.Counter(t for i,t in ev);single=sorted(t for t,n in counts.items()if n==1)
  if len(single)<2:continue
  vt,tt=single[-2:];train=sorted(([i,t]for i,t in ev if t<vt),key=lambda v:(v[1],v[0]))
  if len(set(t for i,t in train))<2:continue
  independent_kept.append(dict(user_id=uid,train=train,validation=next([i,t]for i,t in ev if t==vt),test=next([i,t]for i,t in ev if t==tt),test_between=sorted(([i,t]for i,t in ev if vt<t<tt),key=lambda v:(v[1],v[0]))))
 assert independent_kept==derived

 suite=load(root/'suite-results.json');assert not suite['failures'] and len(suite['results'])==12
 for f in (root/'workers').glob('worker-*.json'):
  w=load(f);assert w['status']=='complete' and not w['failures'] and len(w['results'])==6
 modes=['sum','uw','gradnorm','pcgrad'];seeds=[42,101,202];records=[];checks=[];curves={};scores={};diagnostics=[];ranks={};diagnostic_curves={}
 for seed in seeds:
  for mode in modes:
   key=f'{mode}_seed{seed}';d=root/'trials'/key;r=load(d/'trial-result.json');q=load(d/'quantizer.json');cc=load(d/'training-curves.json');curves[key]=cc;sel=r['selection']
   assert r==suite['results'][key] and sel['training_epochs']==30 and sel['validation_checks']==6 and sel['optimizer_updates']==9420 and sel['amp_skipped_updates']==0
   assert len(cc)==30 and all(v['updates']==314*v['epoch'] and v['train_examples']==80268 for v in cc)
   assert [v['epoch']for v in cc if 'validation'in v]==[5,10,15,20,25,30]
   best=max((v for v in cc if 'validation'in v),key=lambda v:v['validation']['ndcg@10']);assert sel['best_epoch']==best['epoch'] and sel['validation_ndcg10']==best['validation']['ndcg@10']
   assert q['steps']==3000 and q['warmup']['steps']==3500 and q['warmup']['codebooks_unchanged'] and q['warmup']['fresh_rq_optimizer_verified'] and q['warmup']['neural_transfer_verified']
   assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed'][str(seed)]
   assert load(d/'generator.json')['initial_state_sha256']==c['expected_initial_state_sha256_by_seed'][str(seed)]
   assert load(d/'representation.json')['source_features_sha256']==c['modality_input']['features_sha256']
   assert load(d/'sid-catalogue.json')['original_item_ids']==ids
   codes=np.load(d/'semantic-codes.npy');sid=np.load(d/'semantic-ids.npy');np.testing.assert_array_equal(SidIndex(codes,width=256).sids,sid);assert len(np.unique(sid,axis=0))==1273
   with np.load(d/'codebooks-final.npz')as z:np.testing.assert_array_equal(z['counts'],np.stack([np.bincount(codes[:,i],minlength=256)for i in range(3)]))
   lookup={tuple(s):i for i,s in enumerate(sid)}
   def score(path,split,expected):
    with np.load(path)as z:
     np.testing.assert_array_equal(z['user_ids'],cases[split][1]);np.testing.assert_array_equal(z['target_item_ids'],cases[split][2]);pred=z['semantic_ids'];t=z['target_item_ids'];items=np.array([[lookup.get(tuple(s),-1)for s in row]for row in pred])
    hit=(pred==sid[t,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0);values={}
    for k in [5,10]:
     ok=(rank>0)&(rank<=k);values[f'ndcg@{k}']=np.where(ok,1/np.log2(np.maximum(rank,1)+1),0.);values[f'recall@{k}']=ok.astype(float)
    for k,v in values.items():np.testing.assert_allclose(v.mean(),expected[k],atol=1e-12,rtol=0)
    invalid=int((items<0).sum());dup=sum(len(row[row>=0])-len(set(row[row>=0]))for row in items)
    assert invalid==expected['invalid_predictions'] and dup==expected['duplicate_predictions']
    checks.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),invalid=invalid,duplicates=dup))
    return values['ndcg@10'],items
   for v in cc:
    if 'validation'in v:
     sc,items=score(d/f"validation-epoch-{v['epoch']:03d}.npz",'validation',v['validation'])
     if v['epoch']==sel['best_epoch']:scores[mode,seed,'validation']=sc;ranks[mode,seed,'validation']=items
   scores[mode,seed,'test'],ranks[mode,seed,'test']=score(d/'test-predictions.npz','test',r['test'])
   ev=load(d/'mtl-learning-curves.json');assert len(ev)==13;diagnostic_curves[key]=ev;cos=np.array([v['shared_encoder_gradient_cosines']for v in ev])[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]
   diag=dict(mode=mode,seed=seed,weights=q['mtl_final_weights'],encoder_cos_min=float(cos.min()),encoder_cos_median=float(np.median(cos)),encoder_negative_pairs=int((cos<0).sum()),encoder_pair_probes=int(cos.size),projection_totals=q['mtl_projection_totals'],reconstruction_mse=q['head_reconstruction_mse'],unique_triples=len(np.unique(codes,axis=0)),used_codes=[len(np.unique(codes[:,i]))for i in range(3)])
   if mode=='pcgrad':diag.update(projections_at_first_500=ev[0]['total_projections'],projected_updates_at_first_500=ev[0]['projected_updates'])
   diagnostics.append(diag)
   records.append(dict(mode=mode,seed=seed,val=sel['validation_ndcg10'],test=r['test']['ndcg@10'],recall10=r['test']['recall@10'],best_epoch=sel['best_epoch'],mean_epoch_seconds=float(np.mean([v['train_seconds']for v in cc])),tokenizer_seconds=q['fit_seconds']+q['warmup']['seconds']))
 aggregate=[]
 for mode in modes:
  rr=[r for r in records if r['mode']==mode];base=[r for r in records if r['mode']=='sum'];row=dict(mode=mode)
  for split,key in [('val','val'),('test','test')]:
   vv=np.array([r[key]for r in rr]);bb=np.array([r[key]for r in base]);row.update({split+'_mean':float(vv.mean()),split+'_sd':float(vv.std(ddof=1)),split+'_relative':float(vv.mean()/bb.mean()-1),split+'_wins':int((vv>bb).sum()),split+'_relative_by_seed':(vv/bb-1).tolist()})
  row['recall10_mean']=float(np.mean([r['recall10']for r in rr]));aggregate.append(row)
 rng=np.random.default_rng(1110503);comparisons=[]
 for mode in modes[1:]:
  for split in ['validation','test']:
   ds=np.stack([scores[mode,s,split]-scores['sum',s,split]for s in seeds]);delta=ds.mean(0);boot=np.array([rng.choice(delta,len(delta),replace=True).mean()for _ in range(5000)])
   common=np.stack([(scores[mode,s,split]>0)&(scores['sum',s,split]>0)for s in seeds]);gained=np.stack([(scores[mode,s,split]>0)&(scores['sum',s,split]==0)for s in seeds]);lost=np.stack([(scores[mode,s,split]==0)&(scores['sum',s,split]>0)for s in seeds])
   decomposition={k:float(np.where(mask,ds,0).mean())for k,mask in [('common_hits',common),('gained_hits',gained),('lost_hits',lost)]};assert abs(sum(decomposition.values())-delta.mean())<1e-12
   comparisons.append(dict(mode=mode,split=split,ndcg_delta_decomposition=decomposition,delta=float(delta.mean()),conditional_user95=np.quantile(boot,[.025,.975]).tolist(),better_users_mean_seed=int((delta>0).sum()),worse_users_mean_seed=int((delta<0).sum()),ties_users_mean_seed=int((delta==0).sum()),delta_by_seed=ds.mean(1).tolist()))
 fixed=[]
 for e in [5,10,15,20,25,30]:
  for mode in modes:fixed.append(dict(epoch=e,mode=mode,val_mean=float(np.mean([next(v['validation']['ndcg@10']for v in curves[f'{mode}_seed{s}']if v['epoch']==e)for s in seeds]))))
 result=dict(executable=man['git_sha'],run_minutes=man['duration_seconds']/60,users=903,train_events=82234,train_examples=80268,models=records,aggregate=aggregate,diagnostics=diagnostics,comparisons=comparisons,prediction_files_checked=len(checks),fixed_epoch_validation=fixed,scope='3 coupled pipeline seeds on same903users/targets; conditional paired-user CI excludes training/runtime seed uncertainty, target clustering, multiplicity and previous holdout inspection. New singleton cohort is not directly comparable with old scores. Diagnostics probe last shared encoder weight, PCGrad totals count all6500optimizer calls across AE and RQ.')
 for name,data in [('summary',result),('verification',checks),('training-curves',curves),('mtl-curves',diagnostic_curves)]: (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 with (out/'metrics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 rr=[]
 for split in ['validation','test']:
  for j,u in enumerate(users):
   row=dict(split=split,user_id=u.user_id,target=int(cases[split][2][j]))
   for mode in modes:row[mode]=float(np.mean([scores[mode,s,split][j]for s in seeds]))
   rr.append(row)
 with(out/'users.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for mode in modes:
  rr=[r for r in records if r['mode']==mode];axes[0].plot(seeds,[r['test']for r in rr],marker='o',label=mode)
  f=[r for r in fixed if r['mode']==mode];axes[1].plot([r['epoch']for r in f],[r['val_mean']for r in f],marker='o',label=mode)
  for i,s in enumerate(seeds):
   ev=diagnostic_curves[f'{mode}_seed{s}'];axes[2].plot([v['step']for v in ev],[np.mean(np.array(v['shared_encoder_gradient_cosines'])[np.triu_indices(3,1)])for v in ev],color=f'C{modes.index(mode)}',alpha=.4,label=mode if i==0 else None)
 axes[0].set(xlabel='Pipeline seed',ylabel='Selected test NDCG@10',xticks=seeds);axes[1].set(xlabel='TIGER epoch',ylabel='Mean validation NDCG@10');axes[2].set(xlabel='AE+RQ optimizer calls',ylabel='Mean encoder gradient cosine')
 for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'singleton-mtl-results.png',dpi=160);fig.savefig(out/'singleton-mtl-results.pdf')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
