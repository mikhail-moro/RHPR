"""CPU replay of ALS objective ablations and their existing seed42 controls."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(control,channel,objectives,data,cf,out):
 out.mkdir(parents=True,exist_ok=True)
 from rhpr.gr.resources import verify_resources
 from rhpr.gr.ml1m_clean_mtl import validate
 cases={}
 for split,n in [('train',151166),('validation',2786),('test',2786)]:
  with np.load(data/(split+'-cases.npz'))as z:cases[split]=tuple(z[k]for k in ['contexts','user_ids','targets'])
  assert cases[split][0].shape==(n,20)
 configs=[];manifests=[];records=[];scores={};ranks={};checks=[];curves={};diagnostics=[]
 runs=[(control,['sum_seed42'],'20a4c08534cd6853e3fd64340f2dc429e6bed3a6'),(channel,['als_seed42','shuffled_seed42'],'6ecbdaa0b54bd8d353270af203cf6d9d01646a24'),(objectives,['als_gradnorm_seed42','als_no_cf_reconstruction_seed42'],'a102ceac9a374f5c2627cc3c09cb673afdecf1e5')]
 for root,names,commit in runs:
  c=load(root/'experiment.json');validate(c);plan=load(root/'plan.json');manifest=load(root/'run-manifest.json');suite=load(root/'suite-results.json');metrics=load(root/'metrics.json')
  assert c==plan['config']==manifest['config']and manifest['git_sha']==commit and not suite['failures']and metrics['complete']==len(suite['results'])
  for v in metrics.values():assert isinstance(v,(str,int,float,bool))and(not isinstance(v,(int,float))or np.isfinite(v))
  verify_resources(data,c['movie_input']['manifest_sha256'])
  for split,arr in cases.items():assert case_digest(arr)==c['data']['case_sha256'][split]
  assert sha(root/'features/shared-modalities/features.npz')==plan['features_sha256']=='fdb8e49551ba896032b93d97a9be51adeb5a776fd160dc199ac5b874b77198e4'
  if 'collaborative_input'in c:verify_resources(cf,c['collaborative_input']['manifest_sha256'])
  configs.append(c);manifests.append(manifest)
  for name in names:
   p=root/'trials'/name;r=load(p/'trial-result.json');q=load(p/'quantizer.json');curve=load(p/'training-curves.json');sel=r['selection'];curves[name]=curve
   assert r==suite['results'][name]and len(curve)==30 and sel['optimizer_updates']==8880 and sel['amp_skipped_updates']==0 and sel['validation_checks']==6
   assert sel['training_precision']=='fp16'and sel['evaluation_precision']=='fp32'and sel['training_examples_per_epoch']==151166
   assert [x['epoch']for x in curve if 'validation'in x]==[5,10,15,20,25,30]
   best=max((x for x in curve if 'validation'in x),key=lambda x:x['validation']['ndcg@10']);assert sel['best_epoch']==best['epoch']and sel['validation_ndcg10']==best['validation']['ndcg@10']
   assert all(x['updates']==296*x['epoch']and x['train_examples']==151166 for x in curve)
   assert q['steps']==3000 and q['warmup']['steps']==3500 and q['precision']=='fp32'and all(q['warmup'][k]for k in ['codebooks_unchanged','neural_transfer_verified','fresh_rq_optimizer_verified'])
   assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed']['42']
   assert r['initial_state_sha256']==c['expected_initial_state_sha256_by_seed']['42']
   assert load(p/'sid-catalogue.json')['original_item_ids']==plan['source_ids']==load(data/'source-ids.json')
   if name!='sum_seed42':
    prov=load(p/'collaborative-provenance.json');assert prov['factors_sha256']==sha(cf/(prov['variant']+'.npy'))and prov['diagnostics']['no_val_test_read']and prov['diagnostics']['train_case_sha256']==c['data']['case_sha256']['train']
    assert load(p/'modality-normalization.json')['people']['observed_items']==2777
   if name=='als_no_cf_reconstruction_seed42':
    assert q['mtl_final_weights']is None and q['config']['reconstruction_weights']=={'people':0.}and 'mtl_mode'not in q['config']
    assert q['warmup']['reconstruction_weights']['people']==0 and q['active_modalities']==['text','poster','plot','people']
   if name=='als_gradnorm_seed42':assert q['config']['mtl_mode']=='gradnorm'and abs(sum(q['mtl_final_weights'])-4)<1e-5
   codes=np.load(p/'semantic-codes.npy');sid=np.load(p/'semantic-ids.npy');np.testing.assert_array_equal(SidIndex(codes,width=256).sids,sid)
   with np.load(p/'codebooks-final.npz')as z:np.testing.assert_array_equal(z['counts'],np.stack([np.bincount(codes[:,i],minlength=256)for i in range(3)]))
   lookup={tuple(v):i for i,v in enumerate(sid)}
   for split,epoch,expected in [('validation',x['epoch'],x['validation'])for x in curve if 'validation'in x]+[('test',0,r['test'])]:
    f=p/(f'validation-epoch-{epoch:03d}.npz'if split=='validation'else'test-predictions.npz')
    with np.load(f)as z:
     np.testing.assert_array_equal(z['user_ids'],cases[split][1]);np.testing.assert_array_equal(z['target_item_ids'],cases[split][2]);pred=z['semantic_ids'];assert pred.shape==(2786,20,4)and np.isfinite(z['scores']).all()
    decoded=np.array([[lookup.get(tuple(v),-1)for v in row]for row in pred]);hit=(pred==sid[cases[split][2],None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0)
    for k in [5,10]:
     ok=(rank>0)&(rank<=k);ndcg=np.where(ok,1/np.log2(np.maximum(rank,1)+1),0.)
     np.testing.assert_allclose(ndcg.mean(),expected[f'ndcg@{k}'],atol=1e-12,rtol=0);np.testing.assert_allclose(ok.mean(),expected[f'recall@{k}'],atol=1e-12,rtol=0)
    invalid=int((decoded<0).sum());dup=sum(len(v[v>=0])-len(set(v[v>=0]))for v in decoded);assert invalid==expected['invalid_predictions']and dup==expected['duplicate_predictions']
    checks.append(dict(model=name,split=split,epoch=epoch,sha256=sha(f),invalid=invalid,duplicates=dup))
    if split=='test'or epoch==sel['best_epoch']:scores[name,split]=ndcg;ranks[name,split]=rank
   records.append(dict(model=name,validation=sel['validation_ndcg10'],test=r['test']['ndcg@10'],best_epoch=sel['best_epoch'],mean_epoch_seconds=sel['optimizer_training_seconds']/30))
   diagnostics.append(dict(model=name,weights=q['mtl_final_weights'],fixed_weights=q['reconstruction_weights'],mse=q['head_reconstruction_mse'],used_codes=[len(np.unique(codes[:,i]))for i in range(3)],unique_triples=len(np.unique(codes,axis=0)),max_collision=q['semantic_code_statistics']['max_collision_group']))
 for c in configs[1:]:
  for k in ['training','generator','quantizer','data','movie_input','representation','plot_encoder','plot_assets','modern_runtime']:assert c[k]==configs[0][k],k
 rng=np.random.default_rng(20261105);comp=[]
 for a in ['als_gradnorm_seed42','als_no_cf_reconstruction_seed42']:
  for b in ['sum_seed42','als_seed42']:
   for split in ['validation','test']:
    d=scores[a,split]-scores[b,split];boot=[]
    for _ in range(50):boot.extend(d[rng.integers(0,2786,(100,2786))].mean(1).tolist())
    comp.append(dict(model=a,reference=b,split=split,delta=float(d.mean()),relative_pct=float(100*d.mean()/scores[b,split].mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
 result=dict(records=records,comparisons=comp,diagnostics=diagnostics,prediction_files_checked=len(checks),duration_minutes=manifests[-1]['duration_seconds']/60,finished_at=manifests[-1]['finished_at'],content_features_identical=True,all_fits_complete=True,amp_skips=0,scope='One pipeline seed42; conditional user bootstrap excludes training seed/target-item dependence. No new training/inference. No-CF loss does not remove CF inputs; reconstruction total scale also changes.')
 for name,value in [('summary',result),('verification',checks),('training-curves',curves)]: (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 with(out/'users.csv').open('w')as f:
  fields=['split','user_id','target']+[n for _,names,_ in runs for n in names];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for split in ['validation','test']:
   for i,u in enumerate(cases[split][1]):w.writerow(dict(split=split,user_id=int(u),target=int(cases[split][2][i]),**{n:float(scores[n,split][i])for n in fields[3:]}))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(12,4));labels=['Content','ALS sum','Shuffled ALS','ALS GradNorm','ALS lambda0']
 for i,r in enumerate(records):
  curve=[x for x in curves[r['model']]if 'validation'in x];ax[0].plot([x['epoch']for x in curve],[x['validation']['ndcg@10']for x in curve],label=labels[i],marker='.')
  ax[1].bar(i,r['test'],color=f'C{i}');ax[1].text(i,r['test']+.0005,f"{r['test']:.4f}",ha='center',fontsize=8)
 ax[0].set(xlabel='Epoch',ylabel='Validation NDCG@10');ax[0].legend(fontsize=8);ax[1].set(xticks=range(5),xticklabels=labels,ylabel='Selected test NDCG@10',ylim=(0,.075));ax[1].tick_params(axis='x',rotation=20);fig.suptitle('ALS objective ablations — seed42,30epochs / val5');fig.tight_layout();fig.savefig(out/'objective-comparison.png',dpi=180);fig.savefig(out/'objective-comparison.pdf');plt.close(fig)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['control','channel','objectives','data','cf','output']:p.add_argument(name,type=Path)
 a=p.parse_args();analyze(a.control,a.channel,a.objectives,a.data,a.cf,a.output)
