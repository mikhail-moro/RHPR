"""Independent CPU replay of the fixed-index ML1M SID stability study."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from scripts.research.analyze_mini_transfer import decode,load,sha
from rhpr.gr.generator_stability import validate_completion
from rhpr.gr.ml1m_stability_suite import validate,summarize
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex
parser=argparse.ArgumentParser(description='Independently replay fixed-index ML1M validation predictions on CPU')
for name in ['root','data','parent42','parent101','output']:
 parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args();root=args.root;data=args.data;out=args.output;out.mkdir(parents=True,exist_ok=True)
c=load(root/'experiment.json');validate(c);m=load(root/'metrics.json');manifest=load(root/'run-manifest.json');plan=load(root/'plan.json');suite=load(root/'suite-results.json');refs=load(root/'frozen-indexes/references.json')
assert manifest['git_sha']=='2e3e56f7512150fa44f847c3377760582542cc46' and manifest['config']==plan['config']==c
assert m['completed']==18 and m['indexes']==6 and m['generator_seeds']==3 and m['test_evaluated'] is False and not suite['failures'] and len(suite['results'])==18
assert (root/'summary.md').is_file() and len(plan['assignments'])==2 and all(len(x['trials'])==9 for x in plan['assignments'])
with tempfile.TemporaryDirectory() as t:
 assert summarize(c,suite['results'],t)==m
 assert load(Path(t)/'stability-summary.json')==load(root/'stability-summary.json')
workers=[load(p) for p in sorted((root/'workers').glob('worker-*.json'))]
assert len(workers)==2 and all(len(w['results'])==9 and not w['failures'] for w in workers)
assert {k:v for w in workers for k,v in w['results'].items()}==suite['results']
for split in ['train','validation']:
 with np.load(data/(split+'-cases.npz'),allow_pickle=False) as z: cases=tuple(z[k] for k in ['contexts','user_ids','targets'])
 assert case_digest(cases)==c['data']['case_sha256'][split]
assert sha(args.parent42/'features/shared-modalities/features.npz')==c['feature_sha256']
assert sha(args.parent101/'features/shared-modalities/features.npz')==c['feature_sha256']
checks=[];rows=[];users={};invalid=duplicates=0
for i in c['index_ids']:
 seed=int(i.split('_seed')[1]);ref=refs[i];rp=root/'frozen-indexes'/i
 assert ref==load(rp/'stability-index.json') and ref['features_sha256']==c['feature_sha256'] and ref['index_id']==i and ref['tokenizer_seed']==seed
 p=root/'frozen-indexes'/(i+'-semantic-ids.npy');assert sha(p)==ref['sids_sha256'];sid=np.load(p,allow_pickle=False);assert sid.shape==(2874,4)
 idx=SidIndex.from_full_sids(sid,width=256);diagnostics=idx.diagnostics();diagnostics.pop('collision_suffix_order'); expected_diagnostics=dict(ref['diagnostics']);expected_diagnostics.pop('collision_suffix_order');assert diagnostics==expected_diagnostics
 if i.startswith('multimodal'):
  old=(args.parent42 if seed==42 else args.parent101)/'trials'/f'sum_seed{seed}'/'semantic-ids.npy'
  assert sha(old)==sha(p)==c['parents'][str(seed)]['artifacts'][f'sum_seed{seed}']['semantic-ids.npy']
  assert ref['active_modalities']==['text','poster','plot']
 elif i.startswith('meta_rqvae'):
  q=load(rp/'quantizer.json');assert q['seed']==seed and q['steps']==3000 and q['precision']=='fp32' and q['active_modalities']==['text']
  assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed'][str(seed)]==ref['quantizer_initial_state_sha256']
  assert q['warmup']['steps']==3500 and q['warmup']['codebooks_unchanged'] and q['warmup']['fresh_rq_optimizer_verified'] and q['warmup']['neural_transfer_verified']
 elif i.startswith('meta_kmeans'):
  q=load(rp/'quantizer.json');assert q['levels']==3 and q['codebook_size']==256 and q['seed']==seed and q['n_init']==1 and q['max_iter']==100
 for s in c['generator_seeds']:
  tid=f'narrow_{i}_gen{s}';d=root/'trials'/tid;rec=suite['results'][tid];sel=load(d/'training-selection.json');curve=load(d/'training-curves.json');g=load(d/'generator.json')
  assert load(d/'trial-result.json')==rec and rec['seed']==s and rec['index_id']==i and rec['architecture']=='narrow'
  assert all(sel[k]==v for k,v in rec['selection'].items())
  assert g['seed']==s and g['config']==c['generator'] and g['parameters']==1375040 and g['frozen_index'] and not g['test_evaluated']
  assert g['initial_state_sha256']==rec['initial_state_sha256']==c['initialization_pins']['narrow'][str(s)] and g['sids_sha256']==rec['sids_sha256']==sha(p)
  flags=validate_completion(sel,curve,c);assert all(rec[k]==v for k,v in flags.items())
  for x in curve:
   if 'validation' not in x:continue
   epoch=x['epoch'];vp=d/f'validation-epoch-{epoch:03d}.npz';v,ch=decode(vp,sid,cases,x['validation']);checks.append(dict(index=i,seed=s,epoch=epoch,path=str(vp.relative_to(root)),**ch));invalid+=ch['invalid'];duplicates+=ch['duplicates'];users[f'{tid}_e{epoch}']=v
   np.testing.assert_allclose(v.mean(),rec['epochs'][str(epoch)],atol=1e-12,rtol=0)
  rows.append(dict(index=i,seed=s,stopping_reason=sel['stopping_reason'],stop_epoch=sel['training_epochs'],best_epoch=sel['best_epoch'],epoch30=rec['epochs']['30'],best=sel['validation_ndcg10'],total_seconds=sel['training_seconds'],amp_skips=sel['amp_skipped_updates'],plateau=flags['plateau_reached'],censored=flags['budget_censored']))
assert len(rows)==18 and len(checks)==sum(r['stop_epoch']//10 for r in rows)
np.savez_compressed(out/'user-ndcg.npz',**users)
result=dict(status='verified',source_sha=manifest['git_sha'],duration_seconds=manifest['duration_seconds'],completed=len(rows),prediction_files=len(checks),plateau=sum(r['plateau'] for r in rows),censored=sum(r['censored'] for r in rows),amp_skips=sum(r['amp_skips'] for r in rows),invalid=invalid,duplicates=duplicates,rows=rows,indexes={i:dict(sha256=refs[i]['sids_sha256'],diagnostics=refs[i]['diagnostics']) for i in c['index_ids']})
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');(out/'prediction-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','indexes']},indent=2))
