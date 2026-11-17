"""CPU-only replay and checkpoint audit of the three completed ML1M Last-8 windows."""
import argparse
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex,evaluate_ranked_sids
from rhpr.gr.sid_study import state_digest

torch.set_num_threads(1)
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/tmp/ml1m-last8-stopped-v2'))
parser.add_argument('--output',type=Path,default=Path('docs/gr-ml1m-last8-partial-results.json'))
args=parser.parse_args();root=args.root
c=json.loads((root/'experiment.json').read_text())
raw={s:cases(root/'prepared-cases',s) for s in ('train','validation','test')}
assert {s:case_digest(a) for s,a in raw.items()}==c['expected_raw_case_sha256']
sidfile=root/'frozen-indexes/items-42.npy';sid=np.load(sidfile)
sidsha=hashlib.sha256(sidfile.read_bytes()).hexdigest()
assert sidsha=='80422aa36fbcabbd0c1133a0695547513c1a795f908c72b4ed412e9cc2c2b672'
index=SidIndex.from_full_sids(sid,width=256)
rows={};predictions=0
for w in (8,32,128):
 d=root/f'trials/recent{w}_42';r=json.loads((d/'trial-result.json').read_text())
 assert r['initial_state_sha256']==c['expected_initial_state_sha256'] and r['sid_file_sha256']==sidsha
 assert r['raw_case_sha256']=={s:case_digest(raw[s]) for s in ('train','validation')}
 assert r['test_case_sha256']==case_digest(raw['test'])
 curve=json.loads((d/'training-curves.json').read_text());sel=r['selection'];ep=sel['best_epoch']
 assert [a['epoch'] for a in curve]==list(range(1,sel['training_epochs']+1))
 assert all(a['updates']+a['amp_skipped_updates']==95*a['epoch'] for a in curve)
 best=max((a for a in curve if 'validation' in a),key=lambda a:a['validation']['ndcg@10'])
 assert best['epoch']==ep and best['validation']['ndcg@10']==sel['validation_ndcg10']
 for split,path,expected in [('validation',d/f'validation-epoch-{ep:03d}.npz',best['validation']),
                             ('test',d/'test-best-predictions.npz',r['test']['metrics'])]:
  with np.load(path) as z:
   np.testing.assert_array_equal(z['user_ids'],raw[split][1]);np.testing.assert_array_equal(z['target_item_ids'],raw[split][2])
   pred=z['semantic_ids'];scores=z['scores']
  assert pred.shape==(6040,20,4) and np.isfinite(scores).all()
  actual=evaluate_ranked_sids(index,pred,raw[split][2],cutoffs=(5,10))
  for k,value in actual.items(): assert abs(value-expected[k])<1e-12,(w,split,k)
  predictions+=len(pred)*20
 bestpath=d/'generator-best.pt'
 assert hashlib.sha256(bestpath.read_bytes()).hexdigest()==r['test']['checkpoint_sha256']
 states={}
 for name in ('best','last'):
  ck=torch.load(d/f'generator-{name}.pt',map_location='cpu',weights_only=False)
  assert all(torch.isfinite(v).all() for v in ck['model'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
  if name=='best': assert ck['epoch']==ep
  else:
   assert ck['epoch']==sel['training_epochs']
   assert all(k in ck for k in ('optimizer','scaler','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng'))
  states[name]=dict(epoch=ck['epoch'],model_sha256=state_digest(ck['model']))
  del ck
 rows[str(w)]=dict(validation_ndcg10=sel['validation_ndcg10'],test_ndcg10=r['test']['metrics']['ndcg@10'],
   best_epoch=ep,stopped_epoch=sel['training_epochs'],stopping_reason=sel['stopping_reason'],
   warm_epoch_seconds=r['timing']['warm_epoch_training_median_seconds'],forward_gflops=r['forward']['supported_forward_flops_batch1']/1e9,
   total_training_and_validation_seconds=sel['training_seconds'],checkpoints=states)
unfinished=json.loads((root/'trials/recent256_42/training-curves.json').read_text())
out=dict(kernel='carasik/rhpr-ml1m-last8-windows/2',status='CANCEL_ACKNOWLEDGED',source_commit='5c1af46bddb1a848877acead858ec855e77201ed',
 raw_local_directory=str(root),completed_windows=rows,remaining_windows=[16,64],excluded_window=256,excluded_last_completed_epoch=unfinished[-1]['epoch'],
 sid_file_sha256=sidsha,raw_case_sha256=c['expected_raw_case_sha256'],replayed_selected_prediction_files=6,replayed_paths=predictions,
 checkpoints_finite_and_full_resume_state_present=True,gpu_compile_repair_successful=True,additional_gpu_inference=False,
 note='One generator/index seed; windows16/64 pending. Window256 stopped by user and not resumed. Conditional on current target and timestamp tie protocol.')
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
