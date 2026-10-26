"""Frozen probe identity guards and two actual TIGER workers without refitting."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(n) for n in ['numpy','torch','hydra','lightning','transformers'])

class ConfigTests(unittest.TestCase):
 def test_two_final_gradnorm_indices_preserve_completed_generator_protocol(self):
  from rhpr.gr.generator_suite import assignments
  c=json.loads((ROOT/'experiments/amazon_beauty_field_gradnorm_tiger_2t4.json').read_text());old=json.loads((ROOT/'experiments/amazon_beauty_field_ae_rqkmeans_tiger_2t4.json').read_text())
  for k in ['seed','data','generator','training','evaluation','execution','expected_initial_state_sha256','expected_train_cases_sha256']:self.assertEqual(c[k],old[k])
  self.assertEqual(c['quantizer']['kind'],'frozen_ae_rqkmeans');self.assertEqual(len(assignments(c,2)),2)
  self.assertEqual([t['quantizer_seed'] for t in c['trials']],[42,43])
  for t in c['trials']:
   self.assertEqual(t['expected_variant'],'gradnorm');self.assertEqual(t['ae_steps'],5000)
   for k in ['semantic_codes','codebooks','kmeans_report','ae_report']:self.assertIn('sha256',t[k])
  self.assertIn('3500',c['controls']['budget_note'])

@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
 def test_saved_probe_to_two_workers_and_identity_rejection_without_kmeans_or_ae_fit(self):
  script=r'''
import sys,json,numpy as np
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.generator_suite import run_generator_suite,prepare_indices
from rhpr.gr.data import file_sha256
from rhpr.gr.sid import SidIndex
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(3)]))
data_spec=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
parent=root/'parent';parent.mkdir();q=dict(kind='ae_residual_kmeans',num_hierarchies=3,codebook_size=8,latent_dim=6,n_init=1,max_iter=300,tol=1e-4,cpu_threads=1,normalize_latents=False,normalize_residuals=False)
policy=dict(mode='gradnorm',alpha=1.5,weight_learning_rate=.025)
source_trials=[dict(id=f'gradnorm_s{s}',variant='gradnorm',seed=s,loss_policy=policy) for s in [42,43]]
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='parent',config=dict(data=data_spec,kmeans=q,trials=source_trials))))
(parent/'input-provenance.json').write_text(json.dumps(dict(original_item_ids=list(range(24)))))
def pin(p):return dict(path=p.relative_to(parent).as_posix(),sha256=file_sha256(p),size_bytes=p.stat().st_size)
trials=[]
for seed in [42,43]:
 d=parent/f'trials/gradnorm_s{seed}/checkpoints/step-05000';d.mkdir(parents=True)
 codes=np.stack([np.arange(24)%8,np.arange(24)//8,np.full(24,seed%2)],1)
 np.save(d/'semantic-codes.npy',codes);np.save(d/'codebooks.npy',np.random.default_rng(seed).normal(size=(3,8,6)).astype('float32'))
 diag=SidIndex(codes,width=8).diagnostics()
 (d/'kmeans.json').write_text(json.dumps(dict(seed=seed,kind='ae_residual_kmeans',spec=q,diagnostics=dict(items=24,semantic_unique_ids=diag['semantic_unique_ids'],max_collision_group=diag['max_collision_group']))))
 (d/'autoencoder.json').write_text(json.dumps(dict(seed=seed,quantization_enabled=False,steps=5000,latent_dim=6,checkpoint_sha256=f'ae{seed}',loss_policy=policy)))
 t=dict(id=f'gradnorm_s{seed}',parent_trial=f'gradnorm_s{seed}',expected_variant='gradnorm',quantizer_seed=seed,freeze_policy='ae',ae_steps=5000,ae_checkpoint_sha256=f'ae{seed}')
 for k,n in [('semantic_codes','semantic-codes.npy'),('codebooks','codebooks.npy'),('kmeans_report','kmeans.json'),('ae_report','autoencoder.json')]:t[k]=pin(d/n)
 trials.append(t)
c=dict(seed=42,name='fixture',kind='gr_frozen_sid_generator_suite',data=data_spec,resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='parent',files={n:{k:v for k,v in pin(parent/n).items() if k!='path'} for n in ['run-manifest.json','input-provenance.json']}),quantizer={**q,'kind':'frozen_ae_rqkmeans'},
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,patience=2,patience_epochs=2,validation_interval=2,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),trials=trials,controls={'reused':True})
out=root/'out';out.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.ae_rqkmeans.fit_residual_kmeans',side_effect=AssertionError('KMeans fit called')),patch('rhpr.gr.field_autoencoder.fit_field_autoencoder',side_effect=AssertionError('AE fit called')):
 m=validate_metrics(run_generator_suite(c,out))
assert m['runs']==2 and not m['quantizer_retrained'] and not m['autoencoder_retrained']
suite=json.loads((out/'suite-results.json').read_text());assert not suite['failures']
for t in trials:
 d=out/'indices'/t['id'];origin=json.loads((d/'index-provenance.json').read_text());assert origin['fit_method']=='reuse_saved_probe_codes_and_books' and not origin['quantizer_retrained']
 assert file_sha256(d/'codebooks.npy')==t['codebooks']['sha256'];assert file_sha256(d/'semantic-codes.npy')==t['semantic_codes']['sha256']
 r=suite['results'][t['id']];assert r['selection']['epochs']==2 and r['test_evaluations']==1 and not r['quantizer_retrained']
 assert r['selection']['model_examples_consumed']==18
 assert (out/'trials'/t['id']/'generator-best.pt').is_file()
assert not list(out.rglob('autoencoder.pt')) and not list(out.rglob('quantizer.pt'))
for key,value in [('ae_steps',3500),('expected_variant','title_only'),('ae_checkpoint_sha256','wrong')]:
 bad={**c,'trials':[{**trials[0],key:value}]}
 try:prepare_indices(bad,parent,list(range(24)),root/key)
 except ValueError as e:assert 'probe identity' in str(e)
 else:raise AssertionError('wrong candidate accepted')
'''
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=180);self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':unittest.main()
