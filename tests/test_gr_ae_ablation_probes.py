"""Single-intervention pairs, reconstruction math and mixed-shape AE workers."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(n) for n in ['numpy','torch','hydra','lightning','transformers','sklearn','scipy','matplotlib'])

class ConfigTests(unittest.TestCase):
 def test_nine_paired_interventions_reuse_controls_without_generator(self):
  from rhpr.gr.ae_warmup_probe import validate_probe,trial_settings
  from rhpr.gr.ae_suite import assignments
  c=json.loads((ROOT/'experiments/amazon_beauty_field_ae_ablation_probes_2t4.json').read_text());validate_probe(c)
  self.assertEqual(len(c['trials']),18);self.assertEqual(len(c['control_diagnostics']),16)
  self.assertEqual({t['variant'] for t in c['trials']},{'latent128','slower_lr','frozen_bn','no_price','cosine_title','title_only','uw_lr003','dwa','gradnorm'})
  for job in assignments(c,2):self.assertEqual(len({t['seed'] for t in job['trials']}),1)
  for t in c['trials']:
   a,k=trial_settings(c,t);self.assertEqual(a['latent_dim'],k['latent_dim']);self.assertEqual(a['snapshot_steps'],[0,100,250,500,1000,2000,3500,5000])
   self.assertEqual(k['codebook_size'],256)
  bad=copy.deepcopy(c);bad['trials'].pop()
  with self.assertRaises(ValueError):validate_probe(bad)
  with self.assertRaises(ValueError):trial_settings(c,{**c['trials'][0],'quantizer_overrides':{'batch_size':32}})

@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
 def test_loss_math_frozen_statistics_and_mixed_shape_workers(self):
  script=r'''
import sys,json,numpy as np,torch
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.fields import prepare_fields,tensor_fields
from rhpr.gr.field_autoencoder import build_field_autoencoder,configure_objective,fit_field_autoencoder
from rhpr.gr.ae_warmup_probe import run_warmup_probe
from rhpr.gr.data import file_sha256
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);torch.set_num_threads(1);source=prepare_grid_source(sys.argv[1],root/'grid')
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(24)]
titles=np.random.default_rng(7).normal(size=(24,8)).astype('float32');schema,arrays=prepare_fields(texts,title_embeddings=titles)
base=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,latent_dim=6,batch_size=8,max_steps=2,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2,snapshot_steps=[0,2],loss_policy={'mode':'fixed_normalized'})
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4,title={'dimension':8});recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
model,_,_=build_field_autoencoder(source,recipe,schema,base,4);fields=tensor_fields(arrays,slice(0,8),'cpu');model.eval();pred=model(fields)[2]
configure_objective(model,arrays,{'mode':'fixed_normalized','title_loss':'cosine'})
loss,parts,_=model.objective(pred,fields)
expected=(1-torch.nn.functional.cosine_similarity(pred['title'],fields['title'],dim=-1)).mean()
torch.testing.assert_close(parts['title'],expected)
mean=torch.as_tensor(arrays['title'].mean(0)).expand_as(torch.as_tensor(arrays['title']))
constant=(1-torch.nn.functional.cosine_similarity(torch.as_tensor(arrays['title']),mean,dim=-1)).mean()
assert abs(model.loss_normalizers['title']-float(constant))<1e-6
configure_objective(model,arrays,{'mode':'fixed_normalized','field_weights':{'price':0.}})
price=pred['price'].detach().clone().requires_grad_();total,parts,_=model.objective({**pred,'price':price},fields)
assert model.task_weights()[2]==0;total.backward();assert price.grad.abs().sum()==0
for bad in [{'mode':'fixed_normalized','field_weights':{'price':-1.}}, {'mode':'fixed_normalized','title_loss':'l1'}, {'mode':'uncertainty','field_weights':{'price':0.}}]:
 try:configure_objective(model,arrays,bad)
 except ValueError:pass
 else:raise AssertionError('invalid objective accepted')
# Freeze running stats, keep affine parameters learning; actual lower LR recorded.
_,r,_=fit_field_autoencoder(texts,rep,{**base,'freeze_batchnorm_running_stats':True,'neural_learning_rate':.0003},42,'cpu',source,recipe,root/'frozen',title_embeddings=titles)
a=torch.load(root/'frozen/checkpoints/step-00000/autoencoder.pt',weights_only=False);b=torch.load(root/'frozen/autoencoder.pt',weights_only=False)
keys=[k for k in a['model'] if any(x in k for x in ['running_mean','running_var','num_batches_tracked'])];assert keys
assert all(torch.equal(a['model'][k],b['model'][k]) for k in keys)
affine=[k for k in a['model'] if k.startswith('normalization_layer') and k.endswith('weight')]
assert any(not torch.equal(a['model'][k],b['model'][k]) for k in affine)
assert r['learning_rate']==.0003
# Paired cosine, UW and larger-latent paths in the real isolated-worker suite.
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=texts[i]) for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(6)]))
p=root/'parent';p.mkdir();np.savez_compressed(p/'catalogue-fields.npz',title=titles,title_observed=np.ones(24,dtype=bool));(p/'run-manifest.json').write_text(json.dumps({'git_sha':'parent'}))
variants=[('wide',{'latent_dim':12},{'mode':'fixed_normalized'}),('cosine',{}, {'mode':'fixed_normalized','title_loss':'cosine'}),('uw',{}, {'mode':'uncertainty','uw_learning_rate':.03})]
trials=[dict(id=f'{v}_s{s}',variant=v,seed=s,quantizer_overrides=o,loss_policy=l) for v,o,l in variants for s in [42,43]]
c=dict(kind='gr_ae_warmup_probe',name='fixture',seed=42,ablation_suite=True,compare_initialization_by_seed=True,compare_initialization_by_shape=True,
 data=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']}),resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parent=dict(kernel='owner/fixture',path=str(p),source_code_sha='parent',files={n:dict(sha256=file_sha256(p/n),size_bytes=(p/n).stat().st_size) for n in ['run-manifest.json','catalogue-fields.npz']}),
 representation=rep,quantizer=base,kmeans=dict(kind='ae_residual_kmeans',num_hierarchies=3,codebook_size=4,latent_dim=6,n_init=1,max_iter=50,tol=1e-4,cpu_threads=1,normalize_latents=False,normalize_residuals=False),training=dict(device='cpu',precision='fp32'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),trials=trials)
out=root/'out';out.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])):
 metrics=validate_metrics(run_warmup_probe(c,out))
assert metrics['trials']==6 and metrics['index_probes']==12 and not metrics['generator_trained']
rows=json.loads((out/'probe-curves.json').read_text());assert len({r['variant'] for r in rows})==3
for r in rows:
 d=out/'trials'/r['trial']/'checkpoints'/f'step-{r["step"]:05d}'
 assert np.load(d/'codebooks.npy').shape==(3,4,r['latent_dim'])
 assert 'mse' in r['reconstruction']['title'] and 'cosine_similarity' in r['reconstruction']['title']
assert (out/'ablation-train_proxy_gain_bits.png').is_file()
'''
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=180)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':unittest.main()
