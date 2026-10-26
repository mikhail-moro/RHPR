"""Actual two-process native-RQ fit, GradNorm hook and checkpoint contracts."""
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
READY=SOURCE and all(importlib.util.find_spec(n) for n in ('torch','numpy','hydra','lightning','transformers'))


class ConfigTests(unittest.TestCase):
    def test_two_quantizers_use_completed_ae_objectives_and_frozen_title_parent(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_field_rq_balancing_2t4.json').read_text())
        ae=json.loads((ROOT/'experiments/amazon_beauty_field_ae_nonuw_2t4.json').read_text())
        for key in ['seed','data','parent','execution','training']:
            self.assertEqual(c[key],ae[key])
        self.assertEqual(c['quantizer'],{**ae['quantizer'],'kind':'grid_field_rqvae'})
        self.assertEqual([t['id'] for t in c['trials']],['fixed_normalized','gradnorm'])
        for control in c['ae_controls'].values():
            self.assertEqual(control['initial_state_sha256'],c['expected_initial_state_sha256'])
            self.assertEqual(control['steps'],5000)
        self.assertEqual(c['kind'],'gr_field_rq_suite')


@unittest.skipUnless(READY,'requires native GRID and dependencies')
class NativeTests(unittest.TestCase):
    def test_parallel_rq_modes_keep_quantization_and_gradnorm_reloads_after_real_backward(self):
        script=r'''
import json,sys,numpy as np,torch
from pathlib import Path
from lightning.pytorch import Trainer
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.ae_suite import assignments,launch_workers
from rhpr.gr.study_parallel import atomic_json
from rhpr.gr.field_autoencoder import configure_objective,build_field_autoencoder
from rhpr.gr.field_rqvae import export_field_codes,fit_field_rqvae,probe_field_codebooks
from rhpr.gr.fields import prepare_fields
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.sid_study import state_digest
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid');recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32');schema,arrays=prepare_fields(texts,title_embeddings=titles)
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=t) for i,t in enumerate(texts)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=0,train=[0,1,2],validation=3,test=4)]))
np.savez(root/'titles.npz',title=titles,title_observed=np.ones(32,dtype=bool))
spec=dict(kind='grid_field_rqvae',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=12,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=4)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
torch.manual_seed(42);ae,_,_=build_field_autoencoder(source,recipe,schema,spec,4);expected=state_digest(ae.state_dict())
config=dict(seed=42,training=dict(device='cpu'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),data={},representation=rep,quantizer=spec,expected_initial_state_sha256=expected,
 trials=[dict(id='fixed',loss_policy=dict(mode='fixed_normalized')),dict(id='gradnorm',loss_policy=dict(mode='gradnorm',alpha=1.5,weight_learning_rate=.025))])
output=root/'output';(output/'workers').mkdir(parents=True)
plan=dict(config=config,assignments=assignments(config,0),output=str(output),resources=str(root/'resources'),source=str(source),recipe=str(recipe),parent_fields=str(root/'titles.npz'))
path=output/'workers/plan.json';atomic_json(path,plan);summary=launch_workers(path)
assert not summary['failures'] and len(summary['results'])==2
assert len({v['initial_full_state_sha256'] for v in summary['results'].values()})==1
for name,r in summary['results'].items():
 assert r['initial_state_sha256']==expected and r['steps']==12
 assert r['reconstruction_updates']==12 and r['initialization_only_batches']>0
 assert r['final_batch_loss_components']['quantization']>0
 assert r['final_batch_loss_components']['quantization_weight']==1
 assert r['final_batch_loss_components']['reconstruction_weight']==1
 assert r['semantic_code_statistics']['items']==32
 directory=output/'trials'/name;state=torch.load(directory/'quantizer.pt',map_location='cpu',weights_only=False)
 restored,_=build_rqvae(source,recipe,20,spec,feature_schema=schema,field_dim=4)
 configure_objective(restored,arrays,state['loss_policy']);restored.load_state_dict(state['model'],strict=True)
 for layer,flag in zip(restored.quantization_layer_list,state['initialized']):layer.is_initialized=flag;layer.is_initial_step=False
 restored.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
 codes,embeddings,metrics=export_field_codes(restored,arrays,'cpu',8)
 np.testing.assert_array_equal(codes,np.load(directory/'semantic-codes.npy'))
 assert abs(metrics['title']['loss']-r['field_reconstruction']['title']['loss'])<1e-8
 assert restored.task_weights().detach().tolist()==list(r['field_weights'].values())
 if name=='gradnorm':
  assert not torch.equal(restored.task_balancer.logits.detach(),torch.ones(4))
  assert int(restored.task_balancer.passes)>=1
  assert [g['lr'] for g in r['optimizer_param_groups']]==[.001,.025]
  before={k:v.clone() for k,v in restored.state_dict().items()};rng=torch.get_rng_state().clone()
  stats=probe_field_codebooks(restored,arrays,'cpu',8,directory,12)
  assert all(torch.equal(before[k],v) for k,v in restored.state_dict().items())
  assert torch.equal(rng,torch.get_rng_state())
  # The recorded gradient norm targets prove the weighting hook ran after
  # zero_grad and its gradients survived the native manual backward/step.
  curves=json.loads((directory/'field-learning-curves.json').read_text())
  assert curves[-1]['balancing_diagnostics']['auxiliary_loss']>=0
  assert len(curves[-1]['balancing_diagnostics']['gradient_norms'])==4
bad={**spec,'expected_initial_state_sha256':'wrong','loss_policy':dict(mode='fixed_normalized')}
try:fit_field_rqvae(texts,rep,bad,42,'cpu',source,recipe,root/'bad',title_embeddings=titles)
except ValueError as error:assert 'initial neural weights' in str(error)
else:raise AssertionError('wrong control initialization accepted')
assert not (root/'bad/quantizer.pt').exists()
'''
        with tempfile.TemporaryDirectory() as folder:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,folder],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
