"""Loss-policy interventions and the actual isolated subprocess execution path."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("RHPR_GRID_SOURCE")
READY = SOURCE and all(importlib.util.find_spec(n) for n in ("torch", "numpy", "hydra", "lightning", "transformers"))


class AssignmentTests(unittest.TestCase):
    def test_no_gpu_fallback_and_disjoint_trial_ownership(self):
        from rhpr.gr.ae_suite import assignments
        config = json.loads((ROOT / "experiments/amazon_beauty_field_ae_loss_suite_2t4.json").read_text())
        with self.assertRaisesRegex(RuntimeError, "no hardware fallback"):
            assignments(config, 1)
        jobs = assignments(config, 2)
        self.assertEqual([[t['id'] for t in a['trials']] for a in jobs],
                         [['uw_lr_003', 'fixed_normalized'], ['uw_lr_01', 'title_only']])
        config['trials'][1]['id'] = config['trials'][0]['id']
        with self.assertRaisesRegex(ValueError, "unique safe"):
            assignments(config, 2)


@unittest.skipUnless(READY, "Requires pinned GRID and native dependencies")
class NativeSuiteTests(unittest.TestCase):
    def run_script(self, script, timeout=150):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', script, SOURCE, directory],
                                    capture_output=True, text=True, timeout=timeout)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_loss_math_optimizer_isolation_and_checkpoint_replay(self):
        self.run_script(r'''
import sys,json,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.fields import prepare_fields,tensor_fields
from rhpr.gr.field_autoencoder import build_field_autoencoder,configure_objective,make_autoencoder_optimizer,fit_field_autoencoder,evaluate_autoencoder
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32')
schema,arrays=prepare_fields(texts,title_embeddings=titles)
spec=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=2,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
def build():
 torch.manual_seed(42)
 return build_field_autoencoder(source,recipe,schema,spec,4)
model,factory,_=build();initial={k:v.clone() for k,v in model.state_dict().items()}
configure_objective(model,arrays,dict(mode='uncertainty',uw_learning_rate=.03))
optimizer=make_autoencoder_optimizer(model,factory)
assert [g['lr'] for g in optimizer.param_groups]==[.001,.03]
ids=[id(p) for g in optimizer.param_groups for p in g['params']]
assert len(ids)==len(set(ids))==len(list(model.parameters()))
assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
fields=tensor_fields(arrays,slice(0,8),'cpu')
_,_,pred=model(fields);total,losses,counts=model.objective(pred,fields)
torch.testing.assert_close(total,model.uncertainty(torch.stack(list(losses.values())),torch.tensor([True]*4)))
configure_objective(model,arrays,dict(mode='fixed_normalized',normalizer_floor=1e-6))
assert abs(model.loss_normalizers['brand']-np.log(4))<1e-6
assert abs(model.loss_normalizers['title']-np.square(titles-titles.mean(0)).mean())<1e-7
total,losses,_=model.objective(pred,fields)
torch.testing.assert_close(total,sum(losses[n]/model.loss_normalizers[n] for n in model.field_names))
assert all(not p.requires_grad for p in model.uncertainty.parameters())
configure_objective(model,arrays,dict(mode='title_only'))
model.zero_grad(set_to_none=True);_,_,pred=model(fields);total,losses,_=model.objective(pred,fields)
torch.testing.assert_close(total,losses['title']);total.backward()
for name in ['brand','categories','price']:
 assert all(p.grad is None or not p.grad.any() for p in model.field_heads.heads[name].parameters())
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
assert model.task_weights().tolist()==[0.,0.,0.,1.]
assert all(p.grad is None for p in model.uncertainty.parameters())
for policy in [dict(mode='invalid'),dict(mode='title_only',uw_learning_rate=.1),dict(mode='uncertainty',uw_learning_rate=-1),dict(mode='fixed_normalized',normalizer_floor=0)]:
 try:configure_objective(model,arrays,policy)
 except ValueError:pass
 else:raise AssertionError(policy)
_,report,_=fit_field_autoencoder(texts,rep,{**spec,'loss_policy':dict(mode='fixed_normalized')},42,'cpu',source,recipe,root/'fit',title_embeddings=titles)
state=torch.load(root/'fit/autoencoder.pt',map_location='cpu',weights_only=False)
restored,_,_=build();configure_objective(restored,arrays,state['loss_policy']);restored.load_state_dict(state['model'])
capture={};metrics=evaluate_autoencoder(restored,arrays,'cpu',8,capture=capture)
np.testing.assert_array_equal(np.concatenate(capture['latent']),np.load(root/'fit/latent-embeddings.npy'))
assert not metrics['uncertainty_active'] and metrics['field_weights']==report['field_weights']
assert state['loss_normalizers']==restored.loss_normalizers
''')

    def test_two_processes_four_trials_identical_initialization_and_failure_isolation(self):
        self.run_script(r'''
import sys,json,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.ae_suite import assignments,launch_workers
from rhpr.gr.study_parallel import atomic_json
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
data=root/'resources/data';data.mkdir(parents=True)
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=t) for i,t in enumerate(texts)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=0,train=[0,1,2],validation=3,test=4)]))
np.savez(root/'titles.npz',title=np.random.default_rng(7).normal(size=(32,8)).astype('float32'),title_observed=np.ones(32,dtype=bool))
config=dict(seed=42,training=dict(device='cpu'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),data={},
 representation=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4),
 quantizer=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=2,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2),
 trials=[dict(id='uw',loss_policy=dict(mode='uncertainty',uw_learning_rate=.03)),dict(id='fixed',loss_policy=dict(mode='fixed_normalized')),dict(id='title',loss_policy=dict(mode='title_only')),dict(id='bad',loss_policy=dict(mode='invalid'))])
output=root/'output';(output/'workers').mkdir(parents=True)
plan=dict(config=config,assignments=assignments(config,0),output=str(output),resources=str(root/'resources'),source=str(source),recipe=str(Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'),parent_fields=str(root/'titles.npz'))
path=output/'workers/plan.json';atomic_json(path,plan)
try:launch_workers(path)
except RuntimeError:pass
else:raise AssertionError('invalid trial should fail suite')
summary=json.loads((output/'suite-results.json').read_text())
assert set(summary['results'])=={'uw','fixed','title'} and len(summary['failures'])==1
assert len({r['initial_state_sha256'] for r in summary['results'].values()})==1
assert summary['failures'][0]['trial']=='bad'
assert all((output/f'trials/{name}/autoencoder.pt').is_file() for name in ['uw','fixed','title'])
assert not list(output.rglob('semantic-codes.npy'))
for slot in [0,1]:
 status=json.loads((output/f'workers/worker-{slot}.json').read_text())
 assert status['hardware']=='cpu' and status['status']=='complete'
# Rerun a fresh output, now all four succeed, and exercise success aggregation.
config['trials'][-1]['loss_policy']=dict(mode='uncertainty',uw_learning_rate=.1)
output=root/'success';(output/'workers').mkdir(parents=True)
plan.update(output=str(output),assignments=assignments(config,0));path=output/'workers/plan.json';atomic_json(path,plan)
summary=launch_workers(path)
assert len(summary['results'])==4 and not summary['failures'] and summary['parallel_workers']==2
assert len({r['initial_state_sha256'] for r in summary['results'].values()})==1
assert all(r['steps']==2 and r['quantization_enabled'] is False for r in summary['results'].values())
''', timeout=180)


if __name__ == '__main__':
    unittest.main()
