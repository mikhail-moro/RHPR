"""Formula fixtures, isolated weight gradients and adaptive checkpoint replay."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = os.environ.get('RHPR_GRID_SOURCE')
READY = SOURCE and all(importlib.util.find_spec(n) for n in ('torch','numpy','hydra','lightning','transformers'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class FormulaTests(unittest.TestCase):
    def test_dwa_softmax_and_gradnorm_stopped_target_match_libmtl_formulas(self):
        import torch
        from rhpr.gr.vendor.libmtl_balancing import dwa_weights, gradnorm_objective
        weights=dwa_weights(torch.tensor([2.,4.]),torch.tensor([1.,4.]),2.)
        torch.testing.assert_close(weights,2*torch.softmax(torch.tensor([.25,.5]),dim=0))
        logits=torch.zeros(2,requires_grad=True)
        loss,targets=gradnorm_objective(2*logits.softmax(0),torch.tensor([1.,3.]),torch.ones(2))
        torch.testing.assert_close(targets,torch.tensor([2.,2.]))
        self.assertFalse(targets.requires_grad)
        loss.backward();torch.testing.assert_close(logits.grad,torch.tensor([-2.,2.]))

    def test_full_pass_observed_means_and_no_neural_gradient_pollution(self):
        import torch
        from rhpr.gr.vendor.libmtl_balancing import TaskBalancer
        bal=TaskBalancer(2,dict(mode='gradnorm',alpha=1.5))
        bal.observe(torch.tensor([2.,4.]),[2,1]);bal.observe(torch.tensor([4.,2.]),[1,2]);bal.finish_pass()
        torch.testing.assert_close(bal.reference,torch.tensor([8/3,8/3]))
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        values=torch.stack([p.square().sum(),p.sum()])
        bal.backward_weights(values,p,[2,2])
        self.assertIsNone(p.grad)
        self.assertIsNotNone(bal.logits.grad)
        self.assertTrue(torch.isfinite(bal.logits.grad).all())
        total=(values*bal.multipliers().detach()).sum();total.backward()
        torch.testing.assert_close(p.grad,torch.tensor([3.,5.]))
        bal.logits.grad=None;bal.backward_weights(torch.stack([p.sum(),p.sum()*0]),p,[2,0])
        self.assertIsNone(bal.logits.grad)
        self.assertTrue(bal.diagnostics['weight_update_skipped_missing_task'])
        dwa=TaskBalancer(2,dict(mode='dwa',temperature=2.))
        dwa.observe(torch.tensor([2.,4.]),[1,1]);dwa.finish_pass()
        torch.testing.assert_close(dwa.multipliers(),torch.ones(2))
        dwa.observe(torch.tensor([1.,4.]),[1,1]);dwa.finish_pass()
        torch.testing.assert_close(dwa.multipliers(),2*torch.softmax(torch.tensor([.25,.5]),dim=0))


@unittest.skipUnless(READY, 'requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_all_new_policies_train_and_adaptive_state_reloads_exactly(self):
        script=r'''
import sys,json,numpy as np,torch
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.fields import prepare_fields
from rhpr.gr.field_autoencoder import fit_field_autoencoder,build_field_autoencoder,configure_objective,evaluate_autoencoder
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid');recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32');schema,arrays=prepare_fields(texts,title_embeddings=titles)
spec=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=12,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=4)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
initial=set()
for mode in ['equal','fixed_sqrt','dwa','gradnorm']:
 output=root/mode;_,report,_=fit_field_autoencoder(texts,rep,{**spec,'loss_policy':dict(mode=mode)},42,'cpu',source,recipe,output,title_embeddings=titles)
 initial.add(report['initial_state_sha256']);assert report['steps']==12 and report['learning_rate']==.001
 assert all(np.isfinite(v['loss']) for v in report['field_reconstruction'].values())
 if mode=='equal':assert set(report['field_weights'].values())=={1.}
 if mode=='fixed_sqrt':assert abs(report['field_weights']['brand']-1/np.sqrt(np.log(4)))<1e-6
 state=torch.load(output/'autoencoder.pt',map_location='cpu',weights_only=False)
 restored,_,_=build_field_autoencoder(source,recipe,schema,spec,4)
 configure_objective(restored,arrays,state['loss_policy']);restored.load_state_dict(state['model'],strict=True)
 capture={};metrics=evaluate_autoencoder(restored,arrays,'cpu',8,capture=capture)
 np.testing.assert_array_equal(np.concatenate(capture['latent']),np.load(output/'latent-embeddings.npy'))
 assert metrics['field_weights']==report['field_weights']
 if mode in ['dwa','gradnorm']:
  bal=restored.task_balancer;assert int(bal.passes)==3 and bool((bal.reference>0).all())
  assert not torch.allclose(bal.multipliers(),torch.ones(4))
  curves=json.loads((output/'field-learning-curves.json').read_text())
  assert curves[-1]['balancing_diagnostics']['complete_catalogue_passes']==3
 if mode=='gradnorm':
  assert [g['lr'] for g in report['optimizer_param_groups']]==[.001,.025]
  assert any(k.startswith('task_balancer.logits') for k in state['model'])
  assert all(torch.isfinite(v).all() for v in state['model'].values())
assert len(initial)==1
'''
        with tempfile.TemporaryDirectory() as folder:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,folder],capture_output=True,text=True,timeout=150)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':
    unittest.main()
