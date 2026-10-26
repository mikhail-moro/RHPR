"""Latent-size wiring, matched controls and dimension-aware diagnostics."""
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
    def test_only_latent_size_changes_against_completed_rq_controls(self):
        old=json.loads((ROOT/'experiments/amazon_beauty_field_rq_balancing_2t4.json').read_text())
        new=json.loads((ROOT/'experiments/amazon_beauty_field_rq_latent128_2t4.json').read_text())
        for key in ['seed','data','parent','execution','training','representation','trials','kaggle']:
            self.assertEqual(new[key],old[key])
        self.assertEqual(new['quantizer'],{**old['quantizer'],'latent_dim':128})
        self.assertNotIn('expected_initial_state_sha256',new)
        self.assertNotIn('ae_controls',new)
        self.assertEqual(set(new['rq64_controls']),{'fixed_normalized','gradnorm'})
        self.assertTrue(all(r['latent_dim']==64 and r['steps']==5000 for r in new['rq64_controls'].values()))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires NumPy')
class GeometryTests(unittest.TestCase):
    def test_dimensionless_norms_and_relative_errors_have_independent_reference(self):
        import numpy as np
        from rhpr.gr.codebook_diagnostics import diagnose_codebooks
        z=np.array([[1.,1.],[3.,3.]])
        books=np.array([[[0.,0.],[2.,2.]]]);codes=np.array([[0],[1]])
        r=diagnose_codebooks(codes,z,books)
        self.assertEqual(r['latent_dim'],2)
        self.assertAlmostEqual(r['latent_norm_mean_over_sqrt_dim'],2)
        self.assertAlmostEqual(r['quantized_norm_mean_over_sqrt_dim'],1)
        self.assertAlmostEqual(r['quantization_mse_over_zero_predictor_mse'],.2)
        self.assertAlmostEqual(r['quantization_mse_over_mean_predictor_mse'],1)
        degenerate=diagnose_codebooks(codes,np.zeros_like(z),np.zeros_like(books))
        self.assertIsNone(degenerate['quantization_mse_over_zero_predictor_mse'])
        self.assertIsNone(degenerate['quantization_mse_over_mean_predictor_mse'])


@unittest.skipUnless(READY,'requires native GRID and dependencies')
class NativeTests(unittest.TestCase):
    def test_both_128d_objectives_train_all_books_and_reload_exact_codes(self):
        script=r'''
import sys,json,numpy as np,torch
from pathlib import Path
from lightning.pytorch import Trainer
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.field_autoencoder import configure_objective
from rhpr.gr.field_rqvae import fit_field_rqvae,export_field_codes,probe_field_codebooks
from rhpr.gr.fields import prepare_fields
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid');recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32');schema,arrays=prepare_fields(texts,title_embeddings=titles)
spec=dict(kind='grid_field_rqvae',num_hierarchies=3,codebook_size=4,latent_dim=128,batch_size=8,max_steps=12,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=4)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
hashes=set();full_hashes=set()
for mode in ['fixed_normalized','gradnorm']:
 out=root/mode;codes,report,_=fit_field_rqvae(texts,rep,{**spec,'loss_policy':dict(mode=mode)},42,'cpu',source,recipe,out,title_embeddings=titles)
 assert report['latent_dim']==128 and report['input_dim']==20 and report['reconstruction_updates']==12
 assert report['resolved_model']['encoder']['output_dim']==128
 assert report['resolved_model']['decoder']['input_dim']==128
 assert report['resolved_model']['quantization_layer']['n_features']==128
 assert report['resolved_model']['quantization_layer']['initializer']['clustering_module']['n_features']==128
 hashes.add(report['initial_state_sha256']);full_hashes.add(report['initial_full_state_sha256'])
 state=torch.load(out/'quantizer.pt',map_location='cpu',weights_only=False)
 restored,_=build_rqvae(source,recipe,20,spec,feature_schema=schema,field_dim=4)
 configure_objective(restored,arrays,state['loss_policy']);restored.load_state_dict(state['model'],strict=True)
 for book,flag in zip(restored.quantization_layer_list,state['initialized']):
  assert tuple(book.centroids.shape)==(4,128)
  book.is_initialized=flag;book.is_initial_step=False
 restored.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
 replay,_,_=export_field_codes(restored,arrays,'cpu',8);np.testing.assert_array_equal(codes,replay)
 stats=probe_field_codebooks(restored,arrays,'cpu',8,out,12)
 assert stats['latent_dim']==128 and np.isfinite(stats['quantization_mse_over_zero_predictor_mse'])
 assert abs(stats['latent_norm_mean']/np.sqrt(128)-stats['latent_norm_mean_over_sqrt_dim'])<1e-8
 if mode=='gradnorm':assert not torch.equal(restored.task_balancer.logits.detach(),torch.ones(4))
assert len(hashes)==len(full_hashes)==1
for invalid in [0,-1,True,1.5,'128']:
 try:build_rqvae(source,recipe,20,{**spec,'latent_dim':invalid},feature_schema=schema,field_dim=4)
 except ValueError as e:assert 'latent_dim' in str(e)
 else:raise AssertionError(invalid)
'''
        with tempfile.TemporaryDirectory() as folder:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,folder],capture_output=True,text=True,timeout=150)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
