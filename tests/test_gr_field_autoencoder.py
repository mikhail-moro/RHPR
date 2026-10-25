"""Continuous control initialization, gradient route and checkpoint contracts."""
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
READY = SOURCE and all(importlib.util.find_spec(name) for name in ("torch", "numpy", "hydra", "lightning", "transformers"))


class ConfigTests(unittest.TestCase):
    def test_only_the_rq_block_is_removed_from_the_5k_control(self):
        original = json.loads((ROOT / "experiments/amazon_beauty_field_title_rqvae_5k_diagnostic.json").read_text())
        control = json.loads((ROOT / "experiments/amazon_beauty_field_title_ae_5k_control.json").read_text())
        for key in ("data", "seed", "representation", "training", "generator", "evaluation"):
            self.assertEqual(control[key], original[key])
        self.assertEqual(control["quantizer"], {**original["quantizer"], "kind": "continuous_field_autoencoder"})
        self.assertEqual(control["pipeline"]["stage"], "continuous_autoencoder")


@unittest.skipUnless(READY, "Requires pinned GRID and native dependencies")
class NativeControlTests(unittest.TestCase):
    def test_runner_exits_before_any_sid_or_generator_work(self):
        from unittest.mock import patch
        import numpy as np
        from rhpr.gr.rqvae_experiment import run_rqvae_baseline
        from rhpr.metrics import validate_metrics
        config = json.loads((ROOT / "experiments/amazon_beauty_field_title_ae_5k_control.json").read_text())
        config["training"]["cpu_threads"] = 1
        report = {"steps": 5000, "elapsed_seconds": 1.,
            "field_reconstruction": {name: {"loss": .5} for name in config["representation"]["fields"]}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(os.environ), patch("torch.cuda.is_available", return_value=True), \
                 patch("rhpr.offline_runtime.resolve_dataset_directory", return_value=root), \
                 patch("rhpr.gr.resources.verify_resources", return_value={}), \
                 patch("rhpr.gr.resources.unpack_grid", return_value=root), \
                 patch("rhpr.gr.grid.prepare_grid_source", return_value=root), \
                 patch("rhpr.gr.rqvae_experiment.load_beauty", return_value=(['x']*2, [], [0,1], {'items':2})), \
                 patch("rhpr.gr.rqvae_experiment._encode_titles", return_value=(np.zeros((2,8),dtype='float32'),np.ones(2,dtype=bool),{'seconds':.1})), \
                 patch("rhpr.gr.field_autoencoder.fit_field_autoencoder", return_value=(None,report,{'seconds':.1})), \
                 patch("rhpr.gr.field_rqvae.fit_field_rqvae", side_effect=AssertionError('RQ fit called')), \
                 patch("rhpr.gr.sid.SidIndex", side_effect=AssertionError('SID constructed')), \
                 patch("rhpr.gr.grid.build_grid_generator", side_effect=AssertionError('generator constructed')), \
                 patch("rhpr.gr.rqvae_experiment._evaluate", side_effect=AssertionError('test accessed')):
                metrics = validate_metrics(run_rqvae_baseline(config,root))
            self.assertEqual(metrics['stage'],'continuous_autoencoder')
            self.assertFalse(metrics['quantization_enabled'])
            self.assertEqual(metrics['test_evaluations'],0)
            self.assertTrue((root/'autoencoder-catalogue.json').is_file())

    def test_neural_weights_match_rq_but_forward_has_no_books_and_reloads_exactly(self):
        script = r'''
import json,sys,numpy as np,torch
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.field_autoencoder import build_field_autoencoder,fit_field_autoencoder,evaluate_autoencoder
from rhpr.gr.fields import prepare_fields,tensor_fields
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32')
schema,arrays=prepare_fields(texts,title_embeddings=titles)
spec=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,batch_size=8,
 max_steps=4,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
torch.manual_seed(42)
native,_=build_rqvae(source,recipe,20,spec,feature_schema=schema,field_dim=4)
torch.manual_seed(42)
model,factory,_=build_field_autoencoder(source,recipe,schema,spec,4)
for key,value in model.state_dict().items():torch.testing.assert_close(value,native.state_dict()[key],rtol=0,atol=0)
assert not hasattr(model,'quantization_layer_list')
assert not any('quantization' in name or 'centroids' in name for name,_ in model.named_parameters())
fields=tensor_fields(arrays,slice(0,8),'cpu')
model.train()
features,latent,predicted=model(fields)
for name,output in predicted.items():
 torch.testing.assert_close(output,model.field_heads(model.decoder(latent))[name])
loss,_,_=model.objective(predicted,fields);loss.backward()
for name in ['feature_encoder','encoder','decoder','field_heads','uncertainty']:
 grads=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads)
 assert sum(float(g.abs().sum()) for g in grads)>0
before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
model.field_heads.heads['title'].eval();modes=[m.training for m in model.modules()]
evaluate_autoencoder(model,arrays,'cpu',8)
assert torch.equal(rng,torch.get_rng_state()) and modes==[m.training for m in model.modules()]
assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
codes,report,_=fit_field_autoencoder(texts,rep,spec,42,'cpu',source,recipe,root/'fit',title_embeddings=titles)
assert codes is None and report['steps']==4 and report['optimizer']=='Adagrad'
assert report['learning_rate']==.001 and report['quantization_enabled'] is False
assert not (root/'fit/semantic-codes.npy').exists() and not (root/'fit/quantizer.pt').exists()
assert [r['step'] for r in json.loads((root/'fit/reconstruction-learning-curves.json').read_text())]==[0,2,4]
state=torch.load(root/'fit/autoencoder.pt',map_location='cpu',weights_only=False)
restored,_,_=build_field_autoencoder(source,recipe,state['feature_schema'],spec,4)
restored.load_state_dict(state['model'],strict=True)
capture={};metrics=evaluate_autoencoder(restored,arrays,'cpu',8,capture=capture)
np.testing.assert_array_equal(np.concatenate(capture['latent']),np.load(root/'fit/latent-embeddings.npy'))
for name in rep['fields']:assert abs(metrics['field_reconstruction'][name]['loss']-report['field_reconstruction'][name]['loss'])<1e-8
assert not torch.equal(state['model']['feature_encoder.brand.weight'],native.state_dict()['feature_encoder.brand.weight'])
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, directory],
                capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
