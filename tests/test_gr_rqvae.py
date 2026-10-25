"""Native RQ-VAE gradients/checkpoint export and the requested fast run contract."""
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
READY = SOURCE and all(importlib.util.find_spec(n) for n in
    ("torch", "numpy", "hydra", "lightning", "transformers"))


class RunContractTests(unittest.TestCase):
    def test_fast_prefix_recipe_restores_early_baseline_without_changing_quantizer(self):
        config = json.loads((ROOT / "experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json").read_text())
        old = json.loads((ROOT / "experiments/amazon_beauty_gr_baseline.json").read_text())
        previous = json.loads((ROOT / "experiments/amazon_beauty_rqvae_tiger_plain_ff_fp16_20epochs.json").read_text())
        self.assertEqual(config["generator"], old["generator"])
        self.assertEqual(config["data"], old["data"])
        self.assertEqual(config["quantizer"], previous["quantizer"])
        self.assertEqual(config["representation"], previous["representation"])
        for field in ("batch_size", "physical_batch_size", "learning_rate", "weight_decay", "grad_clip_norm"):
            self.assertEqual(config["training"][field], old["training"][field])
        self.assertEqual(config["training"]["sampling"], "all_train_prefixes")
        self.assertEqual(config["training"]["precision"], "fp16")
        self.assertEqual(config["training"]["max_epochs"], 20)
        self.assertEqual(config["training"]["min_epochs"], 20)
        self.assertEqual(config["training"]["validation_interval"], 10)
        self.assertFalse(config["training"]["validate_first_epoch"])

    def test_requested_recipe_preserves_old_generator_and_removes_heavy_diagnostics(self):
        config = json.loads((ROOT / "experiments/amazon_beauty_rqvae_tiger_fp16_20epochs.json").read_text())
        old = json.loads((ROOT / "experiments/amazon_beauty_grid_epochs_fp16_b512.json").read_text())
        self.assertEqual(config["generator"], old["generator"])
        self.assertEqual(config["training"]["max_epochs"], 20)
        self.assertEqual(config["training"]["validation_interval_epochs"], 10)
        self.assertEqual(config["training"]["precision"], "fp16")
        self.assertEqual(config["quantizer"]["num_hierarchies"], 3)
        self.assertEqual(config["quantizer"]["codebook_size"], 256)
        self.assertEqual(config["quantizer"]["max_steps"], 3000)
        self.assertEqual(config["representation"]["model"], "sentence-transformers/sentence-t5-base")
        self.assertEqual(config["representation"]["parent_file"], "item-embeddings.npy")
        self.assertNotIn("diagnostics", config)
        self.assertNotIn("distance_profile", config)
        self.assertFalse(config["parent"]["reuse_tokenizer"])


@unittest.skipUnless(READY, "Requires pinned GRID source and GR dependencies")
class NativeRQTests(unittest.TestCase):
    def test_completed_fit_checkpoint_survives_export_failure(self):
        script = r'''
import sys,torch,numpy as np
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import fit_grid_rqvae
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
spec=dict(num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=1,
    init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist')
values=np.random.default_rng(3).normal(size=(32,8)).astype('float32')
with patch('rhpr.gr.rqvae.export_codes',side_effect=RuntimeError('export failed')):
    try:
        fit_grid_rqvae(values,spec,42,'cpu',source,
            Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml',root/'fit')
    except RuntimeError as error:
        assert str(error)=='export failed'
    else:raise AssertionError('export failure hidden')
state=torch.load(root/'fit/quantizer.pt',map_location='cpu',weights_only=False)
assert state['steps']==1 and all(state['initialized'])
assert state['optimizer'] and state['model']
assert not (root/'fit/semantic-codes.npy').exists()
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, folder],
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cuda_fit_export_survives_lightning_teardown(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("Requires CUDA for the actual post-fit device transition")
        script = r'''
import sys,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import fit_grid_rqvae
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
spec=dict(num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=3,
    init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist')
codes,report=fit_grid_rqvae(np.random.default_rng(3).normal(size=(32,8)).astype('float32'),
    spec,42,'cuda:0',source,Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml',root/'fit')
assert codes.shape==(32,3) and report['steps']==3
assert (root/'fit/quantizer.pt').is_file() and (root/'fit/semantic-codes.npy').is_file()
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, folder],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_fit_reload_gradient_flow_and_generator_training(self):
        script = r'''
import json,sys,torch,numpy as np
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.trainer.states import TrainerFn
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.rqvae import fit_grid_rqvae,build_rqvae,export_codes
from rhpr.gr.sid import SidIndex
from rhpr.gr.data import UserSequence
from rhpr.gr.epoch_training import train_epoch_generator
from rhpr.gr.experiment import _evaluate
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
values=np.random.default_rng(3).normal(size=(32,8)).astype('float32')
spec=dict(num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=3,
    init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist')
codes,report=fit_grid_rqvae(values,spec,42,'cpu',source,recipe,root/'fit')
assert codes.shape==(32,3) and report['steps']==3
assert np.isfinite(report['reconstruction_mse_normalized_input'])
state=torch.load(root/'fit/quantizer.pt',map_location='cpu',weights_only=False)
probed_codes,probed_report=fit_grid_rqvae(values,{**spec,'diagnostics_interval_steps':1},42,'cpu',source,recipe,root/'probed')
probed_state=torch.load(root/'probed/quantizer.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(state['model'][key],value) for key,value in probed_state['model'].items())
np.testing.assert_array_equal(codes,probed_codes)
assert (root/'probed/codebook-diagnostics.json').is_file()
restored,resolved=build_rqvae(source,recipe,8,spec)
restored.load_state_dict(state['model'],strict=True)
for layer,flag in zip(restored.quantization_layer_list,state['initialized']):
    layer.is_initialized=flag;layer.is_initial_step=False
restored.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
again,mse=export_codes(restored,values,'cpu',8)
np.testing.assert_array_equal(codes,again)
assert abs(mse-report['reconstruction_mse_normalized_input'])<1e-8
# Retain real gradients to both MLPs and codebooks for later joint experiments.
from src.data.loading.components.interfaces import ItemData
restored.trainer.state.fn=TrainerFn.FITTING
restored.train()
input_embeddings=torch.tensor(values[:8],requires_grad=True)
_,_,quantization,reconstruction=restored.model_step(ItemData(transformed_features={'input_embedding':input_embeddings}))
(quantization+reconstruction).backward()
assert input_embeddings.grad is not None and torch.isfinite(input_embeddings.grad).all()
assert float(input_embeddings.grad.abs().sum())>0
for parameters in [restored.encoder.parameters(),restored.decoder.parameters(),
                   [layer.centroids for layer in restored.quantization_layer_list]]:
    grads=[p.grad for p in parameters if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0
# The very short tiny fit can collide heavily; use an ample suffix vocabulary
# for the generator fixture rather than altering the exported quantizer codes.
index=SidIndex(codes,width=32)
model=build_grid_generator(index.sids,dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,
    encoder_layers=1,decoder_layers=1,dropout=.15,mlp_layers=2,num_user_bins=None,
    separator_token=True,codebook_size=32,beam_width=2,should_check_prefix=False),source_root=source)
users=[UserSequence(i,(i%32,(i+1)%32,(i+2)%32),(i+3)%32,(i+4)%32) for i in range(4)]
config=dict(seed=42,data={'history_items':2},training=dict(max_epochs=2,
    validation_interval_epochs=2,patience_epochs=2,learning_rate=.001,weight_decay=1e-6,
    precision='fp32',evaluation_precision='fp32',evaluation_batch_size=2,max_seconds=60,
    source_user_batch_size=2,batch_size=4,physical_batch_size=4,sequence_length=12,
    max_augmented_batch=4,grad_clip_norm=None))
out=root/'generator';out.mkdir()
selection=train_epoch_generator(model,users,index,config,out,'cpu')
assert selection['epochs']==2 and selection['validation_checks']==1
assert (out/'generator-best.pt').is_file() and (out/'generator-last.pt').is_file()
assert not (out/'tables').exists() and not (out/'catalogue-scores').exists()
# Fit the prefix adapter with the same actual RQ-VAE SID index and GRID model.
from rhpr.gr.rqvae_experiment import _train_rqvae_generator
prefix_config=dict(seed=42,data={'history_items':2},training=dict(
    sampling='all_train_prefixes',max_epochs=2,min_epochs=2,validation_interval=2,
    validate_first_epoch=False,patience_epochs=2,learning_rate=.001,weight_decay=1e-4,
    precision='fp32',evaluation_precision='fp32',evaluation_batch_size=2,max_seconds=60,
    batch_size=4,physical_batch_size=2,grad_clip_norm=1.0))
prefix=root/'prefix-generator';prefix.mkdir()
selected=_train_rqvae_generator(model,users,index,prefix_config,prefix,'cpu')
assert selected['epochs']==2 and selected['validation_checks']==1
assert selected['model_examples_consumed']==16 and selected['updates']==4
assert selected['checkpoint_metric']=='validation_item_ndcg@10'
assert (prefix/'generator-best.pt').is_file() and (prefix/'generator-last.pt').is_file()
print('Native RQ-VAE reload, all component gradients, and generator fit passed')
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, folder],
                capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
