"""Field targets, missingness, joint gradients and native checkpoint replay."""
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
HAS_TORCH = all(importlib.util.find_spec(name) for name in ("torch", "numpy"))
NATIVE_READY = SOURCE and HAS_TORCH and all(importlib.util.find_spec(name) for name in
    ("hydra", "lightning", "transformers"))


def fixture_texts():
    return ["Title: ignored; Brand: A; Categories: ['Beauty', 'Skin', 'Skin']; Price: 10; ",
            "Title: ignored too; Brand: B; Categories: ['Beauty', 'Hair']; Price: 20; ",
            "Title: anything; Brand: Unknown; Categories: []; Price: Unknown; "]


class ConfigurationTests(unittest.TestCase):
    def test_generator_and_training_remain_controlled(self):
        baseline = json.loads((ROOT / "experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json").read_text())
        config = json.loads((ROOT / "experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json").read_text())
        for key in ("training", "generator", "evaluation", "data"):
            self.assertEqual(config[key], baseline[key])
        self.assertEqual(config["representation"]["fields"], ["brand", "categories", "price"])
        self.assertNotIn("parent", config)
        self.assertEqual(config["kaggle"]["kernel_sources"], [])
        self.assertEqual(config["quantizer"], {**baseline["quantizer"], "kind": "grid_field_rqvae"})


@unittest.skipUnless(HAS_TORCH, "Requires PyTorch and NumPy")
class FieldTests(unittest.TestCase):
    def test_catalogue_scaling_deduplication_missingness_and_schema_replay(self):
        import numpy as np
        from rhpr.gr.fields import prepare_fields
        schema, arrays = prepare_fields(fixture_texts())
        self.assertEqual(schema["price_mean"], 15)
        self.assertEqual(schema["price_scale"], 5)
        np.testing.assert_array_equal(arrays["price"].ravel(), [-1, 1, 0])
        np.testing.assert_array_equal(arrays["brand"], [1, 2, 0])
        self.assertEqual(arrays["categories"][0].sum(), 2)
        self.assertFalse(arrays["categories_observed"][2])
        self.assertFalse(arrays["price_observed"][2])
        _, replay = prepare_fields(fixture_texts()[::-1], schema)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key][::-1], replay[key])
        changed = [text.replace("Title: ignored", "Title: completely different") for text in fixture_texts()]
        _, after_title_change = prepare_fields(changed, schema)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], after_title_change[key])

    def test_missing_targets_cannot_change_loss_and_multilabel_pool_is_a_mean(self):
        import torch
        from rhpr.gr.fields import FieldEncoder, FieldHeads, prepare_fields, tensor_fields
        schema, arrays = prepare_fields(fixture_texts())
        fields = tensor_fields(arrays, slice(None), "cpu")
        encoder = FieldEncoder(schema, 4)
        captured = []
        hook = encoder.projections["categories"].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        encoded = encoder(fields)
        hook.remove()
        expected = encoder.categories.weight[[schema["category_vocabulary"].index(x) for x in ("Beauty", "Skin")]].mean(0)
        torch.testing.assert_close(captured[0][0, :4], expected)
        heads = FieldHeads(schema, 12)
        predictions = heads(encoded)
        losses, counts = heads.losses(predictions, fields)
        self.assertEqual(counts, {"brand": 2, "categories": 2, "price": 2})
        altered = {name: value.clone() for name, value in predictions.items()}
        for value in altered.values():
            value[2] = 1e6
        other, _ = heads.losses(altered, fields)
        for name in losses:
            torch.testing.assert_close(losses[name], other[name])

    def test_uncertainty_matches_libmtl_and_absent_task_has_zero_gradient(self):
        import torch
        from rhpr.gr.vendor.libmtl_uw import UncertaintyWeighting
        weighting = UncertaintyWeighting(3)
        losses = torch.tensor([2., 3., 4.], requires_grad=True)
        active = torch.tensor([True, False, True])
        value = weighting(losses, active)
        expected = (losses / (2 * weighting.loss_scale.exp()) + weighting.loss_scale / 2)[active].sum()
        torch.testing.assert_close(value, expected)
        value.backward()
        self.assertEqual(float(weighting.loss_scale.grad[1]), 0)
        self.assertEqual(float(losses.grad[1]), 0)
        self.assertTrue(torch.isfinite(weighting.loss_scale.grad).all())


@unittest.skipUnless(NATIVE_READY, "Requires pinned GRID source and dependencies")
class NativeJointTests(unittest.TestCase):
    def test_joint_fit_reload_original_fields_and_all_component_gradients(self):
        script = r'''
import json,sys,numpy as np,torch
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.trainer.states import TrainerFn
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.field_rqvae import fit_field_rqvae,export_field_codes
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.fields import prepare_fields,tensor_fields
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
texts=[f"Title: excluded{i}; Brand: B{i%4}; Categories: ['Beauty', 'C{i%3}']; Price: {i+1}; " for i in range(32)]
spec=dict(kind='grid_field_rqvae',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=4,
    init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist')
representation=dict(kind='field_encoder',fields=['brand','categories','price'],field_dim=4)
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
codes,report,rep=fit_field_rqvae(texts,representation,spec,42,'cpu',source,recipe,root/'fit')
assert codes.shape==(32,3) and report['steps']==4
assert report['input_dim']==12 and rep['shape']==[32,12]
assert all(np.isfinite(v['loss']) for v in report['field_reconstruction'].values())
state=torch.load(root/'fit/quantizer.pt',map_location='cpu',weights_only=False)
restored,resolved=build_rqvae(source,recipe,12,spec,feature_schema=state['feature_schema'],field_dim=4)
restored.load_state_dict(state['model'],strict=True)
for layer,flag in zip(restored.quantization_layer_list,state['initialized']):
    layer.is_initialized=flag;layer.is_initial_step=False
restored.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
from src.data.loading.components.interfaces import ItemData
schema,arrays=prepare_fields(texts,state['feature_schema'])
again,embeddings,losses=export_field_codes(restored,arrays,'cpu',8)
np.testing.assert_array_equal(codes,again)
np.testing.assert_array_equal(embeddings,np.load(root/'fit/item-embeddings.npy'))
for name in losses:assert abs(losses[name]['loss']-report['field_reconstruction'][name]['loss'])<1e-8
restored.trainer.state.fn=TrainerFn.FITTING;restored.train();restored.zero_grad()
fields=tensor_fields(arrays,slice(0,8),'cpu')
_,_,quantization,reconstruction=restored.model_step(ItemData(transformed_features=fields))
(quantization+reconstruction).backward()
for module in [restored.feature_encoder.brand,restored.feature_encoder.categories,
    *restored.feature_encoder.projections.values(),restored.encoder,restored.decoder,
    *restored.field_heads.heads.values(),restored.uncertainty]:
    grads=[p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads),type(module)
    assert sum(float(g.abs().sum()) for g in grads)>0,type(module)
for layer in restored.quantization_layer_list:
    assert layer.centroids.grad is not None and torch.isfinite(layer.centroids.grad).all()
    assert float(layer.centroids.grad.abs().sum())>0
# Field reconstruction alone reaches both the learned feature encoder and RQ encoder.
restored.zero_grad()
_,_,_,reconstruction=restored.model_step(ItemData(transformed_features=fields))
reconstruction.backward()
assert float(restored.feature_encoder.brand.weight.grad.abs().sum())>0
assert sum(float(p.grad.abs().sum()) for p in restored.encoder.parameters())>0
print('joint field encoder/native RQ-VAE checkpoint and gradients passed')
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, directory],
                capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
