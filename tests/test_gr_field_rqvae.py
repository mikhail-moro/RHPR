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
    def test_title_comparison_is_matched_3k_quantizer_only(self):
        configs = [json.loads((ROOT / "experiments" / name).read_text()) for name in
            ["amazon_beauty_rqvae_plain_3k_diagnostic.json", "amazon_beauty_field_title_rqvae_3k_diagnostic.json"]]
        for c in configs:
            self.assertEqual(c["pipeline"]["stage"], "quantizer_only")
            self.assertEqual(c["quantizer"]["max_steps"], 3000)
            self.assertEqual(c["quantizer"]["diagnostics_interval_steps"], 500)
        for key in ("seed", "data", "training", "generator"):
            self.assertEqual(configs[0][key], configs[1][key])
        self.assertEqual(configs[1]["representation"]["fields"], ["brand", "categories", "price", "title"])
        self.assertEqual(configs[1]["representation"]["title"]["revision"], configs[0]["representation"]["revision"])

    def test_5k_run_changes_only_budget_and_explicit_diagnostics(self):
        old = json.loads((ROOT / "experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json").read_text())
        new = json.loads((ROOT / "experiments/amazon_beauty_field_rqvae_tiger_5k.json").read_text())
        for key in ("training", "generator", "evaluation", "data", "representation", "seed"):
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["quantizer"], {**old["quantizer"], "max_steps": 5000, "diagnostics_interval_steps": 500})

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
    def test_frozen_title_is_concatenated_directly_and_masked_mse_uses_fixed_target(self):
        import numpy as np
        import torch
        from rhpr.gr.fields import FieldEncoder, FieldHeads, prepare_fields, tensor_fields
        titles = np.random.default_rng(1).normal(size=(3, 768)).astype("float32")
        schema, arrays = prepare_fields(fixture_texts(), title_embeddings=titles, title_observed=[True, True, False])
        encoder = FieldEncoder(schema, 64)
        fields = tensor_fields(arrays, slice(None), "cpu")
        encoded = encoder(fields)
        self.assertEqual(encoded.shape, (3, 960))
        torch.testing.assert_close(encoded[:, -768:], fields["title"])
        heads = FieldHeads(schema, 960)
        predictions = heads(encoded.detach())
        fields["title"].requires_grad_(True)
        losses, counts = heads.losses(predictions, fields)
        self.assertEqual(counts["title"], 2)
        expected = (predictions["title"][:2] - fields["title"][:2].detach()).square().mean()
        torch.testing.assert_close(losses["title"], expected)
        losses["title"].backward()
        self.assertIsNone(fields["title"].grad)
        self.assertIsNotNone(heads.heads["title"].weight.grad)
        with self.assertRaisesRegex(ValueError, "requires frozen title"):
            prepare_fields(fixture_texts(), schema)

    def test_title_encoder_receives_only_titles_and_quantizer_stage_accepts_measured_overflow(self):
        import numpy as np
        from unittest.mock import patch
        from rhpr.gr.rqvae_experiment import _encode_titles, run_rqvae_baseline
        from rhpr.gr.codebook_diagnostics import semantic_statistics
        from rhpr.metrics import validate_metrics
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def encode(config, resources, texts, output, device):
                self.assertEqual(texts, ["ignored", "ignored too", "anything"])
                return np.zeros((3, 768), dtype="float32"), {"seconds": 0.1}
            with patch("rhpr.gr.rqvae_experiment._encode_items", side_effect=encode):
                _, mask, meta = _encode_titles({}, root, fixture_texts(), root, "cpu")
                self.assertTrue(mask.all())
                self.assertEqual(meta["input_scope"], "title_only")
                self.assertTrue(meta["frozen"])
            config = json.loads((ROOT / "experiments/amazon_beauty_field_title_rqvae_3k_diagnostic.json").read_text())
            config["representation"]["fields"] = ["brand", "categories", "price"]
            config["training"]["cpu_threads"] = 1
            codes = np.zeros((257, 3), dtype="int64")
            def fit(*args, **kwargs):
                np.save(root / "semantic-codes.npy", codes)
                return codes, {"steps": 3000, "elapsed_seconds": 1., "used_codes_per_level": [1, 1, 1],
                    "semantic_code_statistics": semantic_statistics(codes, 256)}, {"seconds": 0.1}
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("rhpr.offline_runtime.resolve_dataset_directory", return_value=root), \
                 patch("rhpr.gr.resources.verify_resources", return_value={}), \
                 patch("rhpr.gr.resources.unpack_grid", return_value=root), \
                 patch("rhpr.gr.grid.prepare_grid_source", return_value=root), \
                 patch("rhpr.gr.rqvae_experiment.load_beauty", return_value=(['x']*257, [], list(range(257)), {'items':257})), \
                 patch("rhpr.gr.field_rqvae.fit_field_rqvae", side_effect=fit), \
                 patch("rhpr.gr.grid.build_grid_generator", side_effect=AssertionError("generator constructed")), \
                 patch("rhpr.gr.rqvae_experiment._evaluate", side_effect=AssertionError("test accessed")):
                metrics = validate_metrics(run_rqvae_baseline(config, root))
            self.assertFalse(metrics["generator_trained"])
            self.assertEqual(metrics["test_evaluations"], 0)
            self.assertEqual(metrics["max_collision_group"], 257)
            self.assertFalse(metrics["suffix_representable"])
            self.assertEqual(json.loads((root / "tokenizer-status.json").read_text())["status"], "collision_overflow")

    def test_codebook_occupancy_geometry_and_overflow_do_not_require_resolved_sids(self):
        import numpy as np
        from rhpr.gr.codebook_diagnostics import diagnose_codebooks, semantic_statistics
        # A collapse with overflow in the complete code, not just its prefix.
        codes = np.array([[0, 0], [0, 0], [0, 0], [1, 1]], dtype=np.uint8)
        books = np.array([[[0.], [1.]], [[0.], [1.]]])
        encoded = np.array([[0.], [0.], [0.], [2.]])
        stats = diagnose_codebooks(codes, encoded, books)
        self.assertEqual(stats["semantic_unique_ids"], 2)
        self.assertEqual(stats["max_collision_group"], 3)
        self.assertFalse(stats["suffix_representable"])
        self.assertEqual(stats["prefixes"][-1]["items_in_overflow_groups"], 3)
        self.assertEqual(stats["levels"][0]["counts"], [3, 1])
        self.assertAlmostEqual(stats["levels"][0]["entropy_bits"], 0.8112781244591328)
        self.assertEqual(stats["quantization_mse"], 0)
        self.assertEqual(stats["levels"][0]["residual_mse_after"], .25)
        # Large intermediate groups do not imply complete-SID overflow.
        separated = semantic_statistics(np.array([[0, 0], [0, 0], [0, 1], [0, 1]]), 2)
        self.assertTrue(separated["suffix_representable"])
        with self.assertRaisesRegex(ValueError, "outside codebook"):
            semantic_statistics(np.array([[2, 0]]), 2)

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
    def test_title_native_fit_restores_four_field_losses_and_exact_codes(self):
        script = r'''
import sys,numpy as np,torch
from pathlib import Path
from lightning.pytorch import Trainer
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.field_rqvae import fit_field_rqvae,export_field_codes
from rhpr.gr.rqvae import build_rqvae
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
titles=np.random.default_rng(7).normal(size=(32,8)).astype('float32')
spec=dict(kind='grid_field_rqvae',num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=4,
 init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2)
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4)
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
codes,report,_=fit_field_rqvae(texts,rep,spec,42,'cpu',source,recipe,root/'fit',title_embeddings=titles)
assert report['input_dim']==20 and 'title' in report['field_reconstruction']
assert len(report['field_weights'])==4
state=torch.load(root/'fit/quantizer.pt',map_location='cpu',weights_only=False)
model,_=build_rqvae(source,recipe,20,spec,feature_schema=state['feature_schema'],field_dim=4)
model.load_state_dict(state['model'],strict=True)
for layer,flag in zip(model.quantization_layer_list,state['initialized']):
 layer.is_initialized=flag;layer.is_initial_step=False
model.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
arrays=dict(np.load(root/'fit/catalogue-fields.npz'))
again,embeddings,losses=export_field_codes(model,arrays,'cpu',8)
np.testing.assert_array_equal(again,codes)
np.testing.assert_array_equal(embeddings[:,-8:],titles)
assert np.isfinite(losses['title']['loss']) and losses['title']['observed_items']==32
assert abs(losses['title']['loss']-report['field_reconstruction']['title']['loss'])<1e-8
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-c", script, SOURCE, directory],
                capture_output=True, text=True, timeout=150)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
# Diagnostic probes must leave fit state, BatchNorm, parameters and RNG intact.
from rhpr.gr.field_rqvae import probe_field_codebooks
restored.trainer.state.fn=TrainerFn.FITTING;restored.train();restored.field_heads.heads['brand'].eval()
modes=[m.training for m in restored.modules()]
before={k:v.clone() for k,v in restored.state_dict().items()};rng=torch.get_rng_state().clone()
stats=probe_field_codebooks(restored,arrays,'cpu',8,root/'probe',4)
assert restored.trainer.state.fn==TrainerFn.FITTING
assert modes==[m.training for m in restored.modules()]
assert torch.equal(rng,torch.get_rng_state())
assert all(torch.equal(before[k],v) for k,v in restored.state_dict().items())
assert stats['items']==32 and len(stats['levels'])==3
assert (root/'probe/codebook-snapshots/step-00004.npz').is_file()
# A second identical fit with probes enabled must have the same final weights.
probed_spec={**spec,'diagnostics_interval_steps':1}
probed_codes,probed_report,_=fit_field_rqvae(texts,representation,probed_spec,42,'cpu',source,recipe,root/'probed-fit')
probed_state=torch.load(root/'probed-fit/quantizer.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(state['model'][k],v) for k,v in probed_state['model'].items())
np.testing.assert_array_equal(codes,probed_codes)
curve=json.loads((root/'probed-fit/codebook-learning-curves.json').read_text())
assert curve and curve[-1]['step']==4
loss_curve=json.loads((root/'probed-fit/field-learning-curves.json').read_text())
assert [r['step'] for r in loss_curve]==[4]
components=loss_curve[-1]['loss_components']
assert 'codebook_1' in loss_curve[-1]['gradient_norms']
assert abs(loss_curve[-1]['total_loss']-(components['quantization']*components['quantization_weight']+
    components['weighted_reconstruction']*components['reconstruction_weight']))<1e-5
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
