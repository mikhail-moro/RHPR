from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


FEATURES = ("conv4", "output_gate", "skip_d")


@unittest.skipIf(torch is None, "Torch is required")
class QueryMambaModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def compressor(self, feature):
        from rhpr.sasrec.query_mamba_models import MambaFeatureQuerySSMCompressor
        return MambaFeatureQuerySSMCompressor(d_model=8, summary_tokens=2,
            tail_size=6, query_context_size=4, n_heads=2, d_inner=4,
            d_state=3, gate_rank=3, dropout=0., dt_max=.01,
            query_source="old_suffix", feature=feature).eval()

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0, msg=name)

    def test_ragged_summaries_are_causal_ignore_padding_and_train_every_new_parameter(self):
        from rhpr.sasrec.query_proxy_protocol import hpo_optimizer
        torch.manual_seed(972)
        values = torch.randn(3, 21, 8)
        values[:, 2] = 0  # Observed OOV events remain valid sequence positions.
        lengths = torch.tensor([17, 13, 6])
        mask = torch.arange(21)[None, :] < lengths[:, None]
        changed = values.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1000
        # Each row's complete supervised raw tail must be invisible to summaries.
        for row, length in enumerate(lengths.tolist()):
            changed[row, max(0, length - 6):length] = torch.randn_like(
                changed[row, max(0, length - 6):length]) * 100
        extra_names = {"conv4": ("history_conv.weight", "history_conv.bias"),
                       "output_gate": ("output_gate.weight", "output_gate.bias"),
                       "skip_d": ("D",)}
        for feature in FEATURES:
            with self.subTest(feature=feature):
                compressor = self.compressor(feature)
                inputs = values.clone().requires_grad_()
                result, occupied, sources = compressor(inputs, mask)
                altered = compressor(changed, mask)
                torch.testing.assert_close(result[:2, :2], altered[0][:2, :2], rtol=0, atol=0)
                self.assertEqual(occupied.sum(dim=1).tolist(), [8, 8, 6])
                self.assertEqual(sources[:2, :2].tolist(), [[10, 10], [6, 6]])
                self.assertEqual(int(result[~occupied].count_nonzero()), 0)
                for row, length in enumerate(lengths.tolist()):
                    solo = compressor(inputs[row:row + 1, :length], mask[row:row + 1, :length])
                    width = int(occupied[row].sum())
                    torch.testing.assert_close(result[row, :width], solo[0][0], rtol=3e-5, atol=3e-6)
                loss = (result[:2, :2].square() * torch.linspace(.3, 1.7, 8)).sum()
                loss.backward()
                self.assertGreater(float(inputs.grad[:2, 2].abs().sum()), 0)
                for row, length in enumerate(lengths.tolist()):
                    old = max(0, length - 6)
                    self.assertEqual(int(inputs.grad[row, old:].count_nonzero()), 0)
                parameters = dict(compressor.named_parameters())
                before = {name: parameters[name].detach().clone() for name in extra_names[feature]}
                for name in extra_names[feature]:
                    gradient = parameters[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(bool(torch.isfinite(gradient).all()), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)
                # The unchanged optimizer convention must own the new modules.
                model = torch.nn.Module()
                model.backbone = torch.nn.Linear(8, 8)
                model.compressor = compressor
                optimizer = hpo_optimizer(model, {"learning_rate": .001, "weight_decay": 1e-5},
                    {"backbone_lr": .001, "compressor_lr": .001})
                owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
                for name in extra_names[feature]:
                    self.assertIn(id(parameters[name]), owned)
                optimizer.step()
                for name, expected in before.items():
                    self.assertFalse(torch.equal(expected, parameters[name]), name)
                self.assertEqual(compressor.last_scan_backend, "cpu_reference")

    def test_conv_has_no_lookahead_and_feeds_both_history_projections(self):
        torch.manual_seed(971)
        compressor = self.compressor("conv4")
        memory = torch.randn(2, 10, 8)
        queries = torch.randn(2, 2, 8)
        changed = memory.clone()
        changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100
        before = compressor.history_features(memory)
        after = compressor.history_features(changed)
        torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)
        # Explicit left-padded kernel establishes alignment at the first event.
        explicit = torch.nn.functional.silu(torch.nn.functional.conv1d(
            torch.nn.functional.pad(memory.transpose(1, 2), (3, 0)),
            compressor.history_conv.weight, compressor.history_conv.bias,
            groups=8)).transpose(1, 2)
        torch.testing.assert_close(before, explicit, rtol=0, atol=0)
        observed = {}
        handles = [module.register_forward_pre_hook(
            lambda _module, args, name=name: observed.update({name: args[0].detach().clone()}))
            for name, module in (("values", compressor.value_projection),
                                 ("gates", compressor.input_gate))]
        try:
            compressor.build_scan_inputs(memory, queries)
        finally:
            for handle in handles:
                handle.remove()
        for projected in observed.values():
            torch.testing.assert_close(projected, before, rtol=0, atol=0)
        self.assertEqual(set(observed), {"values", "gates"})

    def test_endpoint_epilogues_match_full_sequence_then_gather_and_gradients(self):
        torch.manual_seed(973)
        endpoints = torch.tensor([6, 3])
        rows = torch.arange(2)
        for feature in ("output_gate", "skip_d"):
            with self.subTest(feature=feature):
                endpoint_model = self.compressor(feature)
                full_model = copy.deepcopy(endpoint_model)
                outputs = torch.randn(2, 2, 7, 4)
                memory = torch.randn(2, 7, 8)
                values = torch.randn(2, 2, 7, 4)
                endpoint_tensors = [tensor.clone().requires_grad_() for tensor in (outputs, memory, values)]
                full_tensors = [tensor.clone().requires_grad_() for tensor in (outputs, memory, values)]
                y, m, u = endpoint_tensors
                actual = endpoint_model.endpoint_features(y[rows, :, endpoints], m[rows, endpoints],
                                                          u[rows, :, endpoints])
                y, m, u = full_tensors
                if feature == "skip_d":
                    complete = y + full_model.D[None, None, None, :] * u
                else:
                    complete = y * torch.nn.functional.silu(full_model.output_gate(m))[:, None, :, :]
                expected = complete[rows, :, endpoints]
                # Endpoint and full-sequence Linear calls can select different
                # GEMM kernels across supported Torch versions. Their FP32
                # operation order need not be bitwise equal.
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                weights = torch.randn_like(actual)
                (actual * weights).square().sum().backward()
                (expected * weights).square().sum().backward()
                for left, right in zip(endpoint_tensors, full_tensors):
                    if left.grad is None or right.grad is None:
                        self.assertIs(left.grad, right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=3e-6, atol=2e-6)
                for name in ("D",) if feature == "skip_d" else ("output_gate.weight", "output_gate.bias"):
                    torch.testing.assert_close(dict(endpoint_model.named_parameters())[name].grad,
                        dict(full_model.named_parameters())[name].grad, rtol=2e-6, atol=2e-6)

    def test_parent_query_generation_and_every_original_initializer_are_unchanged(self):
        from rhpr.sasrec.query_ssm import QuerySSMCompressor
        for feature in FEATURES:
            with self.subTest(feature=feature):
                torch.manual_seed(974)
                expected = QuerySSMCompressor(d_model=8, summary_tokens=2, tail_size=6,
                    query_context_size=4, n_heads=2, d_inner=4, d_state=3,
                    gate_rank=3, dropout=0., dt_max=.01, query_source="old_suffix").eval()
                torch.manual_seed(974)
                model = self.compressor(feature)
                self.assert_state_equal(expected.state_dict(),
                    {name: model.state_dict()[name] for name in expected.state_dict()})
                dt = torch.nn.functional.softplus(model.delta_projection.bias.detach())
                self.assertGreaterEqual(float(dt.min()), .0001 - 1e-9)
                self.assertLessEqual(float(dt.max()), .01 + 1e-8)
                self.assertTrue(model.delta_projection.bias._no_reinit)
                inputs, valid = torch.randn(2, 15, 8), torch.ones(2, 15, dtype=torch.bool)
                before, after = expected._inputs(inputs, valid), model._inputs(inputs, valid)
                torch.testing.assert_close(before[3], after[3], rtol=0, atol=0)
                torch.testing.assert_close(before[5], after[5], rtol=0, atol=0)
                if feature == "skip_d":
                    self.assertEqual(model.D.dtype, torch.float32)
                    torch.testing.assert_close(model.D, torch.ones(4), rtol=0, atol=0)

    def test_factory_preserves_fresh_backbone_shared_compressor_and_following_rng(self):
        from rhpr.sasrec.query_mamba_models import build_mamba_model
        from rhpr.sasrec.query_mamba_protocol import mamba_source_config, mamba_trials
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        config = json.loads((ROOT / "experiments/yambda_query_mamba_ablation.json").read_text())
        variant = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
        expected = build_qssm_model(mamba_source_config(config), variant,
            config["control_history_config"]["selected_source"]["trial"], 20)
        rng = torch.get_rng_state().clone()
        for trial in mamba_trials(config, variant):
            with self.subTest(trial=trial["name"]):
                model = build_mamba_model(config, variant, trial, 20)
                self.assert_state_equal(expected.state_dict(),
                    {name: model.state_dict()[name] for name in expected.state_dict()})
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(model.compressor.query_source, "old_suffix")
                if trial["feature"] == "none":
                    self.assertEqual(set(expected.state_dict()), set(model.state_dict()))
                else:
                    self.assertEqual(model.compressor.feature, trial["feature"])
        with self.assertRaises(ValueError):
            build_mamba_model(config, variant, dict(mamba_trials(config, variant)[1], feature="combined"), 20)

    def test_feature_cost_separates_unhooked_work_from_already_observed_gate_projection(self):
        for feature in FEATURES:
            with self.subTest(feature=feature):
                model = self.compressor(feature)
                cost = model.feature_compute_flops(old_length=10, batch_size=3)
                conv = 2 * 3 * 10 * 8 * 4 if feature == "conv4" else 0
                endpoint = 3 * 2 * 4 * (2 if feature == "skip_d" else 1) if feature != "conv4" else 0
                self.assertEqual(cost["unhooked_feature_flops"], conv + endpoint)
                self.assertEqual(cost["hooked_feature_projection_flops"], 2 * 3 * 8 * 4 if feature == "output_gate" else 0)
                extra = set(model.state_dict()) - {name for name in model.state_dict()
                                                   if name not in ("history_conv.weight", "history_conv.bias",
                                                                   "output_gate.weight", "output_gate.bias", "D")}
                parameters = dict(model.named_parameters())
                self.assertEqual(sum(parameters[name].numel() for name in extra),
                    {"conv4": 8 * 5, "output_gate": 8 * 4 + 4, "skip_d": 4}[feature])
                for bad in (0, True, -1):
                    with self.assertRaises(ValueError):
                        model.feature_compute_flops(bad)
        with self.assertRaises(ValueError):
            self.compressor("none")

    def test_runtime_audit_requires_cuda_and_a_registered_single_feature(self):
        from rhpr.sasrec.query_mamba_models import audit_mamba_feature
        for feature in FEATURES:
            with self.subTest(feature=feature), self.assertRaisesRegex(RuntimeError, "actual CUDA"):
                audit_mamba_feature(feature, "cpu")
        with self.assertRaises(ValueError):
            audit_mamba_feature("combined", "cpu")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_native_feature_models_match_reference_forward_gradients_and_autocast(self):
        from rhpr.sasrec.query_mamba_models import audit_mamba_feature
        from rhpr.sasrec.query_ssm import native_selective_scan_available
        if not native_selective_scan_available():
            self.skipTest("native selective scan is unavailable")
        for feature in FEATURES:
            with self.subTest(feature=feature):
                before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                report = audit_mamba_feature(feature, "cuda")
                self.assertTrue(report["passed"])
                self.assertTrue(report["autocast_fp16"]["all_parameter_gradients_finite"])
                self.assertTrue(report["entire_raw_tail_gradient_zero"])
                self.assertFalse(report["python_gpu_fallback"])
                self.assertTrue(report["new_feature_gradient_norms"])
                self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
                self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
