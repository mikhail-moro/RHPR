from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QueryBaselineModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_compressor_architecture.json").read_text())
        config["standalone_models"] = {
            "mamba4rec": {"d_state": 32, "d_conv": 4, "expand": 2},
            "fuxi_linear": {"chunk_size": 128, "temporal_heads": 8, "positional_dim": 32},
        }
        config["baseline_batches"] = {
            name: {"training": 16 if name == "fuxi_linear" else 32,
                   "proxy_evaluation": 32,
                   "full_evaluation": {"mamba4rec": 8, "fuxi_linear": 4}.get(name, 32)}
            for name in ("recent128", "query_only_m4", "mamba4rec", "fuxi_linear")}
        return config

    def variant(self, name):
        from rhpr.sasrec.query_baseline_models import _FRESH_VARIANTS
        return next(copy.deepcopy(row) for row in _FRESH_VARIANTS if row["name"] == name)

    def trial(self):
        return {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}

    def assert_state_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0, msg=name)

    def test_recent_and_query_only_keep_parent_weights_outputs_and_rng(self):
        from rhpr.sasrec.query_baseline_models import build_baseline_model
        from rhpr.sasrec.query_only_protocol import QUERY_ONLY_KIND, build_query_only_model
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model
        config = self.config()
        items = torch.arange(1, 270)[None, :] % 70 + 1
        mask = torch.ones_like(items, dtype=torch.bool)
        for name in ("recent128", "query_only_m4"):
            with self.subTest(name=name):
                variant = self.variant(name)
                if name == "recent128":
                    expected = build_prefix_model(config["parent"]["experiment_config"], variant, 80).eval()
                else:
                    expected = build_query_only_model(dict(config, kind=QUERY_ONLY_KIND), variant, 80).eval()
                expected_rng = torch.get_rng_state().clone()
                model = build_baseline_model(config, variant, self.trial(), 80).eval()
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assert_state_equal(expected.state_dict(), model.state_dict())
                with torch.no_grad():
                    for before, after in zip(expected.encode(items, mask), model.encode(items, mask)):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertEqual(model.position_embedding.num_embeddings, 32768)
                self.assertFalse(model.baseline_metadata["shared_item_embedding_copied"])

    def test_mamba_builder_preserves_native_initialization_and_dimension_policy(self):
        from rhpr.sasrec.query_baseline_models import build_baseline_model
        from rhpr.sasrec.mamba import Mamba4RecModel
        # Existing exact official CPU fixture is test-only; production still
        # requires the native CUDA extension, exercised by the preflight.
        from test_mamba_model import _cpu_native_mamba
        config = self.config()
        metadata = {"execution_path": "test_only_exact_official_reference"}
        with patch("rhpr.sasrec.mamba._require_native_cuda_runtime", return_value=metadata), \
                patch("rhpr.vendor.mamba4rec.native.Mamba", _cpu_native_mamba()):
            torch.manual_seed(config["seed"])
            expected = Mamba4RecModel(num_items=80, max_seq_len=32768, d_model=128,
                n_heads=4, n_layers=2, ffn_multiplier=4, dropout=.2, use_time_features=False,
                mamba_d_state=32, mamba_d_conv=4, mamba_expand=2)
            expected_rng = torch.get_rng_state().clone()
            actual = build_baseline_model(config, self.variant("mamba4rec"), self.trial(), 80)
        self.assert_state_equal(expected.state_dict(), actual.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertIsNone(actual.position_embedding)
        self.assertFalse(hasattr(actual, "compressor"))
        for layer in actual.mamba_layers:
            self.assertTrue(layer.mamba.dt_proj.bias._no_reinit)
            self.assertTrue(layer.mamba.A_log._no_weight_decay)
            self.assertTrue(layer.mamba.D._no_weight_decay)
            delta = torch.nn.functional.softplus(layer.mamba.dt_proj.bias.detach())
            self.assertGreaterEqual(float(delta.min()), .001 - 1e-7)
            self.assertLessEqual(float(delta.max()), .1 + 1e-7)
        self.assertTrue(actual.baseline_metadata["native_special_initialization_preserved"])

    def test_fuxi_builder_matches_native_model_and_keeps_full_positional_table(self):
        from rhpr.sasrec.fuxi import FuXiLinearModel
        from rhpr.sasrec.query_baseline_models import build_baseline_model
        config = self.config()
        torch.manual_seed(config["seed"])
        expected = FuXiLinearModel(num_items=80, max_seq_len=32768, d_model=128,
            n_heads=4, n_layers=2, ffn_multiplier=4, dropout=.2, use_time_features=False,
            chunk_size=128, fuxi_temporal_heads=8, fuxi_positional_dim=32)
        expected_rng = torch.get_rng_state().clone()
        actual = build_baseline_model(config, self.variant("fuxi_linear"), self.trial(), 80)
        self.assert_state_equal(expected.state_dict(), actual.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertFalse(hasattr(actual, "compressor"))
        self.assertEqual(actual.encoder.layers[0]._channel_p._emb.shape, (32768, 32))
        self.assertEqual(actual.fuxi_time_mode, "zero_payload_all_channels_no_future_timestamps")
        self.assertFalse(actual.baseline_metadata["paper_temporal_recipe_reproduced"])

    def test_optimizer_owns_every_parameter_once_and_keeps_all_parameter_decay(self):
        from rhpr.sasrec.query_baseline_models import baseline_optimizer, build_baseline_model
        config = self.config()
        for name, expected_groups in (("recent128", ["backbone"]),
                                      ("query_only_m4", ["backbone", "compressor"]),
                                      ("fuxi_linear", ["backbone"])):
            with self.subTest(name=name):
                model = build_baseline_model(config, self.variant(name), self.trial(), 40)
                next(model.parameters())._no_weight_decay = True
                optimizer = baseline_optimizer(model, config["training"], self.trial())
                self.assertEqual([group["component"] for group in optimizer.param_groups], expected_groups)
                owned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
                self.assertEqual(len(owned), len(set(owned)))
                self.assertEqual(set(owned), {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
                for group in optimizer.param_groups:
                    self.assertEqual(group["lr"], .001)
                    self.assertEqual(group["weight_decay"], config["training"]["weight_decay"])
                    self.assertFalse(group["foreach"])

    def test_builder_and_optimizer_reject_unregistered_recipes(self):
        from rhpr.sasrec.query_baseline_models import baseline_optimizer, build_baseline_model
        config = self.config()
        for trial in (dict(self.trial(), compressor_lr=.003), dict(self.trial(), name="other"),
                      dict(self.trial(), backbone_lr=True)):
            with self.subTest(trial=trial), self.assertRaises(ValueError):
                build_baseline_model(config, self.variant("recent128"), trial, 40)
            with self.assertRaises(ValueError):
                baseline_optimizer(torch.nn.Linear(2, 2), config["training"], trial)
        with self.assertRaises(ValueError):
            build_baseline_model(config, {"name": "mamba4rec", "kind": "recent", "summary_tokens": 0},
                                 self.trial(), 40)

    def test_budget_prices_four_shapes_and_only_two_distinct_full_validations(self):
        from rhpr.sasrec.query_baseline_models import conservative_baseline_budget
        config = self.config()
        profile = {"training": {"windows_per_second": 100.},
                   "proxy_evaluation": {"anchors_per_second": 100.},
                   "full_evaluation": {"anchors_per_second": 50.}}
        profiles = {name + "-fixed_lr": copy.deepcopy(profile)
                    for name in config["baseline_batches"]}
        row = conservative_baseline_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 10 ** 12})
        self.assertEqual(row["estimated_remaining_work_seconds"], 4 * (600. + 6.) + 2 * 2.)
        self.assertEqual(row["conservative_total_seconds"], 100. + 1.25 * (2428. + 480.))
        self.assertEqual(row["fresh_full_transfer_evaluations"], 2)
        self.assertEqual(row["short_control_duplicate_full_evaluations"], 0)
        self.assertEqual(row["test_evaluations"], 0)
        self.assertEqual(row["full_training_models"], 0)
        self.assertTrue(row["fits"])
        profiles["mamba4rec-fixed_lr"] = {"status": "preflight_out_of_memory"}
        failed = conservative_baseline_budget(config, profiles, 0., 1000, {"validation_anchors": 100})
        self.assertEqual(failed["preflight_failed_shapes"], 1)
        self.assertEqual(failed["fresh_full_transfer_evaluations"], 1)
        profiles["fuxi_linear-fixed_lr"]["full_evaluation"] = {"status": "out_of_memory"}
        optional = conservative_baseline_budget(config, profiles, 0., 1000, {"validation_anchors": 100})
        self.assertEqual(optional["unsupported_optional_full_transfers"], 1)
        self.assertEqual(optional["fresh_full_transfer_evaluations"], 0)
        profiles["fuxi_linear-fixed_lr"]["training"]["windows_per_second"] = float("nan")
        with self.assertRaises(ValueError):
            conservative_baseline_budget(config, profiles, 0., 1000, {"validation_anchors": 100})

    def test_preflight_uses_registered_batches_real_updates_and_deduplicates_short_control(self):
        from rhpr.sasrec.query_baseline_models import preflight_baseline
        config = self.config()
        config["training"].update(amp=False, effective_batch_size=2)
        config["baseline_batches"]["query_only_m4"]["training"] = 2
        calls, daily_calls = [], []
        model = torch.nn.Linear(2, 2)
        model.architecture = "sasrec"
        model.baseline_metadata = {"initialization_policy": "test"}

        def train(model, cohort, windows, *, optimizer, batch_builder, physical_batch, **kwargs):
            calls.append((batch_builder.keywords["old_length"], physical_batch))
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                    "windows_per_second": 128., "seconds": 1.}

        def daily(model, *args, context_length, batch_size, **kwargs):
            daily_calls.append((context_length, batch_size))
            return {"anchors_per_second": 100.}

        protocol = SimpleNamespace(baseline_batch_arrays=lambda *a, **k: {},
            baseline_context_cap=lambda *a: 256,
            baseline_evaluation_batch=lambda *a: 32)
        module = "rhpr.sasrec.query_baseline_models."
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"rhpr.sasrec.query_baseline_protocol": protocol}))
            for name, kwargs in (
                ("build_baseline_model", {"return_value": model}),
                ("fixed_prefix_schedule", {"return_value": (1, 2)}),
                ("_representative_windows", {"return_value": (1, 2)}),
                ("train_prefix_pass", {"side_effect": train}),
                ("_profile_daily_rate", {"side_effect": daily}),
                ("_batch", {"return_value": (torch.ones((1, 256)),)}),
                ("_observe_prefix_flops", {"return_value": (None, {"history_flops": 100})}),
            ):
                stack.enter_context(patch(module + name, **kwargs))
            result = preflight_baseline(config,
                SimpleNamespace(num_items=10, users=[SimpleNamespace(train=list(range(1400)), user_id=1)]),
                None, self.variant("query_only_m4"), self.trial(), torch.device("cpu"))
        self.assertEqual(calls, [(1024, 2), (1024, 2)])
        self.assertEqual(daily_calls, [(256, 32)])
        self.assertTrue(result["full_evaluation"]["reused_identical_input_profile"])
        self.assertFalse(result["full_evaluation"]["separate_forward_performed"])
        self.assertFalse(result["profile_weights_reused"])
        self.assertTrue(result["optimizer_state_allocated_during_all_measurements"])
        self.assertTrue(result["representative_b1_arithmetic"]["full"]["reused_identical_input_cost"])

    def test_optional_native_full_oom_keeps_primary_preflight_and_uses_family_costs(self):
        from rhpr.sasrec.query_baseline_models import preflight_baseline
        config, daily_calls, updates = self.config(), [], []
        config["training"].update(amp=False, effective_batch_size=16)
        model = torch.nn.Linear(2, 2)
        model.architecture, model.d_model = "fuxi_linear", 2
        model.baseline_metadata = {"initialization_policy": "native_test"}

        def train(model, cohort, windows, *, optimizer, physical_batch, **kwargs):
            updates.append(physical_batch)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                    "windows_per_second": 128., "seconds": 1.}

        def daily(model, *args, context_length, batch_size, **kwargs):
            daily_calls.append((context_length, batch_size))
            if context_length == 32768:
                raise torch.OutOfMemoryError("test optional full-history capacity failure")
            return {"anchors_per_second": 100.}

        protocol = SimpleNamespace(baseline_batch_arrays=lambda *a, **k: {},
            baseline_context_cap=lambda config, variant, old: 32768 if old is None else old + 128,
            baseline_evaluation_batch=lambda config, variant, old: 4 if old is None else 32)
        module = "rhpr.sasrec.query_baseline_models."
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"rhpr.sasrec.query_baseline_protocol": protocol}))
            for name, kwargs in (
                ("build_baseline_model", {"return_value": model}),
                ("fixed_prefix_schedule", {"return_value": tuple(range(16))}),
                ("_representative_windows", {"return_value": tuple(range(16))}),
                ("train_prefix_pass", {"side_effect": train}),
                ("_profile_daily_rate", {"side_effect": daily}),
                ("_major_flops", {"return_value": {"per_user": 1234, "scope": "test major matmuls"}}),
            ):
                stack.enter_context(patch(module + name, **kwargs))
            dense_cost = stack.enter_context(patch(module + "_observe_prefix_flops",
                side_effect=AssertionError("standalone models must not use dense SASRec FLOPs")))
            result = preflight_baseline(config,
                SimpleNamespace(num_items=10, users=[SimpleNamespace(train=list(range(1400)), user_id=1)]),
                None, self.variant("fuxi_linear"), self.trial(), torch.device("cpu"))
        self.assertEqual(updates, [16, 16])
        self.assertEqual(daily_calls, [(1152, 32), (32768, 4)])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["full_evaluation"]["status"], "out_of_memory")
        self.assertFalse(result["full_evaluation"]["automatic_precision_or_batch_fallback"])
        self.assertEqual(result["compressor_parameters"], 0)
        self.assertEqual(result["representative_b1_arithmetic"]["proxy"]["forward_major_matmul_flops"], 1234)
        self.assertFalse(result["representative_b1_arithmetic"]["full"]["total_forward_flops_available"])
        dense_cost.assert_not_called()


if __name__ == "__main__":
    unittest.main()
