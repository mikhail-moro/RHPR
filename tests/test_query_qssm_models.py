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
class QueryQSSMModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_qssm_search.json").read_text())

    def variants(self, config):
        from rhpr.sasrec.query_qssm_protocol import qssm_variants
        return qssm_variants(config)

    def trials(self, config, variant):
        from rhpr.sasrec.query_qssm_protocol import qssm_trials
        return qssm_trials(config, variant)

    def assert_state_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0, msg=name)

    def test_k4_default_is_exact_completed_inner64_control_including_rng_and_outputs(self):
        from rhpr.sasrec.query_architecture_models import build_architecture_model
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        config = self.config()
        architecture = config["reference_search"]["experiment_config"]
        variant = self.variants(config)[0]
        old_trial = next(row for row in architecture["architecture_trials"][variant["name"]]
                         if row["name"] == "q_inner64")
        expected = build_architecture_model(architecture, variant, old_trial, 60).eval()
        expected_rng = torch.get_rng_state().clone()
        model = build_qssm_model(config, variant, self.trials(config, variant)[0], 60).eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assert_state_equal(expected.state_dict(), model.state_dict())
        self.assertEqual(model.position_embedding.num_embeddings, 32768)
        items = torch.arange(1, 161)[None, :] % 60 + 1
        valid = torch.ones_like(items, dtype=torch.bool)
        with torch.no_grad():
            for before, after in zip(expected.encode(items, valid), model.encode(items, valid)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_every_k_preserves_backbone_rng_and_dt_ranges_change_only_delta_bias(self):
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        config = self.config()
        expected_backbone = expected_rng = None
        for variant in self.variants(config):
            with self.subTest(k=variant["summary_tokens"]):
                default, slow = (build_qssm_model(config, variant, trial, 30)
                                 for trial in self.trials(config, variant))
                backbone = {key: value for key, value in default.state_dict().items()
                            if not key.startswith("compressor.")}
                if expected_backbone is None:
                    expected_backbone = backbone
                    expected_rng = torch.get_rng_state().clone()
                self.assert_state_equal(expected_backbone, backbone)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                expected = {key: value for key, value in default.state_dict().items()
                            if key != "compressor.delta_projection.bias"}
                self.assert_state_equal(expected, {key: slow.state_dict()[key] for key in expected})
                self.assertEqual(default.compressor.summary_tokens, variant["summary_tokens"])
                self.assertEqual(default.compressor.query_source, "old_suffix")
                for model, upper in ((default, .1), (slow, .01)):
                    dt = torch.nn.functional.softplus(model.compressor.delta_projection.bias.detach())
                    self.assertGreaterEqual(float(dt.min()), .0001 - 1e-9)
                    self.assertLessEqual(float(dt.max()), upper + 1e-8)
                    self.assertTrue(model.compressor.delta_projection.bias._no_reinit)
                self.assertFalse(torch.equal(default.compressor.delta_projection.bias,
                                             slow.compressor.delta_projection.bias))

    def test_all_k_cpu_scan_has_finite_gradients_and_causal_ragged_packing(self):
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        config = self.config()
        for variant in self.variants(config):
            with self.subTest(k=variant["summary_tokens"]):
                compressor = build_qssm_model(config, variant, self.trials(config, variant)[1], 20).compressor.eval()
                values = torch.randn(3, 163, 128, requires_grad=True)
                lengths = torch.tensor([163, 137, 128])
                mask = torch.arange(163)[None, :] < lengths[:, None]
                output, valid, sources = compressor(values, mask)
                k = variant["summary_tokens"]
                self.assertEqual(int(valid[0].sum()), k + 128)
                self.assertEqual(int(valid[1].sum()), min(k, 9) + 128)
                self.assertEqual(int(valid[2].sum()), 128)
                self.assertEqual(sources[0, :k].tolist(), [34] * k)
                self.assertEqual(int(output[~valid].count_nonzero()), 0)
                self.assertTrue(bool(torch.isfinite(output).all()))
                (output[0, :k].square() * torch.linspace(.2, 1.8, 128)).sum().backward()
                self.assertGreater(float(values.grad[0, :35].abs().sum()), 0)
                self.assertEqual(int(values.grad[0, 35:].count_nonzero()), 0)
                for name in ("A_log", "value_projection.weight", "input_gate.weight", "query_gate.weight",
                             "delta_projection.weight", "write_projection.weight", "read_projection.weight"):
                    gradient = dict(compressor.named_parameters())[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(bool(torch.isfinite(gradient).all()), name)
                    self.assertGreater(float(gradient.abs().sum()), 0, name)
                changed = values.detach().clone()
                changed[0, 35:] = torch.randn_like(changed[0, 35:]) * 100
                changed[~mask] = torch.randn_like(changed[~mask]) * 100
                with torch.no_grad():
                    altered = compressor(changed, mask)
                    torch.testing.assert_close(output[0, :k], altered[0][0, :k], rtol=0, atol=0)
                    solo = compressor(values.detach()[0:1], mask[0:1])
                    torch.testing.assert_close(output[0, :k], solo[0][0, :k], rtol=2e-5, atol=2e-6)
                self.assertEqual(compressor.last_scan_backend, "cpu_reference")

    def test_builder_rejects_unknown_trials_and_changes_to_registered_fixed_dimensions(self):
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        config = self.config()
        variant = self.variants(config)[0]
        trial = self.trials(config, variant)[0]
        for changed in (dict(trial, name="foreign"), dict(trial, compressor_lr=.003),
                        dict(trial, d_inner=True), dict(trial, dt_max=float("nan"))):
            with self.subTest(trial=changed), self.assertRaises(ValueError):
                build_qssm_model(config, variant, changed, 20)
        malformed = copy.deepcopy(config)
        malformed["hpo"]["trials"][0]["d_inner"] = True
        with self.assertRaisesRegex(ValueError, "fixes d_inner"):
            build_qssm_model(malformed, variant, malformed["hpo"]["trials"][0], 20)

    def test_budget_prices_seven_exact_shapes_and_explicit_optional_reserve(self):
        from rhpr.sasrec.query_qssm_models import conservative_qssm_budget
        config = self.config()
        profile = {"training": {"windows_per_second": 100.},
                   "proxy_evaluation": {"anchors_per_second": 100.}}
        profiles = {f"{variant['name']}-{trial['name']}": copy.deepcopy(profile)
                    for variant in self.variants(config) for trial in self.trials(config, variant)
                    if (variant["name"], trial["name"]) != ("query_ssm_m4", "dt_default")}
        result = conservative_qssm_budget(config, profiles, 100., 1000,
                                         {"validation_anchors": 100, "test_anchors": 10**12})
        self.assertEqual(result["estimated_remaining_work_seconds"], 7 * 606.)
        self.assertEqual(result["fixed_checkpoint_statistics_overhead_seconds"], 840.)
        self.assertEqual(result["optional_full_transfer_reserve_seconds"], 2400.)
        self.assertEqual(result["conservative_total_seconds"], 100. + 1.25 * (4242. + 840. + 2400.))
        self.assertEqual(result["priced_fresh_shapes"], 7)
        self.assertEqual(result["registered_configurations"], 8)
        self.assertEqual(result["reused_controls"], 1)
        self.assertEqual(result["test_evaluations"], 0)
        self.assertTrue(result["fits"])
        profiles["query_ssm_m32-dt_slow"]["training"]["windows_per_second"] = 1.
        self.assertFalse(conservative_qssm_budget(config, profiles, 100., 1000,
                                                {"validation_anchors": 100})["fits"])
        self.assertFalse(result["auto_reduces_trials_or_passes"])
        profiles["query_ssm_m32-dt_slow"] = {"status": "preflight_out_of_memory"}
        failed = conservative_qssm_budget(config, profiles, 0., 1000, {"validation_anchors": 100})
        self.assertEqual(failed["preflight_failed_shapes"], 1)
        self.assertEqual(failed["priced_fresh_shapes"], 6)
        profiles["query_ssm_m16-dt_default"]["proxy_evaluation"]["anchors_per_second"] = True
        with self.assertRaises(ValueError):
            conservative_qssm_budget(config, profiles, 0., 1000, {"validation_anchors": 100})

    def _mock_preflight(self, *, unsuccessful=False, numerical=False):
        from rhpr.sasrec.query_qssm_models import preflight_qssm
        config, calls, daily_calls = self.config(), [], []
        config["training"].update(amp=False, effective_batch_size=2)
        variant = self.variants(config)[-1]
        config["qssm_batches"][variant["name"]]["training"] = 2
        model = torch.nn.Module()
        model.backbone, model.compressor = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)

        def train(model, cohort, windows, *, optimizer, batch_builder, physical_batch, **kwargs):
            calls.append((batch_builder.keywords["old_length"], physical_batch))
            if not unsuccessful:
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step()
            return {"successful_updates": 0 if unsuccessful else 1,
                    "amp_skipped_update_attempts": 0, "windows_per_second": 100.}

        def daily(model, *args, context_length, batch_size, **kwargs):
            daily_calls.append((context_length, batch_size))
            if numerical:
                raise RuntimeError("evaluation produced nonfinite final states")
            return {"anchors_per_second": 100.}

        module = "rhpr.sasrec.query_qssm_models."
        with ExitStack() as stack:
            for name, kwargs in (("build_qssm_model", {"return_value": model}),
                ("fixed_prefix_schedule", {"return_value": (1, 2)}),
                ("_representative_windows", {"return_value": (1, 2)}),
                ("train_prefix_pass", {"side_effect": train}),
                ("_profile_daily_rate", {"side_effect": daily}),
                ("_batch", {"return_value": (torch.ones((1, 1152)),)}),
                ("_observe_prefix_flops", {"return_value": (None, {"history_flops": 100})})):
                stack.enter_context(patch(module + name, **kwargs))
            result = preflight_qssm(config,
                SimpleNamespace(num_items=10, users=[SimpleNamespace(train=list(range(1400)), user_id=1)]),
                None, variant, self.trials(config, variant)[1], torch.device("cpu"))
        return result, calls, daily_calls

    def test_preflight_measures_two_real_adam_updates_only_proxy_and_exact_k(self):
        result, calls, daily_calls = self._mock_preflight()
        self.assertEqual(calls, [(1024, 2), (1024, 2)])
        self.assertEqual(daily_calls, [(1152, 32)])
        self.assertEqual(result["summary_tokens"], 32)
        self.assertEqual(result["full_evaluation"]["status"], "deferred_to_selected_finalists")
        self.assertEqual(result["full_evaluation"]["batch_size"], 4)
        self.assertEqual(set(result["representative_b1_arithmetic"]), {"proxy"})
        self.assertFalse(result["automatic_precision_or_batch_fallback"])
        self.assertTrue(result["optimizer_state_allocated_during_all_measurements"])

    def test_preflight_rejects_no_adam_update_and_normalizes_nonfinite_evaluation(self):
        with self.assertRaisesRegex(FloatingPointError, "two effective Adam updates"):
            self._mock_preflight(unsuccessful=True)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite final states"):
            self._mock_preflight(numerical=True)


if __name__ == "__main__":
    unittest.main()
