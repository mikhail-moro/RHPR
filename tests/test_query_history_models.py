from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QueryHistoryModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self, family="qssm"):
        return json.loads((ROOT / f"experiments/yambda_history_continuation_{family}.json").read_text())

    def assert_state_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0, msg=name)

    def test_qssm_builder_exactly_reconstructs_selected_search_shape_and_rng(self):
        from rhpr.sasrec.query_history_models import build_history_model
        from rhpr.sasrec.query_qssm_models import build_qssm_model
        from rhpr.sasrec.query_qssm_protocol import qssm_trials, qssm_variants

        source = json.loads((ROOT / "experiments/yambda_query_qssm_search.json").read_text())
        variant = qssm_variants(source)[-1]
        expected = build_qssm_model(source, variant, qssm_trials(source, variant)[1], 20)
        expected_rng = torch.get_rng_state().clone()
        actual = build_history_model(self.config(), 20)
        self.assert_state_equal(expected.state_dict(), actual.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(actual.compressor.summary_tokens, 32)
        self.assertEqual(actual.compressor.query_source, "old_suffix")
        self.assertEqual(actual.position_embedding.num_embeddings, 32768)
        self.assertFalse(actual.history_source_checkpoint_loaded)

    def test_fuxi_builder_exactly_reconstructs_selected_native_shape_and_rng(self):
        from rhpr.sasrec.query_baseline_models import build_baseline_model
        from rhpr.sasrec.query_history_models import build_history_model

        source = json.loads((ROOT / "experiments/yambda_query_baseline_screen.json").read_text())
        expected = build_baseline_model(source,
            {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0},
            {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}, 20)
        expected_rng = torch.get_rng_state().clone()
        actual = build_history_model(self.config("fuxi"), 20)
        self.assert_state_equal(expected.state_dict(), actual.state_dict())
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(actual.encoder.layers[0]._channel_p._emb.shape, (32768, 32))
        self.assertFalse(actual.history_source_checkpoint_loaded)

    def test_builder_rejects_unregistered_family_and_architecture_changes(self):
        from rhpr.sasrec.query_history_models import build_history_model

        for family in ("qssm", "fuxi"):
            for key in ("architecture", "selected_source"):
                config = self.config(family)
                if key == "architecture":
                    config[key][next(iter(config[key]))] = 999
                else:
                    config[key]["trial"]["compressor_lr"] = .003
                with self.subTest(family=family, key=key), self.assertRaises(ValueError):
                    build_history_model(config, 20)
        config = self.config()
        config["family"] = "other"
        with self.assertRaises(ValueError):
            build_history_model(config, 20)

    def test_optimizer_resets_state_owns_every_parameter_and_keeps_native_decay_policy(self):
        from rhpr.sasrec.query_history_models import build_history_model, history_optimizer

        for family, groups in (("qssm", ["backbone", "compressor"]), ("fuxi", ["backbone"])):
            config = self.config(family)
            model = build_history_model(config, 20)
            next(model.parameters())._no_weight_decay = True
            optimizer = history_optimizer(model, config["training"])
            self.assertEqual([row["component"] for row in optimizer.param_groups], groups)
            self.assertFalse(optimizer.state)
            owned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
            self.assertEqual(len(owned), len(set(owned)))
            self.assertEqual(set(owned), {id(parameter) for parameter in model.parameters()})
            for group in optimizer.param_groups:
                self.assertEqual(group["lr"], .001)
                self.assertEqual(group["weight_decay"], .00001)
                self.assertFalse(group["foreach"])
        for key in ("learning_rate", "weight_decay"):
            changed = dict(config["training"], **{key: .003})
            with self.assertRaises(ValueError):
                history_optimizer(model, changed)

    def tiny_model(self, family="qssm"):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_items = 20
                self.item_embedding = torch.nn.Embedding(21, 4)
                if family == "qssm":
                    self.compressor = torch.nn.Linear(4, 4)
                else:
                    self.projection = torch.nn.Linear(4, 4)
                    self.architecture, self.d_model = "fuxi_linear", 4
                self.dropout = torch.nn.Dropout(.2)
                self.history_source_checkpoint_loaded = True

            def encode(self, items, mask):
                projection = self.compressor if family == "qssm" else self.projection
                states = self.dropout(projection(self.item_embedding(items)))
                sources = torch.arange(items.shape[1]).expand_as(items).masked_fill(~mask, -1)
                return states, mask, sources

        return TinyModel()

    def tiny_arrays(self, count, length):
        items = np.tile(np.arange(length) % 19 + 1, (count, 1)).astype(np.int64)
        return {"item_ids": items, "targets": items % 20 + 1,
                "valid_mask": np.ones_like(items, dtype=np.bool_),
                "negative_ids": np.tile(np.array([[19, 20]], dtype=np.int64), (count, 1))}

    def test_capacity_warmup_runs_real_backward_without_mutating_weights_or_adam(self):
        from rhpr.sasrec.query_history_models import _capacity_backward, history_optimizer

        config = self.config()
        config["training"].update(amp=False, supervised_targets=2, num_negatives=2)
        model = self.tiny_model()
        before = copy.deepcopy(model.state_dict())
        optimizer = history_optimizer(model, config["training"])
        row = _capacity_backward(model, None, (1,), config=config, variant={},
            device=torch.device("cpu"), optimizer=optimizer,
            batch_builder=lambda *args: self.tiny_arrays(1, 9))
        self.assert_state_equal(before, model.state_dict())
        self.assertFalse(optimizer.state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(row["input_padded_length"], 9)
        self.assertEqual(row["optimizer_updates"], 0)
        self.assertTrue(row["forward_and_backward_performed"])

    def _preflight(self, arm_name, *, numerical=False, fresh=False, failed_update=False, family="qssm"):
        from rhpr.sasrec.query_history_models import preflight_history
        from rhpr.sasrec.query_history_protocol import history_arms

        config = self.config(family)
        config["training"].update(amp=False, effective_batch_size=2,
                                  supervised_targets=2, num_negatives=2, raw_tail_size=2)
        arms = [dict(row, physical_batch_size=1) for row in history_arms(config)] if family == "qssm" else history_arms(config)
        arm = next(row for row in arms if row["name"] == arm_name)
        model = self.tiny_model(family)
        model.history_source_checkpoint_loaded = not fresh
        before = copy.deepcopy(model.state_dict())
        windows = (SimpleNamespace(user_id=1, end=2000), SimpleNamespace(user_id=2, end=4000))
        calls, daily_calls = [], []

        def arrays(cohort, selected, supplied_config, selected_arm):
            self.assertEqual(selected_arm, arm)
            calls.append((tuple(row.end for row in selected), selected_arm["old_length"]))
            return self.tiny_arrays(len(selected), 6 if selected_arm["old_length"] == 1024 else 12)

        def daily(actual_model, *args, context_length, batch_size, **kwargs):
            self.assertIs(actual_model, model)
            daily_calls.append((context_length, batch_size))
            self.assertFalse(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))
            if numerical:
                raise RuntimeError("evaluation produced nonfinite final states")
            return {"anchors_per_second": 100.}

        module = "rhpr.sasrec.query_history_models."
        with ExitStack() as stack:
            stack.enter_context(patch("rhpr.sasrec.query_history_protocol.history_arms", return_value=arms))
            stack.enter_context(patch("rhpr.sasrec.query_history_protocol.history_batch_arrays", side_effect=arrays))
            schedule = stack.enter_context(patch(module + "fixed_prefix_schedule", return_value=windows))
            stack.enter_context(patch(module + "_representative_windows", return_value=windows))
            stack.enter_context(patch(module + "_profile_daily_rate", side_effect=daily))
            stack.enter_context(patch(module + "_batch", return_value=(torch.ones((1, 6)),)))
            stack.enter_context(patch(module + "_observe_prefix_flops", return_value=(None, {"history_flops": 123})))
            stack.enter_context(patch(module + "_major_flops", return_value={"per_user": 123, "scope": "test_scope"}))
            if failed_update:
                stack.enter_context(patch(module + "train_prefix_pass", return_value={
                    "successful_updates": 0, "amp_skipped_update_attempts": 1}))
            row = preflight_history(config,
                SimpleNamespace(num_items=20, users=[SimpleNamespace(train=list(range(4001)), user_id=2)]),
                None, arm, torch.device("cpu"), load_model=lambda: model)
            schedule.assert_called_once_with(unittest.mock.ANY, seed=42, epoch=61)
        return row, calls, daily_calls

    def test_preflight_uses_real_source_weights_two_effective_updates_and_distinct_history_shapes(self):
        for arm, old, cap, evaluation_batch, retained in (
                ("short", 1024, 1152, 32, 1152), ("full", None, 32768, 4, 4000)):
            with self.subTest(arm=arm):
                row, calls, daily_calls = self._preflight(arm)
                self.assertEqual(calls, [((4000,), old), ((2000,), old), ((4000,), old),
                                         ((2000,), old), ((4000,), old)])
                self.assertEqual(daily_calls, [(cap, evaluation_batch)])
                self.assertEqual(row["warmup"]["successful_updates"], 1)
                self.assertEqual(row["training"]["successful_updates"], 1)
                self.assertEqual(row["capacity_warmup"]["optimizer_updates"], 0)
                self.assertEqual(row["representative_b1_arithmetic"]["input_tokens"], retained)
                self.assertTrue(row["source_checkpoint_weights_loaded"])
                self.assertFalse(row["fresh_initializer_profiled"])
                self.assertFalse(row["profile_weights_reused"])
                self.assertFalse(row["automatic_precision_or_batch_fallback"])

    def test_fuxi_full_only_preflight_measures_registered_b2_without_a_short_profile(self):
        row, calls, daily_calls = self._preflight("full", family="fuxi")
        self.assertEqual(calls, [((4000, 2000), None), ((2000, 4000), None), ((2000, 4000), None)])
        self.assertEqual(daily_calls, [(32768, 4)])
        self.assertEqual(row["physical_batch_size"], 2)
        self.assertEqual(row["capacity_warmup"]["physical_batch_size"], 2)
        self.assertEqual(row["warmup"]["successful_updates"], 1)
        self.assertEqual(row["training"]["successful_updates"], 1)
        self.assertEqual(row["compressor_parameters"], 0)
        self.assertFalse(row["automatic_precision_or_batch_fallback"])

    def test_preflight_restores_rng_and_rejects_fresh_models_failed_updates_and_nonfinite_evaluation(self):
        random.seed(93)
        np.random.seed(94)
        torch.manual_seed(95)
        # Fixture construction intentionally consumes RNG, so isolate the model
        # factory in the test to check the preflight boundary itself.
        from rhpr.sasrec.query_history_models import preflight_history
        from rhpr.sasrec.query_history_protocol import history_arms

        config = self.config()
        config["training"].update(amp=False, effective_batch_size=2)
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()

        def loader():
            random.random()
            np.random.random()
            model = self.tiny_model()
            model.history_source_checkpoint_loaded = False
            return model

        with self.assertRaisesRegex(ValueError, "trained source checkpoint"):
            preflight_history(config, None, None, history_arms(config)[1], torch.device("cpu"), load_model=loader)
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "trained source checkpoint"):
            self._preflight("full", fresh=True)
        with self.assertRaisesRegex(FloatingPointError, "two effective Adam updates"):
            self._preflight("full", failed_update=True)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite final states"):
            self._preflight("full", numerical=True)

    def test_budget_prices_both_twenty_pass_arms_and_zero_ten_twenty_validation(self):
        from rhpr.sasrec.query_history_models import conservative_history_budget

        config = self.config()
        profiles = {"short": {"status": "success", "training": {"windows_per_second": 100.},
                              "evaluation": {"anchors_per_second": 100.}},
                    "full": {"status": "success", "training": {"windows_per_second": 10.},
                             "evaluation": {"anchors_per_second": 20.}}}
        row = conservative_history_budget(config, profiles, 100., 1000,
                                          {"validation_anchors": 100, "test_anchors": 10**12})
        self.assertEqual(row["estimated_remaining_work_seconds"], 200. + 3. + 2000. + 15.)
        self.assertEqual(row["fixed_checkpoint_statistics_overhead_seconds"], 240.)
        self.assertEqual(row["conservative_total_seconds"], 100. + 1.25 * (2218. + 240.))
        self.assertEqual(row["validation_evaluations_per_arm"], 3)
        self.assertTrue(row["fits"])
        self.assertEqual(row["test_evaluations"], 0)
        self.assertFalse(row["auto_reduces_arms_or_passes"])
        profiles["full"]["training"]["windows_per_second"] = 1.
        self.assertFalse(conservative_history_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100})["fits"])
        for invalid in (True, 0., -1., float("nan"), float("inf")):
            profiles["full"]["training"]["windows_per_second"] = invalid
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                conservative_history_budget(config, profiles, 100., 1000, {"validation_anchors": 100})
        profiles["full"] = {"status": "preflight_out_of_memory"}
        with self.assertRaisesRegex(ValueError, "all registered history shapes"):
            conservative_history_budget(config, profiles, 100., 1000, {"validation_anchors": 100})

    def test_fuxi_full_only_budget_omits_short_work_and_replays_old_b1_cost_as_counterfactual(self):
        from rhpr.sasrec.query_history_models import conservative_history_budget

        # Actual successful T4 rates from the earlier paired B1-full run.
        # Repricing its full work alone is only a B1 counterfactual: the new B2
        # run must obtain fresh B2 capacity and throughput measurements before
        # its real gate. This fixture makes no claim about B2 performance.
        profiles = {
            "short": {"status": "success",
                      "training": {"windows_per_second": 115.6207218272389},
                      "evaluation": {"anchors_per_second": 350.5684793812835}},
            "full": {"status": "success",
                     "training": {"windows_per_second": 4.216048214902611},
                     "evaluation": {"anchors_per_second": 19.48255995350387}},
        }
        config = self.config("fuxi")
        self.assertEqual(config["training"]["budget_seconds"], 32400)
        self.assertEqual(config["training"]["physical_batch_sizes"], [2])
        self.assertEqual(config["training"]["effective_batch_size"], 128)
        self.assertEqual(config["continuation"]["short_control"], "historical_only")
        old_config = copy.deepcopy(config)
        old_config["training"]["budget_seconds"] = 28800
        old_config["training"]["physical_batch_sizes"] = [1, 16]
        old_config["continuation"].pop("short_control")
        args = (profiles, 126.906860048, 4427, {"validation_anchors": 26335})
        old = conservative_history_budget(old_config, *args)
        recovered = conservative_history_budget(config, *args)

        self.assertFalse(old["fits"])
        self.assertTrue(recovered["fits"])
        self.assertAlmostEqual(old["conservative_total_seconds"], 32985.67693579488, places=7)
        expected_full_work = 20 * 4427 / profiles["full"]["training"]["windows_per_second"]
        expected_full_work += 3 * 26335 / profiles["full"]["evaluation"]["anchors_per_second"]
        self.assertAlmostEqual(recovered["estimated_remaining_work_seconds"], expected_full_work)
        self.assertAlmostEqual(recovered["conservative_total_seconds"],
                               126.906860048 + 1.25 * (expected_full_work + 120.))
        self.assertEqual(recovered["registered_configurations"], 1)
        self.assertEqual(recovered["fixed_checkpoint_statistics_overhead_seconds"], 120.)
        self.assertEqual(recovered["passes_per_arm"], 20)
        self.assertEqual(recovered["validation_evaluations_per_arm"], 3)
        self.assertEqual(config["continuation"]["validation_passes"], [0, 10, 20])
        self.assertEqual(recovered["safety_factor"], 1.25)
        self.assertFalse(recovered["auto_reduces_arms_or_passes"])

        # A historical short reference adds no new short training or validation
        # cost. Even an unusable or absent short profile must not enter pricing.
        for short in ({"status": "numerical_failed"},
                      {"status": "success", "training": {"windows_per_second": 0.}}):
            changed = dict(profiles, short=short)
            self.assertEqual(conservative_history_budget(config, changed, *args[1:]), recovered)
        self.assertEqual(conservative_history_budget(config, {"full": profiles["full"]}, *args[1:]), recovered)
        for invalid in (True, 0., -1., float("nan"), float("inf")):
            changed = copy.deepcopy(profiles)
            changed["full"]["training"]["windows_per_second"] = invalid
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                conservative_history_budget(config, changed, *args[1:])


if __name__ == "__main__":
    unittest.main()
