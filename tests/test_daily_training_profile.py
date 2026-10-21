from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch required")
class DailyTrainingProfileTest(unittest.TestCase):
    def test_training_selects_worst_endpoint_and_rejects_partial_or_skipped_batches(self):
        from rhpr.sasrec.daily_training_profile import _choose_training_batch

        rows = []
        for batch, budget, rate in [(8, 1, 100.), (8, 32, 20.), (16, 1, 50.), (16, 32, 40.)]:
            rows.append({"status": "ok", "physical_batch_size": batch,
                         "summary_tokens": budget, "amp_skipped_updates": 0,
                         "train_windows_per_second_with_prepare_h2d_sync": rate})
        self.assertEqual(_choose_training_batch(rows, (1, 32)), (16, 40.))
        rows[-1]["amp_skipped_updates"] = 1
        self.assertEqual(_choose_training_batch(rows, (1, 32)), (8, 20.))
        rows[1]["status"] = "oom"
        with self.assertRaises(RuntimeError):
            _choose_training_batch(rows, (1, 32))

    def fixture(self):
        config = {"seed": 42, "summary_budgets": [32, 16, 8, 4],
                  "training": {"raw_tail_size": 128, "supervised_targets": 32,
                      "raw_context_length": 16384, "effective_batch_size": 128,
                      "physical_batch_sizes": [8, 16], "amp": True},
                  "benchmark": {"batch_sizes": [1, 8, 32], "evaluation_batch_size": 32,
                                "item_chunk_size": 65536}}
        variant = {"name": "mixed_pair_mlp", "kind": "pair_mlp", "summary_tokens": 32,
                   "mixed_budgets": [1, 2, 4, 8, 16, 32]}
        cohort = SimpleNamespace(users=tuple(range(257)), num_items=1_049_612)
        data = SimpleNamespace(summary={"validation_anchors": 26335, "test_anchors": 25571,
                                       "validation_users": 4384, "test_users": 4349})
        return config, variant, cohort, data

    def test_forecast_counts_all_six_arms_and_anchors_not_users(self):
        from rhpr.sasrec.daily_training_profile import profile_daily_variant

        config, variant, cohort, data = self.fixture()
        models = []
        def build(*args):
            model = SimpleNamespace(compressor=SimpleNamespace(summary_tokens=32, tail_size=159))
            model.to = lambda device: model
            models.append(model)
            return model
        def train(model, cohort, schedule, profile, context, physical, *args):
            self.assertEqual(context, 16384)
            self.assertEqual(profile["tail_size"], 32)
            rates = {(8, 1): 100., (8, 32): 20., (16, 1): 50., (16, 32): 40.}
            return {"status": "ok", "physical_batch_size": physical, "amp_skipped_updates": 0,
                    "train_windows_per_second_with_prepare_h2d_sync": rates[physical, model.compressor.summary_tokens]}
        def rate(model, *args, batch_size, **kwargs):
            budget = model.compressor.summary_tokens
            value = 100. / budget if batch_size == 8 else 60. - 40. * (budget == 32)
            return {"status": "ok", "batch_size": batch_size, "sample_anchors": 128,
                    "wall_seconds": 128 / value, "anchors_per_second": value,
                    "quality_used_for_selection": False}
        module = "rhpr.sasrec.daily_training_profile."
        with patch(module + "build_quality_model", side_effect=build), \
             patch(module + "select_summary_budget", side_effect=lambda m, k: setattr(m.compressor, "summary_tokens", k)), \
             patch(module + "_optimizer"), patch(module + "_scaler"), patch(module + "_cleanup"), \
             patch(module + "_train_shape", side_effect=train), \
             patch(module + "_measure_daily_rate", side_effect=rate) as measured:
            result = profile_daily_variant(config, variant, cohort, data, (None,) * 128, torch.device("cpu"))
        self.assertEqual(result["training_batch_size"], 16)
        self.assertEqual(result["evaluation_batch_size"], 32)
        self.assertAlmostEqual(result["estimated_train_pass_seconds"], 257 / 40.)
        inverse_rates = 5 / 60. + 1 / 20.
        self.assertAlmostEqual(result["estimated_validation_seconds"], 26335 * inverse_rates)
        self.assertAlmostEqual(result["estimated_test_seconds"], 25571 * inverse_rates)
        self.assertEqual(result["profiled_training_budgets"], [1, 32])
        self.assertEqual(len(result["eval_cases"]), 6)
        self.assertEqual(result["hstu_normalization_length"], 191)
        self.assertEqual(len(models), 5)  # Fresh model for each shape/endpoint, then evaluation.
        self.assertEqual(measured.call_count, 12)
        self.assertTrue(all(m.compressor.mixed_budget_choices == (1, 2, 4, 8, 16, 32) for m in models))

    def test_measured_daily_rate_rebuilds_lazy_contexts_and_counts_actual_support(self):
        from rhpr.sasrec.daily_training_profile import _measure_daily_rate

        sizes = []
        def evaluate(model, examples, **kwargs):
            rows = list(examples)
            sizes.append(len(rows))
            return {"per_anchor": rows, "metrics": {"ndcg10": 99.}}
        module = "rhpr.sasrec.daily_training_profile."
        with patch(module + "iter_daily_evaluation_examples", side_effect=lambda *args, **kw: iter(range(17))) as iterator, \
             patch(module + "evaluate_daily_quality", side_effect=evaluate), \
             patch(module + "_sync"), patch(module + "time.perf_counter", side_effect=[2., 4.]):
            result = _measure_daily_rate(None, None, None, context_length=16384, batch_size=8,
                device=torch.device("cpu"), amp=False, item_chunk_size=5, sample_count=128)
        self.assertEqual(sizes, [8, 17])
        self.assertEqual(iterator.call_count, 2)
        self.assertEqual(result["sample_anchors"], 17)
        self.assertEqual(result["anchors_per_second"], 8.5)
        self.assertNotIn("metrics", result)
        self.assertFalse(result["quality_used_for_selection"])

    def test_no_common_successful_endpoint_batch_fails_before_evaluation(self):
        from rhpr.sasrec.daily_training_profile import profile_daily_variant

        config, variant, cohort, data = self.fixture()
        model = SimpleNamespace(compressor=SimpleNamespace(summary_tokens=32), to=lambda device: model)
        module = "rhpr.sasrec.daily_training_profile."
        with patch(module + "build_quality_model", return_value=model), \
             patch(module + "select_summary_budget"), patch(module + "_optimizer"), \
             patch(module + "_scaler"), patch(module + "_cleanup"), \
             patch(module + "_train_shape", side_effect=torch.OutOfMemoryError), \
             patch(module + "_measure_daily_rate") as evaluate:
            with self.assertRaises(RuntimeError):
                profile_daily_variant(config, variant, cohort, data, (None,) * 128, torch.device("cpu"))
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
