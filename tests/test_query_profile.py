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
class QueryProfileTest(unittest.TestCase):
    def test_native_audit_rejects_cpu_and_unavailable_extension(self):
        from rhpr.sasrec.query_profile import audit_native_query_scan
        with self.assertRaisesRegex(RuntimeError, "actual CUDA device"):
            audit_native_query_scan(torch.device("cpu"))
        module = "rhpr.sasrec.query_profile."
        with patch(module + "torch.cuda.is_available", return_value=True), \
             patch(module + "importlib.import_module", side_effect=ImportError("missing scan")):
            with self.assertRaisesRegex(RuntimeError, "no GPU fallback"):
                audit_native_query_scan(torch.device("cuda:0"))

    def test_native_audit_rejects_python_shim_before_any_cuda_work(self):
        from rhpr.sasrec.query_profile import audit_native_query_scan
        extension = SimpleNamespace(__file__="/tmp/selective_scan_cuda.py", fwd=lambda: None, bwd=lambda: None)
        interface = SimpleNamespace(selective_scan_fn=lambda: None)
        module = "rhpr.sasrec.query_profile."
        with patch(module + "torch.cuda.is_available", return_value=True), \
             patch(module + "importlib.import_module", side_effect=[extension, interface]):
            with self.assertRaisesRegex(RuntimeError, "compiled selective_scan_cuda extension"):
                audit_native_query_scan(torch.device("cuda:0"))

    def test_audit_comparison_rejects_nonfinite_and_inaccurate_gradients(self):
        from rhpr.sasrec.query_profile import _comparison
        expected = torch.tensor([1., 2.])
        result = _comparison(expected, expected + 1e-6, name="gradient", atol=1e-5, rtol=1e-5)
        self.assertTrue(result["passed"])
        self.assertNotIn("tensor", result)
        for candidate in (torch.tensor([1., float("nan")]), expected + 1.):
            with self.assertRaisesRegex(RuntimeError, "native Query SSM audit"):
                _comparison(expected, candidate, name="gradient", atol=1e-5, rtol=1e-5)

    def test_summary_only_loss_reaches_critical_recurrence_and_query_parameters(self):
        from rhpr.sasrec.query_profile import _critical_gradient_norms
        from rhpr.sasrec.query_ssm import QuerySSMCompressor
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(918273)
            for conditioning in ("recent", "slot_only"):
                model = QuerySSMCompressor(d_model=8, summary_tokens=2, tail_size=6,
                    query_context_size=4, n_heads=2, d_state=4, d_inner=8, gate_rank=4,
                    write_conditioning=conditioning, dropout=0.)
                mask = torch.arange(13)[None, :] < torch.tensor([13, 11])[:, None]
                inputs = torch.randn(2, 13, 8).masked_fill(~mask[:, :, None], 0).requires_grad_()
                output = model(inputs, mask)[0]
                (output[:, :2] * torch.randn(2, 2, 8)).sum().backward()
                norms = _critical_gradient_norms(model, inputs, (7, 5), 4)
                self.assertTrue(all(value > 0 for value in norms.values()))
                model.A_log.grad.zero_()
                with self.assertRaisesRegex(RuntimeError, "zero or nonfinite"):
                    _critical_gradient_norms(model, inputs, (7, 5), 4)

    def fixture(self, kind="query_ssm"):
        config = {"seed": 42,
                  "training": {"raw_tail_size": 128, "supervised_targets": 32,
                      "raw_context_length": 8159, "serving_context_length": 8128,
                      "effective_batch_size": 128, "physical_batch_sizes": [8, 16, 32], "amp": True},
                  "benchmark": {"batch_sizes": [1, 8, 32], "evaluation_batch_size": 32,
                                "item_chunk_size": 65536}}
        variant = {"name": kind, "kind": kind, "summary_tokens": 0 if kind == "recent" else 4}
        cohort = SimpleNamespace(users=tuple(range(257)), num_items=1_049_612)
        data = SimpleNamespace(summary={"validation_anchors": 26335, "test_anchors": 25571})
        return config, variant, cohort, data

    def test_batch_selection_requires_actual_updates_and_finite_rates(self):
        from rhpr.sasrec.query_profile import _choose_training_batch
        rows = [{"status": "ok", "physical_batch_size": batch,
                 "measured_successful_updates": successful, "amp_skipped_updates": skipped,
                 "train_windows_per_second_with_prepare_h2d_sync": rate}
                for batch, successful, skipped, rate in
                [(8, 1, 0, 30.), (16, 0, 0, 500.), (32, 1, 1, 400.), (64, 1, 0, float("nan"))]]
        self.assertEqual(_choose_training_batch(rows), (8, 30.))
        with self.assertRaisesRegex(RuntimeError, "successful optimizer update"):
            _choose_training_batch(rows[1:])

    def test_long_and_recent_profiles_use_distinct_training_and_serving_contexts(self):
        from rhpr.sasrec.query_profile import profile_query_variant
        for kind in ("query_ssm", "recent"):
            with self.subTest(kind=kind):
                config, variant, cohort, data = self.fixture(kind)
                models, train_contexts, eval_contexts = [], [], []
                def build(*args):
                    model = SimpleNamespace(compressor=SimpleNamespace(tail_size=-1))
                    model.to = lambda device: model
                    models.append(model)
                    return model
                def train(model, cohort, schedule, profile, context, physical, *args):
                    self.assertEqual(model.compressor.tail_size, 159)
                    self.assertEqual(profile["tail_size"], 32)
                    self.assertEqual(profile["warmup_microbatches"], 3)
                    train_contexts.append(context)
                    return {"status": "ok", "physical_batch_size": physical,
                            "measured_successful_updates": 1, "amp_skipped_updates": 0,
                            "train_windows_per_second_with_prepare_h2d_sync": float(physical)}
                def rate(model, *args, context_length, batch_size, **kwargs):
                    self.assertEqual(model.compressor.tail_size, 128)
                    self.assertEqual(kwargs["sample_count"], 128)
                    self.assertIs(model, models[-1])
                    self.assertEqual(len(models), 4)
                    eval_contexts.append(context_length)
                    value = 100. if batch_size == 8 else 200.
                    return {"status": "ok", "batch_size": batch_size, "sample_anchors": 128,
                            "wall_seconds": 128 / value, "anchors_per_second": value,
                            "quality_used_for_selection": False}
                module = "rhpr.sasrec.query_profile."
                with patch(module + "_build_query_model", side_effect=build), \
                     patch(module + "_optimizer"), patch(module + "_scaler"), patch(module + "_cleanup"), \
                     patch(module + "_train_shape", side_effect=train), \
                     patch(module + "_measure_daily_rate", side_effect=rate):
                    result = profile_query_variant(config, variant, cohort, data, (None,) * 128, torch.device("cpu"))
                self.assertEqual(train_contexts, [159 if kind == "recent" else 8159] * 3)
                self.assertEqual(eval_contexts, [128 if kind == "recent" else 8128] * 2)
                self.assertEqual(result["training_batch_size"], 32)
                self.assertEqual(result["evaluation_batch_size"], 32)
                self.assertAlmostEqual(result["estimated_train_pass_seconds"], 257 / 32.)
                self.assertAlmostEqual(result["estimated_validation_seconds"], 26335 / 200.)
                self.assertAlmostEqual(result["estimated_test_seconds"], 25571 / 200.)
                self.assertFalse(result["profile_weights_reused_for_training"])
                self.assertEqual(len({id(model) for model in models}), 4)

    def test_training_failure_aborts_before_evaluation(self):
        from rhpr.sasrec.query_profile import profile_query_variant
        config, variant, cohort, data = self.fixture()
        model = SimpleNamespace(compressor=SimpleNamespace(tail_size=-1), to=lambda device: model)
        module = "rhpr.sasrec.query_profile."
        with patch(module + "_build_query_model", return_value=model), \
             patch(module + "_optimizer"), patch(module + "_scaler"), patch(module + "_cleanup"), \
             patch(module + "_train_shape", side_effect=FloatingPointError("nonfinite measured norm")), \
             patch(module + "_measure_daily_rate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "successful optimizer update"):
                profile_query_variant(config, variant, cohort, data, (None,) * 128, torch.device("cpu"))
        evaluate.assert_not_called()

    def test_evaluation_oom_uses_other_parallel_batch_without_quality_selection(self):
        from rhpr.sasrec.query_profile import profile_query_variant
        config, variant, cohort, data = self.fixture()
        def build(*args):
            model = SimpleNamespace(compressor=SimpleNamespace(tail_size=-1))
            model.to = lambda device: model
            return model
        def rate(*args, batch_size, **kwargs):
            if batch_size == 32:
                raise torch.OutOfMemoryError("too big")
            return {"status": "ok", "batch_size": 8, "anchors_per_second": 10.,
                    "quality_used_for_selection": False}
        module = "rhpr.sasrec.query_profile."
        with patch(module + "_build_query_model", side_effect=build), \
             patch(module + "_optimizer"), patch(module + "_scaler"), patch(module + "_cleanup"), \
             patch(module + "_train_shape", side_effect=lambda *a: {
                 "status": "ok", "physical_batch_size": a[5], "measured_successful_updates": 1,
                 "amp_skipped_updates": 0, "train_windows_per_second_with_prepare_h2d_sync": 2.}), \
             patch(module + "_measure_daily_rate", side_effect=rate):
            result = profile_query_variant(config, variant, cohort, data, (None,) * 128, torch.device("cpu"))
        self.assertEqual(result["evaluation_batch_size"], 8)
        self.assertEqual(result["evaluation"][-1]["status"], "oom")


if __name__ == "__main__":
    unittest.main()
