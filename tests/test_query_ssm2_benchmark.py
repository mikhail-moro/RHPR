from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QuerySSM2BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_ssm2_benchmark.json").read_text())

    def test_registration_pins_three_short_native_arms_full_catalogue_and_no_quality_work(self):
        from rhpr.sasrec.query_ssm2_benchmark import validate_ssm2_benchmark_config
        config = self.config()
        validate_ssm2_benchmark_config(config)
        for changed in (dict(config["measurement"], measured_updates=50),
                        dict(config["measurement"], physical_batch_size=16),
                        dict(config["measurement"], quality_training=True)):
            with self.assertRaises(ValueError):
                validate_ssm2_benchmark_config(dict(config, measurement=changed))
        for field, value in (("seed", 43), ("arms", config["arms"][:2])):
            with self.assertRaises(ValueError):
                validate_ssm2_benchmark_config(dict(config, **{field: value}))
        changed = copy.deepcopy(config)
        changed["initializer"]["architecture"]["d_inner"] = 128
        with self.assertRaises(ValueError):
            validate_ssm2_benchmark_config(changed)
        self.assertEqual(config["kaggle"]["kernel_sources"], [])
        self.assertEqual(len(config["kaggle"]["dataset_sources"]), 2)

    def test_models_preserve_exact_original_backbone_positions_and_use_no_checkpoint_weights(self):
        from rhpr.sasrec.query_ssm2_benchmark import benchmark_arms, build_benchmark_model, _backbone_hash
        config = self.config()
        reference = None
        with patch("torch.load", side_effect=AssertionError("no source weights may be read")):
            for arm in benchmark_arms():
                model = build_benchmark_model(config, arm, 20)
                digest = _backbone_hash(model)
                reference = reference or digest
                self.assertEqual(digest, reference)
                self.assertEqual(model.position_embedding.num_embeddings, 32768)
                self.assertEqual(model.compressor.summary_tokens, 32)
                self.assertEqual(model.compressor.d_inner, 64)
                self.assertEqual(model.compressor.query_source, "old_suffix")
                self.assertFalse(model.history_source_checkpoint_loaded)
                if arm["backend"] == "mamba2_ssd":
                    self.assertEqual(model.compressor.head_dim, arm["head_dim"])
                    self.assertEqual(model.compressor.c_conditioning, "query")

    def test_actual_batch_preparation_keeps_shifted_targets_and_identity_bound_negatives(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_rotating_protocol import rotating_schedule, rotating_batch_arrays
        from rhpr.sasrec.query_ssm2_benchmark import prepare_benchmark_batches
        config = self.config()
        users = tuple(LongHistoryUser(uid, np.arange(1200, dtype=np.int32) % 20 + 1,
            np.arange(1200, dtype=np.uint32)) for uid in range(896))
        cohort = LongHistoryCohort(users, np.arange(20, dtype=np.int64), {})
        batches, metadata = prepare_benchmark_batches(config, cohort)
        self.assertEqual(len(batches), 7)
        self.assertEqual(metadata["distinct_users"], 896)
        self.assertTrue(metadata["same_arrays_for_every_arm"])
        windows = rotating_schedule(cohort, seed=42, epoch=1)[:128]
        expected = rotating_batch_arrays(cohort, windows, config,
            {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}, old_length=1024)
        for key in expected:
            np.testing.assert_array_equal(batches[0][key], expected[key])
        np.testing.assert_array_equal(batches[0]["targets"][:, :-1], batches[0]["item_ids"][:, 1:])
        again = prepare_benchmark_batches(config, cohort)[1]
        self.assertEqual(metadata["batch_arrays_sha256"], again["batch_arrays_sha256"])
        with self.assertRaisesRegex(ValueError, "distinct actual users"):
            prepare_benchmark_batches(config, LongHistoryCohort(users[:128], cohort.catalog_item_ids, {}))

    def test_full_update_really_accumulates_four_microbatches_and_allocates_adam_state(self):
        from rhpr.sasrec.query_ssm2_benchmark import _full_update
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(1.))
                self.calls = []
            def encode(self, ids, mask):
                self.calls.append(len(ids))
                return self.value * ids.float()
        model = TinyModel()
        arrays = {"item_ids": np.ones((128, 3), dtype=np.int64),
            "targets": np.ones((128, 3), dtype=np.int64), "valid_mask": np.ones((128, 3), dtype=np.bool_),
            "negative_ids": np.ones((128, 512), dtype=np.int64)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with patch("rhpr.sasrec.query_ssm2_benchmark._loss", side_effect=lambda model, encoded, *args: encoded.square().mean()):
            row = _full_update(model, arrays, self.config(), torch.device("cpu"), optimizer, scaler)
        self.assertEqual(model.calls, [32] * 4)
        self.assertEqual(row["successful_updates"], 1)
        self.assertEqual(row["amp_skipped_updates"], 0)
        self.assertEqual(row["loss"], 1.)
        self.assertTrue(optimizer.state)
        self.assertLess(float(model.value.detach()), 1.)
        self.assertTrue(all(value.grad is None for value in model.parameters()))

    def test_failed_native_arm_is_reported_without_cpu_or_precision_retry(self):
        from rhpr.sasrec.query_ssm2_benchmark import benchmark_arms, benchmark_one_arm
        with patch("rhpr.sasrec.query_ssm2.audit_native_query_ssm2", side_effect=AssertionError("T4 SSD unsupported")) as audit:
            with patch("rhpr.sasrec.query_ssm2_benchmark.build_benchmark_model", side_effect=AssertionError("must stop before large model")):
                row = benchmark_one_arm(self.config(), benchmark_arms()[1], [], 20, torch.device("cuda"), float("inf"))
        self.assertEqual(audit.call_count, 1)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["failure"]["stage"], "native_audit")
        self.assertEqual(row["failure"]["message"], "T4 SSD unsupported")
        self.assertFalse(row["automatic_fallback"])

    def test_short_runner_preserves_successful_baseline_when_ssd_fails(self):
        from rhpr.sasrec.query_ssm2_benchmark import run_query_ssm2_benchmark
        config = self.config()
        cohort = SimpleNamespace(num_items=1049612,
            summary={key: value for key, value in config["data"].items() if key != "prepared_path"})
        full = {"wall_seconds": {"p50": 1.}, "peak_allocated_gib": 5.}
        comp = {"wall_seconds": {"p50": .1}}
        def measure(config, arm, *args):
            if arm["name"] == "qssm1_native":
                return {"status": "success", "fresh_backbone_sha256": "same", "full_training": full,
                        "compressor_only": comp}
            return {"status": "failed", "failure": {"stage": "native_audit", "message": "unsupported kernel"}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("rhpr.sasrec.query_ssm2_benchmark.resolve_dataset_directory", return_value=Path(temporary)), \
                 patch("rhpr.sasrec.query_ssm2_benchmark.load_prepared_cohort", return_value=cohort), \
                 patch("rhpr.sasrec.query_ssm2_benchmark.prepare_benchmark_batches", return_value=([], {})), \
                 patch("rhpr.sasrec.query_ssm2_benchmark._versions", return_value={"gpu_name": "Tesla T4"}), \
                 patch("rhpr.sasrec.query_ssm2_benchmark.benchmark_one_arm", side_effect=measure) as benchmark:
                metrics = run_query_ssm2_benchmark(config, output)
            self.assertEqual(benchmark.call_count, 3)
            self.assertTrue(metrics["benchmark_complete"])
            self.assertFalse(metrics["all_arms_succeeded"])
            self.assertEqual(metrics["qssm1_native_update_p50_seconds"], 1.)
            self.assertNotIn("qssm2_h32_full_step_speedup", metrics)
            report = json.loads((output / "query-ssm2-benchmark.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["arms"]["qssm1_native"]["status"], "success")
            self.assertFalse(report["quality_metrics_computed"])
            json.dumps(metrics, allow_nan=False)
            self.assertIn("unsupported kernel", (output / "query-ssm2-benchmark.md").read_text())

    def test_runner_forbids_cpu_timing_fallback(self):
        from rhpr.sasrec.query_ssm2_benchmark import run_query_ssm2_benchmark
        with patch("torch.cuda.is_available", return_value=False), tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "CPU timing fallback is forbidden"):
                run_query_ssm2_benchmark(self.config(), Path(temporary))


if __name__ == "__main__":
    unittest.main()
