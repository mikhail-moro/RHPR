from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class PrefixBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self, kind):
        from rhpr.sasrec.model import SASRecModel, TailOnlySequenceCompressor
        from rhpr.sasrec.query_ssm import make_query_compressor
        model = SASRecModel(num_items=11, max_seq_len=64, d_model=8, n_heads=2,
                            n_layers=1, ffn_multiplier=4, dropout=0)
        if kind == "recent":
            model.compressor = TailOnlySequenceCompressor(3)
        else:
            model.compressor = make_query_compressor(kind=kind, d_model=8, summary_tokens=2,
                tail_size=3, query_context_size=4, query_source="old_suffix", n_heads=2,
                d_state=3, d_inner=4, gate_rank=3)
        return model

    def cohort(self, lengths=(11, 17, 23)):
        return SimpleNamespace(users=[SimpleNamespace(user_id=index, train=np.arange(length) % 12)
                                      for index, length in enumerate(lengths)])

    def config(self, sizes=None):
        return {"training": {"amp": False, "raw_tail_size": 3}, "benchmark": {
            "batch_sizes": sizes or [1, 2], "warmup": 0, "iterations": 1,
            "item_chunk_size": 3, "quality_cutoffs": [10, 100]}}

    def test_length_stratification_preserves_complete_prefix_and_recent_control(self):
        from rhpr.sasrec.query_prefix_benchmark import _serving_contexts
        cohort = self.cohort((23, 11, 17))
        full, lengths = _serving_contexts(cohort, 2, recent=False, tail_size=3, max_seq_len=64)
        self.assertEqual(lengths, [11, 23])
        self.assertEqual([len(context) for context in full], lengths)
        median, _ = _serving_contexts(cohort, 1, recent=False, tail_size=3, max_seq_len=64)
        self.assertEqual(len(median[0]), 17)
        recent, lengths = _serving_contexts(cohort, 2, recent=True, tail_size=3, max_seq_len=64)
        self.assertEqual([len(context) for context in recent], [3, 3])
        np.testing.assert_array_equal(recent[-1], full[-1][-3:])
        with self.assertRaisesRegex(ValueError, "cropping is forbidden"):
            _serving_contexts(cohort, 2, recent=False, tail_size=3, max_seq_len=20)

    def test_cross_attention_flops_include_both_attentions_and_sasrec_dense_work(self):
        from rhpr.sasrec.long_quality_benchmark import _batch
        from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
        model = self.model("query_cross_attention").eval()
        batch = _batch([np.arange(17) % 12], model, torch.device("cpu"))
        with torch.inference_mode():
            _, cost = _observe_prefix_flops(model, batch, torch.device("cpu"), False)
        expected = (4 * (2 + 4) * 8**2 + 4 * 2 * 4 * 8
                    + 4 * (2 + 14) * 8**2 + 4 * 2 * 14 * 8
                    + 2 * 2 * (8 * 32 + 32 * 8))
        self.assertEqual(cost["compressor_flops"], expected)
        self.assertEqual(cost["encoder_length"], 5)
        self.assertEqual(cost["encoder_valid_tokens"], 5)
        self.assertEqual(cost["encoder_flops"], (8 + 4 * 4) * 5 * 8**2 + 4 * 5**2 * 8)
        self.assertEqual(cost["scoring_flops"], 2 * 11 * 8)
        self.assertEqual(cost["scan_arithmetic_flops"], 0)

    def test_ssm_counts_unhooked_delta_and_reports_scan_separately(self):
        from rhpr.sasrec.long_quality_benchmark import _batch
        from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
        model = self.model("query_ssm").eval()
        batch = _batch([np.arange(17) % 12], model, torch.device("cpu"))
        with torch.inference_mode():
            _, cost = _observe_prefix_flops(model, batch, torch.device("cpu"), False)
        expected = (4 * (2 + 4) * 8**2 + 4 * 2 * 4 * 8  # Query attention.
                    + 2 * 2 * (8 * 32 + 32 * 8)  # FFN.
                    + 2 * 14 * 8 * 4 + 2 * 14 * 8 * 3  # Shared value and input gate.
                    + 2 * 2 * 8 * 3 + 2 * 2 * 14 * 3 * 3  # Query gate and write.
                    + 2 * 2 * 8 * 3 + 2 * 2 * 4 * 8  # Read and output.
                    + 2 * 2 * 14 * 3 * 4)  # Delta F.linear, unseen by hooks.
        self.assertEqual(cost["compressor_flops"], expected)
        streams = 2 * 14 * 4
        self.assertEqual(cost["scan_arithmetic_flops"], streams * (7 * 3 - 1))
        self.assertEqual(cost["scan_exp_evaluations"], streams * 3)
        self.assertEqual(cost["scan_softplus_evaluations"], streams)
        self.assertEqual(cost["full_arithmetic_including_scan_flops"],
                         cost["full_flops"] + cost["scan_arithmetic_flops"])

    def test_real_cpu_serving_keeps_mode_occupied_zero_and_finite_flat_metrics(self):
        from rhpr.sasrec.query_prefix_benchmark import benchmark_prefix_quality
        for kind in ("recent", "query_cross_attention", "query_ssm"):
            with self.subTest(kind=kind):
                model = self.model(kind).train()
                result = benchmark_prefix_quality(model, self.cohort(), config=self.config(),
                                                  device=torch.device("cpu"))
                self.assertTrue(model.training)
                self.assertEqual(result["cases"]["b1"]["complete_observed_lengths"], [17])
                case = result["cases"]["b2"]
                self.assertEqual(case["status"], "ok")
                self.assertEqual(case["complete_observed_lengths"], [11, 23])
                self.assertEqual(case["input_lengths"], [3, 3] if kind == "recent" else [11, 23])
                self.assertEqual(case["estimated_flops"]["encoder_valid_tokens"], 6 if kind == "recent" else 10)
                self.assertEqual(set(case["stages"]), {"history", "scoring", "full"})
                self.assertTrue(all(math.isfinite(value) for value in result["metrics"].values()))
                self.assertGreater(result["metrics"]["full_b2_users_per_second"], 0)
                self.assertFalse(result["metadata"]["cache"])
                self.assertEqual(result["metadata"]["k"], 100)
                json.dumps(result, allow_nan=False)

    def test_oom_is_explicit_and_cannot_remove_all_parallel_measurements(self):
        from rhpr.sasrec.query_prefix_benchmark import benchmark_prefix_quality, _observe_prefix_flops
        model = self.model("recent").train()
        def observe(*args):
            if len(args[1][0]) == 4:
                raise torch.OutOfMemoryError("simulated batch memory capacity")
            return _observe_prefix_flops(*args)
        with patch("rhpr.sasrec.query_prefix_benchmark._observe_prefix_flops", side_effect=observe):
            result = benchmark_prefix_quality(model, self.cohort(), config=self.config([1, 2, 4]),
                                              device=torch.device("cpu"))
            self.assertEqual(result["cases"]["b4"]["status"], "oom")
            self.assertEqual(result["cases"]["b4"]["stages"], {})
            self.assertFalse(any("b4" in key for key in result["metrics"]))
            with self.assertRaisesRegex(RuntimeError, "no parallel"):
                benchmark_prefix_quality(model, self.cohort(), config=self.config([1, 4]),
                                         device=torch.device("cpu"))
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
