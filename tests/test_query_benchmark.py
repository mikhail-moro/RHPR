from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch required")
class QueryBenchmarkTest(unittest.TestCase):
    def test_native_scan_cost_is_not_lost_in_matmul_hooks(self):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.long_quality_benchmark import _batch, _observe_flops
        from rhpr.sasrec.query_ssm import make_query_compressor

        compressor = make_query_compressor(kind="query_ssm", d_model=8, summary_tokens=2,
            tail_size=4, query_context_size=4, n_heads=2, d_inner=4, d_state=3, gate_rank=3)
        model = HSTUModel(num_items=20, max_seq_len=8, d_model=8, n_heads=2,
            n_layers=1, ffn_multiplier=4, dropout=0,
            custom_sequence_compressor=compressor, normalization_length=6).eval()
        batch = _batch([np.arange(1, 9)], model, torch.device("cpu"))
        with torch.no_grad():
            _, flops = _observe_flops(model, batch, torch.device("cpu"), False)
        # Query MHA1792 + output FFN2048 + history/query/readout projections912
        # + bias-free delta F.linear192 (which has no nn.Linear forward hook).
        self.assertEqual(flops["compressor_flops"], 4944)
        # B*K*old*d_inner=32 streams; per-stream arithmetic7*d_state-1=20.
        self.assertEqual(flops["scan_arithmetic_flops"], 640)
        self.assertEqual(flops["scan_exp_evaluations"], 96)
        self.assertEqual(flops["scan_softplus_evaluations"], 32)
        self.assertEqual(flops["compressor_arithmetic_including_scan_flops"], 5584)
        self.assertEqual(flops["full_arithmetic_including_scan_flops"], flops["full_flops"] + 640)

    def test_serving_benchmarks_common_parallel_batches_and_records_oom(self):
        from rhpr.sasrec.query_pilot import _benchmark_query_serving

        config = {"training": {"amp": True}, "benchmark": {
            "batch_sizes": [1, 8, 32], "warmup": 3, "iterations": 8, "item_chunk_size": 65536}}
        for large_status in ("ok", "oom"):
            with self.subTest(large_status=large_status):
                calls = []
                def measure(model, contexts, *, batch_size, **kwargs):
                    calls.append(batch_size)
                    return {"metrics": {"full_b1_wall_p50_ms": 1., f"full_b{batch_size}_wall_p50_ms": 2.},
                            "cases": {"b1": {"batch_size": 1}, f"b{batch_size}": {"batch_size": batch_size}},
                            "metadata": {"cache": False}}
                preflight = {"evaluation_batch_size": 8, "evaluation": [
                    {"batch_size": 8, "status": "ok"}, {"batch_size": 32, "status": large_status}]}
                with patch("rhpr.sasrec.query_pilot.benchmark_long_quality", side_effect=measure):
                    result = _benchmark_query_serving(object(), [], config=config,
                        preflight=preflight, device=torch.device("cpu"))
                self.assertEqual(calls, [8, 32] if large_status == "ok" else [8])
                self.assertEqual(result["metadata"]["measured_batch_sizes"], [1, 8, 32] if large_status == "ok" else [1, 8])
                self.assertEqual(len(result["metadata"]["skipped_batch_sizes"]), int(large_status == "oom"))
                self.assertEqual(result["metadata"]["requested_batch_sizes"], [1, 8, 32])


if __name__ == "__main__":
    unittest.main()
