from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class HSTUBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_flops_count_valid_projection_rows_and_both_attention_dimensions(self):
        from rhpr.sasrec.benchmark import hstu_encoder_flops

        layer = SimpleNamespace(
            _uvqk=torch.empty(8, 2 * 2 * (3 + 5)),
            _o=torch.nn.Linear(2 * 5, 8),
            _num_heads=2, _attention_dim=3, _linear_dim=5,
        )
        # Two sequences of lengths 2 and 4: six jagged projection rows,
        # but dense attention GEMMs execute over B=2 and padded T=4.
        expected = 2 * 6 * (8 * 32 + 10 * 8) + 2 * 2 * 4**2 * 2 * (3 + 5)
        self.assertEqual(hstu_encoder_flops(
            batch_size=2, sequence_length=4, valid_tokens=6, layers=[layer]), expected)
        with self.assertRaises(ValueError):
            hstu_encoder_flops(batch_size=2, sequence_length=4, valid_tokens=9, layers=[layer])

    def test_native_hstu_benchmark_records_packing_and_architecture(self):
        from rhpr.sasrec.benchmark import benchmark_inference, hstu_encoder_flops
        from rhpr.sasrec.hstu import HSTUModel

        model = HSTUModel(num_items=12, max_seq_len=6, d_model=32,
                          n_heads=2, n_layers=1, ffn_multiplier=4, dropout=0.0,
                          compressor_type="tail_only", tail_size=3)
        with tempfile.TemporaryDirectory() as directory:
            metrics = benchmark_inference(
                model, [np.array([1, 2]), np.array([3, 4, 5, 6])],
                device=torch.device("cpu"), amp=False, output_dir=Path(directory),
                config={"smoke_test": True, "warmup": 0, "iterations": 2,
                        "throughput_iterations": 1, "batch_size": 2, "fixed_lengths": []},
            )
            samples = json.loads((Path(directory) / "benchmark_samples.json").read_text())
            shape = samples["real"]["throughput_shapes"][0]
            self.assertEqual(shape["raw_padded_length"], 4)
            self.assertEqual(shape["nominal"]["encoder_length"], 3)
            self.assertEqual(shape["nominal"]["encoder_projection_rows"], 5)
            expected = hstu_encoder_flops(batch_size=2, sequence_length=3,
                                          valid_tokens=5, layers=model.encoder.layers)
            self.assertEqual(metrics["inference_encoder_b2_dense_matmul_flops"], expected / 2)
            metadata = json.loads((Path(directory) / "benchmark.json").read_text())["metadata"]
            self.assertEqual(metadata["architecture"], "hstu")
            self.assertEqual(metadata["model"]["hstu_normalization_length"], 6)
            self.assertIn("SiLU", metadata["flop_excludes"])


if __name__ == "__main__":
    unittest.main()
