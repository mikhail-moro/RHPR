from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - optional local dependency
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class SASRecBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.threads)

    def _model(self):
        from rhpr.sasrec.model import SASRecModel

        return SASRecModel(
            num_items=7, max_seq_len=6, d_model=8, n_heads=2,
            n_layers=1, ffn_multiplier=2, dropout=0.0,
        )

    def test_dense_flops_use_two_per_multiply_add(self) -> None:
        from rhpr.sasrec.benchmark import dense_encoder_flops

        dimensions = {"batch_size": 1, "d_model": 256, "n_layers": 4, "ffn_multiplier": 4}
        self.assertEqual(dense_encoder_flops(sequence_length=1028, **dimensions), 10_796_204_032)
        self.assertEqual(dense_encoder_flops(sequence_length=578, **dimensions), 5_004_869_632)
        with self.assertRaises(ValueError):
            dense_encoder_flops(sequence_length=0, **dimensions)

    def test_chunked_catalogue_matches_dense_ranking_and_excludes_padding(self) -> None:
        from rhpr.sasrec.benchmark import score_catalogue

        model = self._model()
        states = torch.zeros((2, 8))
        states[0, 0], states[1, 1] = 1.0, 1.0
        with torch.no_grad():
            model.item_embedding.weight.zero_()
            model.item_embedding.weight[0].fill_(10_000)  # Padding must never be scored.
            model.item_embedding.weight[1:, 0] = torch.arange(1, 8)
            model.item_embedding.weight[1:, 1] = torch.arange(7, 0, -1)
        scores, ids = score_catalogue(model, states, k=3, item_chunk_size=2)
        expected_scores, expected_ids = (states @ model.item_embedding.weight[1:].T).topk(3, dim=1)
        torch.testing.assert_close(scores, expected_scores)
        torch.testing.assert_close(ids, expected_ids + 1)
        self.assertEqual(scores.dtype, torch.float32)
        _, all_ids = score_catalogue(model, states[:1], k=10, item_chunk_size=1)
        self.assertEqual(all_ids.shape, (1, 7))

    def test_regular_benchmark_rejects_insufficient_samples(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least 200"):
                benchmark_inference(
                    self._model(), [np.array([1, 2])], device=torch.device("cpu"),
                    amp=False, output_dir=Path(directory), config={"iterations": 2},
                )

    def test_smoke_benchmark_records_separate_stages_shapes_and_finite_metrics(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference, dense_encoder_flops

        model = self._model()
        model.train()
        with tempfile.TemporaryDirectory() as directory:
            metrics = benchmark_inference(
                model, [np.array([1, 2]), np.array([3, 4, 5, 6])],
                device=torch.device("cpu"), amp=False, output_dir=Path(directory),
                k=3, item_chunk_size=2,
                config={"smoke_test": True, "warmup": 0, "iterations": 2,
                        "throughput_iterations": 2, "batch_size": 2, "fixed_lengths": [6],
                        "real_sample_size": 2},
            )
            self.assertTrue(model.training)
            self.assertTrue(metrics["inference_benchmark_smoke_test"])
            for value in metrics.values():
                self.assertIsInstance(value, (float, int, bool))
                self.assertTrue(math.isfinite(value))
            self.assertGreater(metrics["inference_history_b1_p50_ms"], 0)
            self.assertGreater(metrics["inference_scoring_b1_p95_ms"], 0)
            self.assertGreater(metrics["inference_full_b2_users_per_second"], 0)
            metadata = json.loads((Path(directory) / "benchmark.json").read_text())["metadata"]
            self.assertFalse(metadata["encoder_amp"])
            self.assertEqual(metadata["multiply_add_flops"], 2)
            samples = json.loads((Path(directory) / "benchmark_samples.json").read_text())
            self.assertEqual(set(samples), {"n6", "real"})
            self.assertEqual(len(samples["real"]["b1"]["full"]["samples_ms"]), 2)
            shapes = samples["real"]["throughput_shapes"]
            self.assertEqual(shapes[0]["raw_padded_length"], 4)
            self.assertEqual(shapes[0]["nominal"]["encoder_length"], 4)
            expected = dense_encoder_flops(batch_size=2, sequence_length=4, d_model=8,
                                           n_layers=1, ffn_multiplier=2) / 2
            self.assertEqual(metrics["inference_encoder_b2_dense_matmul_flops"], expected)
            self.assertEqual(samples["n6"]["throughput_shapes"][0]["batch_size"], 2)
            self.assertEqual(metrics["inference_b1_length_513to1028_context_count"], 0)
            self.assertNotIn("inference_full_b1_length_513to1028_p95_ms", metrics)

    def test_throughput_keeps_physical_batch_fixed_when_real_sample_has_remainder(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference

        with tempfile.TemporaryDirectory() as directory:
            metrics = benchmark_inference(
                self._model(), [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])],
                device=torch.device("cpu"), amp=False, output_dir=Path(directory),
                config={"smoke_test": True, "warmup": 0, "iterations": 3,
                        "throughput_iterations": 2, "batch_size": 2, "fixed_lengths": []},
            )
            samples = json.loads((Path(directory) / "benchmark_samples.json").read_text())["real"]
            self.assertEqual(metrics["inference_real_context_count"], 3)
            self.assertEqual(samples["throughput_dropped_context_count"], 1)
            self.assertEqual([shape["batch_size"] for shape in samples["throughput_shapes"]], [2])
            self.assertEqual(samples["throughput"]["full"]["users"], 4)

    def test_shape_hooks_count_executed_projection_rows_not_masked_padding(self) -> None:
        from rhpr.sasrec.benchmark import _prepare_batch, _profile_shapes

        class PrefixPair(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(16, 8)

            def forward(self, embeddings, valid_mask):
                lengths = valid_mask.sum(dim=1)
                outputs, positions = [], []
                for row, length in enumerate(lengths.tolist()):
                    pairs = length // 2
                    output = self.projection(embeddings[row, :2 * pairs].reshape(pairs, 16))
                    source = torch.arange(1, 2 * pairs, 2)
                    if length % 2:
                        output = torch.cat((output, embeddings[row, length - 1:length]))
                        source = torch.cat((source, torch.tensor([length - 1])))
                    outputs.append(output)
                    positions.append(source)
                padded = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
                source_positions = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True, padding_value=-1)
                return padded, source_positions >= 0, source_positions

        model = self._model()
        model.compressor = PrefixPair()
        model.eval()
        batch = _prepare_batch([np.array([1, 2, 3]), np.array([4, 5, 6, 7])], None, torch.device("cpu"))
        with torch.inference_mode():
            _profile_shapes(model, batch, torch.device("cpu"), False)
        self.assertEqual(batch["nominal"]["encoder_length"], 2)
        self.assertEqual(batch["nominal"]["projection_rows"], 3)
        self.assertEqual(batch["nominal"]["compressor_flops"], 2 * 3 * 16 * 8)

    def test_disabled_benchmark_does_not_create_artifacts(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            self.assertEqual(benchmark_inference(self._model(), [], device=torch.device("cpu"),
                                                amp=False, output_dir=output, config={"enabled": False}), {})
            self.assertFalse(output.exists())

    def test_time_pair_benchmark_uses_observed_buckets_and_actual_pair_shapes(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference
        from rhpr.sasrec.model import SASRecModel

        model = SASRecModel(num_items=7, max_seq_len=6, d_model=8, n_heads=2,
                            n_layers=1, ffn_multiplier=2, dropout=0.0,
                            compressor_type="pair_mlp", tail_size=2,
                            use_time_features=True, num_gap_buckets=4)
        contexts = [np.array([1, 2, 3, 4, 5, 6]), np.array([2, 3, 4])]
        gaps = [np.array([0, 1, 2, 3, 1, 2]), np.array([0, 2, 1])]
        options = {"smoke_test": True, "warmup": 0, "iterations": 2,
                   "throughput_iterations": 1, "batch_size": 2, "fixed_lengths": [6]}
        with tempfile.TemporaryDirectory() as directory:
            metrics = benchmark_inference(model, contexts, gap_ids=gaps, device=torch.device("cpu"),
                                           amp=False, output_dir=Path(directory), config=options)
            self.assertEqual(metrics["inference_compressor_b1_n6_dense_matmul_flops"], 4 * 2 * 8**2)
            shapes = json.loads((Path(directory) / "benchmark_samples.json").read_text())["real"]["throughput_shapes"]
            self.assertEqual(shapes[0]["nominal"]["encoder_length"], 4)
            self.assertEqual(shapes[0]["nominal"]["projection_rows"], 2)
            with self.assertRaisesRegex(ValueError, "requires observed gap"):
                benchmark_inference(model, contexts, device=torch.device("cpu"), amp=False,
                                    output_dir=Path(directory), config=options)

    def test_optional_profiler_is_separate_and_reports_uncounted_ops(self) -> None:
        from rhpr.sasrec.benchmark import benchmark_inference

        with tempfile.TemporaryDirectory() as directory:
            benchmark_inference(self._model(), [np.array([1, 2, 3])], device=torch.device("cpu"),
                                amp=False, output_dir=Path(directory),
                                config={"smoke_test": True, "warmup": 0, "iterations": 1,
                                        "throughput_iterations": 1, "batch_size": 1,
                                        "fixed_lengths": [], "profile": True})
            profile = json.loads((Path(directory) / "benchmark_profiler.json").read_text())
            self.assertTrue(any(row["reported_flops"] > 0 for row in profile["operators"]))
            self.assertTrue(profile["operators_without_reported_flops"])
            self.assertIn("do not add profiler total", profile["attention_manual_method"])
            self.assertTrue((Path(directory) / "benchmark_profiler_trace.json").is_file())


if __name__ == "__main__":
    unittest.main()
