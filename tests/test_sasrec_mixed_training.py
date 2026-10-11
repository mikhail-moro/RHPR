from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import polars as pl
    import torch
except ImportError:
    pl = torch = None


@unittest.skipIf(torch is None, "Torch is required")
class MixedDepthSettingsTest(unittest.TestCase):
    def test_rejects_untrained_modes_invalid_probabilities_and_test_selection(self) -> None:
        from rhpr.sasrec.experiment import _mixed_depth_settings

        model = {"compressor_type": "recursive_pair_mlp"}
        evaluation = {"evaluate_test": False}
        settings = {"probabilities": [0.25] * 4, "validation_interval": 5}
        probabilities, interval = _mixed_depth_settings({"mixed_depth": settings}, model, evaluation)
        self.assertEqual(interval, 5)
        self.assertEqual(probabilities.tolist(), [0.25] * 4)
        self.assertIsNone(_mixed_depth_settings({}, {}, {}))
        for values in ([0.5, 0.5], [0.25] * 3 + [float("nan")],
                       [float("inf"), 0.25, 0.25, 0.25], [0.5, 0.5, 0, 0],
                       [-0.25, 0.25, 0.5, 0.5], [0.2] * 4, [True, 0.25, 0.25, 0.25]):
            with self.subTest(probabilities=values), self.assertRaises(ValueError):
                _mixed_depth_settings({"mixed_depth": {"probabilities": values}}, model, evaluation)
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                _mixed_depth_settings({"mixed_depth": {**settings, "validation_interval": interval}},
                                      model, evaluation)
        with self.assertRaises(ValueError):
            _mixed_depth_settings({"mixed_depth": settings}, {"compressor_type": "pair_mlp"}, evaluation)
        with self.assertRaises(ValueError):
            _mixed_depth_settings({"mixed_depth": settings}, model, {"evaluate_test": True})

    def test_private_depth_rng_does_not_advance_other_random_streams(self) -> None:
        from rhpr.sasrec.experiment import _sample_mixed_depth

        probabilities = torch.tensor([0.25] * 4, dtype=torch.float64)
        depth_rng = torch.Generator().manual_seed(42 + 271)
        negative_rng = torch.Generator().manual_seed(42 + 173)
        order_rng = torch.Generator().manual_seed(42)
        torch.manual_seed(928)
        states = [torch.get_rng_state().clone(), negative_rng.get_state().clone(), order_rng.get_state().clone()]
        draws = [_sample_mixed_depth(probabilities, depth_rng) for _ in range(80)]
        for before, after in zip(states, (torch.get_rng_state(), negative_rng.get_state(), order_rng.get_state())):
            self.assertTrue(torch.equal(before, after))
        replay_rng = torch.Generator().manual_seed(42 + 271)
        replay = []
        for _ in range(80):
            torch.rand(7)
            torch.randint(1, 20, (5,), generator=negative_rng)
            torch.randperm(8, generator=order_rng)
            replay.append(_sample_mixed_depth(probabilities, replay_rng))
        self.assertEqual(draws, replay)
        self.assertEqual(set(draws), {0, 1, 2, 3})


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class MixedDepthTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def test_non_finite_loss_stops_before_backward_or_checkpoint(self) -> None:
        from rhpr.sasrec import experiment

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "listens.parquet"
            pl.DataFrame({"uid": [1], "item_ids": [[1, 2, 3, 4, 5, 6, 1, 2]]}).write_parquet(path)
            config = {"data": {"path": str(path)},
                "model": {"max_seq_len": 8, "d_model": 32, "n_heads": 1,
                    "n_layers": 1, "ffn_multiplier": 1, "tail_size": 2,
                    "compressor_type": "recursive_pair_mlp"},
                "loss": {"name": "sampled_softmax", "num_negatives": 4, "tail_size": 2},
                "training": {"device": "cpu", "amp": False, "workers": 0,
                    "physical_batch_size": 1, "effective_batch_size": 1, "max_epochs": 15,
                    "mixed_depth": {"probabilities": [0.25] * 4, "validation_interval": 5}},
                "evaluation": {"evaluate_test": False},
            }
            for value in (float("nan"), float("inf")):
                with self.subTest(value=value), \
                     patch.object(experiment, "next_item_loss", return_value=torch.tensor(value, requires_grad=True)) as loss, \
                     patch.object(experiment.torch, "save") as save, redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(FloatingPointError, "epoch 1, batch 1, depth"):
                        experiment.run_yambda_experiment(config, directory / "output")
                    loss.assert_called_once()
                    save.assert_not_called()

    def test_real_training_mean_checkpoint_schedule_predictions_and_deterministic_benchmarks(self) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.model import SASRecModel

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "listens.parquet"
            pl.DataFrame({"uid": [11, 12, 13, 14], "item_ids": [
                [(i + 7 * u) % 20 + 1 for i in range(length)]
                for u, length in enumerate((10, 15, 21, 26))
            ]}).write_parquet(path)
            config = {"name": "tiny-mixed", "seed": 42, "deterministic": True,
                "data": {"path": str(path)},
                "model": {"max_seq_len": 24, "d_model": 32, "n_heads": 1,
                    "n_layers": 1, "ffn_multiplier": 1, "dropout": 0.0,
                    "tail_size": 4, "compressor_type": "recursive_pair_mlp", "compression_depth": 1},
                "loss": {"name": "sampled_softmax", "num_negatives": 4, "tail_size": 4},
                "training": {"device": "cpu", "amp": False, "workers": 0,
                    "physical_batch_size": 2, "effective_batch_size": 4, "max_epochs": 11,
                    "early_stopping_patience": 1, "keep_checkpoint": True,
                    "progress_interval_batches": 0,
                    "mixed_depth": {"probabilities": [0.25] * 4, "validation_interval": 5}},
                "evaluation": {"batch_size": 2, "item_chunk_size": 7,
                    "evaluate_test": False, "progress_interval_batches": 0},
                "benchmark": {"enabled": True, "smoke_test": True, "warmup": 0,
                    "iterations": 1, "throughput_iterations": 1, "batch_size": 2,
                    "real_sample_size": 2, "fixed_lengths": [24]},
            }
            output = directory / "output"
            training_shapes = []
            final_states = []
            validation_calls = []
            partial_history_lengths = []
            original_evaluate = experiment._full_catalog_ndcg
            original_load = torch.load
            scores = ([0.2, 0.5, 0.5, 0.5], [0.6, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1])

            def build_model(**kwargs):
                model = SASRecModel(**kwargs)

                def inspect_encoder(module, inputs):
                    if model.training:
                        training_shapes.append((model.compression_depth, inputs[0].shape[0], inputs[0].shape[1]))

                model.encoder.register_forward_pre_hook(inspect_encoder)
                return model

            def evaluate(model, loader, **kwargs):
                self.assertEqual(kwargs["phase"], "validation")
                depth = model.compression_depth
                if kwargs.get("prediction_rows") is not None:
                    final_states.append({name: value.detach().cpu().clone()
                                         for name, value in model.state_dict().items()})
                    return original_evaluate(model, loader, **kwargs)
                partial_history_lengths.append(len(json.loads((output / "history.json").read_text())))
                row = len(validation_calls) // 4
                validation_calls.append(depth)
                kwargs["details"].update({"recall_at_10": scores[row][depth], "groups": {}})
                return scores[row][depth], len(loader.dataset)

            with patch.object(experiment, "SASRecModel", side_effect=build_model), \
                 patch.object(experiment, "_full_catalog_ndcg", side_effect=evaluate), \
                 patch.object(experiment.torch, "load", wraps=original_load) as load, \
                 redirect_stdout(io.StringIO()):
                metrics = experiment.run_yambda_experiment(config, output)
                self.assertEqual(load.call_count, 1)

            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 11)  # patience=1 must not stop this screen.
            self.assertEqual([entry["epoch"] for entry in history if entry["validation_performed"]], [5, 10, 11])
            self.assertEqual(validation_calls, [0, 1, 2, 3] * 3)
            self.assertEqual(partial_history_lengths, [4] * 4 + [9] * 4 + [10] * 4)
            self.assertEqual(metrics["best_epoch"], 5)  # depth0 alone would select epoch11.
            self.assertAlmostEqual(metrics["validation_mean_ndcg_at_10"], 0.425)
            self.assertEqual(metrics["epochs_trained"], 11)
            self.assertFalse(metrics["early_stopping_enabled"])
            self.assertFalse(metrics["test_evaluated"])
            self.assertNotIn("test_ndcg_at_10", metrics)
            for entry in history:
                stats = entry["mixed_depth_training"]
                self.assertEqual(sum(row["batches"] for row in stats.values()), 2)
                self.assertEqual(sum(row["users"] for row in stats.values()), 4)
                for row in stats.values():
                    self.assertLessEqual(row["encoder_valid_tokens"], row["encoder_padded_tokens"])
            for depth in range(4):
                shapes = [(batch, width) for mode, batch, width in training_shapes if mode == depth]
                self.assertEqual(metrics[f"depth_{depth}_training_batches"], len(shapes))
                self.assertEqual(metrics[f"depth_{depth}_training_users"], sum(batch for batch, _ in shapes))
                self.assertEqual(metrics[f"depth_{depth}_training_encoder_padded_tokens"],
                                 sum(batch * width for batch, width in shapes))

            checkpoint = original_load(output / "best-model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(len(final_states), 4)
            for state in final_states:
                self.assertEqual(state.keys(), checkpoint.keys())
                self.assertTrue(all(torch.equal(state[name], checkpoint[name]) for name in state))
            predictions = pl.read_parquet(output / "predictions.parquet")
            self.assertEqual(predictions.height, 16)
            self.assertEqual(set(predictions["compression_depth"].to_list()), {0, 1, 2, 3})
            self.assertEqual(set(predictions["model"].to_list()), {f"depth_{depth}" for depth in range(4)})
            checksum = hashlib.sha256((output / "best-model.pt").read_bytes()).hexdigest()
            for depth in range(4):
                run = output / "runs" / f"depth_{depth}"
                benchmark = json.loads((run / "benchmark.json").read_text())
                provenance = benchmark["metadata"]["config"]
                self.assertEqual(provenance["compression_depth"], depth)
                self.assertEqual(provenance["checkpoint_sha256"], checksum)
                self.assertEqual(provenance["checkpoint"], str(output / "best-model.pt"))
                mode_metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(mode_metrics["best_epoch"], 5)
                self.assertEqual(mode_metrics["validation_examples"], 4)
                self.assertIn("inference_full_b1_p50_ms", mode_metrics)
                rows = predictions.filter(pl.col("compression_depth") == depth)
                self.assertAlmostEqual(rows["ndcg_at_10"].mean(), metrics[f"depth_{depth}_validation_ndcg_at_10"], places=6)
            for value in metrics.values():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
