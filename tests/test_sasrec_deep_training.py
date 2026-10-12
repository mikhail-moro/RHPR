from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import polars as pl
    import torch
except ImportError:
    pl = torch = None


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class Recent128TrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def test_validation_interval_is_strict_and_defaults_to_each_epoch(self) -> None:
        from rhpr.sasrec.experiment import _validation_interval

        self.assertEqual(_validation_interval({}), 1)
        self.assertEqual(_validation_interval({"validation_interval": 5}), 5)
        for value in (0, -1, True, 1.5, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validation_interval({"validation_interval": value})

    def test_recent_training_keeps_original_tail_targets_and_selects_only_scheduled_checks(self) -> None:
        self._exercise_recent_training(max_epochs=15, expected_checks=[5, 10, 15],
                                      scores=[0.1, 0.3, 0.2], expected_best=10)

    def test_final_epoch_is_validated_even_between_scheduled_checks(self) -> None:
        self._exercise_recent_training(max_epochs=6, expected_checks=[5, 6],
                                      scores=[0.1, 0.3], expected_best=6)

    def _exercise_recent_training(self, *, max_epochs, expected_checks, scores, expected_best) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.diagnostics import context_hash
        from rhpr.sasrec.losses import build_loss_mask
        from rhpr.sasrec.model import SASRecModel

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "listens.parquet"
            histories = [[(i + 7 * user) % 20 + 1 for i in range(length)]
                         for user, length in enumerate((8, 24, 82))]
            pl.DataFrame({"uid": [11, 12, 13], "item_ids": histories}).write_parquet(path)
            config = {"seed": 42, "deterministic": True, "data": {"path": str(path)},
                "model": {"max_seq_len": 64, "d_model": 32, "n_heads": 1,
                    "n_layers": 1, "ffn_multiplier": 1, "dropout": 0.0,
                    "tail_size": 4, "compressor_type": "tail_only"},
                "loss": {"name": "sampled_softmax", "num_negatives": 4, "tail_size": 4},
                "training": {"device": "cpu", "amp": False, "workers": 0,
                    "physical_batch_size": 2, "effective_batch_size": 4,
                    "max_epochs": max_epochs, "validation_interval": 5,
                    "early_stopping_patience": max_epochs + 1, "keep_checkpoint": True,
                    "progress_interval_batches": 0},
                "evaluation": {"batch_size": 2, "item_chunk_size": 7,
                    "evaluate_test": False, "progress_interval_batches": 0},
                "diagnostics": {"enabled": True},
                "benchmark": {"enabled": True, "smoke_test": True, "warmup": 0,
                    "iterations": 1, "throughput_iterations": 1, "batch_size": 2,
                    "real_sample_size": 2, "fixed_lengths": [64]},
            }
            output = directory / "output"
            training_widths = []
            supervised_lengths = []
            validation_history_lengths = []
            final_state = {}
            original_loss = experiment.next_item_loss
            original_evaluate = experiment._full_catalog_ndcg
            original_load = torch.load

            def build_model(**kwargs):
                model = SASRecModel(**kwargs)

                def inspect_encoder(module, inputs):
                    if model.training:
                        training_widths.append(int(inputs[0].shape[1]))

                model.encoder.register_forward_pre_hook(inspect_encoder)
                return model

            def loss(**kwargs):
                mask = build_loss_mask(kwargs["valid_mask"], kwargs["source_positions"],
                                       kwargs["original_lengths"], kwargs["loss_tail_size"])
                self.assertTrue(torch.equal(mask, kwargs["valid_mask"]))
                for row, raw_length in enumerate(kwargs["original_lengths"].tolist()):
                    expected = torch.arange(max(0, raw_length - 4), raw_length)
                    actual = kwargs["source_positions"][row, mask[row]]
                    self.assertTrue(torch.equal(expected, actual.cpu()))
                    supervised_lengths.append(int(mask[row].sum()))
                return original_loss(**kwargs)

            def evaluate(model, loader, **kwargs):
                self.assertEqual(kwargs["phase"], "validation")
                if kwargs.get("prediction_rows") is not None:
                    self.assertEqual(kwargs["model_label"], "recent128")
                    final_state.update({name: value.detach().cpu().clone()
                                        for name, value in model.state_dict().items()})
                    return original_evaluate(model, loader, **kwargs)
                validation_history_lengths.append(len(json.loads((output / "history.json").read_text())))
                score = scores[len(validation_history_lengths) - 1]
                kwargs["details"].update({"recall_at_10": score, "groups": {}})
                return score, len(loader.dataset)

            with patch.object(experiment, "SASRecModel", side_effect=build_model), \
                 patch.object(experiment, "next_item_loss", side_effect=loss), \
                 patch.object(experiment, "_full_catalog_ndcg", side_effect=evaluate), \
                 patch.object(experiment.torch, "load", wraps=original_load) as load, \
                 redirect_stdout(io.StringIO()):
                metrics = experiment.run_yambda_experiment(config, output)
                self.assertEqual(load.call_count, 1)

            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), max_epochs)
            self.assertEqual([row["epoch"] for row in history if row["validation_performed"]], expected_checks)
            self.assertEqual(validation_history_lengths, [epoch - 1 for epoch in expected_checks])
            for row in history:
                if not row["validation_performed"]:
                    self.assertNotIn("validation_ndcg_at_10", row)
                    self.assertEqual(row["validation_seconds"], 0.0)
            self.assertEqual(metrics["best_epoch"], expected_best)
            self.assertEqual(metrics["epochs_trained"], max_epochs)
            self.assertEqual(metrics["validation_interval"], 5)
            self.assertEqual(metrics["checkpoint_selection_metric"], "validation_ndcg_at_10")
            self.assertFalse(metrics["test_evaluated"])
            self.assertNotIn("test_ndcg_at_10", metrics)
            self.assertAlmostEqual(metrics["training_seconds"], sum(row["training_seconds"] for row in history))
            self.assertEqual(len(training_widths), 2 * max_epochs)
            self.assertTrue(all(width <= 4 for width in training_widths))
            self.assertEqual(len(supervised_lengths), 3 * max_epochs)
            self.assertTrue(all(length == 4 for length in supervised_lengths))

            checkpoint = original_load(output / "best-model.pt", map_location="cpu", weights_only=True)
            self.assertTrue(all(torch.equal(final_state[name], value) for name, value in checkpoint.items()))
            checksum = hashlib.sha256((output / "best-model.pt").read_bytes()).hexdigest()
            self.assertEqual(metrics["checkpoint_sha256"], checksum)
            provenance = json.loads((output / "benchmark.json").read_text())["metadata"]["config"]
            self.assertEqual(provenance["checkpoint_sha256"], checksum)
            predictions = pl.read_parquet(output / "predictions.parquet")
            self.assertEqual(predictions.height, 3)
            self.assertEqual(set(predictions["model"].to_list()), {"recent128"})
            for row in predictions.to_dicts():
                index = row["user_id"] - 11
                expected_raw_context = histories[index][:-2][-64:]
                self.assertEqual(row["context_hash"], context_hash(expected_raw_context))
                self.assertEqual(row["target"], histories[index][-2])


if __name__ == "__main__":
    unittest.main()
