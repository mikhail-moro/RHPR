from __future__ import annotations

import hashlib
import io
import json
import math
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
class HSTUTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def test_architecture_selection_keeps_existing_labels_and_rejects_unsupported_hstu_inputs(self) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.model import SASRecModel

        arguments = {"max_seq_len": 24, "d_model": 32, "n_heads": 1,
                     "n_layers": 1, "ffn_multiplier": 1, "dropout": 0.0, "tail_size": 4}
        self.assertIsInstance(experiment._build_model(arguments, num_items=20), SASRecModel)
        self.assertEqual(experiment._model_label(arguments), "full")
        self.assertEqual(experiment._model_label({**arguments, "compressor_type": "tail_only"}), "recent128")
        for architecture in ("sasrec", "hstu"):
            self.assertEqual(experiment._model_label({**arguments, "architecture": architecture}),
                             f"{architecture}_24")
            self.assertEqual(experiment._model_label({**arguments, "architecture": architecture,
                                                      "compressor_type": "tail_only"}),
                             f"{architecture}_4")
        for invalid in ({"architecture": "unknown"}, {"architecture": []},
                        {"architecture": "hstu", "use_time_features": True},
                        {"architecture": "hstu", "compressor_type": "recursive_pair_mlp"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                experiment._build_model({**arguments, **invalid}, num_items=20)

    def test_actual_hstu_training_preserves_supervision_cadence_predictions_and_checkpoint_reload(self) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.diagnostics import context_hash
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.losses import build_loss_mask

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "listens.parquet"
            histories = [[(i + 7 * user) % 20 + 1 for i in range(length)]
                         for user, length in enumerate((10, 19, 30))]
            pl.DataFrame({"uid": [11, 12, 13], "item_ids": histories}).write_parquet(path)
            for compressor, context in (("identity", 24), ("tail_only", 4)):
                with self.subTest(compressor=compressor):
                    config = {"seed": 42, "deterministic": True, "data": {"path": str(path)},
                        "model": {"architecture": "hstu", "max_seq_len": 24,
                            "d_model": 32, "n_heads": 1, "n_layers": 1,
                            "ffn_multiplier": 1, "dropout": 0.0, "tail_size": 4,
                            "compressor_type": compressor, "use_time_features": False,
                            "hstu_linear_dim": 16, "hstu_attention_dim": 16},
                        "loss": {"name": "sampled_softmax", "num_negatives": 4, "tail_size": 4},
                        "training": {"device": "cpu", "amp": False, "workers": 0,
                            "physical_batch_size": 2, "effective_batch_size": 4,
                            "max_epochs": 3, "validation_interval": 2,
                            "early_stopping_patience": 4, "keep_checkpoint": True,
                            "progress_interval_batches": 0},
                        "evaluation": {"batch_size": 2, "item_chunk_size": 7,
                            "evaluate_test": False, "progress_interval_batches": 0},
                        "diagnostics": {"enabled": True},
                        "benchmark": {"enabled": True, "smoke_test": True, "warmup": 0,
                            "iterations": 1, "throughput_iterations": 1, "batch_size": 2,
                            "real_sample_size": 2, "fixed_lengths": [24]},
                    }
                    output = directory / compressor
                    original_build = experiment._build_model
                    original_loss = experiment.next_item_loss
                    built_models = []
                    loss_rows = []

                    def build(*args, **kwargs):
                        model = original_build(*args, **kwargs)
                        self.assertIsInstance(model, HSTUModel)
                        built_models.append(model)
                        return model

                    def loss(**kwargs):
                        mask = build_loss_mask(kwargs["valid_mask"], kwargs["source_positions"],
                                               kwargs["original_lengths"], kwargs["loss_tail_size"])
                        for row, raw_length in enumerate(kwargs["original_lengths"].tolist()):
                            expected = torch.arange(max(0, raw_length - 4), raw_length)
                            actual = kwargs["source_positions"][row, mask[row]]
                            self.assertTrue(torch.equal(expected, actual.cpu()))
                            loss_rows.append(int(mask[row].sum()))
                        return original_loss(**kwargs)

                    with patch.object(experiment, "_build_model", side_effect=build), \
                         patch.object(experiment, "next_item_loss", side_effect=loss), \
                         redirect_stdout(io.StringIO()):
                        metrics = experiment.run_yambda_experiment(config, output)
                        # This explicit helper is not invoked during the validation-only run.
                        test_metrics = experiment.evaluate_saved_checkpoint(config, output / "best-model.pt")

                    history = json.loads((output / "history.json").read_text())
                    self.assertEqual([row["epoch"] for row in history if row["validation_performed"]], [2, 3])
                    self.assertEqual(len(history), 3)
                    self.assertEqual(len(loss_rows), 9)
                    self.assertEqual(set(loss_rows), {4})
                    self.assertFalse(metrics["test_evaluated"])
                    self.assertNotIn("test_ndcg_at_10", metrics)
                    self.assertEqual(metrics["architecture"], "hstu")
                    self.assertEqual(metrics["model_label"], f"hstu_{context}")
                    self.assertEqual(metrics["hstu_linear_dim"], 16)
                    self.assertEqual(metrics["hstu_attention_dim"], 16)
                    self.assertEqual(metrics["hstu_normalization_length"], 24)
                    self.assertTrue(math.isfinite(test_metrics["test_ndcg_at_10"]))
                    self.assertEqual(test_metrics["test_examples"], 3)
                    self.assertEqual(len(built_models), 2)
                    first, reloaded = built_models
                    self.assertEqual(metrics["backbone_upstream_revision"], first.upstream_revision)
                    self.assertEqual(metrics["attention_implementation"], first.attention_implementation)
                    self.assertTrue(all(torch.equal(value, reloaded.state_dict()[name])
                                        for name, value in first.state_dict().items()))
                    self.assertEqual(metrics["num_model_parameters"],
                                     sum(parameter.numel() for parameter in first.parameters()))
                    self.assertEqual(metrics["num_model_parameters"], metrics["num_trainable_parameters"])
                    self.assertEqual(metrics["num_model_parameters"],
                                     metrics["num_item_embedding_parameters"] + metrics["num_non_item_parameters"])
                    self.assertGreater(metrics["num_encoder_parameters"], 0)
                    checksum = hashlib.sha256((output / "best-model.pt").read_bytes()).hexdigest()
                    self.assertEqual(metrics["checkpoint_sha256"], checksum)
                    benchmark = json.loads((output / "benchmark.json").read_text())
                    provenance = benchmark["metadata"]["config"]
                    self.assertEqual(provenance["architecture"], "hstu")
                    self.assertEqual(provenance["model_label"], f"hstu_{context}")
                    self.assertEqual(provenance["checkpoint_sha256"], checksum)
                    self.assertEqual(provenance["num_model_parameters"], metrics["num_model_parameters"])
                    self.assertEqual(provenance["backbone_upstream_revision"], first.upstream_revision)
                    self.assertEqual(provenance["hstu_normalization_length"], 24)
                    self.assertEqual(json.loads((output / "model-config.json").read_text()), config)
                    predictions = pl.read_parquet(output / "predictions.parquet")
                    self.assertEqual(predictions.height, 3)
                    self.assertEqual(set(predictions["model"].to_list()), {f"hstu_{context}"})
                    for row in predictions.to_dicts():
                        index = row["user_id"] - 11
                        self.assertEqual(row["context_hash"], context_hash(histories[index][:-2][-24:]))
                        self.assertEqual(row["target"], histories[index][-2])
                    for value in metrics.values():
                        if isinstance(value, float):
                            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
