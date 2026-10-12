from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import polars as pl
    import torch
except ImportError:
    pl = torch = None


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class Recent128IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def config(self) -> dict:
        return json.loads((ROOT / "experiments/yambda_recent128_15ep.json").read_text())

    def test_config_preserves_objective_full_context_identity_and_epoch_budget(self) -> None:
        from rhpr.sasrec.recent128 import validate_recent128_config

        config = self.config()
        validate_recent128_config(config)
        invalid_cases = []
        for section, key, value in (
            ("model", "compressor_type", "identity"),
            ("model", "max_seq_len", 128),
            ("model", "tail_size", 64),
            ("model", "use_time_features", True),
            ("loss", "tail_size", 64),
            ("loss", "name", "bce"),
            ("loss", "num_negatives", 256),
            ("evaluation", "evaluate_test", True),
            ("evaluation", "max_validation_batches", 1),
            ("training", "max_epochs", 0),
            ("training", "validation_interval", 16),
            ("training", "early_stopping_patience", 1),
            ("training", "max_batches_per_epoch", 1),
            ("training", "mixed_depth", {"probabilities": [0.25] * 4}),
        ):
            invalid = copy.deepcopy(config)
            invalid["base"][section][key] = value
            invalid_cases.append((f"{section}.{key}", invalid))
        invalid = copy.deepcopy(config)
        invalid["seed"] = invalid["base"]["seed"] = 41
        invalid_cases.append(("seed", invalid))
        for label, candidate in invalid_cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_recent128_config(candidate)

    def test_cpu_training_is_fresh_and_uses_common_targets_with_tail_only_attention(self) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.data import YambdaSequences, make_training_pair, split_sequence, validation_example
        from rhpr.sasrec.diagnostics import context_hash
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors
        from rhpr.sasrec.model import SASRecModel
        from rhpr.sasrec.recent128 import INITIALIZATION, run_recent128

        config = self.config()
        config["base"]["model"].update(d_model=32, n_heads=1, n_layers=1, ffn_multiplier=1, dropout=0.0)
        config["base"]["training"].update(device="cpu", amp=False, workers=0,
            max_epochs=2, validation_interval=1, early_stopping_patience=3,
            physical_batch_size=2, effective_batch_size=2, progress_interval_batches=0)
        config["base"]["evaluation"].update(batch_size=2, item_chunk_size=7)
        config["base"]["benchmark"].update(smoke_test=True, warmup=0, iterations=2,
            throughput_iterations=1, batch_size=2, real_sample_size=2, fixed_lengths=[140], profile=False)
        dataset = YambdaSequences(splits=tuple(
            split_sequence(user, np.arange(length + 2) % 20 + 1)
            for user, length in ((11, 2), (12, 24), (13, 132), (14, 140))),
            catalog_item_ids=np.arange(1, 21))
        actual_evaluation = experiment._full_catalog_ndcg
        selection_scores = iter([0.2, 0.4])

        def controlled_selection(*args, **kwargs):
            score, count = actual_evaluation(*args, **kwargs)
            if kwargs.get("prediction_rows") is None:
                score = next(selection_scores)
            return score, count

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            data_path = output / "listens.parquet"
            pl.DataFrame({"uid": [split.user_id for split in dataset.splits],
                "item_ids": [np.concatenate((split.train, [split.validation, split.test])).tolist()
                             for split in dataset.splits]}).write_parquet(data_path)
            config["base"]["data"]["path"] = str(data_path)
            # An existing checkpoint is never an initialization source.
            checkpoint = output / "best-model.pt"
            checkpoint.write_bytes(b"invalid old checkpoint must be overwritten by fresh training")
            with mock.patch.object(experiment, "_full_catalog_ndcg", side_effect=controlled_selection), \
                    mock.patch.object(experiment, "test_example", side_effect=AssertionError("test was read")):
                metrics = run_recent128(config, output, dataset=dataset)
            self.assertTrue(metrics["recent128_complete"])
            self.assertTrue(metrics["training_performed"])
            self.assertFalse(metrics["test_evaluated"])
            self.assertEqual(metrics["epochs_trained"], 2)
            self.assertEqual(metrics["best_epoch"], 2)
            self.assertEqual(metrics["validation_ndcg_at_10"], 0.4)
            self.assertEqual(metrics["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(metrics["source_initialization"], INITIALIZATION)
            self.assertEqual(metrics["loss_tail_size"], 128)
            self.assertEqual(metrics["num_negatives"], 512)
            self.assertEqual(metrics["physical_batch_size"], 2)
            self.assertEqual(metrics["effective_batch_size"], 2)
            for filename in ("metrics.json", "summary.md", "comparison.md", "run-manifest.json",
                    "training-config.json", "model-config.json", "history.json", "best-model.pt",
                    "diagnostic-reference.json", "recent128-results.json", "history-features.json",
                    "history-features.parquet", "predictions.json", "predictions.parquet", "benchmark.json"):
                self.assertTrue((output / filename).is_file(), filename)
            training = json.loads((output / "training-config.json").read_text())
            self.assertEqual(training["source_initialization"], INITIALIZATION)
            self.assertTrue(training["training"]["keep_checkpoint"])
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in history if row["validation_performed"]], [1, 2])
            reference = json.loads((output / "diagnostic-reference.json").read_text())
            self.assertEqual(reference["checkpoint_sha256"], metrics["checkpoint_sha256"])
            self.assertEqual(reference["source_initialization"], INITIALIZATION)
            benchmark = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(benchmark["metadata"]["config"]["checkpoint_sha256"], metrics["checkpoint_sha256"])
            self.assertEqual(benchmark["metadata"]["model"]["compressor_type"], "tail_only")
            features = json.loads((output / "history-features.json").read_text())
            predictions = json.loads((output / "predictions.json").read_text())
            self.assertEqual(len(features), len(dataset.splits))
            self.assertEqual(len(predictions), len(dataset.splits))
            self.assertEqual(pl.read_parquet(output / "predictions.parquet").height, len(dataset.splits))
            by_user = {split.user_id: split for split in dataset.splits}
            for row in predictions:
                items, target = validation_example(by_user[row["user_id"]], 1028)
                self.assertEqual(row["model"], "recent128")
                self.assertEqual(row["target"], target)
                self.assertEqual(row["context_hash"], context_hash(items))
                self.assertEqual(row["reference_context_hash"], context_hash(items))
                self.assertEqual(row["input_context_hash"], context_hash(items[-128:]))
                self.assertEqual(row["input_length"], min(len(items), 128))
                self.assertEqual(row["target_seen_tail"], bool(target in items[-128:]))
                self.assertEqual(row["target_seen_only_old"], bool(target in items[:-128]) and not bool(target in items[-128:]))
            for row in features:
                self.assertEqual(row["output_length_recent128"], row["tail_length"])
                self.assertFalse(any(key.startswith("target") for key in row))
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertFalse(any(key.startswith("compressor.") for key in state))
            model = SASRecModel(num_items=dataset.num_items, **config["base"]["model"]).eval()
            model.load_state_dict(state)
            inputs, targets = make_training_pair(dataset.splits[-1].train, 1028)
            item_ids = torch.from_numpy(inputs).unsqueeze(0)
            raw_targets = torch.from_numpy(targets).unsqueeze(0)
            raw_mask = torch.ones_like(item_ids, dtype=torch.bool)
            with torch.no_grad():
                encoded, mask, sources = model.encode(item_ids, raw_mask)
            self.assertEqual(mask.sum().item(), 128)
            self.assertEqual(sources[mask].tolist(), list(range(len(inputs) - 128, len(inputs))))
            loss_mask = build_loss_mask(mask, sources, raw_mask.sum(dim=1), 128)
            _, selected_targets, selected_mask = supervised_state_tensors(encoded, loss_mask, sources, raw_targets)
            self.assertEqual(selected_targets[selected_mask].tolist(), targets[-128:].tolist())

    def test_failure_leaves_incomplete_manifest_and_metrics(self) -> None:
        from rhpr.sasrec.data import YambdaSequences, split_sequence
        from rhpr.sasrec.recent128 import run_recent128

        dataset = YambdaSequences(splits=(split_sequence(11, np.array([1, 2, 1, 2])),),
                                 catalog_item_ids=np.array([1, 2]))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch("rhpr.sasrec.recent128.run_yambda_experiment", side_effect=TimeoutError("budget")):
                with self.assertRaises(TimeoutError):
                    run_recent128(self.config(), output, dataset=dataset)
            self.assertFalse(json.loads((output / "metrics.json").read_text())["recent128_complete"])
            self.assertFalse(json.loads((output / "run-manifest.json").read_text())["recent128_complete"])
            self.assertIn("Incomplete", (output / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
