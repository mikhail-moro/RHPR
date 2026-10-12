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
class BackboneComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def config(self, architecture: str = "sasrec") -> dict:
        return json.loads((ROOT / f"experiments/yambda_backbone_{architecture}_15ep.json").read_text())

    def test_configs_share_the_recipe_and_reserve_test(self) -> None:
        from rhpr.sasrec.backbone_comparison import validate_backbone_context_config

        sasrec, hstu = self.config(), self.config("hstu")
        for config in (sasrec, hstu):
            validate_backbone_context_config(config)
        sasrec_base, hstu_base = copy.deepcopy(sasrec["base"]), copy.deepcopy(hstu["base"])
        for base in (sasrec_base, hstu_base):
            base.pop("name")
            base["model"].pop("architecture")
        self.assertEqual(sasrec_base, hstu_base)
        invalid = []
        for section, key, value in (
            ("model", "architecture", "unknown"), ("model", "max_seq_len", 128),
            ("model", "compressor_type", "pair_mlp"), ("model", "use_time_features", True),
            ("loss", "num_negatives", 256), ("loss", "tail_size", 64),
            ("evaluation", "evaluate_test", True), ("evaluation", "max_validation_batches", 1),
            ("training", "early_stopping_patience", 1), ("training", "validation_interval", 16),
            ("training", "mixed_depth", {"probabilities": [0.25] * 4}),
            ("training", "max_batches_per_epoch", 1), ("benchmark", "allow_oom_fallback", True),
        ):
            candidate = copy.deepcopy(sasrec)
            candidate["base"][section][key] = value
            invalid.append((f"{section}.{key}", candidate))
        for contexts in ([128], [1028, 128], [True, 1028], [128, 2048]):
            candidate = copy.deepcopy(sasrec)
            candidate["contexts"] = contexts
            invalid.append((str(contexts), candidate))
        for label, candidate in invalid:
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_backbone_context_config(candidate)

    def test_sasrec_two_fresh_contexts_share_evaluation_identity(self) -> None:
        self._check_training("sasrec")

    def test_hstu_two_fresh_contexts_share_evaluation_identity(self) -> None:
        self._check_training("hstu")

    def _check_training(self, architecture: str) -> None:
        from rhpr.sasrec import experiment
        from rhpr.sasrec.backbone_comparison import INITIALIZATION, run_backbone_context
        from rhpr.sasrec.data import YambdaSequences, split_sequence

        config = self.config(architecture)
        config["base"]["model"].update(d_model=32, n_heads=1, n_layers=1, ffn_multiplier=1, dropout=0.0)
        config["base"]["training"].update(device="cpu", amp=False, workers=0, max_epochs=2,
            validation_interval=1, early_stopping_patience=3, physical_batch_size=2,
            effective_batch_size=2, progress_interval_batches=0)
        config["base"]["evaluation"].update(batch_size=2, item_chunk_size=7)
        config["base"]["benchmark"].update(smoke_test=True, warmup=0, iterations=2,
            throughput_iterations=1, batch_size=2, real_sample_size=2, fixed_lengths=[140], profile=False)
        config["bootstrap_samples"] = 10
        dataset = YambdaSequences(splits=tuple(split_sequence(user, np.arange(length + 2) % 20 + 1)
            for user, length in ((11, 2), (12, 24), (13, 132), (14, 140))), catalog_item_ids=np.arange(1, 21))
        actual_evaluation = experiment._full_catalog_ndcg
        selection_scores = iter([0.2, 0.4, 0.2, 0.4])

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
            with mock.patch.object(experiment, "_full_catalog_ndcg", side_effect=controlled_selection), \
                    mock.patch.object(experiment, "test_example", side_effect=AssertionError("test was read")):
                metrics = run_backbone_context(config, output, dataset=dataset)
            self.assertTrue(metrics["backbone_context_complete"])
            self.assertTrue(metrics["training_performed"])
            self.assertFalse(metrics["test_evaluated"])
            self.assertEqual(metrics["architecture"], architecture)
            self.assertEqual(metrics["source_initialization"], INITIALIZATION)
            self.assertEqual(metrics["planned_contexts"], 2)
            self.assertEqual(metrics["raw_max_seq_len"], 1028)
            features = json.loads((output / "history-features.json").read_text())
            self.assertEqual(len(features), len(dataset.splits))
            self.assertFalse(any("cosine" in key or "target" in key for row in features for key in row))
            predictions = pl.read_parquet(output / "predictions.parquet")
            self.assertEqual(predictions.height, 2 * len(dataset.splits))
            self.assertEqual(set(predictions["model"].to_list()), {f"{architecture}_128", f"{architecture}_1028"})
            self.assertEqual(set(predictions["split"].to_list()), {"validation"})
            for rows in predictions.partition_by("user_id"):
                self.assertEqual(rows["context_hash"].n_unique(), 1)
                self.assertEqual(rows["target"].n_unique(), 1)
            checkpoints = []
            for context in (128, 1028):
                directory = output / "runs" / f"context_{context}"
                checkpoints.append(directory / "best-model.pt")
                mode = json.loads((directory / "metrics.json").read_text())
                self.assertEqual(mode["epochs_trained"], 2)
                self.assertEqual(mode["best_epoch"], 2)
                self.assertEqual(mode["architecture"], architecture)
                self.assertEqual(mode["context_budget"], context)
                self.assertEqual(mode["checkpoint_sha256"], hashlib.sha256(checkpoints[-1].read_bytes()).hexdigest())
                self.assertEqual(metrics[f"context_{context}_checkpoint_sha256"], mode["checkpoint_sha256"])
                config_saved = json.loads((directory / "training-config.json").read_text())
                self.assertEqual(config_saved["source_initialization"], INITIALIZATION)
                self.assertEqual(config_saved["model"]["max_seq_len"], 1028)
                self.assertEqual(config_saved["model"]["compressor_type"], "tail_only" if context == 128 else "identity")
                self.assertEqual(config_saved["loss"]["tail_size"], 128)
                self.assertEqual(config_saved["loss"]["num_negatives"], 512)
                self.assertEqual(config_saved["training"]["physical_batch_size"], 2)
                benchmark = json.loads((directory / "benchmark.json").read_text())
                self.assertEqual(benchmark["metadata"]["config"]["checkpoint_sha256"], mode["checkpoint_sha256"])
                rows = pl.read_parquet(directory / "predictions.parquet").to_dicts()
                for row in rows:
                    self.assertEqual(row["input_length"], min(row["reference_window_length"], context))
                    self.assertEqual(row["checkpoint_sha256"], mode["checkpoint_sha256"])
            self.assertEqual(set(output.rglob("best-model.pt")), set(checkpoints))
            analysis = json.loads((output / "backbone-cohorts.json").read_text())
            self.assertEqual(analysis["comparison"], "long1028_minus_short128")
            self.assertEqual(analysis["cohorts"][0]["user_count"], len(dataset.splits))
            self.assertTrue(any(row["cohort"] == "target_seen_only_old" for row in analysis["cohorts"]))
            self.assertEqual(pl.read_parquet(output / "paired-predictions.parquet").height, len(dataset.splits))
            self.assertEqual(pl.read_parquet(output / "cohort-results.parquet").height, len(analysis["cohorts"]))

    def test_join_rejects_context_or_target_changes_and_old_only_ci_handles_zero_reference(self) -> None:
        from rhpr.sasrec.backbone_comparison import _analyze, _check_predictions, _raw_features
        from rhpr.sasrec.data import YambdaSequences, split_sequence
        from rhpr.sasrec.diagnostics import context_hash

        dataset = YambdaSequences(splits=(
            split_sequence(11, [2, 3, 2, 4]),
            split_sequence(12, [1] * 12 + [2] * 128 + [1, 3]),
            split_sequence(13, [1] * 572 + [2] * 128 + [1, 3])), catalog_item_ids=np.array([1, 2, 3, 4]))
        features, observed = _raw_features(dataset)
        predictions = []
        raw_rows = {}
        for context in (128, 1028):
            rows = [{"model": f"sasrec_{context}", "user_id": user, "split": "validation", "seed": 42,
                "time_enabled": False, "context_hash": context_hash(items), "target": target,
                "ndcg_at_10": int(context == 1028 and user != 11), "recall_at_10": int(context == 1028 and user != 11)}
                for user, (items, target) in observed.items()]
            raw_rows[context] = rows
            predictions.extend(_check_predictions(rows, observed=observed, architecture="sasrec", context=context,
                checkpoint_sha256=f"checksum-{context}", best_epoch=15))
        paired, analysis = _analyze(features, predictions, architecture="sasrec", bootstrap_seed=42, samples=100)
        self.assertEqual(len(paired), 3)
        old_only = next(row for row in analysis["cohorts"] if row["cohort"] == "target_seen_only_old")
        self.assertEqual(old_only["user_count"], 2)
        self.assertEqual(old_only["delta_ndcg_at_10"], 1.0)
        self.assertEqual(old_only["delta_ndcg_at_10_ci95"], [1.0, 1.0])
        self.assertIsNone(old_only["relative_delta_ndcg_at_10"])
        self.assertIsNone(old_only["relative_delta_ndcg_at_10_ci95"])
        json.dumps(analysis, allow_nan=False)
        for field, value in (("target", 4), ("context_hash", "different"), ("seed", 41), ("model", "hstu_128")):
            corrupted = copy.deepcopy(raw_rows[128])
            corrupted[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _check_predictions(corrupted, observed=observed, architecture="sasrec", context=128,
                    checkpoint_sha256="checksum", best_epoch=15)

    def test_failed_training_keeps_incomplete_root_artifacts(self) -> None:
        from rhpr.sasrec.backbone_comparison import run_backbone_context
        from rhpr.sasrec.data import YambdaSequences, split_sequence

        dataset = YambdaSequences(splits=(split_sequence(11, np.array([1, 2, 1, 2])),), catalog_item_ids=np.array([1, 2]))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch("rhpr.sasrec.backbone_comparison.run_yambda_experiment", side_effect=TimeoutError("budget")):
                with self.assertRaises(TimeoutError):
                    run_backbone_context(self.config(), output, dataset=dataset)
            self.assertFalse(json.loads((output / "metrics.json").read_text())["backbone_context_complete"])
            self.assertFalse(json.loads((output / "run-manifest.json").read_text())["backbone_context_complete"])
            self.assertIn("Incomplete", (output / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
