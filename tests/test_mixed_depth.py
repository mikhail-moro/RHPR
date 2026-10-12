from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import polars as pl
    import torch
except ImportError:
    pl = torch = None


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class MixedDepthIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def config(self) -> dict:
        return json.loads((ROOT / "experiments/yambda_mixed_depth_15ep.json").read_text())

    def test_config_requires_shared_tail_four_depths_and_validation_only_selection(self) -> None:
        from rhpr.sasrec.mixed_depth import validate_mixed_depth_config

        config = self.config()
        validate_mixed_depth_config(config)
        cases = []

        def invalid(label: str, section: str, key: str, value: object) -> None:
            candidate = copy.deepcopy(config)
            candidate["base"][section][key] = value
            cases.append((label, candidate))

        invalid("mismatched_loss_tail", "loss", "tail_size", 127)
        invalid("mismatched_model_tail", "model", "tail_size", 64)
        invalid("time_features", "model", "use_time_features", True)
        invalid("test_selection", "evaluation", "evaluate_test", True)
        invalid("fixed_pair_only", "model", "compressor_type", "pair_mlp")
        invalid("unsupported_depth", "model", "compression_depth", 4)
        invalid("zero_epochs", "training", "max_epochs", 0)
        missing = copy.deepcopy(config)
        del missing["base"]["loss"]["tail_size"]
        cases.append(("missing_loss_tail", missing))
        for label, probabilities in (
            ("missing_mode", [0.25, 0.25, 0.5]),
            ("zero_mode", [0.0, 0.5, 0.25, 0.25]),
            ("negative_probability", [-0.25, 0.5, 0.5, 0.25]),
            ("unnormalized", [0.2, 0.2, 0.2, 0.2]),
            ("nonfinite", [math.nan, 0.25, 0.25, 0.25]),
        ):
            candidate = copy.deepcopy(config)
            candidate["base"]["training"]["mixed_depth"]["probabilities"] = probabilities
            cases.append((label, candidate))
        for label, interval in (("zero_interval", 0), ("interval_past_training", 16)):
            candidate = copy.deepcopy(config)
            candidate["base"]["training"]["mixed_depth"]["validation_interval"] = interval
            cases.append((label, candidate))
        for label, candidate in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_mixed_depth_config(candidate)

    def test_config_accepts_all_trained_depths_through_six(self) -> None:
        from rhpr.sasrec.mixed_depth import validate_mixed_depth_config

        for count in (4, 5, 6, 7):
            config = self.config()
            config["base"]["training"]["mixed_depth"]["probabilities"] = [1 / count] * count
            config["base"]["model"]["compression_depth"] = count - 1
            with self.subTest(count=count):
                validate_mixed_depth_config(config)
            invalid = copy.deepcopy(config)
            invalid["base"]["model"]["compression_depth"] = count
            with self.subTest(untrained_depth=count), self.assertRaises(ValueError):
                validate_mixed_depth_config(invalid)
        config["base"]["training"]["mixed_depth"]["probabilities"] = [0.125] * 8
        with self.assertRaises(ValueError):
            validate_mixed_depth_config(config)

    def test_shared_checkpoint_selection_uses_mean_even_when_depth_zero_gets_worse(self) -> None:
        import numpy as np

        from rhpr.sasrec import experiment
        from rhpr.sasrec.data import YambdaSequences, split_sequence

        config = copy.deepcopy(self.config()["base"])
        config["model"].update(
            d_model=32, n_heads=1, n_layers=1, ffn_multiplier=1, max_seq_len=140, dropout=0.0,
        )
        config["training"].update(
            device="cpu", amp=False, workers=0, max_epochs=2,
            physical_batch_size=2, effective_batch_size=2, progress_interval_batches=0,
        )
        config["training"]["mixed_depth"]["validation_interval"] = 1
        config["evaluation"].update(batch_size=2, item_chunk_size=7)
        config["loss"]["num_negatives"] = 8
        config["benchmark"]["enabled"] = False
        dataset = YambdaSequences(
            splits=tuple(split_sequence(user, np.arange(length + 2) % 20 + 1)
                         for user, length in ((11, 132), (12, 140))),
            catalog_item_ids=np.arange(1, 21),
        )
        actual_evaluation = experiment._full_catalog_ndcg
        selection_scores = iter([0.9, 0.0, 0.0, 0.9, 0.7, 0.7, 0.7, 0.7])

        def controlled_selection_score(*args, **kwargs):
            ndcg, examples = actual_evaluation(*args, **kwargs)
            # Keep the real forward/catalogue pass and final prediction rows;
            # control only the score used to select the shared checkpoint.
            if kwargs.get("prediction_rows") is None:
                ndcg = next(selection_scores)
            return ndcg, examples

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            data_path = output / "listens.parquet"
            pl.DataFrame({
                "uid": [split.user_id for split in dataset.splits],
                "item_ids": [np.concatenate((split.train, [split.validation, split.test])).tolist()
                             for split in dataset.splits],
            }).write_parquet(data_path)
            config["data"]["path"] = str(data_path)
            with mock.patch.object(experiment, "_full_catalog_ndcg", side_effect=controlled_selection_score):
                metrics = experiment.run_yambda_experiment(config, output, dataset=dataset)
            history = json.loads((output / "history.json").read_text())
            self.assertGreater(history[0]["depth_0_validation_ndcg_at_10"],
                               history[1]["depth_0_validation_ndcg_at_10"])
            self.assertGreater(history[1]["validation_mean_ndcg_at_10"],
                               history[0]["validation_mean_ndcg_at_10"])
            self.assertEqual(metrics["best_epoch"], 2)
            self.assertAlmostEqual(metrics["validation_mean_ndcg_at_10"], 0.7)
            for depth in range(4):
                self.assertEqual(metrics[f"depth_{depth}_best_epoch"], 2)

    def test_cpu_training_uses_one_checkpoint_for_four_modes_and_common_targets(self) -> None:
        self._check_cpu_training_modes(4)

    def test_cpu_training_uses_one_checkpoint_for_seven_modes_and_common_targets(self) -> None:
        self._check_cpu_training_modes(7)

    def _check_cpu_training_modes(self, depth_count: int) -> None:
        from rhpr.runner import execute
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors
        from rhpr.sasrec.model import SASRecModel

        config = self.config()
        depths = tuple(range(depth_count))
        config["base"]["training"]["mixed_depth"]["probabilities"] = [1 / depth_count] * depth_count
        config["base"]["model"]["compression_depth"] = depth_count - 1
        lengths = [2, 24, 132, 140]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_path = directory / "listens.parquet"
            pl.DataFrame({
                "uid": [11, 12, 13, 14],
                "item_ids": [[(i * (user + 1)) % 20 + 1 for i in range(length + 2)]
                             for user, length in enumerate(lengths)],
            }).write_parquet(data_path)
            config["base"]["data"]["path"] = str(data_path)
            config["base"]["model"].update(
                d_model=32, n_heads=1, n_layers=1, ffn_multiplier=1, max_seq_len=140, dropout=0.0,
            )
            config["base"]["training"].update(
                device="cpu", amp=False, workers=0, max_epochs=2,
                physical_batch_size=2, effective_batch_size=2, progress_interval_batches=0,
            )
            config["base"]["training"]["mixed_depth"]["validation_interval"] = 2
            config["base"]["evaluation"].update(batch_size=2, item_chunk_size=7)
            config["base"]["loss"]["num_negatives"] = 8
            config["base"]["benchmark"].update(
                smoke_test=True, warmup=0, iterations=2, throughput_iterations=1,
                batch_size=2, real_sample_size=2, fixed_lengths=[140], profile=False,
            )
            config["bootstrap_samples"] = 10
            config_path = directory / "config.json"
            config_path.write_text(json.dumps(config))
            output = directory / "output"
            result = execute(config_path, output)
            metrics = result["metrics"]
            self.assertTrue(metrics["mixed_depth_complete"])
            self.assertEqual(metrics["planned_depths"], depth_count)
            self.assertFalse(metrics["test_evaluated"])
            self.assertEqual(metrics["loss_tail_size"], 128)
            self.assertTrue((output / "run-manifest.json").is_file())
            checkpoint_path = output / "best-model.pt"
            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(list(output.rglob("best-model.pt")), [checkpoint_path])
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 2)
            self.assertFalse(history[0]["validation_performed"])
            self.assertTrue(history[1]["validation_performed"])
            evaluated = [row for row in history if row["validation_performed"]]
            selected = max(evaluated, key=lambda row: row["validation_mean_ndcg_at_10"])
            self.assertEqual(metrics["best_epoch"], selected["epoch"])
            for row in evaluated:
                self.assertAlmostEqual(
                    row["validation_mean_ndcg_at_10"],
                    sum(row[f"depth_{depth}_validation_ndcg_at_10"] for depth in depths) / depth_count,
                )
            self.assertAlmostEqual(
                metrics["validation_mean_ndcg_at_10"],
                sum(metrics[f"depth_{depth}_validation_ndcg_at_10"] for depth in depths) / depth_count,
            )

            predictions = pl.read_parquet(output / "predictions.parquet")
            self.assertEqual(predictions.height, depth_count * len(lengths))
            self.assertEqual(set(predictions["split"].to_list()), {"validation"})
            self.assertEqual(set(predictions["model"].to_list()), {f"depth_{depth}" for depth in depths})
            self.assertEqual(set(predictions["compression_depth"].to_list()), set(depths))
            self.assertEqual(set(predictions["time_enabled"].to_list()), {False})
            for user_rows in predictions.partition_by("user_id"):
                self.assertEqual(user_rows.height, depth_count)
                self.assertEqual(user_rows["context_hash"].n_unique(), 1)
                self.assertEqual(user_rows["target"].n_unique(), 1)

            self.assertTrue((output / "depth-cohorts.json").is_file())
            self.assertEqual(json.loads((output / "feature_bins.json").read_text())["fit_split"], "validation")
            representations = json.loads((output / "compression-representations.json").read_text())
            self.assertEqual(representations["checkpoint"], "best-model.pt")
            self.assertEqual(representations["sample_size"], len(lengths))
            self.assertEqual(representations["depths"], list(depths))
            representation_rows = representations["rows"]
            self.assertEqual(len(representation_rows), depth_count * len(lengths))
            self.assertEqual(pl.read_parquet(output / "compression-representations.parquet").height,
                             len(representation_rows))
            for row in representation_rows:
                self.assertEqual(row["split"], "validation")
                depth = row["compression_depth"]
                expected_old = (row["old_length"] + 2 ** depth - 1) // 2 ** depth
                self.assertEqual(row["old_output_length"], expected_old)
                self.assertEqual(row["output_length"], min(row["window_length"], 128) + expected_old)
                self.assertEqual(sum(row["source_counts"]), row["window_length"])
                self.assertEqual(row["source_endpoints"][-1], row["window_length"] - 1)

            reference = json.loads((output / "diagnostic-reference.json").read_text())
            self.assertEqual(reference["trained_depths"], list(depths))
            self.assertIn(f"depths0..{depth_count - 1}", reference["checkpoint_selection"])
            features = json.loads((output / "history-features.json").read_text())
            for row in features:
                for depth in depths:
                    expected = row["tail_length"] + (row["old_length"] + 2 ** depth - 1) // 2 ** depth
                    self.assertEqual(row[f"output_length_depth_{depth}"], expected)
            self.assertEqual(json.loads((output / "depth-results.json").read_text())["depths"], list(depths))

            saved = json.loads((output / "model-config.json").read_text())
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            item_ids = torch.zeros((4, 140), dtype=torch.long)
            targets = torch.zeros_like(item_ids)
            mask = torch.arange(140)[None, :] < torch.tensor(lengths)[:, None]
            for user, length in enumerate(lengths):
                item_ids[user, :length] = torch.arange(length) % 20 + 1
                targets[user, :length] = (torch.arange(length) + 1) % 20 + 1
            selected_targets_reference = selected_mask_reference = None
            short_states = None
            revision = result["manifest"]["git_sha"]
            for depth in depths:
                with self.subTest(depth=depth):
                    run = output / "runs" / f"depth_{depth}"
                    per_depth = json.loads((run / "metrics.json").read_text())
                    self.assertEqual(per_depth["validation_examples"], len(lengths))
                    self.assertEqual(per_depth["compression_depth"], depth)
                    self.assertEqual(per_depth["best_epoch"], metrics["best_epoch"])
                    self.assertEqual(per_depth["checkpoint_sha256"], checkpoint_sha)
                    benchmark = json.loads((run / "benchmark.json").read_text())
                    metadata = benchmark["metadata"]
                    self.assertEqual(Path(metadata["checkpoint"]), checkpoint_path)
                    self.assertEqual(metadata["revision"], revision)
                    self.assertEqual(metadata["model"]["compression_depth"], depth)
                    self.assertEqual(metadata["config"]["compression_depth"], depth)
                    self.assertEqual(metadata["config"]["checkpoint_sha256"], checkpoint_sha)
                    self.assertEqual(benchmark["metrics"]["inference_actual_batch_size"], 2)
                    self.assertTrue(benchmark["metrics"]["inference_benchmark_smoke_test"])
                    model = SASRecModel(num_items=20, **saved["model"]).eval()
                    model.load_state_dict(state)
                    model.set_compression_depth(depth)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, state[name])
                    with torch.no_grad():
                        encoded, compressed_mask, sources = model.encode(item_ids, mask)
                        expected_lengths = [min(length, 128) +
                                            (max(length - 128, 0) + 2 ** depth - 1) // 2 ** depth
                                            for length in lengths]
                        self.assertEqual(compressed_mask.sum(dim=1).tolist(), expected_lengths)
                        loss_mask = build_loss_mask(compressed_mask, sources, mask.sum(dim=1), 128)
                        _, selected_targets, selected_mask = supervised_state_tensors(
                            encoded, loss_mask, sources, targets,
                        )
                        if selected_targets_reference is None:
                            selected_targets_reference, selected_mask_reference = selected_targets, selected_mask
                            short_states = model.last_state(item_ids[:2, :24], mask[:2, :24])
                        else:
                            self.assertTrue(torch.equal(selected_targets, selected_targets_reference))
                            self.assertTrue(torch.equal(selected_mask, selected_mask_reference))
                            torch.testing.assert_close(model.last_state(item_ids[:2, :24], mask[:2, :24]), short_states)
                    self.assertEqual(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), checkpoint_sha)


if __name__ == "__main__":
    unittest.main()
