from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import polars as pl
    import torch
except ImportError:
    pl = torch = None


@unittest.skipIf(torch is None or pl is None, "Torch and Polars are required")
class CompressionPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self.previous_threads)

    def config(self) -> dict:
        return json.loads((ROOT / "experiments/yambda_pairwise_compression_pilot.json").read_text())

    def test_rejects_missing_controls_and_test_selection(self) -> None:
        from rhpr.sasrec.pilot import validate_pilot_config
        config = self.config()
        validate_pilot_config(config)
        invalid = copy.deepcopy(config)
        invalid["base"]["evaluation"]["evaluate_test"] = True
        with self.assertRaises(ValueError):
            validate_pilot_config(invalid)
        invalid = copy.deepcopy(config)
        invalid["variants"][0]["use_time_features"] = True
        with self.assertRaises(ValueError):
            validate_pilot_config(invalid)

    def test_two_off_controls_are_accepted_but_unmatched_or_incomplete_sets_are_rejected(self) -> None:
        from rhpr.sasrec.pilot import validate_pilot_config

        config = self.config()
        off = [variant for variant in config["variants"] if not variant["use_time_features"]]
        on = [variant for variant in config["variants"] if variant["use_time_features"]]
        valid = copy.deepcopy(config)
        valid["variants"] = copy.deepcopy(off)
        validate_pilot_config(valid)
        mismatched = [off[0], on[1]]
        repeated_control = copy.deepcopy([off[0], off[0]])
        repeated_control[1]["name"] = "another_full_off"
        repeated_names = copy.deepcopy(off)
        repeated_names[1]["name"] = repeated_names[0]["name"]
        for label, variants in (
            ("one_model", off[:1]), ("three_models", config["variants"][:3]),
            ("unmatched_time", mismatched), ("on_only_missing_reference", on),
            ("missing_pair", repeated_control), ("duplicate_names", repeated_names),
        ):
            with self.subTest(label=label):
                invalid = copy.deepcopy(config)
                invalid["variants"] = copy.deepcopy(variants)
                with self.assertRaises(ValueError):
                    validate_pilot_config(invalid)

    def test_four_variant_cpu_training_evaluation_and_artifacts(self) -> None:
        self._assert_cpu_pilot(self.config())

    def test_two_off_variant_cpu_execution_has_only_matching_off_predictions_and_cohorts(self) -> None:
        config = self.config()
        config["variants"] = [variant for variant in config["variants"] if not variant["use_time_features"]]
        self._assert_cpu_pilot(config)

    def _assert_cpu_pilot(self, config: dict) -> None:
        from rhpr.runner import execute
        from rhpr.sasrec.model import SASRecModel

        variant_count = len(config["variants"])
        expected_time_states = {variant["use_time_features"] for variant in config["variants"]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_path = directory / "listens.parquet"
            lengths = [12, 24, 132, 140]
            pl.DataFrame({"uid": [11, 12, 13, 14],
                "item_ids": [[(i * (u + 1)) % 20 + 1 for i in range(n)] for u, n in enumerate(lengths)],
                "timestamps": [[1000 + 10 * i + i * i for i in range(n)] for n in lengths],
            }).write_parquet(data_path)
            config["base"]["data"]["path"] = str(data_path)
            config["base"]["model"].update(d_model=32, n_heads=1, n_layers=1, ffn_multiplier=1,
                                            max_seq_len=140, dropout=0.0)
            config["base"]["training"].update(device="cpu", amp=False, workers=0, max_epochs=1,
                physical_batch_size=2, effective_batch_size=2, progress_interval_batches=0)
            config["base"]["evaluation"].update(batch_size=2, item_chunk_size=7)
            config["base"]["loss"]["num_negatives"] = 8
            config["base"]["benchmark"].update(smoke_test=True, warmup=0, iterations=2,
                throughput_iterations=1, batch_size=2, real_sample_size=2, fixed_lengths=[140], profile=False)
            config["bootstrap_samples"] = 10
            path = directory / "config.json"
            path.write_text(json.dumps(config))
            output = directory / "output"
            result = execute(path, output)
            self.assertTrue(result["metrics"]["pilot_complete"])
            self.assertFalse(result["metrics"]["test_evaluated"])
            self.assertEqual(result["metrics"]["completed_variants"], variant_count)
            self.assertEqual(result["metrics"]["planned_variants"], variant_count)
            self.assertTrue((output / "run-manifest.json").is_file())
            predictions = pl.read_parquet(output / "predictions.parquet")
            features = pl.read_parquet(output / "history-features.parquet")
            self.assertEqual(predictions.height, 4 * variant_count)  # one final target/user/variant
            self.assertEqual(features.height, 4)
            self.assertEqual(set(predictions["split"].to_list()), {"validation"})
            self.assertEqual(set(predictions["time_enabled"].to_list()), expected_time_states)
            self.assertTrue((output / "cohorts.json").is_file())
            cohorts = json.loads((output / "cohorts.json").read_text())
            self.assertEqual(cohorts["paired_user_rows"], 4 * len(expected_time_states))
            self.assertEqual({row["time_enabled"] for row in cohorts["cohorts"]}, expected_time_states)
            if variant_count == 2:
                self.assertFalse(any(key.startswith(("full_on_", "pair_on_")) for key in result["metrics"]))
                comparison = (output / "comparison.md").read_text()
                self.assertIn("- Time OFF:", comparison)
                self.assertNotIn("- Time ON:", comparison)
            self.assertEqual(json.loads((output / "feature_bins.json").read_text())["fit_split"], "validation")
            for variant in config["variants"]:
                run = output / "runs" / variant["name"]
                saved = json.loads((run / "model-config.json").read_text())
                restored = SASRecModel(num_items=20, **saved["model"])
                restored.load_state_dict(torch.load(run / "best-model.pt", map_location="cpu", weights_only=True))
                metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(metrics["validation_examples"], 4)
                self.assertIn("validation_recall_at_10", metrics)
                self.assertTrue((run / "benchmark.json").is_file())
                self.assertEqual(metrics["loss_tail_size"], 128)


if __name__ == "__main__":
    unittest.main()
