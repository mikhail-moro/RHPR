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
    import numpy as np
    import polars as pl
except ImportError:
    np = pl = None


@unittest.skipIf(np is None or pl is None, "NumPy and Polars are required")
class CatalogProfileTest(unittest.TestCase):
    def config(self) -> dict:
        return json.loads((ROOT / "experiments/yambda_catalog_profile.json").read_text())

    def dataset(self, *, validation=2, test=4):
        from rhpr.sasrec.data import SequenceSplit, YambdaSequences
        return YambdaSequences(splits=(SequenceSplit(user_id=1,
            train=np.asarray([1] * 98 + [2, 3], dtype=np.int64), validation=validation, test=test),),
            catalog_item_ids=np.asarray([90, 20, 10, 5], dtype=np.int64))

    def run_profile(self, dataset, config=None):
        from rhpr.sasrec.catalog_profile import run_catalog_profile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = run_catalog_profile(config or self.config(), output, dataset=dataset)
            thresholds = json.loads((output / "coverage-thresholds.json").read_text())
            frequency = pl.read_parquet(output / "catalog-frequency.parquet").to_dicts()
            contexts = pl.read_parquet(output / "context-retention.parquet").to_dicts()
            retention = json.loads((output / "context-retention.json").read_text())
            self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)
            self.assertTrue(json.loads((output / "run-manifest.json").read_text())["catalog_profile_complete"])
            self.assertTrue((output / "summary.md").is_file())
            return metrics, thresholds, frequency, contexts, retention

    def test_counts_only_all_train_events_and_99_events_is_not_99_items(self) -> None:
        metrics, thresholds, frequency, _, _ = self.run_profile(self.dataset())
        self.assertEqual(metrics["train_event_count"], 100)
        self.assertEqual([row["train_count"] for row in frequency], [98, 1, 1, 0])
        self.assertEqual(metrics["full_catalog_item_count"], 4)
        self.assertEqual(metrics["p99_retained_item_count"], 2)
        self.assertEqual(metrics["p99_retained_item_fraction"], 0.5)
        self.assertEqual(metrics["p99_retained_train_event_fraction"], 0.99)
        self.assertEqual(metrics["catalog_item_count_without_train_events"], 1)
        for threshold in thresholds["thresholds"]:
            self.assertGreaterEqual(threshold["retained_train_event_count"], threshold["required_train_event_count"])
            self.assertLess(threshold["events_retained_with_one_fewer_item"], threshold["required_train_event_count"])

    def test_equal_popularity_uses_external_ids_for_a_partial_cutoff_tie(self) -> None:
        _, _, frequency, _, _ = self.run_profile(self.dataset())
        self.assertEqual([row["mapped_item_id"] for row in frequency if row["keep_99_events"]], [1, 3])
        self.assertEqual([row["popularity_rank"] for row in frequency], [1, 3, 2, 4])
        metrics, _, _, _, _ = self.run_profile(self.dataset(validation=3))
        self.assertEqual(metrics["p99_cutoff_tie_item_count"], 2)
        self.assertEqual(metrics["p99_cutoff_tie_retained_item_count"], 1)

    def test_held_out_target_changes_do_not_affect_counts_or_item_selection(self) -> None:
        first = self.run_profile(self.dataset(validation=2, test=2))
        second = self.run_profile(self.dataset(validation=3, test=3))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[0]["p99_validation_targets_retained"], 0)
        self.assertEqual(second[0]["p99_validation_targets_retained"], 1)
        self.assertTrue(first[0]["test_targets_inspected"] is False)

    def test_test_target_is_never_accessed_and_data_is_not_mutated(self) -> None:
        from rhpr.sasrec.data import YambdaSequences

        class User:
            user_id = 1
            train = np.asarray([1] * 100, dtype=np.int64)
            validation = 2

            @property
            def test(self):
                raise AssertionError("reserved test target was accessed")

        dataset = YambdaSequences(splits=(User(),), catalog_item_ids=np.asarray([10, 20]))
        before = dataset.splits[0].train.copy()
        metrics, _, _, _, _ = self.run_profile(dataset)
        np.testing.assert_array_equal(dataset.splits[0].train, before)
        self.assertFalse(metrics["catalog_filtered"])
        self.assertFalse(metrics["split_changed"])
        self.assertFalse(metrics["training_performed"])
        self.assertFalse(metrics["gpu_used"])

    def test_counting_precedes_window_cropping_and_old_only_target_retention(self) -> None:
        from rhpr.sasrec.data import SequenceSplit, YambdaSequences
        dataset = YambdaSequences(splits=(
            SequenceSplit(1, np.ones(20_000, dtype=np.int64), validation=1, test=3),
            SequenceSplit(2, np.asarray([2] * 4 + [1] * 128), validation=2, test=3)),
            catalog_item_ids=np.asarray([10, 20, 30]))
        metrics, _, _, rows, retention = self.run_profile(dataset)
        self.assertEqual(metrics["train_event_count"], 20_132)
        self.assertEqual(metrics["p99_retained_item_count"], 1)
        self.assertEqual(metrics["p99_validation_target_seen_only_old_count"], 1)
        self.assertEqual(metrics["p99_validation_target_seen_only_old_targets_retained"], 0)
        self.assertEqual(rows[0]["observed_length_raw"], 20_000)
        self.assertEqual(rows[0]["window_length"], 1028)
        self.assertEqual(rows[0]["retained_old_events"], 900)
        self.assertEqual(rows[1]["old_length"], 4)
        self.assertEqual(rows[1]["retained_old_events"], 0)
        self.assertEqual(rows[1]["retained_tail_events"], 128)
        self.assertEqual(metrics["validation_users_losing_entire_old_prefix"], 1)
        self.assertEqual(metrics["validation_users_losing_entire_old_prefix_fraction"], 0.5)
        self.assertEqual(retention["distributions"]["retained_old_fraction"]["median"], 0.5)
        self.assertEqual(retention["distributions"]["retained_old_fraction"]["q25"], 0.25)

    def test_embedding_estimates_include_padding_and_do_not_reduce_sampled_negatives(self) -> None:
        metrics, _, _, _, _ = self.run_profile(self.dataset())
        self.assertEqual(metrics["embedding_fp32_bytes_per_real_item"], 1024)
        self.assertEqual(metrics["embedding_fp32_with_grad_adam_m_v_bytes_per_real_item"], 4096)
        self.assertEqual(metrics["p99_embedding_fp32_full_bytes"], 5 * 1024)
        self.assertEqual(metrics["p99_embedding_fp32_retained_bytes"], 3 * 1024)
        self.assertEqual(metrics["p99_embedding_fp32_with_grad_adam_m_v_full_bytes"], 5 * 4096)
        self.assertEqual(metrics["sampled_negative_count"], 512)
        self.assertFalse(metrics["sampled_negative_count_changed"])
        json.dumps(metrics, allow_nan=False)

    def test_short_histories_have_no_old_prefix_without_nonfinite_metrics(self) -> None:
        from rhpr.sasrec.data import SequenceSplit, YambdaSequences
        dataset = YambdaSequences(splits=(SequenceSplit(1, np.asarray([1, 1, 1]), 1, 2),),
                                 catalog_item_ids=np.asarray([10, 20]))
        metrics, _, _, rows, retention = self.run_profile(dataset)
        self.assertEqual(metrics["validation_users_with_old_prefix"], 0)
        self.assertEqual(metrics["validation_users_losing_entire_old_prefix_fraction"], 0)
        self.assertIsNone(rows[0]["retained_old_fraction"])
        self.assertEqual(retention["distributions"]["retained_old_fraction"]["count"], 0)
        json.dumps(metrics, allow_nan=False)

    def test_full_coverage_excludes_held_out_only_zero_frequency_items(self) -> None:
        config = self.config()
        config["profile"]["diagnostic_coverages"].append(1)
        metrics, _, _, _, _ = self.run_profile(self.dataset(), config)
        self.assertEqual(metrics["p100_retained_item_count"], 3)
        self.assertEqual(metrics["p100_retained_train_event_fraction"], 1)

    def test_loading_uses_common_temporal_leave_two_out_mapping(self) -> None:
        from rhpr.sasrec.catalog_profile import run_catalog_profile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "listens.parquet"
            pl.DataFrame({"uid": [1], "item_ids": [[10] * 99 + [20, 30, 40]]}).write_parquet(path)
            config = self.config()
            config["data"]["path"] = str(path)
            metrics = run_catalog_profile(config, root / "output")
            self.assertEqual(metrics["train_event_count"], 100)
            self.assertEqual(metrics["train_observed_item_count"], 2)
            self.assertEqual(metrics["p99_retained_item_count"], 1)
            self.assertEqual(metrics["p99_validation_targets_retained"], 0)

    def test_invalid_config_and_mapping_rejected(self) -> None:
        from rhpr.sasrec.catalog_profile import validate_catalog_profile_config
        config = self.config()
        validate_catalog_profile_config(config)
        for name, value in (("event_coverage", 0.98), ("diagnostic_coverages", [1.1]),
                            ("diagnostic_coverages", [0.99, 0.99]), ("max_seq_len", True),
                            ("tail_size", 2048), ("embedding_dimension", 0), ("num_negatives", 256)):
            changed = copy.deepcopy(config)
            changed["profile"][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_catalog_profile_config(changed)
        changed = copy.deepcopy(config)
        changed["training"] = {"device": "cuda"}
        with self.assertRaisesRegex(ValueError, "data-only"):
            validate_catalog_profile_config(changed)
        from rhpr.sasrec.data import SequenceSplit, YambdaSequences
        for train, catalog, validation in (([0, 1], [10, 20], 1), ([1, 3], [10, 20], 1),
                                           ([1, 1], [10, 10], 1), ([1, 1], [10, 20], 3)):
            dataset = YambdaSequences((SequenceSplit(1, np.asarray(train), validation, 2),), np.asarray(catalog))
            with self.subTest(train=train, validation=validation), self.assertRaises(ValueError):
                self.run_profile(dataset)


if __name__ == "__main__":
    unittest.main()
