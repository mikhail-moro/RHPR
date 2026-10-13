from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.daily_evaluation_data import (
    daily_context_sha256, iter_daily_evaluation_examples, load_prepared_daily_evaluation,
    prepare_daily_evaluation_data, write_prepared_daily_evaluation,
)
from rhpr.sasrec.history_features import CATALOGUE_FREQUENCY_FEATURE_FIELDS, HISTORY_FEATURE_FIELDS
from rhpr.sasrec.long_evaluation_data import iter_long_evaluation_examples, prepare_long_evaluation_data
from rhpr.sasrec.long_history_data import load_long_history_cohort

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


@unittest.skipIf(pa is None, "pyarrow is not installed")
class DailyEvaluationDataTest(unittest.TestCase):
    def _source(self, directory: str, *, change_future: int = 999, mutate_train: bool = False) -> Path:
        path = Path(directory) / "listens.parquet"
        pq.write_table(pa.table({
            "uid": pa.array([7, 8, 9, 10, 11], type=pa.uint32()),
            "item_id": pa.array([
                [777 if mutate_train else 10, 20, 30, 999, 20, 30, change_future, 10, 30, 888],
                [30, 20, 10, 888, 10, 30], [20, 10, 30, 20, 10], [10, 20, 30], [12, 999],
            ], type=pa.list_(pa.uint32())),
            "timestamp": pa.array([
                [0, 1, 2, 5, 7, 8, 9, 10, 11, 14], [0, 1, 2, 10, 11, 12],
                [0, 1, 2, 6, 9], [0, 1, 2], [1, 8],
            ], type=pa.list_(pa.uint32())),
        }), path)
        return path

    def _prepare(self, source: Path):
        cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
        data = prepare_daily_evaluation_data(source, cohort, validation_end_timestamp=10,
                                            test_end_timestamp=15, day_seconds=1, reference_context_length=3)
        return cohort, data

    def test_first_daily_anchor_oov_and_all_users_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort, data = self._prepare(self._source(directory))
        self.assertEqual([user.user_id for user in data.users], [7, 8, 9, 10])
        self.assertEqual(data.summary["cohort_users"], 4)
        self.assertEqual(data.summary["validation_users"], 2)
        self.assertEqual(data.summary["test_users"], 2)
        self.assertEqual(data.summary["validation_anchors"], 6)
        self.assertEqual(data.summary["test_anchors"], 6)
        self.assertEqual(data.summary["validation_oov_targets"], 2)
        self.assertEqual(data.summary["test_oov_targets"], 2)
        self.assertEqual(data.summary["test_anchors_per_user_buckets"],
                         {"0": 2, "1": 0, "2": 0, "3": 2, "4": 0, "5": 0})
        self.assertEqual(data.anchors[0].anchor_id, "7:5")
        self.assertEqual(data.anchors[0].target, 0)
        np.testing.assert_array_equal(data.users[0].future_items, [0, 2, 3, 0, 1, 3, 0])
        self.assertEqual(cohort.num_items, 3)

    def test_multiple_day_events_select_first_before_mapping_with_stable_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pq.write_table(pa.table({"uid": [1], "item_id": [[30, 10, 20, 999, 10, 20, 888]],
                                     "timestamp": [[2, 0, 1, 11, 11, 12, 21]]}), path)
            cohort = load_long_history_cohort(path, train_end_timestamp=10, max_seq_len=2)
            data = prepare_daily_evaluation_data(path, cohort, validation_end_timestamp=20,
                                                test_end_timestamp=30, day_seconds=10, reference_context_length=3)
            test = list(iter_daily_evaluation_examples(cohort, data, split_name="test", max_seq_len=3))
        self.assertEqual([(a.day, a.future_index, a.target) for a in data.anchors], [(1, 0, 0), (2, 3, 0)])
        np.testing.assert_array_equal(test[0].context, [0, 1, 2])
        np.testing.assert_array_equal(test[0].timestamps, [11, 11, 12])
        self.assertEqual(data.summary["source_selected_users_time_sorted"], 1)

    def test_contexts_include_previous_anchors_and_other_observed_events_not_future(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort, data = self._prepare(self._source(directory))
            test = list(iter_daily_evaluation_examples(cohort, data, split_name="test", max_seq_len=3))
            recent = list(iter_daily_evaluation_examples(cohort, data, split_name="test", max_seq_len=2))
        by_id = {example.anchor_id: example for example in test}
        np.testing.assert_array_equal(by_id["7:10"].context, [2, 3, 0])
        np.testing.assert_array_equal(by_id["7:11"].context, [3, 0, 1])
        np.testing.assert_array_equal(by_id["7:14"].context, [0, 1, 3])
        for full, cropped in zip(test, recent, strict=True):
            self.assertEqual(full.reference_context_hash, cropped.reference_context_hash)
            self.assertEqual(full.reference_context_hash, daily_context_sha256(full.user_id, full.context, full.timestamps))
            np.testing.assert_array_equal(cropped.context, full.context[-2:])
            self.assertLess(int(full.timestamps[-1]), full.target_timestamp)
        test[0].context[0] = 1000
        self.assertEqual(data.users[0].future_items[1], 2)

    def test_first_anchor_per_period_exactly_matches_previous_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort, data = self._prepare(source)
            prior = prepare_long_evaluation_data(source, cohort, validation_end_timestamp=10, test_end_timestamp=15)
            for split in ("validation", "test"):
                first = {}
                for example in iter_daily_evaluation_examples(cohort, data, split_name=split, max_seq_len=3):
                    first.setdefault(example.user_id, example)
                old = list(iter_long_evaluation_examples(cohort, prior, split_name=split, max_seq_len=3))
                self.assertEqual(set(first), {example.user_id for example in old})
                for example in old:
                    self.assertEqual(first[example.user_id].target, example.target)
                    self.assertEqual(first[example.user_id].target_timestamp, example.target_timestamp)
                    np.testing.assert_array_equal(first[example.user_id].context, example.context)

    def test_future_mutation_does_not_change_previous_context_hash_or_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort, data = self._prepare(source)
            changed = self._source(directory, change_future=10)
            other = prepare_daily_evaluation_data(changed, cohort, validation_end_timestamp=10,
                                                 test_end_timestamp=15, day_seconds=1, reference_context_length=3)
            before = next(a for a in data.anchors if a.anchor_id == "7:9")
            after = next(a for a in other.anchors if a.anchor_id == "7:9")
            self.assertEqual(before.reference_context_hash, after.reference_context_hash)
            self.assertNotEqual(before.target, after.target)
            destination = Path(directory) / "daily"
            summary = write_prepared_daily_evaluation(data, cohort, destination)
            rows = pq.read_table(destination / "history-features.parquet").to_pylist()
            row = next(row for row in rows if row["anchor_id"] == "7:9")
            other_destination = Path(directory) / "other"
            write_prepared_daily_evaluation(other, cohort, other_destination)
            other_row = next(row for row in pq.read_table(other_destination / "history-features.parquet").to_pylist()
                             if row["anchor_id"] == "7:9")
            self.assertEqual(row, other_row)
            self.assertEqual(summary["history_features"]["feature_fields"],
                             list(HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS))
            self.assertNotIn("target", row)
            self.assertNotIn("target_timestamp", row)
            # The catalogue count vector uses the four fixed training prefixes only.
            self.assertAlmostEqual(row["recent_mean_log_train_frequency"], np.log1p(4))

    def test_round_trip_and_content_file_or_cohort_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort, data = self._prepare(self._source(directory))
            destination = Path(directory) / "daily"
            summary = write_prepared_daily_evaluation(data, cohort, destination)
            loaded = load_prepared_daily_evaluation(destination, cohort)
            self.assertEqual(loaded.summary, summary)
            self.assertEqual(loaded.anchors, data.anchors)
            for before, after in zip(data.users, loaded.users, strict=True):
                np.testing.assert_array_equal(before.future_items, after.future_items)
            invalid = replace(data, anchors=(replace(data.anchors[0], reference_context_hash="0" * 64), *data.anchors[1:]))
            with self.assertRaisesRegex(ValueError, "context hash"):
                write_prepared_daily_evaluation(invalid, cohort, destination)
            summary["catalog_sha256"] = "0" * 64
            (destination / "daily-evaluation-summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                load_prepared_daily_evaluation(destination, cohort)
            write_prepared_daily_evaluation(data, cohort, destination)
            path = destination / "history-features.parquet"
            content = bytearray(path.read_bytes())
            content[len(content) // 2] ^= 1
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_prepared_daily_evaluation(destination, cohort)

    def test_train_mutation_bad_boundaries_and_invalid_iteration_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort, data = self._prepare(source)
            changed = self._source(directory, mutate_train=True)
            with self.assertRaisesRegex(ValueError, "training prefix differs"):
                prepare_daily_evaluation_data(changed, cohort, validation_end_timestamp=10,
                                              test_end_timestamp=15, day_seconds=1)
            with self.assertRaisesRegex(ValueError, "aligned"):
                prepare_daily_evaluation_data(source, cohort, validation_end_timestamp=10,
                                              test_end_timestamp=15, day_seconds=2)
            with self.assertRaisesRegex(ValueError, "split_name"):
                list(iter_daily_evaluation_examples(cohort, data, split_name="train", max_seq_len=2))
            with self.assertRaisesRegex(ValueError, "positive"):
                list(iter_daily_evaluation_examples(cohort, data, split_name="test", max_seq_len=0))

    def test_empty_periods_keep_cohort_and_serialize_typed_empty_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pq.write_table(pa.table({"uid": [1], "item_id": [[10, 20, 30]],
                                     "timestamp": [[0, 1, 2]]}), path)
            cohort, data = self._prepare(path)
            destination = Path(directory) / "daily"
            summary = write_prepared_daily_evaluation(data, cohort, destination)
            loaded = load_prepared_daily_evaluation(destination, cohort)
            self.assertEqual(summary["validation_anchors"], 0)
            self.assertEqual(summary["test_anchors"], 0)
            self.assertEqual(len(loaded.users), 1)
            self.assertEqual(loaded.anchors, ())
            self.assertEqual(pq.read_table(destination / "history-features.parquet").num_rows, 0)


if __name__ == "__main__":
    unittest.main()
