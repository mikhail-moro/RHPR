from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.long_evaluation_data import (
    iter_long_evaluation_examples, load_prepared_long_evaluation,
    prepare_long_evaluation_data, write_prepared_long_evaluation,
)
from rhpr.sasrec.long_history_data import load_long_history_cohort

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


@unittest.skipIf(pa is None, "pyarrow is not installed")
class LongEvaluationDataTest(unittest.TestCase):
    def _source(self, directory: str, *, mutate_train: bool = False, first_val_id: int = 999) -> Path:
        path = Path(directory) / "listens.parquet"
        # UID 7: unseen first val target, then known events, first test known.
        # UID 8: no validation event, unseen first test target, later warm one.
        # UID 9: validation only. UID 10: no future events. UID 11 ineligible.
        pq.write_table(pa.table({
            "uid": pa.array([7, 8, 9, 10, 11], type=pa.uint32()),
            "item_id": pa.array([
                [10 if not mutate_train else 777, 20, 30, first_val_id, 20, 999, 10, 30, 888],
                [30, 20, 10, 888, 10], [20, 10, 30, 20], [10, 20, 30], [12, 999],
            ], type=pa.list_(pa.uint32())),
            "timestamp": pa.array([[0, 1, 2, 5, 7, 8, 9, 10, 15], [0, 1, 2, 10, 11],
                                   [0, 1, 2, 6], [0, 1, 2], [1, 8]], type=pa.list_(pa.uint32())),
        }), path)
        return path

    def test_first_anchor_oov_and_missing_periods_keep_fixed_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
            data = prepare_long_evaluation_data(source, cohort, validation_end_timestamp=10, test_end_timestamp=15)
        self.assertEqual([user.user_id for user in data.users], [7, 8, 9, 10])
        self.assertEqual(cohort.num_items, 3)
        self.assertEqual(data.users[0].validation_target, 0)
        self.assertEqual(data.users[0].test_target, 3)
        self.assertIsNone(data.users[1].validation_target)
        self.assertEqual(data.users[1].test_target, 0)
        self.assertEqual(data.summary["validation_users"], 2)
        self.assertEqual(data.summary["test_users"], 2)
        self.assertEqual(data.summary["validation_oov_targets"], 1)
        self.assertEqual(data.summary["test_oov_targets"], 1)
        self.assertEqual(data.summary["both_periods_users"], 1)
        np.testing.assert_array_equal(data.users[0].test_history, [0, 2, 0, 1])
        np.testing.assert_array_equal(data.users[0].test_history_timestamps, [5, 7, 8, 9])

    def test_past_only_cropped_context_preserves_unknown_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
            data = prepare_long_evaluation_data(source, cohort, validation_end_timestamp=10, test_end_timestamp=15)
            validation = tuple(iter_long_evaluation_examples(cohort, data, split_name="validation", max_seq_len=3))
            test = tuple(iter_long_evaluation_examples(cohort, data, split_name="test", max_seq_len=5))
            cropped = tuple(iter_long_evaluation_examples(cohort, data, split_name="test", max_seq_len=2))
        self.assertEqual([example.user_id for example in validation], [7, 9])
        np.testing.assert_array_equal(validation[0].context, [1, 2, 3])
        np.testing.assert_array_equal(validation[0].timestamps, [0, 1, 2])
        self.assertEqual(validation[0].target_timestamp, 5)
        np.testing.assert_array_equal(test[0].context, [3, 0, 2, 0, 1])
        np.testing.assert_array_equal(test[0].timestamps, [2, 5, 7, 8, 9])
        np.testing.assert_array_equal(cropped[0].context, [0, 1])
        np.testing.assert_array_equal(test[1].context, [3, 2, 1])
        self.assertEqual(test[1].target_timestamp, 10)
        self.assertEqual(test[0].context.dtype, np.int32)
        self.assertEqual(test[0].timestamps.dtype, np.uint32)
        test[0].context[0] = 1000
        self.assertEqual(cohort.users[0].train[-1], 3)
        self.assertEqual(data.users[0].test_history[-1], 1)

    def test_future_changes_do_not_change_fixed_cohort_and_anchor_never_skips_oov(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
            changed = self._source(directory, first_val_id=888888)
            data = prepare_long_evaluation_data(changed, cohort, validation_end_timestamp=10, test_end_timestamp=15)
        self.assertEqual(data.users[0].validation_target, 0)
        self.assertEqual(data.users[0].validation_timestamp, 5)
        self.assertEqual(data.summary["source_train_prefix_exact_matches"], 4)
        self.assertEqual(data.summary["catalog_sha256"], cohort.summary["catalog_sha256"])

    def test_source_train_mismatch_and_invalid_boundaries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
            with self.assertRaisesRegex(ValueError, "ordered"):
                prepare_long_evaluation_data(source, cohort, validation_end_timestamp=5, test_end_timestamp=15)
            source = self._source(directory, mutate_train=True)
            with self.assertRaisesRegex(ValueError, "training prefix differs"):
                prepare_long_evaluation_data(source, cohort, validation_end_timestamp=10, test_end_timestamp=15)

    def test_round_trip_and_file_or_cohort_fingerprint_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            cohort = load_long_history_cohort(source, train_end_timestamp=5, max_seq_len=2)
            data = prepare_long_evaluation_data(source, cohort, validation_end_timestamp=10, test_end_timestamp=15)
            destination = Path(directory) / "evaluation"
            summary = write_prepared_long_evaluation(data, cohort, destination)
            loaded = load_prepared_long_evaluation(destination, cohort)
            self.assertEqual(loaded.summary, summary)
            self.assertEqual(set(p.name for p in destination.iterdir()), {"evaluation.parquet", "evaluation-summary.json"})
            for before, after in zip(data.users, loaded.users, strict=True):
                self.assertEqual(before.user_id, after.user_id)
                self.assertEqual(before.validation_target, after.validation_target)
                np.testing.assert_array_equal(before.test_history, after.test_history)
            summary["cohort_uid_length_sha256"] = "0" * 64
            (destination / "evaluation-summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                load_prepared_long_evaluation(destination, cohort)
            write_prepared_long_evaluation(data, cohort, destination)
            path = destination / "evaluation.parquet"
            contents = bytearray(path.read_bytes())
            contents[len(contents) // 2] ^= 1
            path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_prepared_long_evaluation(destination, cohort)

    def test_unsorted_source_uses_stable_time_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pq.write_table(pa.table({"uid": [1], "item_id": [[30, 10, 20, 999, 10, 20]],
                                     "timestamp": [[2, 0, 1, 6, 6, 10]]}), path)
            cohort = load_long_history_cohort(path, train_end_timestamp=5, max_seq_len=2)
            data = prepare_long_evaluation_data(path, cohort, validation_end_timestamp=10, test_end_timestamp=15)
        self.assertEqual(data.users[0].validation_target, 0)
        np.testing.assert_array_equal(data.users[0].test_history, [0, 1])
        self.assertEqual(data.summary["source_selected_users_time_sorted"], 1)


if __name__ == "__main__":
    unittest.main()
