from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.long_history_data import (  # noqa: E402
    LongHistoryUser, load_long_history_cohort, load_prepared_cohort, make_long_training_pair,
    write_prepared_cohort,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


@unittest.skipIf(pa is None, "pyarrow is not installed")
class LongHistoryDataTest(unittest.TestCase):
    def _write(self, directory: str, *, held_out_id: int = 888) -> Path:
        path = Path(directory) / "listens.parquet"
        # Singular list columns reproduce the actual sequential/500m schema.
        # Ineligible UID 9 has an otherwise unseen item. Non-organic/zero-play
        # events deliberately remain in the selected training histories.
        table = pa.table({
            "uid": pa.array([8, 7, 9], type=pa.uint32()),
            "item_id": pa.array([[400, 100, 400, held_out_id],
                                 [200, 100, 300, held_out_id, 999], [222]], type=pa.list_(pa.uint32())),
            "timestamp": pa.array([[0, 2, 3, 5], [4, 1, 4, 5, 6], [0]], type=pa.list_(pa.uint32())),
            "is_organic": [[0, 1, 1, 0], [0, 1, 1, 0, 0], [1]],
            "played_ratio_pct": [[0, 100, 100, 0], [0, 100, 100, 0, 0], [100]],
        })
        pq.write_table(table, path, row_group_size=2)
        return path

    def test_strict_train_boundary_catalog_and_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort = load_long_history_cohort(self._write(directory), train_end_timestamp=5,
                                             max_seq_len=2, batch_size=1, output_dir=Path(directory) / "out")
            profile = pq.read_table(Path(directory) / "out" / "per-user-profile.parquet").to_pylist()
            saved = json.loads((Path(directory) / "out" / "cohort-summary.json").read_text())
        self.assertEqual([user.user_id for user in cohort.users], [7, 8])
        np.testing.assert_array_equal(cohort.catalog_item_ids, [100, 200, 300, 400])
        np.testing.assert_array_equal(cohort.users[0].train, [1, 2, 3])
        np.testing.assert_array_equal(cohort.users[0].timestamps, [1, 4, 4])
        np.testing.assert_array_equal(cohort.users[1].train, [4, 1, 4])
        self.assertEqual(cohort.users[0].train.dtype, np.int32)
        self.assertEqual(cohort.users[0].timestamps.dtype, np.uint32)
        self.assertEqual(cohort.num_items, 4)
        self.assertEqual(cohort.train_events, 6)
        self.assertEqual(saved["source_user_count"], 3)
        self.assertEqual(saved["source_event_count"], 10)
        self.assertEqual(saved["source_train_event_count"], 7)
        self.assertEqual(saved["full_shifted_window_endings"], 2)
        self.assertEqual(saved["source_users_time_sorted"], 1)
        self.assertEqual([row["eligible"] for row in profile], [True, True, False])
        self.assertTrue(saved["all_listens_retained"])
        self.assertFalse(saved["held_out_targets_inspected"])

    def test_future_item_changes_cannot_change_mapping_or_cohort_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = load_long_history_cohort(self._write(first, held_out_id=777), train_end_timestamp=5,
                                        max_seq_len=2, batch_size=1)
            b = load_long_history_cohort(self._write(second, held_out_id=999_999), train_end_timestamp=5,
                                        max_seq_len=2, batch_size=3)
        np.testing.assert_array_equal(a.catalog_item_ids, b.catalog_item_ids)
        for key in ("cohort_uid_length_sha256", "cohort_train_timestamps_sha256",
                    "cohort_mapped_train_items_sha256", "catalog_sha256"):
            self.assertEqual(a.summary[key], b.summary[key])

    def test_plural_list_columns_are_accepted_but_flat_events_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequential.parquet"
            pq.write_table(pa.table({"uid": [1], "item_ids": [[10, 20, 30]],
                                     "timestamps": [[0, 1, 2]]}), path)
            cohort = load_long_history_cohort(path, train_end_timestamp=3, max_seq_len=2)
            self.assertEqual(cohort.train_events, 3)
            pq.write_table(pa.table({"uid": [1, 1, 1], "item_id": [10, 20, 30],
                                     "timestamp": [0, 1, 2]}), path)
            with self.assertRaisesRegex(ValueError, "list column"):
                load_long_history_cohort(path, train_end_timestamp=3, max_seq_len=2)

    def test_invalid_source_shapes_and_duplicate_users_fail(self) -> None:
        cases = [
            ({"uid": [1], "item_id": [[1, 2, 3]], "timestamp": [[1, 2]]}, "different lengths"),
            ({"uid": [1, 1], "item_id": [[1, 2, 3], [1, 2, 3]],
              "timestamp": [[1, 2, 3], [1, 2, 3]]}, "distinct"),
            ({"uid": [1], "item_id": [[1, 2, 3]], "timestamp": [[-1, 2, 3]]}, "uint32"),
            ({"uid": [1], "item_id": [[1, None, 3]], "timestamp": [[1, 2, 3]]}, "null"),
            ({"uid": [1], "item_id": [[1, 2, 3]], "timestamp": [[1.0, 2.0, 3.0]]}, "integer"),
        ]
        for source, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "listens.parquet"
                pq.write_table(pa.table(source), path)
                with self.assertRaisesRegex(ValueError, expected):
                    load_long_history_cohort(path, train_end_timestamp=5, max_seq_len=2)

    def test_eligibility_uses_train_length_not_full_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pq.write_table(pa.table({"uid": [1], "item_id": [[1, 2, 3, 4]],
                                     "timestamp": [[1, 2, 5, 6]]}), path)
            with self.assertRaisesRegex(ValueError, "no users have 3 training events"):
                load_long_history_cohort(path, train_end_timestamp=5, max_seq_len=2)
            with self.assertRaisesRegex(ValueError, "full shifted"):
                load_long_history_cohort(path, train_end_timestamp=5, max_seq_len=2, min_train_events=2)

    def test_prepared_train_only_dataset_roundtrip_preserves_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort = load_long_history_cohort(self._write(directory), train_end_timestamp=5,
                                             max_seq_len=2, batch_size=1)
            prepared = Path(directory) / "prepared"
            written_summary = write_prepared_cohort(cohort, prepared, batch_size=1)
            loaded = load_prepared_cohort(prepared, batch_size=2)
            self.assertEqual(set(path.name for path in prepared.iterdir()),
                             {"train.parquet", "catalog.parquet", "cohort-summary.json"})
            self.assertEqual(loaded.summary, written_summary)
            self.assertEqual(set(written_summary["prepared_files"]), {"train.parquet", "catalog.parquet"})
            np.testing.assert_array_equal(loaded.catalog_item_ids, cohort.catalog_item_ids)
            for expected, actual in zip(cohort.users, loaded.users, strict=True):
                self.assertEqual(expected.user_id, actual.user_id)
                np.testing.assert_array_equal(expected.train, actual.train)
                np.testing.assert_array_equal(expected.timestamps, actual.timestamps)
            # The representation publishes mapped train IDs, not future IDs.
            self.assertNotIn(888, loaded.catalog_item_ids)
            self.assertNotIn(999, loaded.catalog_item_ids)

    def test_prepared_file_and_cohort_signature_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort = load_long_history_cohort(self._write(directory), train_end_timestamp=5, max_seq_len=2)
            prepared = Path(directory) / "prepared"
            summary = write_prepared_cohort(cohort, prepared)
            summary_path = prepared / "cohort-summary.json"
            summary["cohort_train_timestamps_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "cohort_train_timestamps_sha256"):
                load_prepared_cohort(prepared)
            write_prepared_cohort(cohort, prepared)
            train_path = prepared / "train.parquet"
            contents = bytearray(train_path.read_bytes())
            contents[len(contents) // 2] ^= 1
            train_path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_prepared_cohort(prepared)

    def test_prepared_boundary_and_catalog_provenance_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cohort = load_long_history_cohort(self._write(directory), train_end_timestamp=5, max_seq_len=2)
            prepared = Path(directory) / "prepared"
            summary = write_prepared_cohort(cohort, prepared)
            summary_path = prepared / "cohort-summary.json"
            summary["train_end_timestamp"] = 4
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "strictly before"):
                load_prepared_cohort(prepared)
            summary["train_end_timestamp"] = 5
            summary["catalog_source"] = "all_source_events"
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_prepared_cohort(prepared)


class LongHistoryWindowTest(unittest.TestCase):
    def test_exclusive_ending_uses_only_training_inputs_and_next_targets(self) -> None:
        user = LongHistoryUser(1, np.arange(1, 8, dtype=np.int32), np.arange(10, 17, dtype=np.uint32))
        first = make_long_training_pair(user, max_seq_len=3, end_index=4)
        final = make_long_training_pair(user, max_seq_len=3, end_index=7)
        np.testing.assert_array_equal(first[0], [1, 2, 3])
        np.testing.assert_array_equal(first[1], [2, 3, 4])
        np.testing.assert_array_equal(first[2], [10, 11, 12])
        np.testing.assert_array_equal(final[0], [4, 5, 6])
        np.testing.assert_array_equal(final[1], [5, 6, 7])
        np.testing.assert_array_equal(final[2], [13, 14, 15])
        first[0][0] = 999
        self.assertEqual(user.train[0], 1)
        for end in (3, 8):
            with self.assertRaisesRegex(ValueError, "full shifted"):
                make_long_training_pair(user, max_seq_len=3, end_index=end)


if __name__ == "__main__":
    unittest.main()
