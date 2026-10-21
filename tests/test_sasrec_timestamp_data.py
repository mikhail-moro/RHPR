from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.data import (  # noqa: E402
    SequenceSplit,
    _gap_ids,
    evaluation_gap_ids,
    load_yambda_sequences,
    split_sequence,
    training_gap_ids,
)


class TimestampDataTest(unittest.TestCase):
    def test_backward_compatible_split_and_missing_time(self) -> None:
        split = SequenceSplit(1, np.asarray([1, 2]), 3, 4)
        self.assertIsNone(split.timestamps)
        with self.assertRaisesRegex(ValueError, "require observed timestamps"):
            training_gap_ids(split, 2)

    def test_gap_buckets_start_zero_and_clipping(self) -> None:
        np.testing.assert_array_equal(
            _gap_ids(np.asarray([0, 0, 1, 4, 11, 11 + 2**40]), num_buckets=32, time_unit=1),
            [32, 0, 1, 2, 3, 31],
        )
        with self.assertRaises(ValueError):
            _gap_ids(np.asarray([2, 1]), num_buckets=32, time_unit=1)
        with self.assertRaises(ValueError):
            _gap_ids(np.asarray([0, 1]), num_buckets=32, time_unit=0)

    def test_crop_retains_incoming_gap_and_excludes_target_timestamp(self) -> None:
        split = split_sequence(4, [1, 2, 3, 4, 5, 6, 7], [0, 1, 4, 11, 26, 57, 120])
        np.testing.assert_array_equal(training_gap_ids(split, 2), [2, 3])
        np.testing.assert_array_equal(evaluation_gap_ids(split, 2, "validation"), [3, 4])
        np.testing.assert_array_equal(evaluation_gap_ids(split, 2, "test"), [4, 5])
        future_changed = split_sequence(4, [1, 2, 3, 4, 5, 6, 7], [0, 1, 4, 11, 26, 1000, 10000])
        np.testing.assert_array_equal(
            evaluation_gap_ids(split, 2, "validation"), evaluation_gap_ids(future_changed, 2, "validation")
        )
        np.testing.assert_array_equal(training_gap_ids(split, 2), training_gap_ids(future_changed, 2))

    def test_timestamp_validation(self) -> None:
        for times in ([0, 2, 1, 3], [0, np.nan, 2, 3], [0, 1, 2], [1.7e12, 1.7e12 + 1, 1.7e12 + 2, 1.7e12 + 3]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                split_sequence(1, [1, 2, 3, 4], times)

    def test_sort_preserves_items_and_timestamps_together_and_explicit_units(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.parquet"
            pl.DataFrame({
                "uid": [7], "item_ids": [[20, 10, 30, 40]],
                "timestamps": [[1_700_000_003_000, 1_700_000_000_000, 1_700_000_007_000, 1_700_000_010_000]],
            }).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "Unix milliseconds"):
                load_yambda_sequences(path)
            split = load_yambda_sequences(path, timestamp_unit="milliseconds").splits[0]
        np.testing.assert_array_equal(split.train, [1, 2])
        np.testing.assert_array_equal(split.timestamps, [1_700_000_000, 1_700_000_003, 1_700_000_007, 1_700_000_010])
        np.testing.assert_array_equal(evaluation_gap_ids(split, 2, "test"), [2, 2])


if __name__ == "__main__":
    unittest.main()
