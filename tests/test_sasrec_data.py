from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.data import (  # noqa: E402
    cold_target_fraction,
    load_yambda_sequences,
    make_training_pair,
    split_sequence,
    test_example,
    validation_example,
)
from rhpr.sasrec.evaluation import ndcg_at_k_single_target  # noqa: E402


class SASRecDataTest(unittest.TestCase):
    def test_leave_two_out_and_shifted_tail(self) -> None:
        split = split_sequence(7, [10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(split.train, [10, 20, 30, 40])
        self.assertEqual(split.validation, 50)
        self.assertEqual(split.test, 60)
        inputs, targets = make_training_pair(split.train, max_seq_len=2)
        np.testing.assert_array_equal(inputs, [20, 30])
        np.testing.assert_array_equal(targets, [30, 40])
        validation_context, validation_target = validation_example(split, 3)
        test_context, test_target = test_example(split, 3)
        np.testing.assert_array_equal(validation_context, [20, 30, 40])
        np.testing.assert_array_equal(test_context, [30, 40, 50])
        self.assertEqual(validation_target, 50)
        self.assertEqual(test_target, 60)

    def test_short_sequences_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_sequence(1, [1, 2])
        with self.assertRaises(ValueError):
            make_training_pair([1], 10)

    def test_ndcg_uses_one_based_rank(self) -> None:
        self.assertEqual(ndcg_at_k_single_target([[8, 4, 2]], [8], k=10), 1.0)
        self.assertAlmostEqual(ndcg_at_k_single_target([[8, 4, 2]], [4], k=10), 1.0 / np.log2(3))
        self.assertEqual(ndcg_at_k_single_target([[8, 4, 2]], [7], k=10), 0.0)

    def test_cold_target_fraction_is_reported(self) -> None:
        splits = (split_sequence(1, [1, 2, 3, 4]), split_sequence(2, [1, 5, 2, 6]))
        self.assertEqual(cold_target_fraction(splits, target="validation"), 0.5)
        self.assertEqual(cold_target_fraction(splits, target="test"), 1.0)

    def test_sequential_parquet_is_sorted_and_catalogue_mapped(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pl.DataFrame(
                {
                    "uid": [11, 12],
                    "item_ids": [[300, 100, 200, 100], [400, 100, 500]],
                    "timestamps": [[3, 1, 2, 4], [1, 2, 3]],
                }
            ).write_parquet(path)
            data = load_yambda_sequences(path)
        np.testing.assert_array_equal(data.catalog_item_ids, [100, 200, 300, 400, 500])
        np.testing.assert_array_equal(data.splits[0].train, [1, 2])
        self.assertEqual(data.splits[0].validation, 3)
        self.assertEqual(data.splits[0].test, 1)

    def test_flat_parquet_is_grouped_without_filtering_repeats(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listens.parquet"
            pl.DataFrame(
                {
                    "uid": [2, 1, 1, 2, 1, 2],
                    "item_id": [20, 10, 11, 21, 10, 22],
                    "timestamp": [2, 3, 1, 1, 2, 3],
                }
            ).write_parquet(path)
            data = load_yambda_sequences(path)
        first_user = next(split for split in data.splits if split.user_id == 1)
        np.testing.assert_array_equal(first_user.train, [2])
        self.assertEqual(first_user.validation, 1)
        self.assertEqual(first_user.test, 1)


if __name__ == "__main__":
    unittest.main()
