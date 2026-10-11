from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.data import split_sequence  # noqa: E402
from rhpr.sasrec.diagnostics import context_hash, extract_history_features, history_feature_row, write_diagnostic_rows  # noqa: E402


class HistoryDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.asarray([[0, 0], [1, 0], [1, 0], [-1, 0], [0, 1], [0, 0]], dtype=np.float64)

    def test_exact_pair_geometry_and_repeat_features(self) -> None:
        row = history_feature_row(
            user_id=1, split_name="validation", item_ids=[1, 2, 3, 4, 1],
            reference_embeddings=self.reference, observed_length_raw=100, tail_size=1,
        )
        self.assertEqual(row["window_length"], 5)
        self.assertEqual(row["old_length"], 4)
        self.assertEqual(row["output_length"], 3)
        self.assertEqual(row["pair_count"], 2)
        self.assertEqual(row["pair_cosine_mean"], 0.5)  # cos(1,2)=1; cos(3,4)=0.
        self.assertEqual(row["pair_cosine_distance_mean"], 0.5)
        self.assertAlmostEqual(row["pair_cosine_distance_p90"], 0.9)
        self.assertAlmostEqual(row["old_angular_spread"], 0.875)
        self.assertEqual(row["exact_pair_repeat_fraction"], 0.0)
        self.assertEqual(row["old_tail_item_overlap"], 1.0)
        self.assertEqual(row["context_hash"], context_hash(np.asarray([1, 2, 3, 4, 1], dtype=np.int32)))

    def test_norm_missingness_and_singleton_handling(self) -> None:
        row = history_feature_row(
            user_id=1, split_name="test", item_ids=[5, 1, 3, 2],
            reference_embeddings=self.reference, observed_length_raw=4, tail_size=1,
        )
        self.assertEqual(row["pair_count"], 1)
        self.assertEqual(row["pair_cosine_valid_count"], 0)
        self.assertIsNone(row["pair_cosine_mean"])
        self.assertEqual(row["reference_norm_invalid_count"], 1)
        self.assertEqual(row["old_angular_spread"], 1.0)
        short = history_feature_row(
            user_id=2, split_name="test", item_ids=[1], reference_embeddings=self.reference,
            observed_length_raw=1, tail_size=128,
        )
        self.assertIsNone(short["old_angular_spread"])
        self.assertIsNone(short["exact_pair_repeat_fraction"])
        self.assertEqual(short["output_length"], 1)
        json.dumps([row, short], allow_nan=False)

    def test_reference_features_use_only_observed_context(self) -> None:
        first = split_sequence(1, [1, 2, 3, 4, 1], [0, 1, 4, 11, 26])
        changed = split_sequence(1, [1, 2, 3, 1, 4], [0, 1, 4, 1000, 10000])
        a = extract_history_features([first], self.reference, max_seq_len=2, tail_size=1)[0]
        b = extract_history_features([changed], self.reference, max_seq_len=2, tail_size=1)[0]
        self.assertEqual(a, b)
        self.assertEqual(a["observed_length_raw"], 3)
        self.assertEqual(a["window_gap_seconds_mean"], 2)
        self.assertEqual(a["window_gap_bucket_mean"], 1.5)
        test = extract_history_features([first], self.reference, max_seq_len=2, split_name="test", tail_size=1)[0]
        self.assertEqual(test["observed_length_raw"], 4)
        self.assertEqual(test["context_hash"], context_hash([3, 4]))

    def test_start_is_excluded_from_gap_statistics(self) -> None:
        row = history_feature_row(
            user_id=1, split_name="validation", item_ids=[1, 1, 1],
            reference_embeddings=self.reference, observed_length_raw=3, tail_size=1,
            gap_ids=np.asarray([32, 0, 1]), gap_seconds=np.asarray([np.nan, 0, 1]),
        )
        self.assertEqual(row["old_gap_bucket_mean"], 0)
        self.assertEqual(row["window_gap_bucket_count"], 2)
        self.assertEqual(row["exact_pair_repeat_fraction"], 1)

    def test_artifacts_are_finite_and_inspectable(self) -> None:
        row = history_feature_row(
            user_id=1, split_name="validation", item_ids=[1],
            reference_embeddings=self.reference, observed_length_raw=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_diagnostic_rows(Path(directory), features=[row], predictions=[])
            self.assertEqual(json.loads(Path(paths["history-features_json"]).read_text()), [row])

    def test_parquet_keeps_typed_nullable_rank_and_features(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        row = history_feature_row(
            user_id=1, split_name="validation", item_ids=[1],
            reference_embeddings=self.reference, observed_length_raw=1,
        )
        predictions = [{"user_id": 1, "top10": [2, 3], "rank_in_top10": None}]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_diagnostic_rows(Path(directory), features=[row], predictions=predictions)
            feature_frame = pl.read_parquet(paths["history-features_parquet"])
            prediction_frame = pl.read_parquet(paths["predictions_parquet"])
        self.assertEqual(feature_frame.schema["pair_cosine_mean"], pl.Float64)
        self.assertEqual(feature_frame.schema["old_gap_seconds_mean"], pl.Float64)
        self.assertEqual(prediction_frame.schema["rank_in_top10"], pl.Int64)
        self.assertEqual(prediction_frame.schema["top10"], pl.List(pl.Int64))


if __name__ == "__main__":
    unittest.main()
