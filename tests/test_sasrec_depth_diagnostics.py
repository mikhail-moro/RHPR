from __future__ import annotations

import builtins
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.depth_diagnostics import analyze_depth_cohorts  # noqa: E402


def feature(user_id: int, *, split: str = "validation", length: int = 300, value: float = 0.5) -> dict:
    old_length = max(0, length - 128)
    return {
        "split": split,
        "user_id": user_id,
        "window_length": length,
        "observed_length_raw": length,
        "old_length": old_length,
        "tail_length": length - old_length,
        "context_hash": f"{split}-{user_id}",
        "old_unique_fraction": value if old_length else None,
        "pair_cosine_distance_mean": value if old_length >= 2 else None,
    }


def prediction(
    row: dict, depth: int, ndcg: float = 0.0, recall: int = 0, *, seed: int = 42, time_enabled: bool = False,
) -> dict:
    return {
        "user_id": row["user_id"],
        "split": row["split"],
        "context_hash": row["context_hash"],
        "target": row["user_id"] + 100,
        "model": f"depth_{depth}",
        "compression_depth": depth,
        "seed": seed,
        "time_enabled": time_enabled,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
    }


def predictions(features: list[dict]) -> list[dict]:
    return [prediction(row, depth) for row in features for depth in range(4)]


def cohort(summary: dict, *, reference: int = 0, depth: int = 1, kind: str = "all", **filters) -> dict:
    return next(
        row for row in summary["cohorts"]
        if row["reference_depth"] == reference and row["compression_depth"] == depth
        and row["cohort_type"] == kind and all(row[key] == value for key, value in filters.items())
    )


class SASRecDepthDiagnosticsTest(unittest.TestCase):
    def analyze(self, features: list[dict], rows: list[dict], **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            return analyze_depth_cohorts(
                features, rows, Path(directory), bootstrap_samples=60, bootstrap_seed=7, **kwargs,
            )

    def test_four_modes_join_by_user_and_hit_outcomes_are_explicit(self) -> None:
        features = [feature(user_id) for user_id in (4, 2, 1, 3)]
        values = {1: ((1.0, 1), (0.0, 0)), 2: ((0.0, 0), (0.5, 1)),
                  3: ((0.5, 1), (0.25, 1)), 4: ((0.0, 0), (0.0, 0))}
        rows = []
        for row in features:
            full, compressed = values[row["user_id"]]
            rows.extend(prediction(row, depth, *(full if depth == 0 else compressed)) for depth in range(4))
        summary = self.analyze(features, rows)
        result = cohort(summary)
        self.assertEqual(summary["paired_user_rows"], 4)
        self.assertEqual(result["user_count"], 4)
        self.assertEqual(result["reference_ndcg_at_10"], 0.375)
        self.assertEqual(result["depth_ndcg_at_10"], 0.1875)
        self.assertEqual(result["delta_ndcg_at_10"], -0.1875)
        self.assertEqual(result["relative_delta_ndcg_at_10_percent"], -50)
        for key in ("reference_hit_depth_miss", "depth_hit_reference_miss", "both_hit", "both_miss"):
            self.assertEqual(result[key], 1)
        reordered = self.analyze(list(reversed(features)), list(reversed(rows)))
        self.assertEqual(summary["cohorts"], reordered["cohorts"])
        self.assertEqual(summary["comparisons"], [
            "depth_1_minus_depth_0", "depth_2_minus_depth_0", "depth_3_minus_depth_0",
            "depth_2_minus_depth_1", "depth_3_minus_depth_1",
        ])

    def test_paired_bootstrap_preserves_constant_differences_across_all_depths(self) -> None:
        features = [feature(1), feature(2)]
        rows = [
            prediction(row, depth, (row["user_id"] - 1) / 8 + depth / 4, 1)
            for row in features for depth in range(4)
        ]
        summary = self.analyze(features, rows)
        for reference, depth in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3)):
            result = cohort(summary, reference=reference, depth=depth)
            expected = (depth - reference) / 4
            np.testing.assert_allclose(result["delta_ndcg_at_10_ci95"], [expected, expected])
            self.assertEqual(result["delta_recall_at_10_ci95"], [0.0, 0.0])
            self.assertEqual(result["both_hit"], 2)
        self.assertNotEqual(cohort(summary)["reference_ndcg_at_10_ci95"][0],
                            cohort(summary)["reference_ndcg_at_10_ci95"][1])

    def test_missing_modes_and_wholly_missing_feature_users_are_rejected(self) -> None:
        features = [feature(1), feature(2)]
        rows = predictions(features)
        incomplete = [rows[:-1], rows[:4]]
        for case in incomplete:
            with self.subTest(rows=len(case)), self.assertRaisesRegex(ValueError, "paired user IDs differ"):
                self.analyze(features, case)

    def test_duplicate_features_and_predictions_rejected(self) -> None:
        features = [feature(1)]
        rows = predictions(features)
        with self.assertRaisesRegex(ValueError, "duplicate feature key"):
            self.analyze(features + features, rows)
        with self.assertRaisesRegex(ValueError, "duplicate prediction key"):
            self.analyze(features, rows + rows[:1])

    def test_hash_target_and_depth_are_required_and_must_match(self) -> None:
        features = [feature(1)]
        rows = predictions(features)
        cases = (
            ("context_hash", "other-context", "context_hash differs"),
            ("context_hash", None, "context_hash must be"),
            ("target", None, "target must be an integer"),
            ("target", 999, "target differs"),
            ("compression_depth", 4, "model must match"),
            ("compression_depth", True, "compression_depth must be an integer"),
            ("model", "pairmlp", "model must match"),
        )
        for field, value, error in cases:
            changed = copy.deepcopy(rows)
            changed[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, error):
                self.analyze(features, changed)
        missing_hash = copy.deepcopy(features)
        del missing_hash[0]["context_hash"]
        with self.assertRaisesRegex(ValueError, "context_hash must be"):
            self.analyze(missing_hash, rows)

    def test_seed_and_time_cannot_pair_between_groups(self) -> None:
        features = [feature(1)]
        for field, value in (("seed", 43), ("time_enabled", True)):
            rows = predictions(features)
            rows[-1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "paired user IDs differ"):
                self.analyze(features, rows)

    def test_unmatched_prediction_and_invalid_metric_rejected(self) -> None:
        features = [feature(1)]
        with self.assertRaisesRegex(ValueError, "prediction has no features"):
            self.analyze(features, predictions([feature(2)]))
        for field, value in (("ndcg_at_10", float("nan")), ("ndcg_at_10", float("inf")),
                             ("recall_at_10", 0.5), ("ndcg_at_10", 1.1)):
            rows = predictions(features)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.analyze(features, rows)

    def test_validation_bins_frozen_for_test_and_ties_do_not_split_users(self) -> None:
        validation = [feature(i, value=value) for i, value in enumerate((0.1, 0.3, 0.7, 0.9))]
        test = [feature(10, split="test", value=1.0)]
        test[0]["pair_cosine_distance_mean"] = 100.0
        summary = self.analyze(validation + test, predictions(validation + test))
        bins = summary["feature_bins"]
        np.testing.assert_allclose(bins["quantile_features"]["pair_cosine_distance_mean"]["edges"],
                                   [0.25, 0.5, 0.75])
        self.assertEqual(bins["validation_user_count"], 4)
        frozen = copy.deepcopy(bins)
        test_only = self.analyze(test, predictions(test), frozen_bins=bins)
        self.assertEqual(bins, frozen)
        self.assertEqual(test_only["feature_bins"], frozen)
        self.assertEqual(cohort(test_only, kind="feature_quantile", feature="old_unique_fraction",
                                quantile="q4")["user_count"], 1)
        # Draws of the test cohort are stable whether validation rows accompany it.
        self.assertEqual(cohort(summary, split="test"), cohort(test_only, split="test"))
        tied = [feature(i, value=0.5) for i in range(4)]
        ties = self.analyze(tied, predictions(tied))
        for name in ("old_unique_fraction", "pair_cosine_distance_mean"):
            self.assertEqual(cohort(ties, kind="feature_quantile", feature=name,
                                    quantile="q1")["user_count"], 4)
            empty = cohort(ties, kind="feature_quantile", feature=name, quantile="q2")
            self.assertEqual(empty["user_count"], 0)
            self.assertIsNone(empty["delta_ndcg_at_10_ci95"])
        with self.assertRaisesRegex(ValueError, "never fit bins on test"):
            self.analyze(test, predictions(test))

    def test_short_histories_stay_in_overall_and_use_correct_actual_compression(self) -> None:
        features = [feature(i, length=length) for i, length in enumerate((128, 129, 130, 512, 513, 2048))]
        summary = self.analyze(features, predictions(features))
        self.assertEqual(cohort(summary)["user_count"], 6)
        self.assertEqual(cohort(summary, kind="actually_compressed")["user_count"], 4)
        counts = {row["length_bin"]: row["user_count"] for row in summary["cohorts"]
                  if row["cohort_type"] == "length" and row["reference_depth"] == 0
                  and row["compression_depth"] == 1}
        self.assertEqual(counts, {"le_128": 1, "129_512": 3, "gt_512": 2})
        self.assertEqual(cohort(summary, reference=1, depth=2)["changed_length_user_count"], 3)
        self.assertEqual(cohort(summary, kind="length", length_bin="le_128")["changed_length_user_count"], 0)
        crosses = [row for row in summary["cohorts"] if row["cohort_type"] == "length_feature_quantile"
                   and row["reference_depth"] == 0 and row["compression_depth"] == 1]
        self.assertEqual(sum(row["user_count"] for row in crosses), 6)
        self.assertEqual({row["feature"] for row in crosses}, {"old_unique_fraction"})

    def test_zero_hits_have_undefined_relative_metric_and_no_safe_label(self) -> None:
        features = [feature(1), feature(2)]
        summary = self.analyze(features, predictions(features))
        result = cohort(summary)
        self.assertEqual(result["both_miss"], 2)
        self.assertEqual(result["delta_ndcg_at_10"], 0)
        self.assertIsNone(result["relative_delta_ndcg_at_10_percent"])
        self.assertIsNone(result["relative_delta_ndcg_at_10_percent_ci95"])
        self.assertEqual(result["relative_delta_ndcg_at_10_bootstrap_defined_samples"], 0)
        self.assertFalse(any("safe" in key for key in result))
        json.dumps(summary, allow_nan=False)

    def test_artifacts_preserve_history_descriptors_and_depth_geometry(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        features = [feature(1, length=1028), feature(2, length=129)]
        features[0]["old_gap_seconds_p90"] = 120.0
        features[1]["old_gap_seconds_p90"] = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = analyze_depth_cohorts(features, predictions(features), output, bootstrap_samples=10)
            self.assertEqual(json.loads((output / "depth-cohorts.json").read_text()), summary)
            self.assertEqual(json.loads((output / "feature_bins.json").read_text()), summary["feature_bins"])
            self.assertEqual(summary["artifacts"], {
                "paired_predictions": "depth-paired_predictions.parquet",
                "cohort_results": "depth-cohort_results.parquet",
            })
            paired = pl.read_parquet(output / summary["artifacts"]["paired_predictions"])
            self.assertEqual(paired["old_gap_seconds_p90"].to_list(), [120.0, None])
            self.assertEqual([paired[f"depth_{depth}_encoder_length"][0] for depth in range(4)],
                             [1028, 578, 353, 241])
            self.assertEqual([paired[f"depth_{depth}_encoder_length"][1] for depth in range(4)],
                             [129, 129, 129, 129])
            table = pl.read_parquet(output / summary["artifacts"]["cohort_results"])
            self.assertEqual(table.schema["delta_ndcg_at_10_ci95"], pl.List(pl.Float64))
            self.assertNotIn("old_gap_seconds_p90", summary["feature_bins"]["quantile_features"])

    def test_json_fallback_and_nonfinite_preserved_feature(self) -> None:
        real_import = builtins.__import__

        def without_polars(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        features = [feature(1)]
        with tempfile.TemporaryDirectory() as directory, patch("builtins.__import__", side_effect=without_polars):
            output = Path(directory)
            summary = analyze_depth_cohorts(features, predictions(features), output, bootstrap_samples=10)
            self.assertEqual(summary["artifacts"]["paired_predictions"], "depth-paired_predictions.json")
            self.assertEqual(summary["artifacts"]["cohort_results"], "depth-cohort_results.json")
            self.assertEqual(json.loads((output / "depth-cohorts.json").read_text()), summary)
            self.assertEqual(json.loads((output / summary["artifacts"]["paired_predictions"]).read_text())[0]["user_id"], 1)
        features[0]["extra_descriptor"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(ValueError):
                analyze_depth_cohorts(features, predictions(features), output, bootstrap_samples=10)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
