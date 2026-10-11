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

from rhpr.sasrec.cohorts import OPTIONAL_QUANTILE_FEATURES, QUANTILE_FEATURES, analyze_cohorts  # noqa: E402


def feature(user_id: int, *, split: str = "validation", length: int = 300, value: float = 0.5) -> dict:
    row = {name: value for name in QUANTILE_FEATURES}
    row.update(
        split=split,
        user_id=user_id,
        window_length=length,
        observed_length_raw=length,
        old_length=max(0, length - 128),
        context_hash=f"{split}-{user_id}",
    )
    return row


def prediction(
    user_id: int,
    model: str,
    ndcg: float,
    recall: float,
    *,
    split: str = "validation",
    seed: int = 42,
    time_enabled: bool = False,
) -> dict:
    return {
        "user_id": user_id,
        "model": model,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "split": split,
        "seed": seed,
        "time_enabled": time_enabled,
        "context_hash": f"{split}-{user_id}",
    }


def all_cohort(summary: dict, *, split: str = "validation") -> dict:
    return next(row for row in summary["cohorts"] if row["cohort_type"] == "all" and row["split"] == split)


class SASRecCohortsTest(unittest.TestCase):
    def analyze(self, features: list[dict], predictions: list[dict], **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            return analyze_cohorts(
                features,
                predictions,
                Path(directory),
                bootstrap_samples=40,
                bootstrap_seed=7,
                **kwargs,
            )

    def test_reordered_rows_join_by_user_id_and_count_four_outcomes(self) -> None:
        features = [feature(user_id) for user_id in (4, 2, 1, 3)]
        predictions = [
            prediction(1, "full", 1.0, 1),
            prediction(2, "full", 0.0, 0),
            prediction(3, "full", 0.5, 1),
            prediction(4, "full", 0.0, 0),
            prediction(4, "pairmlp", 0.0, 0),
            prediction(3, "pairmlp", 0.25, 1),
            prediction(2, "pairmlp", 0.5, 1),
            prediction(1, "pairmlp", 0.0, 0),
        ]
        summary = self.analyze(features, predictions)
        cohort = all_cohort(summary)
        self.assertEqual(cohort["user_count"], 4)
        self.assertEqual(cohort["full_ndcg_at_10"], 0.375)
        self.assertEqual(cohort["pairmlp_ndcg_at_10"], 0.1875)
        self.assertEqual(cohort["delta_ndcg_at_10"], -0.1875)
        self.assertEqual(cohort["delta_recall_at_10"], 0.0)
        for name in ("full_hit_pair_miss", "pair_hit_full_miss", "both_hit", "both_miss"):
            self.assertEqual(cohort[name], 1)
        reordered = self.analyze(list(reversed(features)), list(reversed(predictions)))
        self.assertEqual(cohort, all_cohort(reordered))

    def test_bootstrap_resamples_paired_users(self) -> None:
        features = [feature(1), feature(2)]
        predictions = [
            prediction(1, "full", 0.1, 1),
            prediction(2, "full", 0.3, 1),
            prediction(1, "pairmlp", 0.3, 1),
            prediction(2, "pairmlp", 0.5, 1),
        ]
        cohort = all_cohort(self.analyze(features, predictions))
        np.testing.assert_allclose(cohort["delta_ndcg_at_10_ci95"], [0.2, 0.2])
        self.assertEqual(cohort["delta_recall_at_10_ci95"], [0.0, 0.0])

    def test_unmatched_user_ids_rejected_even_with_equal_lengths(self) -> None:
        predictions = [prediction(1, "full", 0, 0), prediction(2, "pairmlp", 0, 0)]
        with self.assertRaisesRegex(ValueError, "paired user IDs differ"):
            self.analyze([feature(1), feature(2)], predictions)

    def test_seed_and_time_groups_do_not_pair_with_each_other(self) -> None:
        for changed in ({"seed": 43}, {"time_enabled": True}):
            predictions = [prediction(1, "full", 0, 0), prediction(1, "pairmlp", 0, 0, **changed)]
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "paired user IDs differ"):
                self.analyze([feature(1)], predictions)

    def test_predictions_without_features_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prediction has no features"):
            self.analyze([feature(1)], [prediction(2, "full", 0, 0), prediction(2, "pairmlp", 0, 0)])

    def test_duplicate_features_and_predictions_rejected(self) -> None:
        predictions = [prediction(1, "full", 0, 0), prediction(1, "pairmlp", 0, 0)]
        with self.assertRaisesRegex(ValueError, "duplicate feature key"):
            self.analyze([feature(1), feature(1)], predictions)
        with self.assertRaisesRegex(ValueError, "duplicate prediction key"):
            self.analyze([feature(1)], predictions + [predictions[0]])

    def test_context_hash_mismatch_rejected(self) -> None:
        predictions = [prediction(1, "full", 0, 0), prediction(1, "pairmlp", 0, 0)]
        predictions[1]["context_hash"] = "another-history"
        with self.assertRaisesRegex(ValueError, "context_hash differs"):
            self.analyze([feature(1)], predictions)

    def test_different_targets_rejected(self) -> None:
        predictions = [prediction(1, "full", 0, 0), prediction(1, "pairmlp", 0, 0)]
        predictions[0]["target"], predictions[1]["target"] = 10, 11
        with self.assertRaisesRegex(ValueError, "target differs"):
            self.analyze([feature(1)], predictions)

    def test_optional_distance_feature_can_be_missing(self) -> None:
        row = feature(1)
        for name in OPTIONAL_QUANTILE_FEATURES:
            del row[name]
        summary = self.analyze([row], [prediction(1, model, 0, 0) for model in ("full", "pairmlp")])
        for name in OPTIONAL_QUANTILE_FEATURES:
            self.assertEqual(summary["feature_bins"]["quantile_features"][name], {"edges": None, "validation_count": 0})

    def test_bins_fit_validation_only_then_freeze_for_test(self) -> None:
        validation = [feature(i, value=value) for i, value in enumerate((0.1, 0.3, 0.7, 0.9))]
        test = [feature(10, split="test", value=100.0)]
        predictions = [
            prediction(row["user_id"], model, 0, 0, split=row["split"])
            for row in validation + test
            for model in ("full", "pairmlp")
        ]
        summary = self.analyze(validation + test, predictions)
        bins = summary["feature_bins"]
        np.testing.assert_allclose(bins["quantile_features"]["pair_cosine_distance_mean"]["edges"], [0.25, 0.5, 0.75])
        self.assertEqual(set(bins["quantile_features"]), {"pair_cosine_distance_mean", "old_unique_fraction"})
        self.assertEqual(bins["validation_user_count"], 4)
        original = copy.deepcopy(bins)
        test_only = self.analyze(test, predictions[-2:], frozen_bins=bins)
        self.assertEqual(test_only["feature_bins"], original)
        self.assertEqual(bins, original)
        quantile = next(
            row for row in test_only["cohorts"]
            if row["cohort_type"] == "feature_quantile"
            and row["feature"] == "pair_cosine_distance_mean" and row["quantile"] == "q4"
        )
        self.assertEqual(quantile["user_count"], 1)

    def test_test_only_without_frozen_bins_and_incomplete_bins_rejected(self) -> None:
        features = [feature(1, split="test")]
        predictions = [prediction(1, model, 0, 0, split="test") for model in ("full", "pairmlp")]
        with self.assertRaisesRegex(ValueError, "never fit bins on test"):
            self.analyze(features, predictions)
        validation = self.analyze(
            [feature(1)], [prediction(1, model, 0, 0) for model in ("full", "pairmlp")]
        )
        bins = validation["feature_bins"]
        del bins["quantile_features"]["old_unique_fraction"]
        with self.assertRaisesRegex(ValueError, "missing thresholds for old_unique_fraction"):
            self.analyze(features, predictions, frozen_bins=bins)

    def test_fixed_length_bins_and_stratified_quantiles(self) -> None:
        features = [feature(i, length=length) for i, length in enumerate((128, 129, 512, 513, 1028))]
        predictions = [prediction(row["user_id"], model, 0, 0) for row in features for model in ("full", "pairmlp")]
        summary = self.analyze(features, predictions)
        length_counts = {
            row["length_bin"]: row["user_count"] for row in summary["cohorts"] if row["cohort_type"] == "length"
        }
        self.assertEqual(length_counts, {"le_128": 1, "129_512": 2, "513_1028": 2})
        stratified = [row for row in summary["cohorts"] if row["cohort_type"] == "length_feature_quantile"]
        self.assertTrue(stratified)
        self.assertEqual(sum(row["user_count"] for row in stratified if row["feature"] == "pair_cosine_distance_mean"), 5)

    def test_missing_features_and_tied_edges_have_explicit_cohorts(self) -> None:
        features = [feature(1), feature(2)]
        features[0]["pair_cosine_distance_mean"] = None
        features[0]["old_unique_fraction"] = features[1]["old_unique_fraction"] = None
        predictions = [prediction(user_id, model, 0, 0) for user_id in (1, 2) for model in ("full", "pairmlp")]
        summary = self.analyze(features, predictions)
        descriptors = summary["feature_bins"]["quantile_features"]
        self.assertIsNone(descriptors["old_unique_fraction"]["edges"])
        q2 = next(row for row in summary["cohorts"] if row["cohort_type"] == "feature_quantile"
                  and row["feature"] == "pair_cosine_distance_mean" and row["quantile"] == "q2")
        self.assertEqual(q2["user_count"], 0)
        self.assertIsNone(q2["delta_ndcg_at_10_ci95"])
        missing = next(row for row in summary["cohorts"] if row["cohort_type"] == "feature_quantile"
                       and row["feature"] == "pair_cosine_distance_mean" and row["quantile"] == "missing")
        self.assertEqual(missing["user_count"], 1)

    def test_artifacts_written_and_json_fallback_without_polars(self) -> None:
        real_import = builtins.__import__

        def without_polars(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch("builtins.__import__", side_effect=without_polars):
            output = Path(directory)
            summary = analyze_cohorts(
                [feature(1)], [prediction(1, model, 0, 0) for model in ("full", "pairmlp")],
                output, bootstrap_samples=10,
            )
            self.assertEqual(json.loads((output / "cohorts.json").read_text()), summary)
            self.assertEqual(json.loads((output / "feature_bins.json").read_text()), summary["feature_bins"])
            self.assertEqual(summary["artifacts"]["paired_predictions"], "paired_predictions.json")
            self.assertEqual(summary["artifacts"]["cohort_results"], "cohort_results.json")
            for name in summary["artifacts"].values():
                self.assertTrue((output / name).is_file())

    def test_parquet_preserves_other_features_and_nullable_interval_lists(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        features = [feature(1), feature(2)]
        for row in features:
            row.update(pair_cosine_mean=0.9, old_angular_spread=0.2, old_gap_seconds_p90=120.0)
        predictions = [prediction(user_id, model, 0, 0) for user_id in (1, 2) for model in ("full", "pairmlp")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = analyze_cohorts(features, predictions, output, bootstrap_samples=20)
            paired = pl.read_parquet(output / summary["artifacts"]["paired_predictions"])
            self.assertEqual(paired["old_gap_seconds_p90"].to_list(), [120.0, 120.0])
            self.assertEqual(paired["old_angular_spread"].to_list(), [0.2, 0.2])
            self.assertNotIn("old_gap_seconds_p90", summary["feature_bins"]["quantile_features"])
            cohorts = pl.read_parquet(output / summary["artifacts"]["cohort_results"])
            self.assertEqual(cohorts.schema["delta_ndcg_at_10_ci95"], pl.List(pl.Float64))
            empty = cohorts.filter(
                (pl.col("cohort_type") == "feature_quantile")
                & (pl.col("feature") == "pair_cosine_distance_mean")
                & (pl.col("quantile") == "q2")
            ).to_dicts()[0]
            self.assertEqual(empty["user_count"], 0)
            self.assertIsNone(empty["delta_ndcg_at_10_ci95"])
            aggregate = cohorts.filter(pl.col("cohort_type") == "all").to_dicts()[0]
            self.assertEqual(aggregate["delta_ndcg_at_10_ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
