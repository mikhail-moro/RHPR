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

from rhpr.sasrec.depth_diagnostics import LENGTH_BINS  # noqa: E402
from rhpr.sasrec.tail_diagnostics import MODELS, analyze_tail_cohorts  # noqa: E402


def feature(user_id: int, *, length: int = 300, split: str = "validation", value: float = 0.5) -> dict:
    tail = min(length, 128)
    return {
        "user_id": user_id, "split": split, "context_hash": f"full-{split}-{user_id}",
        "window_length": length, "old_length": length - tail, "tail_length": tail,
        "old_unique_fraction": value if length > tail else None,
        "pair_cosine_distance_mean": value if length - tail >= 2 else None,
    }


def frozen_bins() -> dict:
    return {
        "schema_version": 1, "fit_split": "validation", "validation_user_count": 4,
        "length_bins": list(LENGTH_BINS), "quantile_levels": [0.25, 0.5, 0.75],
        "quantile_assignment": "right_closed",
        "quantile_features": {
            name: {"edges": [0.25, 0.5, 0.75], "validation_count": 4}
            for name in ("old_unique_fraction", "pair_cosine_distance_mean")
        },
    }


def prediction(row: dict, model: str, ndcg: float = 0.0, recall: int = 0) -> dict:
    tail = model == "tail_128"
    result = {
        "user_id": row["user_id"], "split": row["split"], "seed": 42, "time_enabled": False,
        "model": model, "target": row["user_id"] + 100,
        "context_hash": row["context_hash"], "reference_context_hash": row["context_hash"],
        "input_context_hash": (f"tail-{row['split']}-{row['user_id']}" if tail and row["old_length"] else row["context_hash"]),
        "input_length": row["tail_length"] if tail else row["window_length"],
        "ndcg_at_10": ndcg, "recall_at_10": recall,
    }
    if not tail:
        result["compression_depth"] = 0 if model == "depth_0" else 3
    return result


def predictions(features: list[dict]) -> list[dict]:
    return [prediction(row, model) for row in features for model in MODELS]


def cohort(summary: dict, *, reference: str = "depth_0", model: str = "tail_128", kind: str = "all", **filters) -> dict:
    return next(row for row in summary["cohorts"] if row["reference_model"] == reference and row["model"] == model
                and row["cohort_type"] == kind and all(row[key] == value for key, value in filters.items()))


class TailDiagnosticsTest(unittest.TestCase):
    def analyze(self, features: list[dict], rows: list[dict], *, bins: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            return analyze_tail_cohorts(features, rows, Path(directory), bootstrap_seed=7, bootstrap_samples=80,
                                        frozen_bins=frozen_bins() if bins is None else bins)

    def test_reordered_three_way_join_has_correct_means_and_hit_transitions(self) -> None:
        features = [feature(i) for i in (4, 2, 1, 3)]
        values = {1: ((1.0, 1), (0.0, 0)), 2: ((0.0, 0), (0.5, 1)),
                  3: ((0.5, 1), (0.25, 1)), 4: ((0.0, 0), (0.0, 0))}
        rows = []
        for row in features:
            reference, tail = values[row["user_id"]]
            rows.extend(prediction(row, model, *(tail if model == "tail_128" else reference)) for model in MODELS)
        summary = self.analyze(features, rows)
        result = cohort(summary)
        self.assertEqual(summary["paired_user_rows"], 4)
        self.assertEqual(result["reference_ndcg_at_10"], 0.375)
        self.assertEqual(result["model_ndcg_at_10"], 0.1875)
        self.assertEqual(result["delta_ndcg_at_10"], -0.1875)
        self.assertEqual(result["relative_delta_ndcg_at_10_percent"], -50)
        for key in ("reference_hit_model_miss", "model_hit_reference_miss", "both_hit", "both_miss"):
            self.assertEqual(result[key], 1)
        self.assertEqual(summary["comparisons"], ["tail_128_minus_depth_0", "tail_128_minus_depth_3", "depth_3_minus_depth_0"])
        self.assertEqual(summary["cohorts"], self.analyze(list(reversed(features)), list(reversed(rows)))["cohorts"])

    def test_bootstrap_resamples_users_jointly_across_all_models(self) -> None:
        features = [feature(1), feature(2)]
        rows = [prediction(row, model, row["user_id"] / 8 + MODELS.index(model) / 4, 1)
                for row in features for model in MODELS]
        summary = self.analyze(features, rows)
        for reference, model, expected in (("depth_0", "tail_128", 0.5), ("depth_3", "tail_128", 0.25),
                                           ("depth_0", "depth_3", 0.25)):
            result = cohort(summary, reference=reference, model=model)
            np.testing.assert_allclose(result["delta_ndcg_at_10_ci95"], [expected, expected])
            self.assertEqual(result["delta_recall_at_10_ci95"], [0.0, 0.0])
        self.assertNotEqual(cohort(summary)["reference_ndcg_at_10_ci95"][0],
                            cohort(summary)["reference_ndcg_at_10_ci95"][1])

    def test_missing_and_duplicate_user_coverage_rejected(self) -> None:
        features = [feature(1), feature(2)]
        rows = predictions(features)
        for incomplete in (rows[:-1], rows[:3]):
            with self.subTest(length=len(incomplete)), self.assertRaisesRegex(ValueError, "paired user IDs differ"):
                self.analyze(features, incomplete)
        with self.assertRaisesRegex(ValueError, "duplicate prediction key"):
            self.analyze(features, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "duplicate feature key"):
            self.analyze(features + features[:1], rows)

    def test_full_hashes_target_seed_time_and_input_lengths_are_strict(self) -> None:
        features = [feature(1)]
        rows = predictions(features)
        cases = (
            (2, "context_hash", "tail-hash", "full context_hash differs"),
            (2, "reference_context_hash", "tail-hash", "full context_hash differs"),
            (2, "reference_context_hash", None, "reference_context_hash must be"),
            (2, "target", 999, "target differs"),
            (2, "seed", 43, "paired user IDs differ"),
            (2, "time_enabled", True, "paired user IDs differ"),
            (2, "input_length", 300, "input_length differs"),
            (1, "input_length", 150, "input_length differs"),
            (2, "input_context_hash", None, "input_context_hash must be"),
            (0, "input_context_hash", "tail-hash", "untruncated input_context_hash differs"),
            (1, "compression_depth", 0, "compression_depth must agree"),
        )
        for index, field, value, error in cases:
            changed = copy.deepcopy(rows)
            changed[index][field] = value
            with self.subTest(field=field, index=index), self.assertRaisesRegex(ValueError, error):
                self.analyze(features, changed)
        with self.assertRaisesRegex(ValueError, "prediction has no features"):
            self.analyze(features, predictions([feature(2)]))

    def test_frozen_bins_never_refit_and_test_only_preserves_membership(self) -> None:
        bins = frozen_bins()
        original = copy.deepcopy(bins)
        rows = [feature(1, split="test", value=100.0)]
        summary = self.analyze(rows, predictions(rows), bins=bins)
        self.assertEqual(bins, original)
        self.assertEqual(summary["feature_bins"], original)
        for name in ("old_unique_fraction", "pair_cosine_distance_mean"):
            self.assertEqual(cohort(summary, kind="feature_quantile", feature=name, quantile="q4")["user_count"], 1)
        invalid = copy.deepcopy(bins)
        invalid["fit_split"] = "test"
        with self.assertRaisesRegex(ValueError, "fitted on validation"):
            self.analyze(rows, predictions(rows), bins=invalid)

    def test_original_length_bins_keep_short_users_and_singleton_prefix_cohort(self) -> None:
        features = [feature(i, length=length) for i, length in enumerate((128, 129, 130, 512, 513, 1028))]
        summary = self.analyze(features, predictions(features))
        self.assertEqual(cohort(summary)["user_count"], 6)
        self.assertEqual(cohort(summary, kind="old_prefix_present")["user_count"], 5)
        self.assertEqual(cohort(summary, kind="actually_compressed")["user_count"], 4)
        self.assertEqual(cohort(summary)["changed_length_user_count"], 5)
        short = cohort(summary, kind="length", length_bin="le_128")
        self.assertEqual(short["user_count"], 1)
        self.assertEqual(short["changed_length_user_count"], 0)
        counts = {row["length_bin"]: row["user_count"] for row in summary["cohorts"]
                  if row["cohort_type"] == "length" and row["comparison"] == "tail_128_minus_depth_0"}
        self.assertEqual(counts, {"le_128": 1, "129_512": 3, "gt_512": 2})
        incorrect_short = predictions([features[0]])
        incorrect_short[-1]["input_context_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "untruncated input_context_hash differs"):
            self.analyze([features[0]], incorrect_short)

    def test_undefined_relative_ratios_and_empty_cohorts_are_null_not_safe(self) -> None:
        features = [feature(1), feature(2)]
        summary = self.analyze(features, predictions(features))
        result = cohort(summary)
        self.assertEqual(result["both_miss"], 2)
        self.assertIsNone(result["relative_delta_ndcg_at_10_percent"])
        self.assertIsNone(result["relative_delta_ndcg_at_10_percent_ci95"])
        self.assertEqual(result["relative_delta_ndcg_at_10_bootstrap_defined_samples"], 0)
        self.assertFalse(any("safe" in key for key in result))
        empty = cohort(summary, kind="feature_quantile", feature="old_unique_fraction", quantile="q1")
        self.assertEqual(empty["user_count"], 0)
        self.assertIsNone(empty["delta_ndcg_at_10_ci95"])
        sparse = [prediction(row, model, 0.5 if row["user_id"] == 1 else 0, int(row["user_id"] == 1))
                  for row in features for model in MODELS]
        result = cohort(self.analyze(features, sparse))
        self.assertEqual(result["relative_delta_ndcg_at_10_percent"], 0)
        self.assertIsNone(result["relative_delta_ndcg_at_10_percent_ci95"])
        self.assertLess(result["relative_delta_ndcg_at_10_bootstrap_defined_samples"], 80)
        json.dumps(summary, allow_nan=False)

    def test_artifacts_distinguish_raw_input_length_and_encoder_length(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars is not installed")
        features = [feature(1, length=1028), feature(2, length=129), feature(3, length=128)]
        features[0]["old_gap_seconds_p90"] = 120.0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = analyze_tail_cohorts(features, predictions(features), output, frozen_bins=frozen_bins(), bootstrap_samples=10)
            self.assertEqual(json.loads((output / "tail-cohorts.json").read_text()), summary)
            self.assertEqual(json.loads((output / "feature_bins.json").read_text()), frozen_bins())
            self.assertEqual(summary["artifacts"], {
                "paired_predictions": "tail-paired_predictions.parquet", "cohort_results": "tail-cohort_results.parquet",
            })
            paired = pl.read_parquet(output / summary["artifacts"]["paired_predictions"])
            self.assertEqual([paired[f"{model}_encoder_length"][0] for model in MODELS], [1028, 241, 128])
            self.assertEqual([paired[f"{model}_input_length"][0] for model in MODELS], [1028, 1028, 128])
            self.assertEqual([paired[f"{model}_encoder_length"][1] for model in MODELS], [129, 129, 128])
            self.assertEqual([paired[f"{model}_encoder_length"][2] for model in MODELS], [128, 128, 128])
            self.assertEqual(paired["old_gap_seconds_p90"].to_list(), [120.0, None, None])
            self.assertEqual(paired["context_hash"].to_list(), paired["reference_context_hash"].to_list())
            table = pl.read_parquet(output / summary["artifacts"]["cohort_results"])
            self.assertEqual(table.schema["delta_ndcg_at_10_ci95"], pl.List(pl.Float64))

    def test_json_fallback_and_nonfinite_input_rejection(self) -> None:
        real_import = builtins.__import__

        def without_polars(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        features = [feature(1)]
        with tempfile.TemporaryDirectory() as directory, patch("builtins.__import__", side_effect=without_polars):
            output = Path(directory)
            summary = analyze_tail_cohorts(features, predictions(features), output, frozen_bins=frozen_bins(), bootstrap_samples=10)
            self.assertEqual(summary["artifacts"]["paired_predictions"], "tail-paired_predictions.json")
            self.assertEqual(summary["artifacts"]["cohort_results"], "tail-cohort_results.json")
            self.assertEqual(json.loads((output / "tail-cohorts.json").read_text()), summary)
        for value in (float("nan"), float("inf")):
            rows = predictions(features)
            rows[0]["ndcg_at_10"] = value
            with self.assertRaises(ValueError):
                self.analyze(features, rows)


if __name__ == "__main__":
    unittest.main()
