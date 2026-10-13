from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Numpy required")
class HistoryFeaturesTest(unittest.TestCase):
    def describe(self, context, timestamps=None, train_event_count=1000, **kwargs):
        from rhpr.sasrec.history_features import history_features
        context = np.asarray(context, dtype=np.int32)
        if timestamps is None:
            timestamps = np.arange(context.size, dtype=np.uint32)
        return history_features(context, np.asarray(timestamps, dtype=np.uint32),
                                train_event_count=train_event_count, **kwargs)

    def test_hand_calculated_identity_statistics_and_event_weighted_overlaps(self):
        old = [1, 1, 1, 2, 2, 3, 0, 0]
        recent = [1, 4, 0, 0] * 32
        result = self.describe(old + recent)
        self.assertAlmostEqual(result["context_oov_fraction"], 66 / 136)
        self.assertEqual(result["old_known_fraction"], .75)
        self.assertEqual(result["recent_known_fraction"], .5)
        self.assertEqual(result["old_unique_fraction"], .5)
        entropy = -sum(p * math.log(p) for p in (.5, 1 / 3, 1 / 6)) / math.log(3)
        self.assertAlmostEqual(result["old_entropy_normalized"], entropy)
        self.assertEqual(result["old_top_item_share"], .5)
        self.assertAlmostEqual(result["old_hhi"], 14 / 36)
        self.assertEqual(result["recent_unique_fraction"], 2 / 64)
        self.assertEqual(result["recent_old_item_overlap_fraction"], .5)
        self.assertEqual(result["old_recent_item_overlap_fraction"], .5)
        self.assertEqual(result["recent_old_last_quarter_item_overlap_fraction"], 0.)
        self.assertEqual(result["old_first_last_quarter_item_jaccard"], 0.)
        self.assertEqual(result["old_region_mean_unique_fraction_16"], 1.)

    def test_quarters_and_regions_follow_observed_slots_including_unknowns(self):
        old = [1, 2, 0, 0] + [3] * 8 + [1, 4, 0, 0]
        result = self.describe(old + [1, 1, 2, 4] * 32)
        self.assertAlmostEqual(result["old_first_last_quarter_item_jaccard"], 1 / 3)
        self.assertEqual(result["recent_old_last_quarter_item_overlap_fraction"], .75)
        # Sixteen singleton chronological regions remain singleton identities;
        # unknown slots cannot shift the first/last-quarter boundaries.
        self.assertEqual(result["old_region_mean_unique_fraction_16"], 1.)

    def test_region_uniqueness_distinguishes_local_repetition_from_global_diversity(self):
        old = np.repeat(np.arange(1, 17, dtype=np.int32), 16)
        result = self.describe(np.concatenate((old, np.ones(128, dtype=np.int32))))
        self.assertEqual(result["old_unique_fraction"], 1 / 16)
        self.assertEqual(result["old_region_mean_unique_fraction_16"], 1 / 16)
        self.assertAlmostEqual(result["old_entropy_normalized"], 1.)
        self.assertEqual(result["old_hhi"], 1 / 16)
        interleaved = np.tile(np.arange(1, 17, dtype=np.int32), 16)
        other = self.describe(np.concatenate((interleaved, np.ones(128, dtype=np.int32))))
        self.assertEqual(other["old_unique_fraction"], result["old_unique_fraction"])
        self.assertEqual(other["old_hhi"], result["old_hhi"])
        self.assertEqual(other["old_region_mean_unique_fraction_16"], 1.)

    def test_durations_and_gap_burstiness_use_only_supplied_observed_times(self):
        result = self.describe(np.ones(132), np.arange(132, dtype=np.uint32) * 3600)
        self.assertAlmostEqual(result["context_duration_days"], 131 / 24)
        self.assertEqual(result["old_duration_days"], 3 / 24)
        self.assertEqual(result["recent_duration_hours"], 127.)
        self.assertEqual(result["context_gap_burstiness"], -1.)
        self.assertEqual(self.describe([1, 2, 3], [0, 0, 10])["context_gap_burstiness"], 0.)
        # Large unsigned timestamps remain correctly ordered and subtractable.
        self.assertEqual(self.describe([1, 2], [2**32 - 2, 2**32 - 1])["recent_duration_hours"], 1 / 3600)

    def test_empty_unknown_singleton_and_constant_time_histories_are_finite(self):
        from rhpr.sasrec.history_features import HISTORY_FEATURE_FIELDS
        for context in ([], [0], [1], [0] * 256, [1] * 256):
            result = self.describe(context, np.full(len(context), 12), train_event_count=0)
            self.assertEqual(tuple(result), HISTORY_FEATURE_FIELDS)
            self.assertTrue(all(type(value) is float and math.isfinite(value) for value in result.values()))
            self.assertEqual(result["context_gap_burstiness"], 0.)
            self.assertEqual(result["log_train_event_count"], 0.)
        all_unknown = self.describe([0] * 256)
        self.assertEqual(all_unknown["context_oov_fraction"], 1.)
        self.assertEqual(all_unknown["old_hhi"], 0.)
        singleton_identity = self.describe([1] * 256)
        self.assertEqual(singleton_identity["old_entropy_normalized"], 0.)
        self.assertEqual(singleton_identity["old_top_item_share"], 1.)
        self.assertEqual(singleton_identity["old_hhi"], 1.)

    def test_global_train_frequency_is_one_based_and_event_weighted(self):
        from rhpr.sasrec.history_features import CATALOGUE_FREQUENCY_FEATURE_FIELDS
        counts = np.array([999, 2, 8, 0])
        result = self.describe([1, 1, 2, 0] + [2, 3, 0, 0] * 32, catalogue_counts=counts)
        self.assertEqual(tuple(result)[-2:], CATALOGUE_FREQUENCY_FEATURE_FIELDS)
        self.assertAlmostEqual(result["old_mean_log_train_frequency"],
                               (2 * math.log(3) + math.log(9)) / 3)
        self.assertAlmostEqual(result["recent_mean_log_train_frequency"], math.log(9) / 2)
        # Unknown observed slots never use the catalogue's entry zero.
        self.assertEqual(self.describe([0], catalogue_counts=counts)["recent_mean_log_train_frequency"], 0.)

    def test_training_activity_is_distinct_from_visible_window_length(self):
        context = np.ones(16384, dtype=np.int32)
        short = self.describe(context, train_event_count=16385)
        long = self.describe(context, train_event_count=25000)
        self.assertEqual(short["log_train_event_count"], math.log1p(16385))
        self.assertEqual(long["log_train_event_count"], math.log1p(25000))
        self.assertEqual({k: v for k, v in short.items() if k != "log_train_event_count"},
                         {k: v for k, v in long.items() if k != "log_train_event_count"})

    def test_rejects_misalignment_nonchronology_and_invalid_domains(self):
        from rhpr.sasrec.history_features import history_features
        good = dict(context=np.array([1, 2], dtype=np.int32),
                    timestamps=np.array([1, 2], dtype=np.uint32), train_event_count=2)
        bad_values = [dict(timestamps=np.array([1])), dict(timestamps=np.array([2, 1])),
                      dict(context=np.array([-1, 2])), dict(context=np.array([1., 2.])),
                      dict(timestamps=np.array([1., 2.])), dict(context=np.array([[1, 2]])),
                      dict(train_event_count=True), dict(train_event_count=-1),
                      dict(catalogue_counts=np.array([0, 1])),
                      dict(catalogue_counts=np.array([0, 1, np.nan])),
                      dict(catalogue_counts=np.array([0, 1, -1])),
                      dict(catalogue_counts=np.array([0, 1, 2j]))]
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                history_features(**{**good, **bad})


if __name__ == "__main__":
    unittest.main()
