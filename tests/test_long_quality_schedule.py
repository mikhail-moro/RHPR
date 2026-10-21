from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required")
class LongQualityScheduleTest(unittest.TestCase):
    def cohort(self, order=None):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(100 + i, (np.arange(45 + i) % 19 + 1).astype(np.int32),
                                      np.arange(45 + i, dtype=np.uint32)) for i in range(5))
        if order is not None:
            users = tuple(users[i] for i in order)
        return LongHistoryCohort(users, np.arange(19, dtype=np.int64), {})

    def test_seed_epoch_uid_endings_ignore_model_rng_and_cohort_order(self):
        from rhpr.sasrec.long_quality_schedule import (
            quality_schedule_fingerprint, quality_window_schedule,
        )
        cohort = self.cohort()
        original = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=32)
        np.random.seed(999)
        np.random.random(500)
        repeated = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=32)
        self.assertEqual(original, repeated)
        reordered = quality_window_schedule(self.cohort([4, 2, 0, 3, 1]), seed=42, epoch=1,
                                             raw_context_length=32)
        identities = lambda windows: [(w.user_id, w.end, w.epoch, w.negative_seed) for w in windows]
        self.assertEqual(identities(original), identities(reordered))
        self.assertNotEqual([w.user_index for w in original], [w.user_index for w in reordered])
        self.assertEqual(quality_schedule_fingerprint(original),
                         quality_schedule_fingerprint(reordered[::-1]))
        self.assertNotEqual(quality_schedule_fingerprint(original), quality_schedule_fingerprint(
            quality_window_schedule(cohort, seed=42, epoch=2, raw_context_length=32)))
        self.assertNotEqual(quality_schedule_fingerprint(original), quality_schedule_fingerprint(
            quality_window_schedule(cohort, seed=43, epoch=1, raw_context_length=32)))
        self.assertEqual(len(original), len(cohort.users))
        self.assertEqual(len({w.user_id for w in original}), len(cohort.users))
        for window in original:
            self.assertGreaterEqual(window.end, 32)
            self.assertLess(window.end, len(cohort.users[window.user_index].train))

    def test_recent_and_long_receive_identical_targets_and_stateless_negatives(self):
        from rhpr.sasrec.long_quality_schedule import quality_batch_arrays, quality_window_schedule
        cohort = self.cohort()
        windows = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=32)
        options = dict(raw_context_length=32, supervised_targets=4, raw_tail_size=8,
                       num_negatives=13)
        long = quality_batch_arrays(cohort, windows, **options)
        recent = quality_batch_arrays(cohort, windows, recent_only=True, **options)
        self.assertEqual(long["item_ids"].shape, (5, 32))
        self.assertEqual(recent["item_ids"].shape, (5, 11))
        self.assertEqual(long["targets"].dtype, np.dtype("int64"))
        self.assertTrue(long["valid_mask"].all())
        for field in ("item_ids", "targets"):
            np.testing.assert_array_equal(long[field][:, -11:], recent[field])
        np.testing.assert_array_equal(long["negative_ids"], recent["negative_ids"])
        shards = [quality_batch_arrays(cohort, windows[:2], **options),
                  quality_batch_arrays(cohort, windows[2:], **options)]
        for field in long:
            np.testing.assert_array_equal(long[field], np.concatenate([s[field] for s in shards]))
        reversed_batch = quality_batch_arrays(cohort, windows[::-1], **options)
        for field in long:
            np.testing.assert_array_equal(long[field][::-1], reversed_batch[field])
        self.assertGreaterEqual(int(long["negative_ids"].min()), 1)
        self.assertLessEqual(int(long["negative_ids"].max()), cohort.num_items)

    def test_hand_checked_shift_burn_in_and_future_target_is_never_input(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.long_quality_schedule import QualityWindow, quality_batch_arrays
        user = LongHistoryUser(7, np.arange(1, 46, dtype=np.int32), np.arange(45, dtype=np.uint32))
        cohort = LongHistoryCohort((user,), np.arange(45, dtype=np.int64), {})
        window = QualityWindow(0, 7, 36, 1, 123)
        options = dict(raw_context_length=32, supervised_targets=4, raw_tail_size=8, num_negatives=3)
        batch = quality_batch_arrays(cohort, [window], **options)
        np.testing.assert_array_equal(batch["item_ids"][0], np.arange(5, 37))
        np.testing.assert_array_equal(batch["targets"][0], np.arange(6, 38))
        np.testing.assert_array_equal(batch["targets"][0, -4:], [34, 35, 36, 37])
        tail = batch["item_ids"][0, -11:]
        prefix = batch["item_ids"][0, :-11]
        self.assertEqual(prefix[-1], 25)
        self.assertEqual(tail[0], 26)
        self.assertEqual(tail[-4], 33)  # First supervised query; its raw context is 26..33.
        self.assertLess(int(prefix[-1]), int(tail[-4]))
        changed = user.train.copy()
        changed[window.end] = 44
        future_changed = replace(cohort, users=(replace(user, train=changed),))
        changed_batch = quality_batch_arrays(future_changed, [window], **options)
        np.testing.assert_array_equal(batch["item_ids"], changed_batch["item_ids"])
        np.testing.assert_array_equal(batch["negative_ids"], changed_batch["negative_ids"])
        np.testing.assert_array_equal(batch["targets"][:, :-1], changed_batch["targets"][:, :-1])
        self.assertEqual(changed_batch["targets"][0, -1], 44)

    def test_one_available_ending_and_both_inclusive_boundaries(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule
        one = LongHistoryUser(5, np.ones(33, dtype=np.int32), np.arange(33, dtype=np.uint32))
        cohort = LongHistoryCohort((one,), np.arange(1, dtype=np.int64), {})
        self.assertEqual(quality_window_schedule(cohort, seed=42, epoch=1,
                                                 raw_context_length=32)[0].end, 32)
        two = replace(one, train=np.ones(34, dtype=np.int32), timestamps=np.arange(34, dtype=np.uint32))
        cohort = replace(cohort, users=(two,))
        ends = {quality_window_schedule(cohort, seed=42, epoch=epoch,
                                         raw_context_length=32)[0].end for epoch in range(30)}
        self.assertEqual(ends, {32, 33})

    def test_summary_reports_true_16k_matched_context_and_descriptor_only_counts(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.long_quality_schedule import quality_schedule_summary, quality_window_schedule
        user = LongHistoryUser(1, np.ones(16_386, dtype=np.int32), np.arange(16_386, dtype=np.uint32))
        cohort = LongHistoryCohort((user,), np.arange(1, dtype=np.int64), {})
        windows = quality_window_schedule(cohort, seed=42, epoch=1)
        summary = quality_schedule_summary(cohort, windows)
        self.assertEqual(summary["training_raw_tail_size"], 159)
        self.assertEqual(summary["training_old_prefix_size"], 16_225)
        self.assertEqual(summary["training_query_raw_context_min"], 128)
        self.assertEqual(summary["training_query_raw_context_max"], 159)
        self.assertEqual(summary["target_presentations"], 32)
        self.assertEqual(summary["available_full_window_endings"], 2)
        self.assertFalse(summary["train_is_exhaustive_event_epoch"])
        repeated = quality_schedule_summary(cohort, windows + windows)
        self.assertEqual(repeated["unique_endpoints"], 1)
        self.assertEqual(repeated["repeated_endpoint_presentations"], 1)
        self.assertEqual(repeated["available_full_window_endings"], 2)
        self.assertEqual(repeated["target_presentations"], 64)

    def test_invalid_users_shapes_bools_and_endpoints_fail_explicitly(self):
        from rhpr.sasrec.long_quality_schedule import quality_batch_arrays, quality_window_schedule
        cohort = self.cohort()
        for parameter in ({"seed": True}, {"seed": -1}, {"epoch": False},
                          {"epoch": -1}, {"raw_context_length": True}, {"raw_context_length": 50}):
            values = dict(seed=42, epoch=1, raw_context_length=32)
            values.update(parameter)
            with self.assertRaises(ValueError):
                quality_window_schedule(cohort, **values)
        with self.assertRaises(ValueError):
            quality_window_schedule(replace(cohort, users=()), seed=42, epoch=1, raw_context_length=32)
        with self.assertRaises(ValueError):
            quality_window_schedule(replace(cohort, users=(cohort.users[0], cohort.users[0])),
                                    seed=42, epoch=1, raw_context_length=32)
        windows = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=32)
        defaults = dict(raw_context_length=32, supervised_targets=4, raw_tail_size=8, num_negatives=7)
        for parameter in ({"raw_tail_size": True}, {"supervised_targets": True},
                          {"num_negatives": True}, {"supervised_targets": 32},
                          {"recent_only": 1}, {"raw_context_length": False}):
            values = {**defaults, **parameter}
            with self.assertRaises(ValueError):
                quality_batch_arrays(cohort, windows, **values)
        for bad in (replace(windows[0], user_index=99), replace(windows[0], user_id=999),
                    replace(windows[0], end=31), replace(windows[0], end=100)):
            with self.assertRaises(ValueError):
                quality_batch_arrays(cohort, [bad], **defaults)

    def test_uniform_budget_floors_shared_passes_and_charges_elapsed_preflight_once(self):
        from rhpr.sasrec.long_quality_schedule import choose_uniform_pass_budget
        costs = [dict(training_pass_seconds=20., validation_pass_seconds=2., evaluation_seconds=10.),
                 dict(training_pass_seconds=30., validation_pass_seconds=3., evaluation_seconds=10.)]
        plan = choose_uniform_pass_budget(costs, budget_seconds=500., preflight_seconds=30.,
                                          fixed_overhead_seconds=10.)
        self.assertEqual(plan["passes"], 6)
        self.assertEqual(plan["models"], 2)
        self.assertEqual(plan["estimated_suite_seconds"], 390.)
        self.assertEqual(plan["conservative_suite_seconds"], 480.)
        self.assertEqual(plan["remaining_budget_seconds"], 20.)
        self.assertFalse(plan["selection_uses_quality_metrics"])
        large = choose_uniform_pass_budget(costs, budget_seconds=100_000.)
        self.assertEqual(large["passes"], 20)
        exact = choose_uniform_pass_budget(costs, budget_seconds=30 + 1.25 * (30 + 5 * 55),
                                           preflight_seconds=30., fixed_overhead_seconds=10.)
        self.assertEqual(exact["passes"], 5)
        with self.assertRaisesRegex(ValueError, "cannot support 5 common passes"):
            choose_uniform_pass_budget(costs, budget_seconds=400., preflight_seconds=30.,
                                       fixed_overhead_seconds=10.)

    def test_budget_rejects_missing_measurements_nan_negative_and_boolean_settings(self):
        from rhpr.sasrec.long_quality_schedule import choose_uniform_pass_budget
        costs = [dict(training_pass_seconds=10., validation_pass_seconds=1.)]
        for parameter in ({"budget_seconds": True}, {"budget_seconds": float("nan")},
                          {"preflight_seconds": -1}, {"min_passes": True}, {"max_passes": False},
                          {"min_passes": 21}, {"safety_factor": .99}, {"safety_factor": False}):
            options = {"budget_seconds": 5000., **parameter}
            with self.assertRaises(ValueError):
                choose_uniform_pass_budget(costs, **options)
        for bad in ([], [{}], [dict(training_pass_seconds=0)],
                    [dict(training_pass_seconds=True)],
                    [dict(training_pass_seconds=float("inf"))],
                    [dict(training_pass_seconds=1, evaluation_seconds=-1)],
                    [dict(training_pass_seconds=1, validation_pass_seconds=False)]):
            with self.assertRaises(ValueError):
                choose_uniform_pass_budget(bad, budget_seconds=5000.)


if __name__ == "__main__":
    unittest.main()
