from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryHistoryProtocolTest(unittest.TestCase):
    def config(self, family="qssm"):
        return json.loads((ROOT / f"experiments/yambda_history_continuation_{family}.json").read_text())

    def test_running_qssm_configuration_remains_byte_identical(self):
        path = ROOT / "experiments/yambda_history_continuation_qssm.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "f893722586dc9af7e011fbea6bc92cbb02e5dd9680eb60a9dcd4a4cfe7fa8faf")

    def test_two_registered_families_share_fixed_endpoint_comparison(self):
        from rhpr.sasrec.query_history_protocol import (
            history_arms, history_context_cap, history_evaluation_batch,
            history_trial, history_variant, validate_history_config,
        )
        for family in ("qssm", "fuxi"):
            config = self.config(family)
            original = copy.deepcopy(config)
            validate_history_config(config)
            variant = history_variant(config)
            self.assertEqual(variant["name"], config["selected_source"]["model"])
            self.assertEqual(variant["summary_tokens"], 32 if family == "qssm" else 0)
            arms = history_arms(config)
            self.assertEqual([a["name"] for a in arms], ["short", "full"] if family == "qssm" else ["full"])
            self.assertEqual([a["old_length"] for a in arms], [1024, None] if family == "qssm" else [None])
            self.assertEqual([a["physical_batch_size"] for a in arms], [32, 1] if family == "qssm" else [2])
            for arm in arms:
                self.assertEqual(history_context_cap(config, arm), 1152 if arm["name"] == "short" else 32768)
                self.assertEqual(history_evaluation_batch(config, arm["name"]), 32 if arm["name"] == "short" else 4)
            trial = history_trial(config)
            self.assertEqual(trial, config["selected_source"]["trial"])
            trial["backbone_lr"] = 9
            arms[0]["old_length"] = 9
            self.assertEqual(config, original)
            self.assertEqual(config["continuation"]["validation_passes"], [0, 10, 20])
            self.assertEqual(config["continuation"]["schedule_epoch_offset"], 60)
            self.assertEqual(config["continuation"]["primary_metric"], "endpoint20_ndcg10")
            self.assertFalse(config["continuation"]["early_stopping"])
            self.assertEqual(config["continuation"].get("short_control", "paired"),
                             "paired" if family == "qssm" else "historical_only")
            self.assertEqual(config["training"]["budget_seconds"], 21600 if family == "qssm" else 32400)
            self.assertEqual(config["training"]["physical_batch_sizes"], [1, 32] if family == "qssm" else [2])

    def test_registration_rejects_length_protocol_optimization_or_source_drift(self):
        from rhpr.sasrec.query_history_protocol import validate_history_config
        mutations = [
            lambda c: c.update(extra=True), lambda c: c.pop("selected_source"),
            lambda c: c.update(family="mamba"), lambda c: c.update(name="other"),
            lambda c: c.update(kind="other"), lambda c: c.update(summary_budgets=[4]),
            lambda c: c["architecture"].update(unregistered=True),
            lambda c: c["data"].update(catalog_sha256="0" * 64),
            lambda c: c["model"].update(n_layers=4), lambda c: c.update(seed=7),
            lambda c: c["parent"].update(source_code_sha="0" * 40),
            lambda c: c["parent"]["experiment_config"]["training"].update(raw_context_length=16384),
            lambda c: c["selected_source"]["checkpoint"].update(sha256="0" * 64),
            lambda c: c["selected_source"].update(source_code_sha="0" * 40),
            lambda c: c["selected_source"]["trial"].update(backbone_lr=.0003),
            lambda c: c["selected_source"]["artifacts"]["validation_ranks"].update(sha256="0" * 64),
            lambda c: c["training"].update(num_negatives=128),
            lambda c: c["training"].update(effective_batch_size=32),
            lambda c: c["training"].update(max_passes=60),
            lambda c: c["training"].update(budget_seconds=7200),
            lambda c: c["training"].update(raw_context_length=16384),
            lambda c: c["continuation"].update(optimizer_state="inherit"),
            lambda c: c["continuation"].update(early_stopping=True),
            lambda c: c["continuation"].update(primary_metric="best_ndcg10"),
            lambda c: c["continuation"].update(validation_passes=[10, 20]),
            lambda c: c["continuation"].update(schedule_epoch_offset=0),
            lambda c: c["continuation"].update(short_control="paired"),
            lambda c: c["continuation"].update(test_evaluations=1),
            lambda c: c["kaggle"].update(kernel_sources=[]),
        ]
        for family in ("qssm", "fuxi"):
            for index, mutate in enumerate(mutations):
                bad = self.config(family)
                mutate(bad)
                with self.subTest(family=family, mutation=index), self.assertRaises(ValueError):
                    validate_history_config(bad)

    def test_arm_helpers_reject_hidden_or_mutated_history_shapes(self):
        from rhpr.sasrec.query_history_protocol import history_arms, history_context_cap
        config = self.config()
        for invalid in ("H2048", 1024, None, True,
                        dict(history_arms(config)[0], old_length=2048),
                        dict(history_arms(config)[1], physical_batch_size=4)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                history_context_cap(config, invalid)

    def test_fuxi_rejects_a_fresh_short_arm_and_keeps_historical_source_pins(self):
        from rhpr.sasrec.query_history_protocol import history_context_cap, validate_history_config
        config = self.config("fuxi")
        with self.assertRaises(ValueError):
            history_context_cap(config, "short")
        before = copy.deepcopy(config["selected_source"])
        for mutate in (
                lambda c: c["continuation"].pop("short_control"),
                lambda c: c["training"].update(physical_batch_sizes=[1]),
                lambda c: c["training"].update(physical_batch_sizes=[2, 16]),
                lambda c: c["training"].update(budget_seconds=36000)):
            bad = copy.deepcopy(config)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_history_config(bad)
        self.assertEqual(config["selected_source"], before)
        self.assertEqual(config["selected_source"]["checkpoint"]["best_epoch"], 60)

    def cohort(self, lengths=(379, 1533, 2637)):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
            np.arange(length, dtype=np.uint32)) for uid, length in enumerate(lengths, 1))
        for user in users:
            user.train[-200] = user.train[-20] = 0  # Real occupied OOVs, not padding.
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_all_arms_preserve_final_targets_query_suffix_and_negative_identities(self):
        from rhpr.sasrec.query_history_protocol import history_arms, history_batch_arrays
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        cohort = self.cohort()
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=61)
        baseline = {}
        for family in ("qssm", "fuxi"):
            config = self.config(family)
            # Keep the semantic test small; helper deliberately does not reload
            # the immutable launch registration for each physical batch.
            config["parent"]["experiment_config"]["training"]["num_negatives"] = 7
            for specification in history_arms(config):
                arm = specification["name"]
                actual = history_batch_arrays(cohort, windows, config, arm)
                if arm not in baseline:
                    baseline[arm] = actual
                else:
                    for field in actual:
                        np.testing.assert_array_equal(actual[field], baseline[arm][field])
                for index, window in enumerate(windows):
                    user = cohort.users[window.user_index]
                    count = int(actual["valid_mask"][index].sum())
                    start = 0 if arm == "full" else max(0, window.end - 1152)
                    np.testing.assert_array_equal(actual["item_ids"][index, :count], user.train[start:window.end])
                    np.testing.assert_array_equal(actual["targets"][index, count-128:count], user.train[-128:])
                    np.testing.assert_array_equal(actual["item_ids"][index, count-256:count-128],
                                                  user.train[window.end-256:window.end-128])
                    self.assertTrue(((actual["item_ids"][index] == 0) & actual["valid_mask"][index]).any())
        np.testing.assert_array_equal(baseline["short"]["negative_ids"], baseline["full"]["negative_ids"])

    def test_full_input_refuses_truncation_and_noncanonical_window_end(self):
        from rhpr.sasrec.query_history_protocol import history_batch_arrays
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        config = self.config()
        config["parent"]["experiment_config"]["training"]["num_negatives"] = 2
        cohort = self.cohort()
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=61)
        for arm in ("short", "full"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                history_batch_arrays(cohort, [replace(windows[0], end=windows[0].end-1)], config, arm)
        too_long = self.cohort((32770,))
        windows = fixed_prefix_schedule(too_long, seed=42, epoch=61)
        for arm in ("short", "full"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                history_batch_arrays(too_long, windows, config, arm)


if __name__ == "__main__":
    unittest.main()
