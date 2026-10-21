from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryQSSMProtocolTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_qssm_search.json").read_text())

    def test_registered_eight_recipes_and_exact_completed_control_adapter(self):
        from rhpr.sasrec.query_qssm_protocol import (
            _baseline_control_config, qssm_trials, qssm_variants, validate_qssm_config,
        )
        config = self.config()
        before = copy.deepcopy(config)
        validate_qssm_config(config)
        variants = qssm_variants(config)
        self.assertEqual([variant["name"] for variant in variants], config["variants"])
        self.assertEqual([variant["summary_tokens"] for variant in variants], [4, 8, 16, 32])
        self.assertEqual(_baseline_control_config(config), json.loads(
            (ROOT / "experiments/yambda_query_baseline_screen.json").read_text()))
        for variant in variants:
            trials = qssm_trials(config, variant)
            self.assertEqual([trial["name"] for trial in trials], ["dt_default", "dt_slow"])
            self.assertEqual([trial["dt_max"] for trial in trials], [.1, .01])
            for trial in trials:
                self.assertEqual((trial["d_inner"], trial["d_state"], trial["gate_rank"]), (64, 16, 32))
                self.assertEqual((trial["dt_min"], trial["backbone_lr"], trial["compressor_lr"]), (.0001, .001, .001))
            trials[0]["d_inner"] = 999
            self.assertEqual(qssm_trials(config, variant["name"])[0]["d_inner"], 64)
        self.assertEqual(config, before)

    def test_registration_rejects_protocol_data_or_artifact_pin_drift(self):
        from rhpr.sasrec.query_qssm_protocol import validate_qssm_config
        mutations = [
            lambda c: c.update(unregistered=True), lambda c: c.pop("reference_search"),
            lambda c: c.update(summary_budgets=[4, 8, 16]),
            lambda c: c.update(variants=c["variants"][::-1]),
            lambda c: c["hpo"].update(proxy_old_length=2048),
            lambda c: c["hpo"].update(min_passes=10),
            lambda c: c["hpo"].update(test_evaluations=1),
            lambda c: c["hpo"]["trials"][0].update(d_inner=128),
            lambda c: c["hpo"]["trials"][1].update(dt_max=.001),
            lambda c: c["training"].update(budget_seconds=14400),
            lambda c: c["training"].update(num_negatives=128),
            lambda c: c["model"].update(n_layers=4),
            lambda c: c["compressor"].update(d_inner=64),
            lambda c: c["data"].update(catalog_sha256="0" * 64),
            lambda c: c["parent"].update(source_code_sha="0" * 40),
            lambda c: c["reference_search"].update(source_code_sha="0" * 40),
            lambda c: c["reference_search"]["models"]["query_ssm_m4"]["checkpoint"].update(sha256="0" * 64),
            lambda c: c["kaggle"].update(kernel_sources=[]),
            lambda c: c["qssm_batches"]["query_ssm_m32"].update(training=16),
            lambda c: c["full_transfer"].update(new_configurations=8),
            lambda c: c["full_transfer"].update(selection_metric="full_validation_ndcg10"),
            lambda c: c["full_transfer"].update(reserve_seconds=0),
        ]
        for mutate in mutations:
            bad = self.config()
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_qssm_config(bad)

    def test_registered_batches_and_lengths_reject_unregistered_shapes(self):
        from rhpr.sasrec.query_qssm_protocol import qssm_evaluation_batch, qssm_trials, qssm_variants
        config = self.config()
        for variant in qssm_variants(config):
            self.assertEqual(qssm_evaluation_batch(config, variant, 1024), 32)
            self.assertEqual(qssm_evaluation_batch(config, variant, None), 4)
            for invalid in (True, 0, 128, 2048, 1024.):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    qssm_evaluation_batch(config, variant, invalid)
        for invalid in ("query_ssm_m64", {"name": "query_ssm_m8", "kind": "query_ssm", "summary_tokens": 4}):
            with self.assertRaises(ValueError):
                qssm_trials(config, invalid)

    def test_every_summary_count_preserves_identical_occupied_inputs_labels_and_negatives(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants
        from rhpr.sasrec.query_qssm_protocol import qssm_batch_arrays, qssm_variants
        config = self.config()
        config["parent"]["experiment_config"]["training"]["num_negatives"] = 7
        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
            np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 1533)))
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})
        for user in users:
            user.train[-200] = user.train[-20] = 0
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=7)
        for old_length in (1024, None):
            reference = hpo_batch_arrays(cohort, windows, config, hpo_variants(config)[1], old_length=old_length)
            for variant in qssm_variants(config):
                actual = qssm_batch_arrays(cohort, windows, config, variant, old_length=old_length)
                for field in reference:
                    np.testing.assert_array_equal(actual[field], reference[field])
                for index, window in enumerate(windows):
                    count = int(actual["valid_mask"][index].sum())
                    user = cohort.users[window.user_index]
                    np.testing.assert_array_equal(actual["targets"][index, count-128:count], user.train[-128:])
                    self.assertTrue(bool(((actual["item_ids"][index] == 0) & actual["valid_mask"][index]).any()))
                with self.assertRaises(ValueError):
                    qssm_batch_arrays(cohort, [replace(windows[0], end=windows[0].end-1)], config,
                        variant, old_length=old_length)

    def candidate(self, trial, score, **fields):
        return dict(trial=copy.deepcopy(trial), trial_name=trial["name"], complete=True,
            status="success", best_validation={"ndcg10": score}, **fields)

    def test_primary_selection_uses_registered_ties_not_full_score_or_result_order(self):
        from rhpr.sasrec.query_qssm_protocol import qssm_trials, select_qssm_candidate
        trials = qssm_trials(self.config(), "query_ssm_m4")
        rows = [self.candidate(trials[0], .2, best_full_validation={"ndcg10": .9}),
                self.candidate(trials[1], .3, best_full_validation={"ndcg10": .1})]
        self.assertEqual(select_qssm_candidate(trials, rows[::-1]), trials[1])
        rows[0]["best_validation"]["ndcg10"] = .3
        self.assertEqual(select_qssm_candidate(trials, rows[::-1]), trials[0])
        rows[1]["best_validation"]["ndcg10"] = .4
        for status in ("training_completed", "success", "full_transfer_failed"):
            rows[1]["status"] = status
            self.assertEqual(select_qssm_candidate(trials, rows), trials[1])

    def test_primary_selection_excludes_nonfinite_failed_unfinished_and_unknown_records(self):
        from rhpr.sasrec.query_qssm_protocol import qssm_trials, select_qssm_candidate
        trials = qssm_trials(self.config(), "query_ssm_m4")
        control = self.candidate(trials[0], .2)
        for score in (float("nan"), float("inf"), -.1, 1.1, True, None):
            self.assertEqual(select_qssm_candidate(trials, [control, self.candidate(trials[1], score)]), trials[0])
        for update in ({"complete": False}, {"status": "numerical_failed"},
                       {"status": "training"}, {"trial": {"name": trials[1]["name"]}}):
            bad = self.candidate(trials[1], .9)
            bad.update(update)
            self.assertEqual(select_qssm_candidate(trials, [control, bad]), trials[0])
        self.assertIsNone(select_qssm_candidate(trials, []))
        for bad in ([control, control], [{"trial_name": "unknown"}]):
            with self.assertRaises(ValueError):
                select_qssm_candidate(trials, bad)

    def test_reuse_filters_before_loader_io_and_preserves_original_trial_provenance(self):
        from rhpr.sasrec.query_qssm_protocol import load_qssm_controls, qssm_trials
        config = self.config()
        source = config["reference_search"]["models"]["query_ssm_m4"]["trial"]
        original = {"trial": copy.deepcopy(source), "trial_name": source["name"],
            "best_checkpoint": {"trial": copy.deepcopy(source), "sha256": "1" * 64},
            "reused_provenance": {"test_ranks_read": False, "weights_copied": True}}
        evaluation = object()
        with patch("rhpr.sasrec.query_qssm_protocol.load_baseline_controls",
                return_value={"query_ssm_m4": copy.deepcopy(original)}) as loader:
            controls = load_qssm_controls(config, evaluation, Path("output"))
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs, {"model_names": ("query_ssm_m4",)})
        self.assertIs(loader.call_args.args[1], evaluation)
        result = controls["query_ssm_m4"]
        expected = qssm_trials(config, "query_ssm_m4")[0]
        self.assertEqual(result["trial"], expected)
        self.assertEqual(result["trial_name"], "dt_default")
        self.assertEqual(result["best_checkpoint"]["trial"], expected)
        self.assertEqual(result["best_checkpoint"]["sha256"], "1" * 64)
        self.assertEqual(result["reused_source_trial"], source)
        self.assertEqual(result["reused_provenance"]["original_trial"], source)
        self.assertFalse(result["reused_provenance"]["test_ranks_read"])
        bad = self.config()
        bad["reference_search"]["models"]["query_ssm_m4"]["checkpoint"]["sha256"] = "0" * 64
        with patch("rhpr.sasrec.query_qssm_protocol.load_baseline_controls") as loader:
            with self.assertRaises(ValueError):
                load_qssm_controls(bad, evaluation, Path("output"))
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
