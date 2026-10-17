from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class Query1KProtocolTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_prefix_1k_search.json").read_text())

    def test_registered_search_is_accepted_without_mutation(self):
        from rhpr.sasrec.query_1k_protocol import validate_1k_config

        config = self.config()
        original = copy.deepcopy(config)
        validate_1k_config(config)
        self.assertEqual(config, original)

    def test_registered_plan_and_pilot_pins_cannot_drift(self):
        from rhpr.sasrec.query_1k_protocol import validate_1k_config

        config = self.config()
        mutations = [
            ("hpo", "proxy_old_length", 2048), ("hpo", "min_passes", 20),
            ("hpo", "max_passes", 30), ("hpo", "patience", 2),
            ("hpo", "full_length_validation", False), ("hpo", "test_evaluations", 1),
            ("hpo", "trials", list(reversed(config["hpo"]["trials"]))),
            ("training", "budget_seconds", 21600), ("training", "max_passes", 100),
            ("training", "early_stopping_min_passes", 20),
            ("training", "early_stopping_patience", 4),
            ("training", "validation_interval", 5),
            ("training", "num_negatives", 128), ("model", "n_layers", 3),
            ("compressor", "d_inner", 64), ("runtime", "target_torch_version", "2.5.1"),
            ("data", "catalog_sha256", "0" * 64),
            ("parent", "source_code_sha", "0" * 40),
            ("parent", "manifest_sha256", "0" * 64),
            ("parent", "checkpoint_root", "/tmp/rhpr-sasrec-prefix-query-pilot"),
            ("parent", "baselines", {}), ("parent", "checkpoints", {}),
            ("kaggle", "kernel_sources", []),
        ]
        for section, key, value in mutations:
            bad = copy.deepcopy(config)
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_1k_config(bad)
        # Mutually changing the parent and child still breaks the completed-run hash.
        bad = copy.deepcopy(config)
        bad["seed"] = bad["parent"]["experiment_config"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "exact completed parent"):
            validate_1k_config(bad)

    def test_unknown_or_missing_registered_fields_are_rejected(self):
        from rhpr.sasrec.query_1k_protocol import validate_1k_config

        config = self.config()
        for section in (None, "hpo", "training", "kaggle", "parent"):
            bad = copy.deepcopy(config)
            target = bad if section is None else bad[section]
            target["unregistered_setting"] = True
            with self.subTest(section=section), self.assertRaises(ValueError):
                validate_1k_config(bad)
        bad = copy.deepcopy(config)
        del bad["benchmark"]
        with self.assertRaises(ValueError):
            validate_1k_config(bad)
        for key, value in (("name", "other-search"), ("kind", "yambda_query_prefix_proxy_hpo")):
            bad = copy.deepcopy(config)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_1k_config(bad)

    def test_selection_uses_full_validation_and_excludes_failed_or_unfinished_trials(self):
        from rhpr.sasrec.query_1k_protocol import select_1k_candidate

        trials = self.config()["hpo"]["trials"]
        results = [
            {"trial_name": "current_lr", "status": "success", "best_proxy_validation": {"ndcg10": .9},
             "best_full_validation": {"ndcg10": .06}},
            {"trial_name": "lower_lr", "status": "failed", "best_full_validation": {"ndcg10": .99}},
            {"trial_name": "faster_compressor", "status": "success", "best_proxy_validation": {"ndcg10": .01},
             "best_full_validation": {"ndcg10": .061}},
        ]
        selected = select_1k_candidate(trials, results)
        self.assertEqual(selected, trials[2])
        self.assertIsNot(selected, trials[2])
        results[2]["status"] = "running"
        self.assertEqual(select_1k_candidate(trials, results), trials[0])
        self.assertIsNone(select_1k_candidate(trials, results[1:]))
        self.assertIsNone(select_1k_candidate(trials, []))

    def test_selection_excludes_invalid_full_scores_and_preserves_zero(self):
        from rhpr.sasrec.query_1k_protocol import select_1k_candidate

        trials = self.config()["hpo"]["trials"]
        for value in (math.nan, math.inf, -math.inf, True, ".06", None, -.01, 1.01):
            records = [{"trial_name": "current_lr", "status": "success",
                        "best_full_validation": {"ndcg10": value}}]
            with self.subTest(value=value):
                self.assertIsNone(select_1k_candidate(trials, records))
        for metrics in (None, {}, [], {"ndcg100": .2}):
            self.assertIsNone(select_1k_candidate(trials, [{"trial_name": "current_lr", "status": "success",
                                                          "best_full_validation": metrics}]))
        self.assertEqual(select_1k_candidate(trials, [{"trial_name": "current_lr", "status": "success",
                                                      "best_full_validation": {"ndcg10": 0.}}]), trials[0])

    def test_selection_ties_use_registered_order_not_result_order(self):
        from rhpr.sasrec.query_1k_protocol import select_1k_candidate

        trials = self.config()["hpo"]["trials"]
        results = [{"trial_name": trial["name"], "status": "success",
                    "best_full_validation": {"ndcg10": .06}} for trial in reversed(trials)]
        self.assertEqual(select_1k_candidate(trials, results), trials[0])

    def test_ambiguous_or_unknown_selection_inputs_are_rejected(self):
        from rhpr.sasrec.query_1k_protocol import select_1k_candidate

        trials = self.config()["hpo"]["trials"]
        good = {"trial_name": "current_lr", "status": "success", "best_full_validation": {"ndcg10": .06}}
        invalid_inputs = [
            (trials + [dict(trials[0])], []), ([{"name": ""}], []),
            ([None], []), (trials, [good, good]),
            (trials, [{"trial_name": "unknown", "status": "success"}]),
            (trials, [None]), ("current_lr", []), (trials, {}),
        ]
        for selected_trials, results in invalid_inputs:
            with self.subTest(trials=selected_trials, results=results), self.assertRaises(ValueError):
                select_1k_candidate(selected_trials, results)


if __name__ == "__main__":
    unittest.main()
