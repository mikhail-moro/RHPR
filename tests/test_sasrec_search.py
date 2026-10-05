from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.search import candidate_config, run_search


class SearchTest(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "experiments/yambda_sasrec_search.json").read_text())
        config["search"].update(n_trials=2, n_finalists=1)
        return config

    def test_conditional_heads_and_base_preserved(self):
        base = self.config()["base"]
        original = copy.deepcopy(base)
        params = dict(d_model=128, n_layers=2, head_dim=32, ffn_multiplier=2,
                      dropout=0.1, learning_rate=0.0005, weight_decay=1e-5)
        candidate = candidate_config(base, params)
        self.assertEqual(candidate["model"]["n_heads"], 4)
        self.assertEqual(candidate["model"]["max_seq_len"], 1028)
        self.assertEqual(base, original)

    def exercise_search(self, expire_after_first=False):
        now = [100.0]
        study = SimpleNamespace(trials=[], enqueue_trial=Mock())

        def optimize(objective, n_trials, **kwargs):
            for number in range(n_trials):
                trial = SimpleNamespace(number=number,
                    suggest_categorical=lambda name, choices: choices[0],
                    suggest_float=lambda name, low, high, **kw: low,
                    set_user_attr=Mock())
                value = objective(trial)
                study.trials.append(SimpleNamespace(number=number,
                    state=SimpleNamespace(name="COMPLETE"), value=value, params={}))
                study.trials[-1].state = "COMPLETE"

        # String-enum stand-in supports both equality and .name, like TrialState.
        class State(str):
            @property
            def name(self):
                return str(self)

        def optimized(objective, n_trials, **kwargs):
            optimize(objective, n_trials, **kwargs)
            for trial in study.trials:
                trial.state = State("COMPLETE")

        study.optimize = optimized
        optuna = SimpleNamespace(__version__="test", create_study=Mock(return_value=study),
            samplers=SimpleNamespace(TPESampler=Mock()),
            trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE="COMPLETE")))
        torch = SimpleNamespace(__version__="test", cuda=SimpleNamespace(
            empty_cache=Mock(), OutOfMemoryError=type("OOM", (RuntimeError,), {})))
        evaluated = Mock(return_value={"test_ndcg_at_10": 0.08, "test_examples": 9238})
        seen = []

        def execute(path, directory):
            candidate = json.loads(path.read_text())
            self.assertFalse(candidate["evaluation"]["evaluate_test"])
            self.assertEqual(candidate["training"]["physical_batch_size"], 8)
            seen.append(candidate["name"])
            if candidate["training"]["keep_checkpoint"]:
                (directory / "best-model.pt").write_bytes(candidate["name"].encode())
            if expire_after_first:
                now[0] = 100000.0
            return {"metrics": {"validation_ndcg_at_10": 0.04 + 0.001 * len(seen),
                                "train_loss_at_best": 0.1, "best_epoch": 1}}

        modules = {"optuna": optuna, "torch": torch,
                   "rhpr.sasrec.experiment": SimpleNamespace(evaluate_saved_checkpoint=evaluated)}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, modules), \
             patch("rhpr.runner.execute", side_effect=execute), \
             patch("rhpr.sasrec.search.time.time", side_effect=lambda: now[0]):
            output = Path(temporary)
            result = run_search(self.config(), output)
            evaluated.assert_called_once()
            self.assertTrue((output / "best-model.pt").is_file())
            self.assertTrue((output / "search-results.json").is_file())
            self.assertNotIn("deadline_unix_seconds", json.loads((output / "best-config.json").read_text())["training"])
            return result, seen

    def test_search_tests_only_selected_winner(self):
        result, seen = self.exercise_search()
        self.assertEqual(len(seen), 6)  # 3 losses, 2 trials, 1 finalist.
        self.assertEqual(result["best_run"], "finalist-0")
        self.assertEqual(result["optuna_trials_completed"], 2)

    def test_budget_returns_completed_checkpoint(self):
        result, seen = self.exercise_search(expire_after_first=True)
        self.assertEqual(seen, ["loss-bce-1"])
        self.assertTrue(result["search_budget_reached"])
        self.assertEqual(result["best_run"], "loss-bce-1")


if __name__ == "__main__":
    unittest.main()
