from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


if torch is not None:
    class QSSMTinyModel(torch.nn.Module):
        def __init__(self, variant, trial):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))
            self.compressor = torch.nn.Linear(1, 1)
            self.family = variant["name"]
            self.summary_tokens = variant["summary_tokens"]
            self.trial = trial["name"]


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryQSSMSearchTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_qssm_search.json").read_text())

    def test_fresh_full_candidates_use_primary_quality_and_registered_tie_order(self):
        from rhpr.sasrec.query_qssm_protocol import qssm_trials, qssm_variants
        from rhpr.sasrec.query_qssm_search import ranked_qssm_results, qssm_full_transfer_candidates

        config = self.config()
        variants = qssm_variants(config)
        trials = {variant["name"]: qssm_trials(config, variant) for variant in variants}
        report = {"trials": {}}
        for index, variant in enumerate(variants):
            family = report["trials"][variant["name"]] = {}
            for offset, trial in enumerate(trials[variant["name"]]):
                family[trial["name"]] = {
                    "trial_name": trial["name"], "trial": trial, "complete": True,
                    "status": "training_completed", "best_validation": {"ndcg10": .7},
                    "best_full_validation": {"ndcg10": .99 - .1 * index - .01 * offset},
                }
        report["trials"][variants[0]["name"]][trials[variants[0]["name"]][0]["name"]].update(
            reused=True, best_validation={"ndcg10": .99}, status="success")
        ordered = ranked_qssm_results(config, variants, trials, report)
        self.assertEqual(ordered[0]["score"], .99)
        expected = {(variants[0]["name"], trials[variants[0]["name"]][1]["name"]),
                    (variants[1]["name"], trials[variants[1]["name"]][0]["name"])}
        self.assertEqual(qssm_full_transfer_candidates(config, variants, trials, report), expected)
        # A failed optional transfer must not exclude a finite primary result.
        first = report["trials"][variants[0]["name"]][trials[variants[0]["name"]][1]["name"]]
        first["status"] = "full_transfer_failed"
        first.pop("best_full_validation")
        self.assertEqual(qssm_full_transfer_candidates(config, variants, trials, report), expected)
        for invalid in (float("nan"), float("inf"), True, -1., 1.1):
            with self.subTest(score=invalid):
                first["best_validation"]["ndcg10"] = invalid
                self.assertNotIn((variants[0]["name"], first["trial_name"]),
                    qssm_full_transfer_candidates(config, variants, trials, report))

    def test_seven_fresh_trials_restore_only_two_best_full_models_and_retain_their_adam_states(self):
        self.exercise_suite()

    def test_optional_full_numerical_failure_preserves_primary_selection_and_second_finalist(self):
        self.exercise_suite(full_numerical_failure=True)

    def test_optional_full_allowance_exhaustion_finishes_primary_without_starting_second_forward(self):
        self.exercise_suite(full_budget_expired=True)

    def exercise_suite(self, *, full_numerical_failure=False, full_budget_expired=False):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_qssm_protocol import qssm_trials, qssm_variants
        from rhpr.sasrec.query_qssm_search import run_query_qssm_search

        config = self.config()
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        variants = qssm_variants(config)
        recipes = {variant["name"]: qssm_trials(config, variant) for variant in variants}
        pairs = [(variant["name"], trial["name"]) for variant in variants
                 for trial in recipes[variant["name"]]]
        control_pair = pairs[0]
        expected_finalists = {pairs[4], pairs[6]}
        short_scores = dict(zip(pairs, (.99, .65, .72, .70, .80, .60, .79, .40)))
        users = tuple(LongHistoryUser(i, np.arange(1400 + i, dtype=np.int32) % 41 + 1,
            np.arange(1400 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=10 ** 12)
        evaluation = SimpleNamespace(summary=summary)
        built, profiled, trained, full_calls, first_batches = [], [], {}, [], {}
        clock = [0.]

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def controls(_config, _evaluation, output):
            from rhpr.sasrec.query_prefix_pilot import _checkpoint_record
            variant, trial = variants[0], recipes[control_pair[0]][0]
            filenames = ["reused-qssm-" + condition + "-anchors.json.gz" for condition in ("short", "full")]
            for name in filenames:
                with gzip.open(output / name, "wt") as stream:
                    json.dump(rows(.99), stream)
            weights = output / "checkpoints" / "reused-qssm.pt"
            torch.save({"state_dict": QSSMTinyModel(variant, trial).state_dict()}, weights)
            return {control_pair[0]: {"status": "success", "complete": True, "trial_name": trial["name"],
                "trial": trial, "best_epoch": 50, "passes_completed": 60,
                "best_validation": {"ndcg10": .99, "ndcg100": .995},
                "best_full_validation": {"ndcg10": .99, "ndcg100": .995},
                "best_validation_rank_file": filenames[0], "full_validation_rank_file": filenames[1],
                "best_checkpoint": _checkpoint_record(weights, output, reused_source=True),
                "reused": True, "resume_checkpoint_retained": False}}

        def build(_config, variant, trial, _items):
            pair = variant["name"], trial["name"]
            self.assertNotEqual(pair, control_pair, "the completed K4 control must never be rebuilt or trained")
            built.append(pair)
            return QSSMTinyModel(variant, trial)

        def preflight(_config, _cohort, _evaluation, variant, trial, _device):
            profiled.append((variant["name"], trial["name"]))
            return {"status": "success", "full_evaluation": {"status": "deferred_to_selected_finalists"}}

        def train(model, _cohort, windows, *, variant, optimizer, batch_builder, physical_batch, **kwargs):
            pair = model.family, model.trial
            epoch = windows[0].epoch
            self.assertEqual(model.summary_tokens, variant["summary_tokens"])
            self.assertEqual(physical_batch, 32)
            arrays = batch_builder(cohort, windows, config, variant)
            self.assertEqual(arrays["item_ids"].shape, (3, 1152))
            if epoch == 1:
                first_batches[pair] = {key: value.copy() for key, value in arrays.items()}
            optimizer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            with torch.no_grad():
                model.marker.fill_(epoch)
            trained[pair] = epoch
            return {"seconds": 1., "loss": .1, "labels": len(users) * 128,
                "successful_updates": 2, "amp_skipped_update_attempts": 0}

        def audited(_examples, old, _references, rows_out, guard):
            guard()
            rows_out.append({"old_length": old})
            yield old

        def evaluate(model, examples, *, batch_size, **kwargs):
            old = list(examples)[0]
            pair = model.family, model.trial
            if old is None:
                self.assertEqual(set(trained), set(pairs) - {control_pair})
                self.assertTrue(all(epoch == 30 for epoch in trained.values()))
                self.assertEqual(float(model.marker.detach()), 10., "restore the selected best weights, not the last pass")
                self.assertEqual(batch_size, 4)
                full_calls.append(pair)
                if full_numerical_failure and pair == pairs[4]:
                    raise RuntimeError("evaluation produced nonfinite final states")
                if full_budget_expired:
                    clock[0] = config["full_transfer"]["reserve_seconds"] + 1.
                score = .1
            else:
                self.assertEqual(old, 1024)
                self.assertEqual(batch_size, 32)
                score = short_scores[pair] - .01 * (int(model.marker.detach()) // 10 - 1)
            return {"metrics": {"ndcg10": score, "ndcg100": score + .001},
                "per_anchor": rows(score), "per_user": []}

        engine = "rhpr.sasrec.query_1k_search."
        search = "rhpr.sasrec.query_qssm_search."
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            replacements = (
                (search + "validate_qssm_config", {"return_value": None}),
                (search + "load_qssm_controls", {"side_effect": controls}),
                (search + "build_qssm_model", {"side_effect": build}),
                (search + "preflight_qssm", {"side_effect": preflight}),
                (search + "conservative_qssm_budget", {"return_value": {"fits": True}}),
                (engine + "_registered_device", {"return_value": torch.device("cpu")}),
                (engine + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (engine + "load_prepared_cohort", {"return_value": cohort}),
                (engine + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (engine + "resolve_dataset_directory", {"side_effect": lambda value: value}),
                (engine + "_validate_complete_histories", {"return_value": None}),
                (engine + "load_proxy_parent_artifacts", {"return_value": (None, {"validation": {"rows": []}}, {}, None)}),
                (engine + "iter_daily_evaluation_examples", {"side_effect": lambda *a, **kw: iter(())}),
                (engine + "audited_hpo_examples", {"side_effect": audited}),
                (engine + "train_prefix_pass", {"side_effect": train}),
                (engine + "evaluate_daily_quality", {"side_effect": evaluate}),
                (engine + "paired_daily_ndcg_bootstrap", {"return_value": {"delta": 0.}}),
            )
            for target, kwargs in replacements:
                stack.enter_context(patch(target, **kwargs))
            if full_budget_expired:
                stack.enter_context(patch(engine + "time", SimpleNamespace(monotonic=lambda: clock[0])))
            with redirect_stdout(io.StringIO()):
                metrics = run_query_qssm_search(config, output)
            report = json.loads((output / "query-qssm-search.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertEqual(set(profiled), set(pairs) - {control_pair})
            self.assertEqual(len(profiled), 7)
            self.assertEqual(set(full_calls), {pairs[4]} if full_budget_expired else expected_finalists)
            self.assertEqual(len(full_calls), 1 if full_budget_expired else 2)
            self.assertEqual(set(built), set(pairs) - {control_pair})
            self.assertEqual(len(built), 8 if full_budget_expired else 9)
            reference = next(iter(first_batches.values()))
            for arrays in first_batches.values():
                for key in reference:
                    np.testing.assert_array_equal(arrays[key], reference[key])
            self.assertEqual(len(manifest["checkpoints"]), 8)
            resume_pairs = set()
            self.assertEqual(len(manifest["resume_checkpoints"]), 2)
            for record in manifest["resume_checkpoints"]:
                saved = torch.load(output / record["path"], weights_only=True)
                pair = record["training_model"], record["trial"]["name"]
                resume_pairs.add(pair)
                self.assertEqual(saved["epoch"], 30)
                self.assertEqual(saved["next_epoch"], 31)
                self.assertEqual(saved["experiment_config"]["active_trial"], record["trial"])
                self.assertEqual(float(saved["state_dict"]["marker"]), 30.)
                states = saved["optimizer_state_dict"]["state"]
                self.assertTrue(states)
                self.assertTrue(all(float(state["step"]) == 30. for state in states.values()))
                self.assertIn("torch_rng_state", saved)
                self.assertIn("scaler_state_dict", saved)
                self.assertIn("early_stopping", saved)
            self.assertEqual(resume_pairs, expected_finalists)
            self.assertEqual(len(list((output / "checkpoints").glob("*-resume.pt"))), 2)
            for name, trial_name in set(pairs) - expected_finalists - {control_pair}:
                result = report["trials"][name][trial_name]
                self.assertEqual(result["status"], "success")
                self.assertTrue(result["full_transfer_not_requested"])
                self.assertNotIn("best_full_validation", result)
                self.assertNotIn("full_validation_rank_file", result)
                self.assertFalse(result["resume_checkpoint_retained"])
            if full_numerical_failure:
                result = report["trials"][pairs[4][0]][pairs[4][1]]
                self.assertEqual(result["status"], "full_transfer_failed")
                self.assertEqual(result["best_validation"]["ndcg10"], .80)
                self.assertNotIn("best_full_validation", result)
                self.assertEqual(report["selected_trials"][pairs[4][0]]["name"], pairs[4][1])
                self.assertEqual(metrics["full_transfer_failed_trials"], 1)
            if full_budget_expired:
                for family, trial_name in expected_finalists:
                    result = report["trials"][family][trial_name]
                    self.assertEqual(result["status"], "full_transfer_failed")
                    self.assertEqual(result["failure"]["exception_type"], "_OptionalFullTransferTimeout")
                    self.assertNotIn("best_full_validation", result)
                    self.assertEqual(report["selected_trials"][family]["name"], trial_name)
                self.assertEqual(metrics["full_transfer_failed_trials"], 2)
            self.assertTrue(metrics["quality_complete"])
            self.assertEqual((metrics["registered_configurations"], metrics["proxy_training_models"],
                metrics["reused_controls"]), (8, 7, 1))
            self.assertEqual(metrics["test_evaluations"], 0)
            self.assertEqual(metrics["full_training_models"], 0)


if __name__ == "__main__":
    unittest.main()
