from __future__ import annotations

from contextlib import ExitStack
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
    class TinyModel(torch.nn.Module):
        def __init__(self, identity):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = torch.nn.Linear(1, 1)
            self.register_buffer("origin", torch.tensor(identity))


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class Query1kSearchTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_prefix_1k_search.json").read_text())

    def test_budget_prices_every_trial_at60_and_never_prices_test_or_full_training(self):
        from rhpr.sasrec.query_1k_search import conservative_1k_budget
        config = self.config()
        profile = {"training": {"windows_per_second": 100.},
            "proxy_evaluation": {"anchors_per_second": 100.},
            "full_evaluation": {"anchors_per_second": 50.}}
        profiles = {name: profile for name in config["variants"]}
        row = conservative_1k_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 10 ** 12})
        self.assertEqual(row["estimated_remaining_work_seconds"], 6 * (600. + 6. + 2.))
        self.assertEqual(row["conservative_total_seconds"], 100. + 1.25 * (3648. + 720.))
        self.assertTrue(row["fits"])
        self.assertEqual(row["test_evaluations"], 0)
        self.assertEqual(row["full_training_models"], 0)
        config["training"]["budget_seconds"] = 1000
        self.assertFalse(conservative_1k_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100})["fits"])
        profiles[config["variants"][0]]["training"]["windows_per_second"] = float("nan")
        with self.assertRaises(ValueError):
            conservative_1k_budget(config, profiles, 0., 1000, {"validation_anchors": 100})

    def test_disposable_preflight_profiles_short_training_and_both_inference_lengths_with_adam(self):
        from rhpr.sasrec.query_1k_search import _preflight_1k
        config, calls, caps = self.config(), [], []
        model = TinyModel(0)

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            calls.append(batch_builder.keywords["old_length"])
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                "windows_per_second": 128., "seconds": 1.}

        def daily(model, *args, context_length, **kwargs):
            caps.append(context_length)
            return {"anchors_per_second": 100.}

        module = "rhpr.sasrec.query_1k_search."
        with ExitStack() as stack:
            for name, kwargs in (
                (module + "build_hpo_model", {"return_value": model}),
                (module + "fixed_prefix_schedule", {"return_value": (1, 2)}),
                (module + "_representative_windows", {"return_value": (1, 2)}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "_profile_daily_rate", {"side_effect": daily}),
            ):
                stack.enter_context(patch(name, **kwargs))
            result = _preflight_1k(config, SimpleNamespace(num_items=10), None,
                {"name": config["variants"][0]}, torch.device("cpu"))
        self.assertEqual(calls, [1024, 1024])
        self.assertEqual(caps, [1152, 32768])
        self.assertEqual(result["profile_trial"]["name"], "current_lr")
        self.assertFalse(result["approves_stability_of_other_recipes"])
        self.assertFalse(result["profile_weights_reused"])
        self.assertTrue(result["optimizer_state_allocated_during_all_measurements"])

    def test_six_fresh_trials_early_stop_and_full_transfer_selects_different_lr_then_proxy(self):
        self.exercise_suite()

    def test_missing_finite_full_family_fails_only_after_all_trials_and_saves_partial_artifacts(self):
        self.exercise_suite(all_q_full_fail=True)

    def test_programming_error_is_not_reclassified_as_failed_trial(self):
        self.exercise_suite(programming_error=True)

    def exercise_suite(self, *, all_q_full_fail=False, programming_error=False):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_1k_search import run_query_1k_search
        config = self.config()
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        users = tuple(LongHistoryUser(i, np.arange(1400 + i, dtype=np.int32) % 41 + 1,
            np.arange(1400 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=100000)
        evaluation = SimpleNamespace(summary=summary)
        models, trained, full_ids, batch_identities = [], [], [], []
        counts, captured, diagnostic_calls = {}, [], []
        proxy_scores = ((.8, .7, .7, .7, .7, .7), (.1, .2, .3, .4, .5, .6),
            (.7, .6, .6, .6, .6, .6), (.8, .8, .8, .8, .8, .8),
            (.5, .5, .5, .5, .5, .5), (.6, .6, .6, .6, .6, .6))
        full_scores = (.1, .9, .2, .1, .8, .3)

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def build(*args):
            model = TinyModel(len(models))
            models.append(model)
            return model

        def batch(*args, old_length, **kwargs):
            self.assertEqual(old_length, 1024)
            return {"item_ids": np.array([[1, 2]]), "negative_ids": np.array([[3]])}

        def capture(device):
            token = {"capture": len(captured)}
            captured.append(token)
            return token

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            identity = int(model.origin)
            counts[identity] = counts.get(identity, 0) + 1
            count = counts[identity]
            self.assertLess(identity, 6, "full-length transfer must never perform training")
            arrays = batch_builder(cohort, windows, config, {})
            self.assertEqual(int(arrays["item_ids"][0, 0]), 1)
            if programming_error and identity == 0:
                raise ValueError("bad architecture")
            if identity == 3 and count == 13:
                # Mimic a partial pass that changed live weights after prior
                # checkpoint10. This state must never replace that resume.
                model.marker.data.fill_(123.)
                raise FloatingPointError("nonfinite prefix training loss")
            trained.append((identity, count))
            batch_identities.append((identity, count,
                [(w.epoch, w.user_id, w.end, w.negative_seed) for w in windows]))
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"loss": 1., "windows": 3, "labels": 384, "successful_updates": 1,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def diagnostic(model, arrays, device, *, amp, rng_state):
            self.assertEqual(rng_state, captured[-1])
            diagnostic_calls.append(int(model.origin))
            return {"complete": True, "failure_replayed": True}

        def audited(examples, old, *args):
            self.assertEqual(list(examples), ["validation"])
            return iter([old])

        def evaluate(model, examples, **kwargs):
            old, identity = list(examples)[0], int(model.origin)
            if old is None:
                self.assertEqual(len(counts), 6, "all six training statuses must resolve before full transfer")
                full_ids.append(identity)
                if identity == 5 or (all_q_full_fail and identity == 4):
                    raise RuntimeError("evaluation produced nonfinite final states")
                score = full_scores[identity]
            else:
                self.assertEqual(old, 1024)
                score = proxy_scores[identity][counts[identity] // 10 - 1]
            from rhpr.sasrec.daily_quality import _aggregate
            anchors = rows(score)
            metrics, per_user = _aggregate(anchors)
            return {"metrics": metrics, "per_user": per_user, "per_anchor": anchors, "metadata": {}}

        profile = {"training": {"windows_per_second": 10000.},
            "proxy_evaluation": {"anchors_per_second": 10000.},
            "full_evaluation": {"anchors_per_second": 10000.}}
        module = "rhpr.sasrec.query_1k_search."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                (module + "validate_1k_config", {}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (output,
                    {"validation": {"rows": []}}, {"recent128": rows(.4)}, {})}),
                (module + "_preflight_1k", {"return_value": profile}),
                (module + "build_hpo_model", {"side_effect": build}),
                (module + "hpo_batch_arrays", {"side_effect": batch}),
                (module + "capture_training_rng", {"side_effect": capture}),
                (module + "diagnose_nonfinite_batch", {"side_effect": diagnostic}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "audited_hpo_examples", {"side_effect": audited}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "paired_daily_ndcg_bootstrap", {"return_value": {"delta_ndcg": 0.}}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            if programming_error:
                with self.assertRaisesRegex(ValueError, "bad architecture"):
                    run_query_1k_search(config, output)
                self.assertEqual(len(models), 1)
                return
            if all_q_full_fail:
                with self.assertRaisesRegex(RuntimeError, "no finite full-H"):
                    run_query_1k_search(config, output)
                metrics = json.loads((output / "metrics.json").read_text())
            else:
                metrics = run_query_1k_search(config, output)
            report = json.loads((output / "query-1k-search.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            states = [torch.load(output / row["path"], weights_only=True) for row in manifest["checkpoints"]]
            resumes = [torch.load(output / row["path"], weights_only=True) for row in manifest["resume_checkpoints"]]
            self.assertEqual(len(states), 6, "the failed trial's earlier valid best10 is retained too")
            self.assertEqual(len(resumes), 2)
            self.assertEqual([r["epoch"] for r in resumes], [40, 40])
            self.assertTrue(all(r["optimizer_state_dict"]["state"] for r in resumes))
            self.assertEqual([r["experiment_config"]["active_trial"]["name"] for r in resumes],
                ["faster_compressor", "faster_compressor"])
            self.assertTrue(all(r["experiment_config"]["active_old_length"] == 1024 for r in resumes))
        self.assertEqual(metrics["quality_complete"], not all_q_full_fail)
        self.assertEqual(metrics["test_evaluations"], 0)
        self.assertEqual(metrics["full_training_models"], 0)
        self.assertEqual(counts, {0: 40, 1: 60, 2: 40, 3: 13, 4: 40, 5: 40})
        self.assertEqual(full_ids, [0, 1, 2, 4, 5])
        self.assertEqual(diagnostic_calls, [3], "full validation failure must not replay an unrelated train batch")
        ca, q = (report["trials"][name] for name in config["variants"])
        self.assertEqual(report["selected_trials"][config["variants"][0]]["name"], "lower_lr")
        self.assertEqual(ca["current_lr"]["best_epoch"], 10)
        self.assertEqual(ca["lower_lr"]["best_epoch"], 60)
        self.assertTrue(ca["current_lr"]["stopped_early"])
        self.assertFalse(ca["lower_lr"]["stopped_early"])
        self.assertEqual(ca["current_lr"]["stopping_reason"], "patience")
        self.assertEqual(ca["lower_lr"]["stopping_reason"], "max_passes")
        self.assertEqual(q["current_lr"]["failure"]["last_completed_training_pass"], 12)
        self.assertEqual(q["current_lr"]["resumable_checkpoint"]["epoch"], 10)
        self.assertFalse(q["current_lr"]["failure"]["partial_pass_state_checkpointed"])
        self.assertFalse(q["current_lr"]["resume_checkpoint_retained"])
        self.assertEqual(q["faster_compressor"]["status"], "full_transfer_failed")
        if not all_q_full_fail:
            self.assertEqual(report["selected_trials"][config["variants"][1]]["name"], "lower_lr")
        first = [identities for _, epoch, identities in batch_identities if epoch == 1]
        self.assertTrue(all(identity == first[0] for identity in first))


if __name__ == "__main__":
    unittest.main()
