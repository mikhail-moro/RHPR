from __future__ import annotations

from contextlib import ExitStack
import copy
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
            self.identity = identity


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryProxyHpoTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_prefix_proxy_hpo.json").read_text())

    def test_trial_selection_is_strict_validation_only_and_ties_follow_config_order(self):
        from rhpr.sasrec.query_proxy_hpo import select_proxy_trial
        trials = self.config()["hpo"]["trials"]
        results = [{"complete": True, "trial": trial, "best_validation": {"ndcg10": score}}
                   for trial, score in zip(trials, (.5, .5, .4))]
        self.assertEqual(select_proxy_trial(trials, results), trials[0])
        changed = copy.deepcopy(results)
        changed[2]["best_validation"]["ndcg10"] = .6
        self.assertEqual(select_proxy_trial(trials, changed), trials[2])
        for broken in (results[:-1], [{**results[0], "complete": False}, *results[1:]],
                       [{**results[0], "best_validation": {"ndcg10": float("nan")}}, *results[1:]]):
            with self.assertRaises(ValueError):
                select_proxy_trial(trials, broken)

    def test_budget_prices_the_fixed_grid_and_rejects_insufficient_time_without_shortening(self):
        from rhpr.sasrec.query_proxy_hpo import conservative_hpo_budget
        config = self.config()
        profile = {"training": {"windows_per_second": 100.}, "daily_evaluation": {"anchors_per_second": 100.}}
        profiles = {name: {phase: profile for phase in ("proxy", "full")} for name in config["variants"]}
        row = conservative_hpo_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 100})
        # 2 families * (3 * (20*10 + 2*1) + 30*10 + 3*1 + 1*1).
        self.assertEqual(row["estimated_remaining_work_seconds"], 1820.)
        self.assertEqual(row["conservative_total_seconds"], 100. + 1.25 * (1820. + 480.))
        self.assertTrue(row["fits"])
        config["training"]["budget_seconds"] = 1000
        self.assertFalse(conservative_hpo_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 100})["fits"])
        self.assertEqual(config["hpo"]["proxy_passes"], 20)
        self.assertEqual(len(config["hpo"]["trials"]), 3)

    def test_disposable_preflight_uses_two_successful_effective_updates_and_real_short_shape(self):
        from rhpr.sasrec.query_proxy_hpo import _preflight_shape
        config, calls = self.config(), []
        model = TinyModel(0)

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            calls.append(batch_builder.keywords["old_length"])
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                "windows_per_second": 128., "seconds": 1.}

        module = "rhpr.sasrec.query_proxy_hpo."
        with ExitStack() as stack:
            for name, kwargs in (
                (module + "build_hpo_model", {"return_value": model}),
                (module + "fixed_prefix_schedule", {"return_value": (1, 2)}),
                (module + "_representative_windows", {"return_value": (1, 2)}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "_profile_daily_rate", {"return_value": {"anchors_per_second": 100.}}),
            ):
                stack.enter_context(patch(name, **kwargs))
            result = _preflight_shape(config, SimpleNamespace(num_items=10), None,
                {"name": config["variants"][0]}, 4096, torch.device("cpu"))
        self.assertEqual(calls, [4096, 4096])
        self.assertFalse(result["profile_weights_reused"])
        self.assertEqual(result["profile_trial"]["name"], "faster_compressor")

    def test_fixed_screening_and_fresh_finalists_finish_both_selections_before_any_test(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_proxy_hpo import run_query_proxy_hpo
        config = self.config()
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        users = tuple(LongHistoryUser(i, np.arange(300 + i, dtype=np.int32) % 41 + 1,
            np.arange(300 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=3)
        evaluation = SimpleNamespace(summary=summary)
        trained, models, events, reads, tested = [], [], [], [], []
        pass_counts = {}
        scores = ((.6, .6), (.6, .55), (.5, .5), (.4, .41), (.7, .7), (.6, .6),
                  (.8, .8, .7), (.7, .9, .9))

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def build(*args):
            model = TinyModel(len(models))
            models.append(model)
            return model

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            pass_counts[model.identity] = pass_counts.get(model.identity, 0) + 1
            count = pass_counts[model.identity]
            old = batch_builder.keywords["old_length"]
            self.assertEqual(old, 4096 if model.identity < 6 else None)
            trained.append((model.identity, count, [(w.epoch, w.user_id, w.end, w.negative_seed) for w in windows]))
            self.assertEqual(float(model.marker.detach()[0]) if count == 1 else 1., 1.)
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            events.append("train")
            return {"loss": 1., "windows": 3, "labels": 384, "successful_updates": 1,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def evaluate(model, examples, **kwargs):
            split = list(examples)[0]
            events.append(split)
            if split == "test":
                self.assertEqual(len(trained), 180)
                self.assertTrue((output / "final-selection.json").exists())
                selection = json.loads((output / "final-selection.json").read_text())
                self.assertEqual(len(selection), 2)
                self.assertEqual(sorted(x["best_epoch"] for x in selection.values()), [10, 20])
                tested.append(float(model.marker.detach()[0]))
                score = .45
            else:
                score = scores[model.identity][pass_counts[model.identity] // 10 - 1]
            from rhpr.sasrec.daily_quality import _aggregate
            anchors = rows(score)
            metrics, users = _aggregate(anchors)
            return {"metrics": metrics, "per_user": users, "per_anchor": anchors, "metadata": {}}

        def read(paths, split):
            reads.append((split, len(trained)))
            self.assertEqual(split, "test")
            self.assertEqual(len(trained), 180)
            return {name: rows(.4) for name in config["parent"]["baselines"]}

        profile = {"training": {"windows_per_second": 10000.}, "daily_evaluation": {"anchors_per_second": 10000.}}
        module = "rhpr.sasrec.query_proxy_hpo."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                (module + "validate_proxy_hpo_config", {}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (output,
                    {split: {"rows": []} for split in ("validation", "test")},
                    {name: rows(.4) for name in config["parent"]["baselines"]}, {})}),
                (module + "_verified_file", {"return_value": output / "pinned.gz"}),
                (module + "_read_baselines", {"side_effect": read}),
                (module + "run_proxy_diagnostics", {"return_value": {"complete": True}}),
                (module + "_preflight_shape", {"return_value": profile}),
                (module + "build_hpo_model", {"side_effect": build}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "audited_hpo_examples", {"side_effect": lambda examples, *a: examples}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "_contrasts", {"return_value": []}),
                ("rhpr.sasrec.query_prefix_benchmark.benchmark_prefix_quality", {"return_value": {
                    "metrics": {"full_b1_wall_p50_ms": .1}, "metadata": {}, "cases": {}}}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            metrics = run_query_proxy_hpo(config, output)
            report = json.loads((output / "query-proxy-hpo.json").read_text())
            states = [torch.load(output / f"checkpoints/{name}.pt", weights_only=True) for name in config["variants"]]
            resumes = [torch.load(output / f"checkpoints/{name}-resume.pt", weights_only=True) for name in config["variants"]]
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
        self.assertTrue(metrics["quality_complete"])
        self.assertEqual(metrics["test_evaluations"], 2)
        self.assertEqual(len(trained), 180)
        self.assertEqual(events.count("validation"), 18)
        self.assertEqual(events.count("test"), 2)
        self.assertEqual(reads, [("test", 180)])
        self.assertEqual(report["selected_trials"][config["variants"][0]]["name"], "current_lr")
        self.assertEqual(report["selected_trials"][config["variants"][1]]["name"], "lower_lr")
        self.assertEqual([state["best_epoch"] for state in states], [10, 20])
        self.assertEqual([float(state["state_dict"]["marker"][0]) for state in states], tested)
        self.assertEqual([r["epoch"] for r in resumes], [30, 30])
        self.assertTrue(all(r["optimizer_state_dict"]["state"] for r in resumes))
        self.assertTrue(all(r["experiment_config"]["active_stage"] == "full_finalist" for r in resumes))
        self.assertEqual([r["experiment_config"]["active_trial"]["name"] for r in resumes], ["current_lr", "lower_lr"])
        self.assertEqual(len(manifest["checkpoints"]), 2)
        self.assertFalse(manifest["proxy_checkpoints_saved"])
        # Every model starts with the same endpoint identities/order/negative seeds.
        first = [row[2] for row in trained if row[1] == 1]
        self.assertTrue(all(row == first[0] for row in first))


if __name__ == "__main__":
    unittest.main()
