from __future__ import annotations

from contextlib import ExitStack
import gzip
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
    class ArchitectureTinyModel(torch.nn.Module):
        def __init__(self, identity):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = torch.nn.Linear(1, 1)
            self.register_buffer("origin", torch.tensor(identity))


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryArchitectureSearchTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_compressor_architecture.json").read_text())
        config["hpo"].update(min_passes=20, max_passes=60, validation_interval=10, patience=2)
        return config

    def test_budget_prices_four_new_shapes_at_distinct_measured_rates_without_control_training(self):
        from rhpr.sasrec.query_architecture_search import conservative_architecture_budget
        config = self.config()
        profiles = {}
        expected = 0.
        for family, trials in config["architecture_trials"].items():
            for index, trial in enumerate(trials[1:], 1):
                rate = 100. * index
                profiles[f"{family}-{trial['name']}"] = {"status": "success",
                    "training": {"windows_per_second": rate},
                    "proxy_evaluation": {"anchors_per_second": rate * 2},
                    "full_evaluation": {"anchors_per_second": rate / 2}}
                expected += 60 * 1000 / rate + 6 * 100 / (rate * 2) + 100 / (rate / 2)
        row = conservative_architecture_budget(config, profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 10 ** 12})
        self.assertEqual(row["estimated_remaining_work_seconds"], expected)
        self.assertEqual(row["conservative_total_seconds"], 100. + 1.25 * (expected + 480.))
        self.assertEqual((row["priced_fresh_shapes"], row["reused_controls"]), (4, 2))
        self.assertEqual(row["test_evaluations"], 0)
        first = next(iter(profiles))
        profiles[first] = {"status": "preflight_out_of_memory"}
        row = conservative_architecture_budget(config, profiles, 100., 1000, {"validation_anchors": 100})
        self.assertEqual((row["priced_fresh_shapes"], row["preflight_failed_shapes"]), (3, 1))
        profiles[first] = {"training": {"windows_per_second": float("nan")},
            "proxy_evaluation": {"anchors_per_second": 1.}, "full_evaluation": {"anchors_per_second": 1.}}
        with self.assertRaises(ValueError):
            conservative_architecture_budget(config, profiles, 0., 1000, {"validation_anchors": 100})

    def test_preflight_uses_exact_trial_two_real_updates_and_reports_each_input_cost(self):
        from rhpr.sasrec.query_architecture_search import _preflight_architecture
        config, calls, caps = self.config(), [], []
        model = ArchitectureTinyModel(0)
        trial = config["architecture_trials"]["query_ssm_m4"][2]
        cohort = SimpleNamespace(num_items=10, users=[SimpleNamespace(user_id=0, train=np.arange(2000))])

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            calls.append(batch_builder.keywords["old_length"])
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                "windows_per_second": 128., "seconds": 1.}

        def daily(model, *args, context_length, **kwargs):
            caps.append(context_length)
            self.assertFalse(model.marker.grad is None)
            return {"anchors_per_second": 100.}

        def batch(contexts, *args):
            return torch.tensor(np.array(contexts)), torch.ones((1, len(contexts[0])), dtype=torch.bool)

        module = "rhpr.sasrec.query_architecture_search."
        with ExitStack() as stack:
            for name, kwargs in (
                (module + "_build_architecture_model", {"return_value": model}),
                (module + "fixed_prefix_schedule", {"return_value": (1, 2)}),
                (module + "_representative_windows", {"return_value": (1, 2)}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "_profile_daily_rate", {"side_effect": daily}),
                (module + "_batch", {"side_effect": batch}),
                (module + "_observe_prefix_flops", {"return_value": (torch.ones((1, 1)), {"scan_arithmetic_flops": 42})}),
            ):
                stack.enter_context(patch(name, **kwargs))
            result = _preflight_architecture(config, cohort, None,
                {"name": "query_ssm_m4"}, trial, torch.device("cpu"))
        self.assertEqual(calls, [1024, 1024])
        self.assertEqual(caps, [1152, 32768])
        self.assertEqual(result["profile_trial"], trial)
        self.assertEqual(result["representative_b1_arithmetic"]["proxy"]["input_tokens"], 1152)
        self.assertEqual(result["representative_b1_arithmetic"]["full"]["input_tokens"], 1999)
        self.assertFalse(result["approves_stability_of_other_shapes"])
        self.assertTrue(result["optimizer_state_allocated_during_all_measurements"])

    def test_preflight_reclassifies_only_known_nonfinite_evaluation_errors(self):
        from rhpr.sasrec.query_architecture_search import _preflight_architecture
        config = self.config()
        trial = config["architecture_trials"]["query_ssm_m4"][2]
        cohort = SimpleNamespace(num_items=10)
        module = "rhpr.sasrec.query_architecture_search."

        def train(model, cohort, windows, *, optimizer, **kwargs):
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                "windows_per_second": 128., "seconds": 1.}

        for message, expected in (("evaluation produced nonfinite final states", FloatingPointError),
                                  ("evaluation produced nonfinite catalogue scores", FloatingPointError),
                                  ("unrelated programming error", RuntimeError)):
            with self.subTest(message=message), ExitStack() as stack:
                for name, kwargs in (
                    (module + "_build_architecture_model", {"return_value": ArchitectureTinyModel(0)}),
                    (module + "fixed_prefix_schedule", {"return_value": (1, 2)}),
                    (module + "_representative_windows", {"return_value": (1, 2)}),
                    (module + "train_prefix_pass", {"side_effect": train}),
                    (module + "_profile_daily_rate", {"side_effect": RuntimeError(message)}),
                ):
                    stack.enter_context(patch(name, **kwargs))
                with self.assertRaisesRegex(expected, message):
                    _preflight_architecture(config, cohort, None,
                        {"name": "query_ssm_m4"}, trial, torch.device("cpu"))

    def test_reuses_two_controls_trains_four_shapes_and_selects_full_validation_with_early_stopping(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_architecture_search import run_query_architecture_search
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
        models, built_trials, preflighted, diagnostics, full_ids, identities = [], [], [], [], [], []
        counts = {}
        proxy_scores = ((.6, .5, .5, .5, .5, .5), (.1, .2, .3, .4, .5, .6),
            (.4, .4, .4, .4, .4, .4), (.4, .4, .4, .4, .4, .4))
        full_scores = (.55, .52, .9, .45)

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def build(current, variant, trial, num_items):
            self.assertNotIn(trial["name"], ("ca_base", "q_base"))
            built_trials.append(trial["name"])
            model = ArchitectureTinyModel(len(models))
            models.append(model)
            return model

        def controls(current, evaluation, output):
            controls = {}
            for name, trials in current["architecture_trials"].items():
                label = f"{name}-{trials[0]['name']}"
                rank_file = label + "-full-anchors.json.gz"
                with gzip.open(output / rank_file, "wt", encoding="utf-8") as stream:
                    json.dump(rows(.5), stream)
                controls[name] = {"status": "success", "complete": True, "trial_name": trials[0]["name"],
                    "trial": dict(trials[0]), "passes": [], "passes_completed": 60, "best_epoch": 50,
                    "best_validation": {"ndcg10": .5}, "best_full_validation": {"ndcg10": .5},
                    "best_validation_rank_file": rank_file, "full_validation_rank_file": rank_file,
                    "best_checkpoint": {"path": "/mounted/verified/" + label + ".pt", "reused": True}}
            return controls

        def preflight(current, cohort, evaluation, variant, trial, device):
            preflighted.append(trial["name"])
            return {"training": {"windows_per_second": 10000.},
                "proxy_evaluation": {"anchors_per_second": 10000.},
                "full_evaluation": {"anchors_per_second": 10000.}}

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            identity = int(model.origin)
            counts[identity] = counts.get(identity, 0) + 1
            count = counts[identity]
            self.assertLess(identity, 4)
            batch_builder(cohort, windows, config, {})
            if identity == 2 and count == 25:
                raise FloatingPointError("nonfinite prefix training loss")
            identities.append((identity, count, [(w.epoch, w.user_id, w.end, w.negative_seed) for w in windows]))
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"loss": 1., "windows": 3, "labels": 384, "successful_updates": 1,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def diagnostic(model, *args, **kwargs):
            diagnostics.append(int(model.origin))
            return {"complete": True}

        def audited(examples, old, *args):
            self.assertEqual(list(examples), ["validation"])
            return iter([old])

        def evaluate(model, examples, **kwargs):
            old, identity = list(examples)[0], int(model.origin)
            if old is None:
                self.assertEqual(len(counts), 4)
                full_ids.append(identity)
                score = full_scores[identity]
            else:
                self.assertEqual(old, 1024)
                score = proxy_scores[identity][counts[identity] // 10 - 1]
            from rhpr.sasrec.daily_quality import _aggregate
            anchors = rows(score)
            metrics, per_user = _aggregate(anchors)
            return {"metrics": metrics, "per_user": per_user, "per_anchor": anchors, "metadata": {}}

        module = "rhpr.sasrec.query_1k_search."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                ("rhpr.sasrec.query_architecture_protocol.validate_architecture_config", {}),
                ("rhpr.sasrec.query_architecture_protocol.load_architecture_controls", {"side_effect": controls}),
                ("rhpr.sasrec.query_architecture_search._preflight_architecture", {"side_effect": preflight}),
                ("rhpr.sasrec.query_architecture_search._build_architecture_model", {"side_effect": build}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (output,
                    {"validation": {"rows": []}}, {"recent128": rows(.4)}, {})}),
                (module + "hpo_batch_arrays", {"return_value": {"item_ids": np.array([[1, 2]])}}),
                (module + "capture_training_rng", {"return_value": {}}),
                (module + "diagnose_nonfinite_batch", {"side_effect": diagnostic}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "audited_hpo_examples", {"side_effect": audited}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "paired_daily_ndcg_bootstrap", {"return_value": {"delta_ndcg": 0.}}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            metrics = run_query_architecture_search(config, output)
            report = json.loads((output / "query-compressor-architecture.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertFalse((output / "query-1k-search.json").exists())
            self.assertEqual(len(manifest["checkpoints"]), 6)
            self.assertEqual(len(manifest["resume_checkpoints"]), 2)
            self.assertEqual(len([row for row in manifest["checkpoints"] if row.get("reused")]), 2)
        self.assertTrue(metrics["quality_complete"])
        self.assertEqual((metrics["registered_configurations"], metrics["proxy_training_models"], metrics["reused_controls"]), (6, 4, 2))
        self.assertEqual(metrics["test_evaluations"], 0)
        self.assertEqual(counts, {0: 30, 1: 60, 2: 25, 3: 30})
        self.assertEqual(preflighted, ["ca_read2", "ca_heads8", "q_inner64", "q_state32"])
        self.assertEqual(built_trials[:4], preflighted)
        self.assertEqual(full_ids, [0, 1, 3])
        self.assertEqual(diagnostics, [2])
        self.assertEqual(report["selected_trials"]["query_cross_attention_m4"]["name"], "ca_read2")
        self.assertEqual(report["selected_trials"]["query_ssm_m4"]["name"], "q_base")
        self.assertEqual(report["trials"]["query_ssm_m4"]["q_inner64"]["resumable_checkpoint"]["epoch"], 20)
        self.assertFalse(report["trials"]["query_ssm_m4"]["q_inner64"]["resume_checkpoint_retained"])
        first = [row for _, epoch, row in identities if epoch == 1]
        self.assertTrue(all(row == first[0] for row in first))
        self.assertTrue(all(trial["backbone_lr"] == trial["compressor_lr"] == .001
            for family in config["architecture_trials"].values() for trial in family))


if __name__ == "__main__":
    unittest.main()
