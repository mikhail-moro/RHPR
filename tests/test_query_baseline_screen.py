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
except ImportError:
    np = torch = None


if torch is not None:
    class NativeTinyModel(torch.nn.Module):
        """A real state dict and parameter, deliberately no compressor module."""
        def __init__(self, family: str):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))
            self.family = family
            self.architecture = family
            self.native_runtime = {"python_reference_fallback": False}


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryBaselineScreenTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_primary_selection_does_not_use_full_transfer_quality(self):
        from rhpr.sasrec.query_baseline_screen import select_native_short_candidate
        trials = [{"name": "first"}, {"name": "second"}]
        results = [{"trial": trials[0], "status": "success", "complete": True,
                    "best_validation": {"ndcg10": .6}, "best_full_validation": {"ndcg10": .1}},
                   {"trial": trials[1], "status": "full_transfer_failed", "complete": True,
                    "best_validation": {"ndcg10": .7}}]
        self.assertEqual(select_native_short_candidate(trials, results), trials[1])
        results[1]["best_validation"]["ndcg10"] = .6
        self.assertEqual(select_native_short_candidate(trials, results), trials[0])
        for score in (float("nan"), float("inf"), True, -1., 1.1):
            with self.subTest(score=score):
                results[1]["best_validation"]["ndcg10"] = score
                self.assertEqual(select_native_short_candidate(trials, results), trials[0])
        results[0]["status"] = "numerical_failed"
        self.assertIsNone(select_native_short_candidate(trials, results))
        with self.assertRaises(ValueError):
            select_native_short_candidate(trials, results[:1])

    def test_four_fresh_backbones_reuse_two_controls_and_restore_native_best_checkpoints(self):
        self.exercise_suite()

    def test_optional_full_oom_retains_primary_and_continues_other_families(self):
        self.exercise_suite(full_oom=True)

    def test_failed_optional_full_preflight_does_not_repeat_unsupported_forward(self):
        self.exercise_suite(preflight_full_failure=True)

    def test_required_primary_preflight_failure_is_isolated_and_suite_reports_missing_family(self):
        self.exercise_suite(primary_preflight_failure=True)

    def test_unexpected_programming_error_propagates(self):
        self.exercise_suite(programming_error=True)

    def test_training_numerical_failure_retains_completed_pass_and_resolves_other_families(self):
        self.exercise_suite(training_failure=True)

    def exercise_suite(self, *, full_oom=False, preflight_full_failure=False,
                       primary_preflight_failure=False, programming_error=False,
                       training_failure=False):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
        from rhpr.sasrec.query_baseline_screen import _protocol_metadata, select_native_short_candidate
        config = json.loads((ROOT / "experiments/yambda_query_compressor_architecture.json").read_text())
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        users = tuple(LongHistoryUser(i, np.arange(1400 + i, dtype=np.int32) % 41 + 1,
            np.arange(1400 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=10 ** 12)
        evaluation = SimpleNamespace(summary=summary)
        families = ("query_cross_attention_m4", "query_ssm_m4", "recent128", "query_only_m4", "mamba4rec", "fuxi_linear")
        variants = [dict(name=name, kind=kind, summary_tokens=4) for name, kind in zip(families,
            ("query_cross_attention", "query_ssm", "recent", "query_only", "mamba4rec", "fuxi_linear"))]
        recipes = {name: dict(name=trial, backbone_lr=.001, compressor_lr=.001)
            for name, trial in zip(families, ("ca_base", "q_inner64", "fixed_lr", "fixed_lr", "fixed_lr", "fixed_lr"))}
        built, trained, evaluations, prepared, profiled, full_restored = [], [], [], [], [], []

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def controls(_config, _evaluation, output):
            result = {}
            for name in families[:2]:
                paths = [f"{name}-{condition}-anchors.json.gz" for condition in ("short", "full")]
                for path, score in zip(paths, (.65, .64)):
                    with gzip.open(output / path, "wt") as stream:
                        json.dump(rows(score), stream)
                result[name] = {"status": "success", "complete": True, "trial_name": recipes[name]["name"],
                    "trial": recipes[name], "best_epoch": 50, "passes_completed": 60,
                    "best_validation": {"ndcg10": .65}, "best_full_validation": {"ndcg10": .64},
                    "best_validation_rank_file": paths[0], "full_validation_rank_file": paths[1],
                    "best_checkpoint": {"path": f"checkpoints/{name}.pt"}, "reused": True}
            return result

        def build(_config, variant, trial, _items):
            self.assertEqual(trial, recipes[variant["name"]])
            built.append(variant["name"])
            return NativeTinyModel(variant["name"])

        def optimizer(model, training, trial):
            return torch.optim.AdamW([{"params": list(model.parameters()), "component": "backbone"}],
                lr=trial["backbone_lr"], weight_decay=training["weight_decay"], foreach=False)

        def preflight(_config, _cohort, _evaluation, variant, trial, _device):
            name = variant["name"]
            profiled.append(name)
            if primary_preflight_failure and name == "mamba4rec":
                raise torch.OutOfMemoryError("primary shape out of memory")
            row = {"status": "success"}
            if preflight_full_failure and name == "fuxi_linear":
                row["full_evaluation"] = {"status": "out_of_memory", "message": "optional shape failed"}
            return row

        def prepare(_cohort, windows, _config, variant, *, old_length):
            prepared.append((variant["name"], old_length))
            return {"item_ids": np.zeros((len(windows), 2), dtype=np.int64)}

        def train(model, _cohort, windows, *, variant, physical_batch, batch_builder, **kwargs):
            name, epoch = variant["name"], windows[0].epoch
            self.assertEqual(physical_batch, 1 if name == "fuxi_linear" else 2)
            self.assertFalse(hasattr(model, "compressor"))
            batch_builder(cohort, windows, config, variant)
            if training_failure and name == "fuxi_linear" and epoch == 2:
                raise FloatingPointError("nonfinite prefix training loss")
            with torch.no_grad():
                model.marker.fill_(epoch)
            trained.append((name, epoch))
            return {"seconds": 1., "loss": .1, "labels": len(users) * 128,
                "successful_updates": 2, "amp_skipped_update_attempts": 0}

        def examples(_examples, old, variant, _references, rows_out, guard):
            guard()
            rows_out.append({"family": variant["name"], "old_length": old})
            yield variant["name"], old

        def evaluate(model, source, *, batch_size, **kwargs):
            name, old = list(source)[0]
            evaluations.append((name, old, batch_size))
            self.assertEqual(name, model.family)
            if old is None:
                # No full model starts before all fresh training is resolved.
                self.assertEqual({name for name, _ in trained}, set(families[2:]) -
                    ({"mamba4rec"} if primary_preflight_failure else set()))
                self.assertEqual(float(model.marker.detach()), 10.)
                full_restored.append(name)
                if name == "fuxi_linear" and full_oom:
                    raise torch.OutOfMemoryError("optional full transfer OOM")
                score = .1
            else:
                if name == "mamba4rec" and programming_error:
                    raise RuntimeError("unexpected shape programming error")
                score = {10: .6, 20: .5, 30: .4}[int(model.marker.detach())]
            return {"metrics": {"ndcg10": score, "ndcg100": score + .01},
                "per_anchor": rows(score), "per_user": []}

        definition = QuerySearchDefinition(validate=lambda _: None, variants=lambda _: variants,
            trials=lambda _config, variant: [recipes[variant["name"]]], build_model=build,
            optimizer=optimizer, preflight=preflight, budget=lambda *_: {"fits": True},
            batch_arrays=prepare, training_batch=lambda _config, variant: 1 if variant["name"] == "fuxi_linear" else 2,
            evaluation_examples=examples,
            evaluation_batch=lambda _config, variant, old: (4 if variant["name"] == "fuxi_linear" else 8) if old is None else 32,
            full_transfer_required=lambda _config, variant: variant["name"] not in ("recent128", "query_only_m4"),
            select_candidate=select_native_short_candidate, protocol_metadata=_protocol_metadata,
            profile_each_trial=True, load_controls=controls, isolate_preflight_failures=True,
            paired_short_validation=True, optional_full_transfer=True, hpo_performed=False,
            report_file="query-baseline-screen.json")
        module = "rhpr.sasrec.query_1k_search."
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            for name, kwargs in (
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "resolve_dataset_directory", {"side_effect": lambda value: value}),
                (module + "_validate_complete_histories", {"return_value": None}),
                (module + "load_proxy_parent_artifacts", {"return_value": (None, {"validation": {"rows": []}}, {}, None)}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, **kw: iter(())}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "paired_daily_ndcg_bootstrap", {"return_value": {"delta": 0.}}),
            ):
                stack.enter_context(patch(name, **kwargs))
            with redirect_stdout(io.StringIO()):
                if programming_error:
                    with self.assertRaisesRegex(RuntimeError, "unexpected shape programming error"):
                        run_registered_query_search(config, output, definition=definition)
                    return
                if primary_preflight_failure or training_failure:
                    failed_family = "mamba4rec" if primary_preflight_failure else "fuxi_linear"
                    with self.assertRaisesRegex(RuntimeError, "native short-context.*" + failed_family):
                        run_registered_query_search(config, output, definition=definition)
                    metrics = json.loads((output / "metrics.json").read_text())
                else:
                    metrics = run_registered_query_search(config, output, definition=definition)
            report = json.loads((output / "query-baseline-screen.json").read_text())
            self.assertEqual(profiled, list(families[2:]))
            self.assertNotIn(families[0], built)
            self.assertNotIn(families[1], built)
            self.assertEqual({length for _, length in prepared}, {1024})
            self.assertFalse(metrics["hpo_performed"])
            self.assertEqual(metrics["reused_controls"], 2)
            self.assertEqual(metrics["test_evaluations"], 0)
            self.assertEqual(metrics["quality_complete"], not (primary_preflight_failure or training_failure))
            self.assertIn("not a fully trained long-history benchmark", report["protocol"]["comparison"])
            for name in ("recent128", "query_only_m4"):
                row = report["trials"][name]["fixed_lr"]
                self.assertEqual(row["best_epoch"], 10)
                self.assertEqual(row["passes_completed"], 30)
                self.assertTrue(row["stopped_early"])
                self.assertEqual(row["full_validation_rank_file"], row["best_validation_rank_file"])
                self.assertEqual(row["best_full_validation"]["ndcg10"], .6)
                self.assertTrue(row["full_validation_reuses_best_proxy"])
                self.assertEqual(row["compressor_parameters"], 0)
                self.assertEqual([group["component"] for group in row["optimizer_groups"]], ["backbone"])
                self.assertFalse(row["model_metadata"]["has_compressor"])
                self.assertNotIn(name, full_restored)
                self.assertEqual(len(report["cases"][name]["paired_matched_short_validation_contrasts"]), 2)
            if full_oom or preflight_full_failure:
                row = report["trials"]["fuxi_linear"]["fixed_lr"]
                self.assertEqual(row["status"], "full_transfer_failed")
                self.assertEqual(report["selected_trials"]["fuxi_linear"], recipes["fuxi_linear"])
                self.assertNotIn("best_full_validation", row)
                self.assertEqual(metrics["full_transfer_failed_trials"], 1)
                if preflight_full_failure:
                    self.assertNotIn("fuxi_linear", full_restored)
                    self.assertFalse(row["full_transfer_failure"]["full_evaluation_attempted"])
            elif not training_failure:
                self.assertIn(("fuxi_linear", None, 4), evaluations)
            if not primary_preflight_failure:
                self.assertIn(("mamba4rec", None, 8), evaluations)
            if training_failure:
                row = report["trials"]["fuxi_linear"]["fixed_lr"]
                self.assertEqual(row["status"], "numerical_failed")
                self.assertEqual(row["passes_completed"], 1)
                self.assertEqual(row["failure"]["stage"], "training")
                self.assertEqual(row["failure"]["last_completed_training_pass"], 1)
                self.assertFalse(row["failure"]["automatic_precision_or_lr_fallback"])
                self.assertNotIn("fuxi_linear", report["selected_trials"])
            elif not primary_preflight_failure:
                self.assertEqual(metrics["primary_successful_families"], 6)
            selection = json.loads((output / "final-selection.json").read_text())
            self.assertEqual(selection["selection_input"], "registered_native_short_context")


if __name__ == "__main__":
    unittest.main()
