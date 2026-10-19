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
    class TinyRotatingModel(torch.nn.Module):
        def __init__(self, family):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = torch.nn.Linear(1, 1)
            self.family = family


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryRotatingWindowsTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_rotating_windows.json").read_text())
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        config["hpo"].update(min_passes=2, max_passes=4, validation_interval=1, patience=2,
                             bootstrap_draws=10)
        return config

    @staticmethod
    def rows(score):
        return [{"user_id": i, "day": day, "anchor_id": f"{i}:{day}",
            "reference_context_hash": "c" * 64, "target_known": True, "rank_cutoff": 100,
            "ndcg10": score + .01 * (day - 286), "ndcg100": score + .02 * (day - 286),
            "hr10": 1., "hr100": 1.} for i in range(3) for day in (286, 287)]

    def run_suite(self, *, fail_training=False, fail_preflight=False, fail_statistics=False,
                  fuxi_later_best=False):
        from rhpr.sasrec.daily_quality import _aggregate
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        from rhpr.sasrec.query_rotating_protocol import rotating_trials, rotating_variants
        from rhpr.sasrec.query_rotating_windows import run_query_rotating_windows

        config = self.config()
        users = tuple(LongHistoryUser(i, np.arange(2600 + i, dtype=np.int32) % 41 + 1,
            np.arange(2600 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=6, test_anchors=10 ** 12)
        evaluation = SimpleNamespace(summary=summary)
        built, profiles, counts, schedules, batches, evaluation_caps = [], [], {}, {}, {}, []

        def build(current, variant, trial, num_items):
            self.assertEqual(trial["name"], "rotating")
            built.append(variant["name"])
            return TinyRotatingModel(variant["name"])

        def controls(current, evaluation, output):
            result = {}
            for variant in rotating_variants(current):
                name = variant["name"]
                trial = rotating_trials(current, variant)[0]
                filename = name + "-historical-short-anchors.json.gz"
                with gzip.open(output / filename, "wt", encoding="utf-8") as stream:
                    json.dump(self.rows(.9), stream)
                result[name] = {"status": "success", "complete": True,
                    "trial_name": "historical_fixed", "trial": trial, "passes": [],
                    "passes_completed": 60, "best_epoch": 20,
                    "best_validation": {"ndcg10": .905, "ndcg100": .91},
                    "best_validation_rank_file": filename, "reused": True,
                    "best_checkpoint": {"path": "/verified/" + name + ".pt", "reused": True}}
            return result

        def preflight(current, cohort, evaluation, variant, trial, device):
            self.assertEqual(trial["name"], "rotating")
            profiles.append(variant["name"])
            if fail_preflight and variant["name"] == "query_ssm_m32":
                raise torch.OutOfMemoryError("isolated capacity failure")
            return {"status": "success", "training": {"windows_per_second": 10000.},
                "proxy_evaluation": {"anchors_per_second": 10000.}}

        def train(model, cohort, windows, *, optimizer, batch_builder, variant, **kwargs):
            name = model.family
            counts[name] = counts.get(name, 0) + 1
            epoch = counts[name]
            arrays = batch_builder(cohort, windows, config, variant)
            batches[name, epoch] = {key: value.copy() for key, value in arrays.items()}
            schedules[name, epoch] = tuple(windows)
            if fail_training and name == "query_ssm_m32" and epoch == 2:
                raise FloatingPointError("isolated nonfinite rotating loss")
            optimizer.zero_grad(set_to_none=True)
            loss = model.marker.square().sum() + model.compressor.weight.square().sum()
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.detach()), "windows": 3, "labels": 384,
                "successful_updates": 1, "amp_skipped_update_attempts": 0,
                "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def audited(examples, old, *args):
            self.assertEqual(list(examples), ["validation"])
            self.assertEqual(old, 1024)
            evaluation_caps.append(old)
            return iter([old])

        def evaluate(model, examples, **kwargs):
            self.assertEqual(list(examples), [1024])
            self.assertEqual(kwargs["batch_size"], 32)
            score = .4 if model.family == "query_ssm_m32" else .6
            if fuxi_later_best and model.family == "fuxi_linear" and counts[model.family] == 1:
                score = .5
            anchors = self.rows(score)
            metrics, per_user = _aggregate(anchors)
            return {"metrics": metrics, "per_user": per_user, "per_anchor": anchors, "metadata": {}}

        module = "rhpr.sasrec.query_1k_search."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                ("rhpr.sasrec.query_rotating_protocol.validate_rotating_config", {}),
                ("rhpr.sasrec.query_rotating_protocol.load_rotating_controls", {"side_effect": controls}),
                ("rhpr.sasrec.query_rotating_models.build_rotating_model", {"side_effect": build}),
                ("rhpr.sasrec.query_rotating_models.preflight_rotating", {"side_effect": preflight}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (output,
                    {"validation": {"rows": []}}, {"recent128": self.rows(.3)}, {})}),
                (module + "capture_training_rng", {"return_value": {}}),
                (module + "diagnose_nonfinite_batch", {"return_value": {"complete": True}}),
                (module + "fixed_prefix_schedule", {"side_effect": AssertionError("old fixed schedule was used")}),
                (module + "prefix_schedule_summary", {"side_effect": AssertionError("old fixed summary was used")}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "audited_hpo_examples", {"side_effect": audited}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            if fail_statistics:
                stack.enter_context(patch("rhpr.sasrec.query_rotating_windows._paired",
                    side_effect=ValueError("invalid paired rank identities")))
                with self.assertRaisesRegex(ValueError, "invalid paired rank identities"):
                    run_query_rotating_windows(config, output)
                metrics = json.loads((output / "metrics.json").read_text())
            else:
                metrics = run_query_rotating_windows(config, output)
            report = json.loads((output / "query-rotating-windows.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            selection = json.loads((output / "final-selection.json").read_text())
            coverage = json.loads((output / "rotating-window-coverage.json").read_text())
            window_report = json.loads((output / "window-schedules.json").read_text())
            with gzip.open(output / "rotating-window-descriptors.json.gz", "rt", encoding="utf-8") as stream:
                descriptors = json.load(stream)
            first = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
            self.assertFalse((output / "query-1k-search.json").exists())
        return SimpleNamespace(metrics=metrics, report=report, manifest=manifest, selection=selection,
            coverage=coverage, window_report=window_report, descriptors=descriptors,
            profiles=profiles, built=built, counts=counts, schedules=schedules, batches=batches,
            evaluation_caps=evaluation_caps, historical_first=first)

    def test_shared_engine_changes_endings_matches_labels_negatives_and_never_trains_controls(self):
        result = self.run_suite()
        self.assertTrue(result.metrics["quality_complete"])
        self.assertFalse(result.metrics["hpo_performed"])
        self.assertEqual(result.built, ["query_ssm_m32", "fuxi_linear"])
        self.assertEqual(result.profiles, result.built)
        self.assertEqual((result.metrics["registered_configurations"], result.metrics["proxy_training_models"],
            result.metrics["reused_controls"]), (4, 2, 2))
        self.assertEqual(result.counts, {"query_ssm_m32": 3, "fuxi_linear": 3})
        self.assertEqual(result.schedules["query_ssm_m32", 1], result.historical_first)
        for epoch in (1, 2, 3):
            self.assertEqual(result.schedules["query_ssm_m32", epoch], result.schedules["fuxi_linear", epoch])
            for key in ("item_ids", "targets", "negative_ids", "valid_mask"):
                np.testing.assert_array_equal(result.batches["query_ssm_m32", epoch][key],
                                               result.batches["fuxi_linear", epoch][key])
        latest = {w.user_id: w.end for w in result.schedules["query_ssm_m32", 1]}
        older = {w.user_id: w.end for w in result.schedules["query_ssm_m32", 2]}
        self.assertTrue(all(older[uid] != end for uid, end in latest.items()))
        self.assertTrue(all(cap == 1024 for cap in result.evaluation_caps))
        self.assertEqual(result.report["full_transfer_candidates"], [])
        self.assertEqual(result.metrics["full_history_inference_evaluations"], 0)
        self.assertEqual(result.metrics["test_evaluations"], 0)
        self.assertEqual(len(result.manifest["checkpoints"]), 4)
        self.assertEqual(len(result.manifest["resume_checkpoints"]), 2)

    def test_primary_always_compares_fresh_checkpoints_even_when_historical_wins(self):
        result = self.run_suite(fuxi_later_best=True)
        self.assertTrue(all(row["name"] == "historical_fixed"
                            for row in result.report["selected_trials"].values()))
        self.assertTrue(all(row["name"] == "rotating"
                            for row in result.report["fresh_selected_trials"].values()))
        primary = result.report["primary_fresh_comparison"]
        self.assertTrue(primary["comparison_available"])
        self.assertFalse(primary["historical_family_winners_used"])
        self.assertEqual(primary["reference"], "fuxi_linear-rotating")
        self.assertEqual(primary["candidate"], "query_ssm_m32-rotating")
        self.assertAlmostEqual(primary["cutoffs"]["10"]["delta_ndcg"], -.2)
        self.assertEqual(primary["cutoffs"]["10"]["confidence"], .95)
        self.assertEqual(primary["cutoffs"]["100"]["draws"], 10)
        self.assertEqual(set(primary["secondary_slice_contrasts"]),
                         {"first_anchor", "later_anchors", "day_286", "day_287"})
        self.assertEqual(result.selection["fresh_checkpoints"]["query_ssm_m32"]["path"],
                         "checkpoints/query_ssm_m32-rotating.pt")
        self.assertTrue(result.selection["historical_winners_do_not_replace_fresh_comparison"])
        for name, comparison in result.report["fresh_vs_historical"].items():
            self.assertFalse(comparison["schedule_effect_is_causally_isolated"])
            self.assertEqual(comparison["only_registered_training_change_is_schedule"], name != "fuxi_linear")
            self.assertEqual(comparison["arithmetic_matches_source"], name != "fuxi_linear")
            self.assertIn("stopping/checkpoint selection", comparison["comparison_scope"])
        self.assertEqual(result.report["fresh_cases"]["query_ssm_m32"]["best_epoch"], 1)
        exposure = primary["fresh_training_exposure"]["query_ssm_m32"]
        self.assertEqual((exposure["selected_checkpoint_epoch"], exposure["completed_training_passes"]), (1, 3))
        self.assertEqual(exposure["successful_updates_to_selected_checkpoint"], 1)
        self.assertEqual(exposure["unique_supervised_target_positions"], 384)
        fuxi_exposure = primary["fresh_training_exposure"]["fuxi_linear"]
        self.assertEqual((fuxi_exposure["selected_checkpoint_epoch"],
                          fuxi_exposure["completed_training_passes"]), (2, 4))
        self.assertEqual(fuxi_exposure["successful_updates_to_selected_checkpoint"], 2)
        self.assertEqual(fuxi_exposure["unique_supervised_target_positions"], 768)
        self.assertTrue(primary["selected_checkpoint_target_coverage_can_differ_under_early_stopping"])

    def test_reports_registered_and_actual_nonoverlapping_coverage_and_daily_trajectories(self):
        result = self.run_suite()
        registered = result.coverage["registered_maximum_passes"]
        self.assertEqual((registered["epochs"], registered["unique_target_positions"]), ([1, 2, 3, 4], 1536))
        self.assertEqual(registered["repeated_endpoint_presentations"], 0)
        for name, row in result.coverage["actual_completed_training"].items():
            self.assertEqual(row["passes_completed"], 3)
            self.assertEqual(row["unique_target_positions"], 1152)
            self.assertTrue(row["completed_passes_only"])
            self.assertFalse(row["failed_partial_pass_included"])
            trajectory = result.report["validation_trajectories"][name]
            self.assertEqual([x["epoch"] for x in trajectory], [1, 2, 3])
            self.assertIn("first_anchor_ndcg10", trajectory[0]["validation"])
            selected = result.coverage["actual_selected_checkpoint"][name]
            self.assertEqual(selected["selected_checkpoint_epoch"], 1)
            self.assertEqual(selected["epochs"], [1])
            self.assertEqual(selected["successful_updates_to_selected_checkpoint"], 1)
            self.assertEqual(selected["unique_target_positions"], 384)
            self.assertLess(selected["unique_target_positions"], row["unique_target_positions"])
        self.assertEqual(len(result.descriptors["descriptors"]), 12)
        self.assertFalse(result.descriptors["contains_raw_item_or_timestamp_arrays"])
        self.assertTrue(result.window_report["custom_training_schedule"])
        self.assertNotIn("original_parent_schedules", result.window_report)
        self.assertIn("registered_training_schedules", result.window_report)
        self.assertNotIn("unique_supervised_target_positions_per_model", result.metrics)

    def test_numerical_failure_isolated_and_historical_winner_does_not_mask_missing_fresh_result(self):
        result = self.run_suite(fail_training=True)
        self.assertFalse(result.metrics["quality_complete"])
        self.assertFalse(result.report["complete"])
        self.assertTrue(result.report["runner_completed"])
        self.assertEqual(result.report["fresh_failed_trials"], ["query_ssm_m32-rotating"])
        self.assertEqual(result.counts, {"query_ssm_m32": 2, "fuxi_linear": 3})
        self.assertFalse(result.report["primary_fresh_comparison"]["comparison_available"])
        self.assertNotIn("cutoffs", result.report["primary_fresh_comparison"])
        self.assertEqual(result.coverage["actual_completed_training"]["query_ssm_m32"]["unique_target_positions"], 384)
        self.assertEqual(result.coverage["actual_completed_training"]["query_ssm_m32"]["passes_completed"], 1)
        self.assertTrue(result.report["fresh_vs_historical"]["fuxi_linear"]["comparison_available"])
        self.assertNotIn("query_ssm_m32", result.selection["fresh_checkpoints"])

    def test_preflight_oom_skips_only_failed_fresh_family_and_marks_incomplete(self):
        result = self.run_suite(fail_preflight=True)
        self.assertFalse(result.metrics["quality_complete"])
        self.assertEqual(result.counts, {"fuxi_linear": 3})
        self.assertEqual(result.profiles, ["query_ssm_m32", "fuxi_linear"])
        self.assertEqual(result.metrics["fresh_failed_trials"], 1)
        self.assertEqual(result.coverage["actual_completed_training"]["query_ssm_m32"]["unique_target_positions"], 0)
        self.assertEqual(result.report["fresh_vs_historical"]["query_ssm_m32"]["status"],
                         "preflight_out_of_memory")

    def test_failed_postprocessing_cannot_leave_stale_complete_marker_or_fresh_selection(self):
        result = self.run_suite(fail_statistics=True)
        self.assertFalse(result.metrics["quality_complete"])
        self.assertFalse(result.report["complete"])
        self.assertFalse(result.report["rotating_statistics_complete"])
        self.assertFalse(result.selection["primary_comparison_available"])
        self.assertEqual(result.selection["fresh_checkpoints"], {})


if __name__ == "__main__":
    unittest.main()
