from __future__ import annotations

from contextlib import ExitStack
import copy
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
    class TinyFeatureCompressor(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)

        def feature_compute_flops(self, old_length, batch_size=1):
            return {"unhooked_feature_flops": 123, "feature_conv_flops": 123,
                    "hooked_feature_projection_flops": 999, "feature_silu_evaluations": 0,
                    "old_length": old_length, "batch_size": batch_size}


    class TinyMambaModel(torch.nn.Module):
        def __init__(self, identity=0):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = TinyFeatureCompressor()
            self.register_buffer("origin", torch.tensor(identity))


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryMambaAblationTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_mamba_ablation.json").read_text())

    def test_budget_prices_three_fresh_shapes_without_control_or_full_history_work(self):
        from rhpr.sasrec.query_mamba_ablation import conservative_mamba_budget
        profiles, expected = {}, 0.
        for i, name in enumerate(("conv4", "output_gate", "skip_d"), 1):
            rate = 100. * i
            profiles["query_ssm_m32-" + name] = {"training": {"windows_per_second": rate},
                "proxy_evaluation": {"anchors_per_second": rate * 2}}
            expected += 60 * 1000 / rate + 6 * 100 / (rate * 2)
        result = conservative_mamba_budget(self.config(), profiles, 100., 1000,
            {"validation_anchors": 100, "test_anchors": 10 ** 20})
        self.assertAlmostEqual(result["estimated_remaining_work_seconds"], expected)
        self.assertAlmostEqual(result["conservative_total_seconds"], 100. + 1.25 * (expected + 360.))
        self.assertEqual((result["registered_configurations"], result["new_training_configurations"],
                          result["reused_controls"], result["priced_fresh_shapes"]), (4, 3, 1, 3))
        self.assertEqual(result["full_history_inference_evaluations"], 0)
        self.assertEqual(result["test_evaluations"], 0)
        self.assertNotIn("optional_full_transfer_reserve_seconds", result)
        profiles["query_ssm_m32-conv4"] = {"status": "preflight_out_of_memory"}
        result = conservative_mamba_budget(self.config(), profiles, 100., 1000, {"validation_anchors": 100})
        self.assertEqual((result["priced_fresh_shapes"], result["preflight_failed_shapes"]), (2, 1))

    def test_budget_rejects_invalid_rates_and_rejects_over_budget_without_cutting_trials(self):
        from rhpr.sasrec.query_mamba_ablation import conservative_mamba_budget
        profiles = {"query_ssm_m32-" + name: {"training": {"windows_per_second": .01},
            "proxy_evaluation": {"anchors_per_second": .01}}
            for name in ("conv4", "output_gate", "skip_d")}
        row = conservative_mamba_budget(self.config(), profiles, 0., 1000, {"validation_anchors": 100})
        self.assertFalse(row["fits"])
        self.assertEqual(row["priced_fresh_shapes"], 3)
        self.assertFalse(row["auto_reduces_trials_or_passes"])
        for rate in (0., -1., float("inf"), float("nan"), True):
            with self.subTest(rate=rate):
                bad = copy.deepcopy(profiles)
                bad["query_ssm_m32-conv4"]["training"]["windows_per_second"] = rate
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    conservative_mamba_budget(self.config(), bad, 0., 1000, {"validation_anchors": 100})

    def test_preflight_two_real_adam_updates_only_short_catalogue_and_adds_unhooked_cost(self):
        from rhpr.sasrec.query_mamba_ablation import preflight_mamba
        from rhpr.sasrec.query_mamba_protocol import mamba_trials, mamba_variants
        config = self.config()
        variant = mamba_variants(config)[0]
        trial = mamba_trials(config, variant)[1]
        model = TinyMambaModel()
        calls, caps = [], []
        cohort = SimpleNamespace(num_items=10,
            users=[SimpleNamespace(user_id=0, train=np.arange(2000))])

        def train(current, cohort, windows, *, optimizer, batch_builder, **kwargs):
            calls.append(batch_builder.keywords["old_length"])
            optimizer.zero_grad(set_to_none=True)
            current.marker.grad = torch.ones_like(current.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0,
                    "windows_per_second": 128., "seconds": 1.}

        def profile(current, *args, context_length, **kwargs):
            caps.append(context_length)
            self.assertIsNotNone(current.marker.grad)
            return {"anchors_per_second": 100.}

        def batch(contexts, *args):
            return torch.tensor(np.array(contexts)), torch.ones((1, len(contexts[0])), dtype=torch.bool)

        cost = {name: 1000 for name in ("compressor_flops", "history_flops", "full_flops",
            "compressor_arithmetic_including_scan_flops", "history_arithmetic_including_scan_flops",
            "full_arithmetic_including_scan_flops")}
        module = "rhpr.sasrec.query_mamba_ablation."
        with ExitStack() as stack:
            for name, kwargs in (
                ("rhpr.sasrec.query_mamba_models.audit_mamba_feature", {"return_value": {"passed": True}, "create": True}),
                ("rhpr.sasrec.query_mamba_models.build_mamba_model", {"return_value": model}),
                (module + "fixed_prefix_schedule", {"return_value": (1, 2)}),
                (module + "_representative_windows", {"return_value": (1, 2)}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "_profile_daily_rate", {"side_effect": profile}),
                (module + "_batch", {"side_effect": batch}),
                (module + "_observe_prefix_flops", {"return_value": (torch.ones((1, 1)), cost)}),
            ):
                stack.enter_context(patch(name, **kwargs))
            result = preflight_mamba(config, cohort, None, variant, trial, torch.device("cpu"))
        self.assertEqual(calls, [1024, 1024])
        self.assertEqual(caps, [1152])
        self.assertEqual(result["full_evaluation"], {"status": "not_registered", "profiled": False})
        arithmetic = result["representative_b1_arithmetic"]["proxy"]
        self.assertEqual(arithmetic["input_tokens"], 1152)
        self.assertEqual(arithmetic["feature_increment"]["old_length"], 1024)
        self.assertEqual(arithmetic["compressor_flops"], 1123)
        self.assertEqual(arithmetic["full_arithmetic_including_scan_flops"], 1123)
        self.assertTrue(result["optimizer_state_allocated_during_all_measurements"])
        self.assertTrue(result["native_feature_audit"]["passed"])
        self.assertFalse(result["profile_weights_reused"])
        self.assertFalse(result["automatic_precision_or_batch_fallback"])

    def test_preflight_reclassifies_known_nonfinite_catalogue_errors_only(self):
        from rhpr.sasrec.query_mamba_ablation import preflight_mamba
        from rhpr.sasrec.query_mamba_protocol import mamba_trials, mamba_variants
        config = self.config()
        variant = mamba_variants(config)[0]
        trial = mamba_trials(config, variant)[1]

        def train(model, *args, optimizer, **kwargs):
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"successful_updates": 1, "amp_skipped_update_attempts": 0}

        module = "rhpr.sasrec.query_mamba_ablation."
        for message, expected in (("evaluation produced nonfinite catalogue scores", FloatingPointError),
                                  ("programming error", RuntimeError)):
            with self.subTest(message=message), ExitStack() as stack:
                for name, kwargs in (
                    ("rhpr.sasrec.query_mamba_models.audit_mamba_feature", {"return_value": {"passed": True}, "create": True}),
                    ("rhpr.sasrec.query_mamba_models.build_mamba_model", {"return_value": TinyMambaModel()}),
                    (module + "fixed_prefix_schedule", {"return_value": ()}),
                    (module + "_representative_windows", {"return_value": ()}),
                    (module + "train_prefix_pass", {"side_effect": train}),
                    (module + "_profile_daily_rate", {"side_effect": RuntimeError(message)}),
                ):
                    stack.enter_context(patch(name, **kwargs))
                with self.assertRaisesRegex(expected, message):
                    preflight_mamba(config, SimpleNamespace(num_items=10), None, variant, trial,
                                    torch.device("cpu"))

    @staticmethod
    def rows(score):
        return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                 "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                 "hr10": 1., "hr100": 1.} for i in range(3)]

    def run_suite(self, *, fail_training=False, fail_preflight=False):
        from rhpr.sasrec.daily_quality import _aggregate
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_mamba_ablation import run_query_mamba_ablation
        from rhpr.sasrec.query_mamba_protocol import mamba_trials, mamba_variants
        config = self.config()
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        config["hpo"].update(min_passes=2, max_passes=4, validation_interval=1, patience=2)
        config["hpo"]["bootstrap_draws"] = 10
        users = tuple(LongHistoryUser(i, np.arange(1400 + i, dtype=np.int32) % 41 + 1,
            np.arange(1400 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=10 ** 10)
        evaluation = SimpleNamespace(summary=summary)
        models, profiles, counts, schedules, contrasts = [], [], {}, [], []

        def build(current, variant, trial, num_items):
            self.assertNotEqual(trial["name"], "base")
            model = TinyMambaModel(len(models))
            models.append(model)
            return model

        def controls(current, evaluation, output):
            trial = mamba_trials(current, mamba_variants(current)[0])[0]
            filename = "verified-base-short-anchors.json.gz"
            with gzip.open(output / filename, "wt", encoding="utf-8") as stream:
                json.dump(self.rows(.5), stream)
            return {"query_ssm_m32": {"status": "success", "complete": True, "trial_name": "base",
                "trial": trial, "passes": [], "passes_completed": 40, "best_epoch": 20,
                "best_validation": {"ndcg10": .5, "ndcg100": .5},
                "best_validation_rank_file": filename,
                "best_checkpoint": {"path": "/verified/base.pt", "reused": True}, "reused": True}}

        def preflight(current, cohort, evaluation, variant, trial, device):
            profiles.append(trial["name"])
            if fail_preflight and trial["name"] == "conv4":
                raise torch.OutOfMemoryError("isolated capacity failure")
            return {"training": {"windows_per_second": 10000.},
                    "proxy_evaluation": {"anchors_per_second": 10000.}}

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            identity = int(model.origin)
            counts[identity] = counts.get(identity, 0) + 1
            batch_builder(cohort, windows, config, mamba_variants(config)[0])
            if fail_training and identity == 1 and counts[identity] == 2:
                raise FloatingPointError("isolated numerical failure")
            schedules.append((identity, counts[identity],
                [(w.epoch, w.user_id, w.end, w.negative_seed) for w in windows]))
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"loss": 1., "windows": 3, "labels": 384, "successful_updates": 1,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def audited(examples, old, *args):
            self.assertEqual(list(examples), ["validation"])
            self.assertEqual(old, 1024)
            return iter([old])

        def evaluate(model, examples, **kwargs):
            self.assertEqual(list(examples), [1024])
            score = (.6, .4, .55)[int(model.origin)]
            anchors = self.rows(score)
            metrics, per_user = _aggregate(anchors)
            return {"metrics": metrics, "per_user": per_user, "per_anchor": anchors, "metadata": {}}

        def bootstrap(reference, candidate, **kwargs):
            contrasts.append(kwargs)
            return {"delta_ndcg": candidate[0]["ndcg10"] - reference[0]["ndcg10"],
                    "confidence": kwargs.get("confidence", .95)}

        module = "rhpr.sasrec.query_1k_search."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                ("rhpr.sasrec.query_mamba_protocol.validate_mamba_config", {}),
                ("rhpr.sasrec.query_mamba_protocol.load_mamba_controls", {"side_effect": controls}),
                ("rhpr.sasrec.query_mamba_models.build_mamba_model", {"side_effect": build}),
                ("rhpr.sasrec.query_mamba_ablation.preflight_mamba", {"side_effect": preflight}),
                ("rhpr.sasrec.query_mamba_ablation.paired_daily_ndcg_bootstrap", {"side_effect": bootstrap}),
                ("rhpr.sasrec.query_mamba_protocol.mamba_batch_arrays", {"return_value": {"item_ids": np.array([[1, 2]])}}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (output,
                    {"validation": {"rows": []}}, {"recent128": self.rows(.4)}, {})}),
                (module + "capture_training_rng", {"return_value": {}}),
                (module + "diagnose_nonfinite_batch", {"return_value": {"complete": True}}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "audited_hpo_examples", {"side_effect": audited}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "paired_daily_ndcg_bootstrap", {"return_value": {"delta_ndcg": 0.}}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            metrics = run_query_mamba_ablation(config, output)
            report = json.loads((output / "query-mamba-ablation.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            selection = json.loads((output / "final-selection.json").read_text())
            self.assertFalse((output / "query-1k-search.json").exists())
        return metrics, report, manifest, selection, profiles, counts, schedules, contrasts

    def test_actual_shared_engine_runs_three_fresh_shapes_and_every_feature_has_paired_short_comparison(self):
        metrics, report, manifest, selection, profiles, counts, schedules, contrasts = self.run_suite()
        self.assertTrue(metrics["quality_complete"])
        self.assertFalse(metrics["hpo_performed"])
        self.assertEqual((metrics["registered_configurations"], metrics["proxy_training_models"],
                          metrics["reused_controls"], metrics["fresh_successful_trials"]), (4, 3, 1, 3))
        self.assertEqual(profiles, ["conv4", "output_gate", "skip_d"])
        self.assertEqual(counts, {0: 3, 1: 3, 2: 3})
        self.assertEqual(len(manifest["checkpoints"]), 4)
        self.assertEqual(len(manifest["resume_checkpoints"]), 2)
        self.assertEqual(len(report["feature_vs_base"]), 3)
        self.assertEqual(len(contrasts), 12)
        self.assertEqual(len([row for row in contrasts if "confidence" in row]), 6)
        self.assertTrue(all(row["comparison_available"] for row in report["feature_vs_base"].values()))
        self.assertAlmostEqual(metrics["conv4_minus_base_ndcg10"], .1)
        self.assertAlmostEqual(metrics["output_gate_minus_base_ndcg10"], -.1)
        self.assertEqual(report["full_transfer_candidates"], [])
        self.assertEqual(metrics["full_history_inference_evaluations"], 0)
        self.assertEqual(metrics["test_evaluations"], 0)
        self.assertEqual(selection["selected_trials"]["query_ssm_m32"]["name"], "conv4")
        self.assertNotIn("paired_parent_full_validation_contrasts", report["cases"]["query_ssm_m32"])
        first = [row for _, epoch, row in schedules if epoch == 1]
        self.assertTrue(all(row == first[0] for row in first))

    def test_one_numerical_failure_keeps_other_results_but_does_not_mark_ablation_complete(self):
        metrics, report, manifest, _, profiles, counts, _, _ = self.run_suite(fail_training=True)
        self.assertFalse(metrics["quality_complete"])
        self.assertFalse(report["complete"])
        self.assertTrue(report["runner_completed"])
        self.assertEqual((metrics["fresh_successful_trials"], metrics["fresh_failed_trials"]), (2, 1))
        self.assertEqual(counts, {0: 3, 1: 2, 2: 3})
        self.assertEqual(report["fresh_failed_trials"], ["query_ssm_m32-output_gate"])
        self.assertFalse(report["feature_vs_base"]["query_ssm_m32-output_gate"]["comparison_available"])
        self.assertTrue(report["feature_vs_base"]["query_ssm_m32-skip_d"]["comparison_available"])
        self.assertEqual(profiles, ["conv4", "output_gate", "skip_d"])

    def test_one_preflight_oom_skips_only_that_shape_and_marks_suite_incomplete(self):
        metrics, report, _, _, profiles, counts, _, _ = self.run_suite(fail_preflight=True)
        self.assertFalse(metrics["quality_complete"])
        self.assertEqual((metrics["fresh_successful_trials"], metrics["fresh_failed_trials"]), (2, 1))
        self.assertEqual(len(counts), 2)
        self.assertEqual(profiles, ["conv4", "output_gate", "skip_d"])
        self.assertEqual(report["feature_vs_base"]["query_ssm_m32-conv4"]["status"], "preflight_out_of_memory")
        self.assertFalse(report["budget"]["auto_reduces_trials_or_passes"])


if __name__ == "__main__":
    unittest.main()
