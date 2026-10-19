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
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QuerySSM2TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_ssm2_training.json").read_text())

    def test_registration_matches_previous_rotating_data_schedule_and_optimization(self):
        from rhpr.sasrec.query_ssm2_training_protocol import validate_ssm2_training_config
        config = self.config()
        prior = json.loads((ROOT / "experiments/yambda_query_rotating_windows.json").read_text())
        validate_ssm2_training_config(config)
        for key in ("data", "schedule", "model", "runtime", "benchmark", "parent"):
            self.assertEqual(config[key], prior[key])
        self.assertEqual(config["training"], dict(prior["training"], budget_seconds=7200))
        self.assertEqual(config["kaggle"]["kernel_sources"], [config["parent"]["kernel"]])
        for key, value in (("d_state", 16), ("head_dim", 128), ("chunk_size", 128), ("summary_tokens", 4)):
            changed = copy.deepcopy(config)
            changed["architecture"][key] = value
            with self.assertRaises(ValueError):
                validate_ssm2_training_config(changed)
        changed = copy.deepcopy(config)
        changed["training"]["learning_rate"] = .003
        with self.assertRaises(ValueError):
            validate_ssm2_training_config(changed)

    def test_fresh_model_retains_original_backbone_and_replaces_true_state_dimension(self):
        from rhpr.sasrec.query_history_models import build_history_model
        from rhpr.sasrec.query_ssm2_training_models import build_ssm2_training_model, ssm2_training_optimizer
        from rhpr.sasrec.query_ssm2_training_protocol import VARIANT, TRIAL
        config = self.config()
        original = build_history_model(config["initializer"], 20)
        with patch("torch.load", side_effect=AssertionError("source weights are forbidden")):
            model = build_ssm2_training_model(config, VARIANT, TRIAL, 20)
        for name, value in original.state_dict().items():
            if not name.startswith("compressor."):
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(model.position_embedding.num_embeddings, 32768)
        self.assertEqual(model.compressor.d_state, 128)
        self.assertEqual(model.compressor.head_dim, 64)
        self.assertEqual(model.compressor.scan_heads, 1)
        self.assertEqual(model.compressor.write_projection.out_features, 128)
        self.assertEqual(model.compressor.read_projection.out_features, 128)
        self.assertEqual(model.compressor.delta_projection.out_features, 1)
        self.assertFalse(model.history_source_checkpoint_loaded)
        optimizer = ssm2_training_optimizer(model, config["training"], TRIAL)
        owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(set(owned), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(owned), len(set(owned)))
        self.assertTrue(all(group["lr"] == .001 and group["weight_decay"] == 1e-5 for group in optimizer.param_groups))

    def test_batch_adapter_equals_original_for_two_different_window_endings(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_rotating_protocol import rotating_schedule, rotating_batch_arrays
        from rhpr.sasrec.query_ssm2_training_protocol import VARIANT, ssm2_training_batch_arrays
        config = self.config()
        users = tuple(LongHistoryUser(uid, np.arange(2600 + uid, dtype=np.int32) % 20 + 1,
            np.arange(2600 + uid, dtype=np.uint32)) for uid in range(3))
        cohort = LongHistoryCohort(users, np.arange(20), {})
        endings = []
        for epoch in (1, 2):
            schedule = rotating_schedule(cohort, seed=42, epoch=epoch)
            actual = ssm2_training_batch_arrays(cohort, schedule, config, VARIANT, old_length=1024)
            expected = rotating_batch_arrays(cohort, schedule, config,
                {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}, old_length=1024)
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
            endings.append({w.user_id: w.end for w in schedule})
        self.assertTrue(all(endings[0][uid] != endings[1][uid] for uid in endings[0]))
        with self.assertRaises(ValueError):
            ssm2_training_batch_arrays(cohort, schedule, config, VARIANT, old_length=None)

    def test_budget_prices_only_one_maximum_trial_and_six_exact_catalogue_validations(self):
        from rhpr.sasrec.query_ssm2_training_models import conservative_ssm2_training_budget
        from rhpr.sasrec.query_ssm2_training_protocol import VARIANT
        config = self.config()
        profiles = {VARIANT["name"] + "-fixed_lr": {"status": "success", "training": {"windows_per_second": 100.},
            "proxy_evaluation": {"anchors_per_second": 200.}}}
        row = conservative_ssm2_training_budget(config, profiles, 100., 4427, {"validation_anchors": 26335})
        expected = 60 * 4427 / 100 + 6 * 26335 / 200
        self.assertAlmostEqual(row["conservative_total_seconds"], 100 + 1.25 * (expected + 120))
        self.assertEqual(row["new_training_configurations"], 1)
        self.assertEqual(row["reused_controls"], 0)
        self.assertFalse(row["auto_reduces_trials_or_passes"])
        profiles[next(iter(profiles))]["training"]["windows_per_second"] = float("nan")
        with self.assertRaises(ValueError):
            conservative_ssm2_training_budget(config, profiles, 0, 4427, {"validation_anchors": 26335})

    def test_registered_engine_runs_one_arm_with_ssd_audit_no_controls_and_no_full_transfer(self):
        from rhpr.sasrec.daily_quality import _aggregate
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_rotating_protocol import rotating_schedule
        from rhpr.sasrec.query_ssm2_training import run_query_ssm2_training
        from rhpr.sasrec.query_ssm2_training_protocol import VARIANT, TRIAL
        config = self.config()
        config["training"].update(effective_batch_size=2, cpu_threads=1, amp=False)
        config["hpo"].update(min_passes=2, max_passes=2, validation_interval=1, patience=2)
        users = tuple(LongHistoryUser(uid, np.arange(2600 + uid, dtype=np.int32) % 20 + 1,
            np.arange(2600 + uid, dtype=np.uint32)) for uid in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(20), summary)
        evaluation = SimpleNamespace(summary={key: config["data"][key] for key in
            ("evaluation_content_sha256", "validation_end_timestamp", "test_end_timestamp", "day_seconds")})
        evaluation.summary.update(validation_anchors=3, test_anchors=10 ** 9)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.tensor(1.))
                self.compressor = torch.nn.Linear(1, 1)
        count, seen, built = 0, [], []
        def build(*args):
            built.append(args[1])
            return Tiny()
        def train(model, cohort, windows, *, optimizer, batch_builder, variant, **kwargs):
            nonlocal count
            count += 1
            self.assertEqual(tuple(windows), rotating_schedule(cohort, seed=42, epoch=count))
            seen.append(batch_builder(cohort, windows, config, variant))
            optimizer.zero_grad(set_to_none=True)
            loss = model.marker.square() + model.compressor.weight.square().sum()
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.detach()), "windows": 3, "labels": 384, "successful_updates": 2,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}
        def evaluate(model, examples, **kwargs):
            self.assertEqual(list(examples), [1024])
            self.assertEqual(kwargs["batch_size"], 32)
            anchors = [{"user_id": uid, "day": 286, "anchor_id": f"{uid}:286", "reference_context_hash": "a" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": .1 * count,
                "ndcg100": .2 * count, "hr10": 1., "hr100": 1.} for uid in range(3)]
            metrics, per_user = _aggregate(anchors)
            return {"metrics": metrics, "per_user": per_user, "per_anchor": anchors, "metadata": {}}
        module = "rhpr.sasrec.query_1k_search."
        own = "rhpr.sasrec.query_ssm2_training."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, options in (
                (own + "validate_ssm2_training_config", {}),
                (own + "build_ssm2_training_model", {"side_effect": build}),
                (own + "preflight_ssm2_training", {"return_value": {"status": "success", "training": {"windows_per_second": 10000.}, "proxy_evaluation": {"anchors_per_second": 10000.}}}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "audit_native_query_scan", {"side_effect": AssertionError("old QSSM1 audit must not run")}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "load_proxy_parent_artifacts", {"return_value": (Path(temporary), {"validation": {"rows": []}}, {}, {})}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *args, **kwargs: iter(["validation"])}),
                (module + "audited_hpo_examples", {"side_effect": lambda examples, old, *args: iter([old])}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
            ):
                stack.enter_context(patch(name, **options))
            audit = stack.enter_context(patch(own + "audit_native_query_ssm2", return_value={"status": "passed"}))
            metrics = run_query_ssm2_training(config, output)
            audit.assert_called_once_with(torch.device("cpu"), head_dim=64, chunk_size=64, d_state=128)
            self.assertEqual(len(built), 1)
            self.assertEqual(count, 2)
            self.assertTrue(metrics["quality_complete"])
            self.assertEqual(metrics["d_state"], 128)
            self.assertEqual(metrics["reused_controls"], 0)
            self.assertEqual(metrics["actual_unique_supervised_target_positions"], 768)
            report = json.loads((output / "query-ssm2-training.json").read_text())
            result = report["trials"][VARIANT["name"]][TRIAL["name"]]
            self.assertEqual(result["status"], "success")
            self.assertNotIn("best_full_validation", result)
            self.assertEqual(result["best_epoch"], 2)
            self.assertTrue((output / result["best_checkpoint"]["path"]).is_file())
            self.assertTrue((output / result["resumable_checkpoint"]["path"]).is_file())
            self.assertEqual(report["protocol"]["full_history"].split(":")[0], "not registered")
            with gzip.open(output / "ssm2-window-descriptors.json.gz", "rt") as stream:
                descriptors = json.load(stream)
            self.assertEqual(len(descriptors["descriptors"]), 6)
            self.assertEqual(metrics["selected_checkpoint_unique_supervised_target_positions"], 768)


if __name__ == "__main__":
    unittest.main()
