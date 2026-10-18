from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
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
    class HistoryTinyModel(torch.nn.Module):
        """Real trainable state; FuXi deliberately has no compressor member."""

        def __init__(self, family):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([-999.], dtype=torch.float32))
            if family == "qssm":
                self.compressor = torch.nn.Linear(1, 1)
            self.family = family
            self.architecture = "sasrec" if family == "qssm" else "fuxi_linear"
            self.history_source_checkpoint_loaded = False


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryHistoryContinuationTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self, family):
        config = json.loads((ROOT / f"experiments/yambda_history_continuation_{family}.json").read_text())
        config["training"].update(effective_batch_size=2, amp=False, cpu_threads=1)
        return config

    def test_qssm_paired_real_adam_restart_fixed_endpoint_and_portable_checkpoints(self):
        self.exercise_suite("qssm")

    def test_fuxi_without_compressor_obeys_the_same_history_continuation_contract(self):
        self.exercise_suite("fuxi")

    def test_all_loaded_preflights_precede_budget_gate_and_failed_gate_never_trains(self):
        self.exercise_suite("qssm", budget_fits=False)

    def test_failed_full_history_retains_successful_short_endpoint_but_cannot_complete_pair(self):
        self.exercise_suite("qssm", fail_full=True)

    def test_required_full_preflight_failure_prohibits_all_quality_training(self):
        self.exercise_suite("fuxi", fail_preflight=True)

    def test_nonfinite_full_validation_retains_finished_short_without_faking_full_quality(self):
        self.exercise_suite("fuxi", fail_full_validation=True)

    def exercise_suite(self, family, *, budget_fits=True, fail_full=False,
                       fail_preflight=False, fail_full_validation=False):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_history_continuation import run_history_continuation
        from rhpr.sasrec.query_history_protocol import history_arms, history_variant
        from rhpr.sasrec.query_prefix_pilot import _checkpoint_record

        config = self.config(family)
        arms = history_arms(config)
        by_name = {arm["name"]: arm for arm in arms}
        users = tuple(LongHistoryUser(i, (np.arange(length) % 41 + 1).astype(np.int32),
            np.arange(length, dtype=np.uint32)) for i, length in enumerate((1400, 1450, 1500)))
        cohort = LongHistoryCohort(users, np.arange(41, dtype=np.int64), {
            key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
                "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256",
                "source_revision")})
        evaluation_summary = {key: config["data"][key] for key in ("evaluation_content_sha256",
            "train_end_timestamp", "validation_end_timestamp", "test_end_timestamp", "day_seconds")}
        evaluation_summary.update(validation_anchors=len(users), validation_users=len(users),
            test_anchors=10 ** 12)
        from types import SimpleNamespace
        evaluation = SimpleNamespace(summary=evaluation_summary)
        events, profiles, optimizers, builds = [], [], [], []
        trained, evaluated, training_rng, arrays_by_arm = (defaultdict(list) for _ in range(4))
        model_arms = {}
        source_value = 7.25

        def build(_config, num_items):
            self.assertEqual(num_items, cohort.num_items)
            model = HistoryTinyModel(family)
            builds.append(model)
            return model

        def prepare(_config, _evaluation, output):
            """Only source discovery is mocked; strict tensor loading stays real."""
            self.assertIs(_evaluation, evaluation)
            output = Path(output)
            variant = history_variant(config)
            model = HistoryTinyModel(family)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(source_value)
            checkpoint_config = copy.deepcopy(config)
            best_epoch = config["selected_source"]["checkpoint"]["best_epoch"]
            source_config = copy.deepcopy(config)
            source_sha = config["selected_source"]["source_code_sha"]
            destination = output / "checkpoints" / "source-tiny.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"format_version": 3, "source_code_sha": source_sha,
                "experiment_config": checkpoint_config, "model_config": config["model"],
                "variant": variant, "num_items": cohort.num_items, "best_epoch": best_epoch,
                "selection_score": .60, "state_dict": model.state_dict()}, destination)
            provenance = {"source_code_sha": source_sha, "weights_copied": True,
                "weight_bytes_verified": True, "test_ranks_read": False,
                "optimizer_state_inherited": False, "scaler_state_inherited": False,
                "rng_state_inherited": False}
            return {"checkpoint_path": str(destination),
                "checkpoint_record": _checkpoint_record(destination, output, copied_reference=True,
                    training_model=variant["name"], best_epoch=best_epoch),
                "source_config": source_config, "source_checkpoint_config": checkpoint_config,
                "source_variant": variant, "source_trial": copy.deepcopy(config["selected_source"]["trial"]),
                "source_best_validation": {"ndcg10": .60}, "source_best_epoch": best_epoch,
                "provenance": provenance, "validation_artifacts": {}, "stopping_policy_replay": {}}

        def profile(_config, _cohort, _evaluation, arm, device, *, load_model):
            self.assertIs(_cohort, cohort)
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(trained, {})
            model = load_model()
            self.assertTrue(model.history_source_checkpoint_loaded)
            for parameter in model.parameters():
                self.assertTrue(torch.equal(parameter.detach(), torch.full_like(parameter, source_value)))
            # Disposable preflight updates must not leak into either arm.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(-123.)
            random.random()
            np.random.random()
            torch.rand(())
            profiles.append(arm["name"])
            events.append(("profile", arm["name"]))
            if fail_preflight and arm["name"] == "full":
                raise FloatingPointError("nonfinite loaded full-history state")
            return {"status": "success", "training_batch_size": arm["physical_batch_size"],
                "evaluation_batch_size": arm["evaluation_batch_size"],
                "estimated_train_pass_seconds": .001, "estimated_validation_seconds": .001}

        def budget(_config, results, *args):
            self.assertEqual(profiles, ["short", "full"])
            self.assertEqual(set(results), {"short", "full"})
            self.assertEqual(trained, {})
            events.append(("budget",))
            return {"fits": budget_fits, "estimated_total_seconds": .1}

        def examples(_cohort, _evaluation, *, split_name, max_seq_len):
            self.assertEqual(split_name, "validation", "the continuation must never access test examples")
            for user in users:
                yield DailyEvaluationExample(user_id=user.user_id, day=286,
                    anchor_id=f"{user.user_id}:286", context=user.train[-max_seq_len:].copy(),
                    timestamps=user.timestamps[-max_seq_len:].copy(), target=2,
                    target_timestamp=286 * 86400, reference_context_hash="c" * 64)

        def train(model, _cohort, windows, *, config, variant, optimizer, scaler,
                  physical_batch, batch_builder, **kwargs):
            self.assertIs(_cohort, cohort)
            arrays = batch_builder(cohort, windows, config, variant)
            name = "short" if arrays["item_ids"].shape[1] == 1152 else "full"
            model_arms[id(model)] = name
            self.assertEqual(physical_batch, by_name[name]["physical_batch_size"])
            epoch = windows[0].epoch
            self.assertTrue(all(window.epoch == epoch for window in windows))
            expected_epoch = 61 + len(trained[name])
            self.assertEqual(epoch, expected_epoch)
            training_rng[name].append((random.random(), float(np.random.random()), float(torch.rand(()))))
            if not trained[name]:
                self.assertEqual(len(optimizer.state), 0, "both arms must restart Adam")
                self.assertFalse(scaler.is_enabled())
                self.assertTrue(model.history_source_checkpoint_loaded)
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                for parameter in model.parameters():
                    self.assertTrue(torch.equal(parameter.detach(), torch.full_like(parameter, source_value)))
                optimizers.append(optimizer)
                arrays_by_arm[name] = {key: value.copy() for key, value in arrays.items()}
            if name == "full" and fail_full and epoch == 62:
                raise FloatingPointError("nonfinite prefix training loss")
            trained[name].append([(window.epoch, window.user_id, window.end, window.negative_seed)
                for window in windows])
            events.append(("train", name, epoch))
            optimizer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            return {"loss": 1., "seconds": .001, "labels": len(users) * 128,
                "successful_updates": 1, "amp_skipped_update_attempts": 0, "loss_scale": 1.}

        def evaluate(model, source, *, batch_size, **kwargs):
            examples_list = list(source)
            self.assertEqual(len(examples_list), len(users))
            lengths = [len(example.context) for example in examples_list]
            name = "short" if max(lengths) == 1152 else "full"
            model_arms[id(model)] = name
            self.assertEqual(batch_size, by_name[name]["evaluation_batch_size"])
            if name == "short":
                self.assertEqual(lengths, [1152] * len(users))
            else:
                self.assertEqual(lengths, [len(user.train) for user in users])
            additional = len(trained[name])
            evaluated[name].append(additional)
            events.append(("validation", name, additional))
            self.assertIn(additional, (0, 10, 20))
            if fail_full_validation and name == "full":
                raise RuntimeError("evaluation produced nonfinite final states")
            # All evaluation RNG must be isolated from the next training pass.
            random.random()
            np.random.random()
            torch.rand(())
            score = {0: .60, 10: .80, 20: .65}[additional] + (.01 if name == "full" else 0.)
            rows = [{"user_id": example.user_id, "day": example.day,
                "anchor_id": example.anchor_id, "reference_context_hash": example.reference_context_hash,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score + .01,
                "hr10": 1., "hr100": 1.} for example in examples_list]
            return {"metrics": {"ndcg10": score, "ndcg100": score + .01},
                "per_anchor": rows, "per_user": []}

        from rhpr.sasrec.daily_evaluation_data import daily_context_sha256
        references = [{**{key: getattr(example, key) for key in ("user_id", "anchor_id", "day",
            "target", "target_timestamp", "reference_context_hash")},
            "serving_context_length": len(example.context),
            "serving_context_sha256": daily_context_sha256(example.user_id, example.context, example.timestamps)}
            for example in examples(cohort, evaluation, split_name="validation", max_seq_len=32768)]
        module = "rhpr.sasrec.query_history_continuation."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            output = Path(temporary)
            for name, kwargs in (
                ("validate_history_config", {"return_value": None}),
                ("_registered_device", {"return_value": torch.device("cpu")}),
                ("audit_native_query_scan", {"return_value": {"passed": True}}),
                ("load_prepared_cohort", {"return_value": cohort}),
                ("load_prepared_daily_evaluation", {"return_value": evaluation}),
                ("resolve_dataset_directory", {"side_effect": lambda value: value}),
                ("_validate_complete_histories", {"return_value": None}),
                ("load_proxy_parent_artifacts", {"return_value": (None,
                    {"validation": {"rows": references}}, {}, None)}),
                ("prepare_history_source", {"side_effect": prepare}),
                ("build_history_model", {"side_effect": build}),
                ("preflight_history", {"side_effect": profile}),
                ("conservative_history_budget", {"side_effect": budget}),
                ("iter_daily_evaluation_examples", {"side_effect": examples}),
                ("train_prefix_pass", {"side_effect": train}),
                ("evaluate_daily_quality", {"side_effect": evaluate}),
                ("paired_daily_ndcg_bootstrap", {"return_value": {"delta_ndcg": .01}}),
                ("diagnose_nonfinite_batch", {"return_value": {"captured": True}}),
            ):
                stack.enter_context(patch(module + name, **kwargs))
            with redirect_stdout(io.StringIO()):
                if not budget_fits or fail_full or fail_preflight or fail_full_validation:
                    with self.assertRaises(RuntimeError):
                        run_history_continuation(config, output)
                    metrics = json.loads((output / "metrics.json").read_text())
                else:
                    metrics = run_history_continuation(config, output)
            report = json.loads((output / "history-continuation.json").read_text())
            self.assertEqual(metrics["test_evaluations"], 0)
            self.assertEqual(profiles, ["short", "full"])
            if fail_preflight:
                self.assertEqual(events, [("profile", "short"), ("profile", "full")])
                self.assertEqual(trained, {})
                self.assertEqual(evaluated, {})
                self.assertFalse(metrics["quality_complete"])
                self.assertEqual(report["preflight"]["full"]["status"], "numerical_failed")
                return
            self.assertEqual(events[:3], [("profile", "short"), ("profile", "full"), ("budget",)])
            if not budget_fits:
                self.assertEqual(trained, {})
                self.assertEqual(evaluated, {})
                self.assertFalse(metrics["quality_complete"])
                return
            self.assertEqual(evaluated["short"], [0, 10, 20])
            self.assertEqual(len(trained["short"]), 20)
            self.assertEqual([windows[0][0] for windows in trained["short"]], list(range(61, 81)))
            short_result = report["arms"]["short"]
            self.assertEqual(short_result["evaluations"]["20"]["metrics"]["ndcg10"], .65)
            self.assertEqual(short_result["evaluations"]["10"]["metrics"]["ndcg10"], .80)
            self.assertTrue((output / short_result["endpoint_checkpoint"]["path"]).is_file())
            self.assertEqual(metrics["quality_complete"], not (fail_full or fail_full_validation))
            if fail_full_validation:
                self.assertEqual(len(optimizers), 1)
                self.assertEqual(trained["full"], [])
                self.assertEqual(report["arms"]["full"]["failure"]["stage"], "initial_validation")
                self.assertFalse(report["complete"])
                return
            self.assertEqual(len(optimizers), 2)
            self.assertIsNot(optimizers[0], optimizers[1])
            self.assertEqual(training_rng["short"][:len(training_rng["full"])], training_rng["full"])
            short_arrays, full_arrays = arrays_by_arm["short"], arrays_by_arm["full"]
            np.testing.assert_array_equal(short_arrays["negative_ids"], full_arrays["negative_ids"])
            for row in range(len(users)):
                short_length = int(short_arrays["valid_mask"][row].sum())
                full_length = int(full_arrays["valid_mask"][row].sum())
                self.assertEqual(short_length, 1152)
                self.assertGreater(full_length, short_length)
                for key in ("item_ids", "targets"):
                    np.testing.assert_array_equal(short_arrays[key][row, short_length - 128:short_length],
                        full_arrays[key][row, full_length - 128:full_length])
            if fail_full:
                self.assertEqual(len(trained["full"]), 1)
                self.assertFalse(report["complete"])
                return
            self.assertEqual(evaluated["full"], [0, 10, 20])
            self.assertEqual(trained["short"], trained["full"])
            self.assertTrue(report["complete"])
            self.verify_checkpoints(output, report, config)

    def verify_checkpoints(self, output, report, config):
        manifest = json.loads((output / "checkpoint-manifest.json").read_text())
        resumes = manifest["resume_checkpoints"]
        self.assertEqual(len(resumes), 2)
        for record in resumes:
            path = Path(record["path"])
            self.assertFalse(path.is_absolute(), "all output checkpoints must be portable")
            payload = torch.load(output / path, map_location="cpu", weights_only=True)
            self.assertTrue(payload["optimizer_state_saved"])
            self.assertEqual(payload["epoch"], 20)
            self.assertEqual(payload["next_epoch"], 21)
            self.assertEqual(record["absolute_schedule_epoch"], 80)
            self.assertEqual(payload["optimizer_class"], "torch.optim.AdamW")
            state = payload["optimizer_state_dict"]["state"]
            self.assertTrue(state)
            self.assertTrue(all(float(row["step"]) == 20. for row in state.values()))
            self.assertNotEqual(float(payload["state_dict"]["marker"]), 7.25)
            self.assertEqual(payload["experiment_config"]["continuation"], config["continuation"])
            active = payload["experiment_config"]
            self.assertEqual(active["active_stage"], "paired_history_continuation")
            self.assertEqual(active["full_length_training_performed"], active["active_arm"]["name"] == "full")
            self.assertIn("torch_rng_state", payload)
            self.assertIn("python_rng_state", payload)
            self.assertIn("numpy_rng_state", payload)
        # The primary result is always the fixed20 endpoint, even when p10 wins
        # the separately reported best-within-budget selection.
        for name, expected in (("short", .65), ("full", .66)):
            result = report["arms"][name]
            self.assertEqual(result["endpoint_validation"]["ndcg10"], expected)
            self.assertEqual(result["best_epoch"], 10)
            endpoint = result["endpoint_checkpoint"]
            best = result["best_checkpoint"]
            endpoint_payload = torch.load(output / endpoint["path"], map_location="cpu", weights_only=True)
            best_payload = torch.load(output / best["path"], map_location="cpu", weights_only=True)
            self.assertNotEqual(float(endpoint_payload["state_dict"]["marker"]),
                float(best_payload["state_dict"]["marker"]))


if __name__ == "__main__":
    unittest.main()
