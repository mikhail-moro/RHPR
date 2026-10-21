from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


if torch is not None:
    class TinyRestartModel(torch.nn.Module):
        """One real Adam parameter, with the serving interface of the HSTU model."""

        def __init__(self, name):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([-999.], dtype=torch.float32))
            self.compressor = torch.nn.Module()
            self.compressor.tail_size = 159
            self.hstu_normalization_length = 163
            self.arm_name = name


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryLrRestartRunnerTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_lr_restart.json").read_text())
        config["training"].update(raw_context_length=167, serving_context_length=136,
            old_prefix_length=8, effective_batch_size=2, physical_batch_sizes=[2],
            num_negatives=7, amp=False, cpu_threads=1, budget_seconds=100000)
        config["benchmark"].update(warmup=0, iterations=1, batch_sizes=[1, 2],
            evaluation_batch_size=2, item_chunk_size=17)
        return config

    def cohort(self, config):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser

        users = tuple(LongHistoryUser(i, (np.arange(185 + i) % 41 + 1).astype(np.int32),
            np.arange(185 + i, dtype=np.uint32)) for i in range(3))
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {
            key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
                "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256")})

    def evaluation(self, config):
        return SimpleNamespace(summary={"reference_context_length": 167,
            "validation_users": 3, "test_users": 3, "validation_anchors": 3, "test_anchors": 3,
            **{key: config["data"][key] for key in ("evaluation_content_sha256",
                "validation_end_timestamp", "test_end_timestamp", "day_seconds")}})

    def examples(self, cohort, evaluation, *, split_name, max_seq_len):
        day = 286 if split_name == "validation" else 293
        for user in cohort.users:
            yield SimpleNamespace(user_id=user.user_id, day=day,
                anchor_id=f"{user.user_id}:{day}", context=user.train[-max_seq_len:],
                timestamps=user.timestamps[-max_seq_len:], target=2,
                target_timestamp=day * 86400, reference_context_hash="c" * 64)

    def quality(self, examples, score):
        from rhpr.sasrec.daily_quality import _aggregate

        rows = [{"user_id": example.user_id, "day": example.day,
            "anchor_id": example.anchor_id, "reference_context_hash": example.reference_context_hash,
            "target_known": True, "rank_cutoff": 100, "hr10": 1., "hr100": 1.,
            "ndcg10": score, "ndcg100": score} for example in examples]
        metrics, users = _aggregate(rows)
        metrics.update(num_items=41, evaluation_wall_seconds=.001)
        return {"metrics": metrics, "per_user": users, "per_anchor": rows, "metadata": {}}

    def run_fixture(self, output, *, improving=False, baseline_mismatch=False, budget_seconds=None):
        """Exercise the runner orchestration, keeping real optimizer and checkpoints."""
        from rhpr.sasrec.query_lr_restart import run_query_lr_restart
        from rhpr.sasrec.query_restart_state import restart_arms

        config = self.config()
        if budget_seconds is not None:
            config["training"]["budget_seconds"] = budget_seconds
        cohort, evaluation = self.cohort(config), self.evaluation(config)
        arms = restart_arms(config)
        by_name = {arm["name"]: arm for arm in arms}
        training = defaultdict(list)
        evaluations = defaultdict(list)
        training_rng = defaultdict(list)
        initial = {}
        test_states = {}
        events = []
        optimizers = []
        parent_values = {"recent128": 11., "query_cross_attention_m4": 22.}

        def build(config, variant, num_items):
            return TinyRestartModel(variant["name"])

        def load(config, parent_name, *, num_items, model=None, **kwargs):
            value = parent_values[parent_name]
            state = {"marker": torch.tensor([value], dtype=torch.float32)}
            if model is not None:
                model.load_state_dict(state)
            return {"state_dict": state, "selection_score": .7 if baseline_mismatch else .6,
                "best_epoch": 48, "format_version": 1, "num_items": num_items,
                "variant": {"name": parent_name, "kind": "recent" if parent_name == "recent128"
                    else "query_cross_attention", "summary_tokens": 0 if parent_name == "recent128" else 4},
                "source_code_sha": config["restart"]["source_code_sha"],
                "model_config": config["model"], "experiment_config": config}

        def train(model, cohort, windows, *, config, variant, optimizer, scaler, **kwargs):
            name = variant["name"]
            model.arm_name = name
            epoch = windows[0].epoch
            self.assertTrue(all(window.epoch == epoch for window in windows))
            draw = (random.random(), float(np.random.random()), float(torch.rand(())))
            training_rng[name].append(draw)
            if not training[name]:
                self.assertEqual(len(optimizer.state), 0, "each LR arm needs a fresh Adam")
                self.assertFalse(scaler.is_enabled())
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                self.assertEqual(optimizer.param_groups[0]["lr"], by_name[name]["learning_rate"])
                initial[name] = {"marker": float(model.marker.detach()[0]),
                    "rng": draw}
                optimizers.append(optimizer)
            training[name].append([(window.epoch, window.user_id, window.end, window.negative_seed)
                                   for window in windows])
            events.append(("train", name, epoch))
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"loss": 1., "windows": len(windows), "labels": len(windows) * 32,
                "successful_updates": 1, "amp_skipped_update_attempts": 0,
                "seconds": .001, "loss_scale": 1.}

        def evaluate(model, examples, **kwargs):
            examples = list(examples)
            split = "validation" if examples[0].day == 286 else "test"
            name = model.arm_name
            # A family-named build can occur when the runner reconstructs a
            # selected model; recover the arm from its saved metadata if needed.
            self.assertIn(name, by_name)
            # Validation deliberately touches all RNGs. Its implementation
            # must preserve the training stream even at intermediate checks.
            random.random()
            np.random.random()
            torch.rand(())
            additional = len(training[name])
            evaluations[name].append((split, additional))
            events.append((split, name, additional))
            self.assertEqual(model.compressor.tail_size, 128)
            expected_length = 128 if by_name[name]["kind"] == "recent" else 136
            self.assertTrue(all(len(example.context) == expected_length for example in examples))
            if split == "test":
                self.assertTrue(all(training[arm["name"]] for arm in arms),
                    "finish validation-based LR selection before any test evaluation")
                test_states[name] = float(model.marker.detach()[0])
            score = .6
            if split == "validation" and improving:
                score += additional * by_name[name]["learning_rate"]
            return self.quality(examples, score)

        def profile(config, arm, cohort, evaluation, schedule, device):
            self.assertTrue(all(window.epoch == 49 for window in schedule))
            return {**arm, "training_batch_size": 2, "evaluation_batch_size": 2,
                "estimated_train_pass_seconds": .001, "estimated_validation_seconds": .001,
                "estimated_test_seconds": .001,
                "evaluation": [{"status": "ok", "batch_size": 2}]}

        context_manifest = {split: {"anchors": 3,
            "ordered_serving_context_sha256": ("a" if split == "validation" else "b") * 64,
            "rows": []} for split in ("validation", "test")}
        module = "rhpr.sasrec.query_lr_restart."
        with ExitStack() as stack:
            for name, kwargs in (
                (module + "validate_restart_config", {}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "iter_daily_evaluation_examples", {"side_effect": self.examples}),
                (module + "_context_manifest", {"return_value": context_manifest}),
                (module + "_load_parent_metadata", {"return_value": {}}),
                (module + "build_query_model", {"side_effect": build}),
                (module + "load_parent_checkpoint", {"side_effect": load}),
                (module + "train_quality_pass", {"side_effect": train}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "_profile_restart_arm", {"side_effect": profile}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "_cleanup", {}),
                ("torch.cuda.get_device_name", {"return_value": "CPU orchestration fixture"}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            metrics = run_query_lr_restart(config, output)
        return {"metrics": metrics, "config": config, "arms": arms, "training": training,
            "evaluations": evaluations, "initial": initial, "test_states": test_states,
            "events": events, "optimizers": optimizers, "training_rng": training_rng}

    def test_four_fresh_arms_share_parent_weights_rng_and_windows_and_keep_baseline_on_ties(self):
        from rhpr.metrics import validate_metrics

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_fixture(output)
            report = json.loads((output / "query-lr-restart-quality.json").read_text())
            self.assertEqual(validate_metrics(result["metrics"]), result["metrics"])
            self.assertTrue(result["metrics"]["optimizer_restart"])
            self.assertFalse(result["metrics"]["exact_parent_optimizer_resume"])
            self.assertEqual(result["metrics"]["test_evaluations"], 2)
            self.assertTrue(report["complete"])
            self.assertEqual(len(result["optimizers"]), 4)
            self.assertEqual(len({id(optimizer) for optimizer in result["optimizers"]}), 4)
            self.assertEqual(len(set(value["rng"] for value in result["initial"].values())), 1)
            python_rng = random.Random(result["config"]["seed"])
            numpy_rng = np.random.RandomState(result["config"]["seed"])
            torch_rng = torch.Generator().manual_seed(result["config"]["seed"])
            expected_draws = [(python_rng.random(), float(numpy_rng.random()),
                float(torch.rand((), generator=torch_rng))) for _ in range(24)]
            expected_test_names = {arm["name"] for arm in result["arms"] if arm["learning_rate"] == .001}
            self.assertEqual(set(result["test_states"]), expected_test_names)
            self.assertEqual(set(report["selections"].values()), expected_test_names)
            self.assertEqual(len(report["cases"]), 2)
            training_order = next(iter(result["training"].values()))
            self.assertEqual([windows[0][0] for windows in training_order], list(range(49, 73)))
            for arm in result["arms"]:
                name = arm["name"]
                with self.subTest(arm=name):
                    self.assertEqual(result["training"][name], training_order)
                    self.assertEqual(result["training_rng"][name], expected_draws)
                    self.assertEqual(result["initial"][name]["marker"],
                        11. if arm["parent_name"] == "recent128" else 22.)
                    expected_checks = [0, 8, 16, 24] if arm["learning_rate"] == .001 else [8, 16, 24]
                    self.assertEqual([additional for split, additional in result["evaluations"][name]
                        if split == "validation"], expected_checks)
                    model = report["models"][name]
                    self.assertEqual(model["best_additional_passes"], 0)
                    self.assertEqual(model["best_epoch"], 48)
                    self.assertEqual(model["additional_passes_completed"], 24)
                    self.assertTrue(model["stopped_early"])
            self.assertEqual(sorted(result["test_states"].values()), [11., 22.])
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertEqual(len(manifest["checkpoints"]), 4)
            self.assertEqual(len(manifest["resume_checkpoints"]), 4)
            self.assertTrue(manifest["optimizer_state_saved"])
            for record in manifest["resume_checkpoints"]:
                saved = torch.load(output / record["path"], weights_only=True)
                self.assertEqual(saved["global_epoch"], 72)
                self.assertEqual(saved["next_global_epoch"], 73)
                self.assertEqual(saved["additional_passes"], 24)
                self.assertTrue(saved["optimizer_restart"])
                self.assertEqual(saved["early_stopping"]["best_additional_passes"], 0)
                self.assertEqual(saved["early_stopping"]["bad_checks"], 3)
                state = saved["optimizer_state_dict"]["state"]
                self.assertEqual(len(state), 1)
                self.assertEqual(float(next(iter(state.values()))["step"]), 24.)
                self.assertIn("exp_avg", next(iter(state.values())))
                self.assertTrue(torch.equal(saved["torch_rng_state"], torch_rng.get_state()))
                self.assertEqual(saved["numpy_rng_state"]["algorithm"], "MT19937")
                self.assertEqual(len(saved["numpy_rng_state"]["keys"]), 624)
                expected_parent = 11. if record["training_model"].startswith("recent128") else 22.
                self.assertLess(float(saved["state_dict"]["marker"][0]), expected_parent)
            events = result["events"]
            first_test = next(index for index, event in enumerate(events) if event[0] == "test")
            self.assertFalse(any(event[0] in {"train", "validation"} for event in events[first_test:]))

    def test_improving_validation_runs_to_96_and_selects_lr_before_two_test_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_fixture(output, improving=True)
            report = json.loads((output / "query-lr-restart-quality.json").read_text())
            selected = {arm["name"] for arm in result["arms"] if arm["learning_rate"] == .003}
            self.assertEqual(set(report["selections"].values()), selected)
            self.assertEqual(set(result["test_states"]), selected)
            self.assertEqual(sum(split == "test" for values in result["evaluations"].values()
                for split, _ in values), 2)
            schedules = list(result["training"].values())
            for arm in result["arms"]:
                name = arm["name"]
                with self.subTest(arm=name):
                    self.assertEqual(result["training"][name], schedules[0])
                    self.assertEqual([windows[0][0] for windows in result["training"][name]], list(range(49, 97)))
                    expected_checks = [8, 16, 24, 32, 40, 48]
                    if arm["learning_rate"] == .001:
                        expected_checks.insert(0, 0)
                    self.assertEqual([additional for split, additional in result["evaluations"][name]
                        if split == "validation"], expected_checks)
                    self.assertEqual(report["models"][name]["best_additional_passes"], 48)
                    self.assertEqual(report["models"][name]["best_epoch"], 96)
                    self.assertFalse(report["models"][name]["stopped_early"])

    def test_parent_validation_mismatch_fails_before_any_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(ValueError, "baseline|validation|parent"):
                self.run_fixture(output, baseline_mismatch=True)
            report = json.loads((output / "query-lr-restart-quality.json").read_text())
            self.assertFalse(report["complete"])
            self.assertTrue(all(not model["passes"] for model in report["models"].values()))

    def test_parent_report_requires_matching_actual_validation_and_test_contexts(self):
        from rhpr.sasrec.query_lr_restart import _load_parent_metadata

        config = self.config()
        contexts = {split: {"anchors": 3,
            "ordered_serving_context_sha256": ("a" if split == "validation" else "b") * 64}
            for split in ("validation", "test")}
        report = {"complete": True, "code_sha": config["restart"]["source_code_sha"],
            "input_contexts": contexts}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "query-pilot-quality.json"
            with patch("rhpr.sasrec.query_lr_restart._parent_output_directory", return_value=root):
                path.write_text(json.dumps(report))
                result = _load_parent_metadata(config, contexts)
                self.assertEqual(set(result), {"recent128", "query_cross_attention_m4"})
                self.assertEqual(result["recent128"]["report_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest())
                for split, field, value in (
                    ("validation", "anchors", 2),
                    ("validation", "ordered_serving_context_sha256", "d" * 64),
                    ("test", "anchors", 2),
                    ("test", "ordered_serving_context_sha256", "d" * 64),
                ):
                    bad = copy.deepcopy(report)
                    bad["input_contexts"][split][field] = value
                    path.write_text(json.dumps(bad))
                    with self.subTest(split=split, field=field), self.assertRaisesRegex(ValueError, "context mismatch"):
                        _load_parent_metadata(config, contexts)
                for field, value in (("complete", False), ("code_sha", "0" * 40)):
                    bad = copy.deepcopy(report)
                    bad[field] = value
                    path.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "parent report"):
                        _load_parent_metadata(config, contexts)

    def test_preflight_requires_full_48_pass_cap_instead_of_shortening_training(self):
        from rhpr.sasrec.query_lr_restart import _restart_budget
        from rhpr.sasrec.query_restart_state import restart_arms

        config = self.config()
        profiles = [{**arm, "estimated_train_pass_seconds": 10.,
            "estimated_validation_seconds": 20., "estimated_test_seconds": 30.}
            for arm in restart_arms(config)]
        budget = _restart_budget(config, profiles, elapsed=15.)
        self.assertEqual(budget["models"], 4)
        self.assertEqual(budget["max_additional_passes"], 48)
        self.assertFalse(budget["cap_reduced_by_preflight"])
        self.assertFalse(budget["selection_uses_quality_metrics"])
        config["training"]["budget_seconds"] = 2000.
        with self.assertRaisesRegex(ValueError, "four.*48-pass caps"):
            _restart_budget(config, profiles, elapsed=15.)


if __name__ == "__main__":
    unittest.main()
