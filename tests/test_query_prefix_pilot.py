from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import random
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
        def __init__(self, name):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = torch.nn.Module()
            self.compressor.tail_size = 128
            self.arm_name = name


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryPrefixPilotTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        config = json.loads((ROOT / "experiments/yambda_sasrec_prefix_query_pilot.json").read_text())
        config["training"].update(effective_batch_size=2, physical_batch_sizes=[1, 2],
            num_negatives=7, amp=False, cpu_threads=1, min_passes=4, max_passes=10,
            validation_interval=2, early_stopping_min_passes=4, early_stopping_patience=2,
            budget_seconds=100000)
        config["benchmark"].update(warmup=0, iterations=1, batch_sizes=[1, 2],
            evaluation_batch_size=2, item_chunk_size=17)
        return config

    def cohort(self, config):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(i, (np.arange(260 + i) % 41 + 1).astype(np.int32),
            np.arange(260 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256")}
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)

    def evaluation(self, config):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationAnchor, DailyEvaluationData, DailyEvaluationUser
        low, high = config["data"]["train_end_timestamp"], config["data"]["validation_end_timestamp"]
        users = tuple(DailyEvaluationUser(i, np.asarray([2, 3], dtype=np.int32),
            np.asarray([low + 1, high + 1], dtype=np.uint32)) for i in range(3))
        anchors = tuple(DailyEvaluationAnchor(i, f"{i}:{day}", day, index, 2, timestamp, "c" * 64)
            for index, (day, timestamp) in enumerate(((286, low + 1), (293, high + 1))) for i in range(3))
        summary = {"reference_context_length": 256, "validation_anchors": 3, "test_anchors": 3,
            **{key: config["data"][key] for key in ("train_end_timestamp", "evaluation_content_sha256",
                "validation_end_timestamp", "test_end_timestamp", "day_seconds")}}
        return DailyEvaluationData(users, anchors, summary)

    def examples(self, cohort, evaluation, *, split_name, max_seq_len):
        day = 286 if split_name == "validation" else 293
        for user in cohort.users:
            yield SimpleNamespace(user_id=user.user_id, day=day, anchor_id=f"{user.user_id}:{day}",
                context=user.train[-max_seq_len:], timestamps=user.timestamps[-max_seq_len:],
                target=2, target_timestamp=day * 86400, reference_context_hash="c" * 64)

    def quality(self, examples, score):
        from rhpr.sasrec.daily_quality import _aggregate
        rows = [{"user_id": example.user_id, "day": example.day, "anchor_id": example.anchor_id,
            "reference_context_hash": example.reference_context_hash, "target_known": True,
            "rank_cutoff": 100, "hr10": 1., "hr100": 1., "ndcg10": score, "ndcg100": score}
            for example in examples]
        metrics, users = _aggregate(rows)
        metrics.update(num_items=41, evaluation_wall_seconds=.001)
        return {"metrics": metrics, "per_user": users, "per_anchor": rows, "metadata": {}}

    def run_fixture(self, output, *, improving=False, budget_seconds=None):
        from rhpr.sasrec.query_prefix_pilot import run_query_prefix_pilot
        from rhpr.sasrec.query_prefix_protocol import prefix_variants
        config = self.config()
        if budget_seconds is not None:
            config["training"]["budget_seconds"] = budget_seconds
        cohort, evaluation = self.cohort(config), self.evaluation(config)
        variants = prefix_variants(config)
        training, training_rng, evaluations = defaultdict(list), defaultdict(list), defaultdict(list)
        events, initial, tested = [], {}, {}
        optimizers = []

        def build(config, variant, num_items):
            return TinyModel(variant["name"])

        def train(model, cohort, windows, *, variant, optimizer, **kwargs):
            name = variant["name"]
            training_rng[name].append((random.random(), float(np.random.random()), float(torch.rand(()))))
            if not training[name]:
                initial[name] = float(model.marker.detach()[0])
                self.assertEqual(len(optimizer.state), 0)
                optimizers.append(optimizer)
            training[name].append([(window.epoch, window.user_id, window.end, window.negative_seed) for window in windows])
            events.append(("train", name))
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            return {"loss": 1., "windows": len(windows), "labels": len(windows) * 128,
                "successful_updates": 1, "amp_skipped_update_attempts": 0, "seconds": .001,
                "windows_per_second": 3000., "labels_per_second": 384000., "loss_scale": 1.}

        def evaluate(model, examples, **kwargs):
            rows = list(examples)
            name = model.arm_name
            split = "validation" if rows[0].day == 286 else "test"
            events.append((split, name))
            random.random(), np.random.random(), torch.rand(())
            epoch = len(training[name])
            evaluations[name].append((split, epoch))
            self.assertEqual(model.compressor.tail_size, 128)
            expected_length = 128 if name == "recent128" else len(cohort.users[0].train)
            self.assertEqual(len(rows[0].context), expected_length)
            if split == "test":
                self.assertTrue(all(training[variant["name"]] for variant in variants))
                self.assertTrue(all((output / f"checkpoints/{variant['name']}.pt").is_file() for variant in variants))
                tested[name] = float(model.marker.detach()[0])
            return self.quality(rows, .5 + epoch * .01 if improving else .5)

        def profile(config, variant, cohort, evaluation, schedule, device):
            random.random(), np.random.random(), torch.rand(())
            self.assertTrue(all(window.epoch == 1 for window in schedule))
            return {**variant, "training_batch_size": 2, "evaluation_batch_size": 2,
                "estimated_train_pass_seconds": .001, "estimated_validation_seconds": .001,
                "estimated_test_seconds": .001}

        contexts = {split: {"anchors": 3, "ordered_serving_context_sha256": "a" * 64, "rows": []}
                    for split in ("validation", "test")}
        module = "rhpr.sasrec.query_prefix_pilot."
        with ExitStack() as stack:
            for name, kwargs in (
                (module + "validate_prefix_config", {}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "iter_daily_evaluation_examples", {"side_effect": self.examples}),
                (module + "_context_manifest", {"return_value": contexts}),
                (module + "build_prefix_model", {"side_effect": build}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "profile_prefix_variant", {"side_effect": profile}),
                (module + "audit_native_query_scan", {"return_value": {"passed": True}}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "_cleanup", {}),
                ("rhpr.sasrec.query_prefix_benchmark.benchmark_prefix_quality", {
                    "return_value": {"metrics": {"full_b1_wall_p50_ms": .1}, "cases": {}, "metadata": {}}}),
                ("torch.cuda.get_device_name", {"return_value": "CPU orchestration fixture"}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            metrics = run_query_prefix_pilot(config, output)
        return {"metrics": metrics, "config": config, "training": training, "training_rng": training_rng,
            "evaluations": evaluations, "initial": initial, "tested": tested,
            "events": events, "optimizers": optimizers}

    def test_matched_fixed_targets_early_stopping_rng_and_exact_resume_are_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_fixture(output)
            report = json.loads((output / "query-prefix-quality.json").read_text())
            self.assertTrue(report["complete"])
            self.assertTrue(result["metrics"]["quality_complete"])
            self.assertEqual(result["metrics"]["test_evaluations"], 3)
            self.assertFalse(result["metrics"]["train_is_exhaustive_event_epoch"])
            self.assertEqual(result["metrics"]["common_max_training_passes"], 10)
            schedules = list(result["training"].values())
            self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
            self.assertTrue(all(len(schedule) == 6 for schedule in schedules))
            self.assertEqual(set(result["initial"].values()), {1.})
            rngs = list(result["training_rng"].values())
            self.assertTrue(all(draws == rngs[0] for draws in rngs))
            expected_rng = random.Random(result["config"]["seed"])
            self.assertEqual([draw[0] for draw in rngs[0]], [expected_rng.random() for _ in range(6)])
            self.assertEqual(len({id(optimizer) for optimizer in result["optimizers"]}), 3)
            first_test = next(index for index, event in enumerate(result["events"]) if event[0] == "test")
            self.assertFalse(any(kind in ("train", "validation") for kind, _ in result["events"][first_test:]))
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertEqual(len(manifest["checkpoints"]), 3)
            self.assertEqual(len(manifest["resume_checkpoints"]), 3)
            for name, model in report["models"].items():
                self.assertEqual(model["best_epoch"], 2)
                self.assertEqual(model["passes_completed"], 6)
                self.assertTrue(model["stopped_early"])
                best = torch.load(output / model["checkpoint"]["path"], weights_only=True)
                resume = torch.load(output / model["resume_checkpoint"]["path"], weights_only=True)
                self.assertEqual(resume["epoch"], 6)
                self.assertEqual(resume["next_epoch"], 7)
                self.assertEqual(resume["early_stopping"]["best_epoch"], 2)
                self.assertEqual(resume["early_stopping"]["bad_checks"], 2)
                state = next(iter(resume["optimizer_state_dict"]["state"].values()))
                self.assertEqual(float(state["step"]), 6.)
                self.assertIn("exp_avg", state)
                self.assertEqual(float(best["state_dict"]["marker"][0]), result["tested"][name])
                self.assertLess(float(resume["state_dict"]["marker"][0]), result["tested"][name])
                self.assertEqual(model["total_labels"], 6 * 3 * 128)

    def test_improving_validation_runs_every_model_to_common_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_fixture(output, improving=True)
            report = json.loads((output / "query-prefix-quality.json").read_text())
            for model in report["models"].values():
                self.assertEqual(model["best_epoch"], 10)
                self.assertEqual(model["passes_completed"], 10)
                self.assertFalse(model["stopped_early"])
            self.assertEqual(sum(split == "test" for rows in result["evaluations"].values() for split, _ in rows), 3)

    def test_insufficient_common_budget_prevents_any_quality_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaisesRegex(ValueError, "budget cannot support"):
                self.run_fixture(output, budget_seconds=1.)
            report = json.loads((output / "query-prefix-quality.json").read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["models"], {})

    def test_cost_cap_rounding_counts_every_validation_and_final_test(self):
        from rhpr.sasrec.query_prefix_pilot import prefix_pass_budget
        config = self.config()
        config["training"].update(min_passes=30, max_passes=100, validation_interval=10, budget_seconds=2700.)
        profiles = [{"estimated_train_pass_seconds": 10., "estimated_validation_seconds": 20.,
            "estimated_test_seconds": 30.} for _ in range(3)]
        budget = prefix_pass_budget(config, profiles, elapsed=10.)
        self.assertEqual(budget["passes"] % 10, 0)
        self.assertEqual(budget["validation_checks_per_model"], budget["passes"] // 10)
        self.assertEqual(budget["model_fixed_evaluation_seconds"], 90.)
        self.assertFalse(budget["selection_uses_quality_metrics"])
        self.assertLessEqual(budget["conservative_suite_seconds"], config["training"]["budget_seconds"])

    def test_complete_history_guard_accepts_reference_shorter_than_forward_and_rejects_cropping(self):
        from rhpr.sasrec.query_prefix_pilot import _validate_complete_histories
        config = self.config()
        cohort, evaluation = self.cohort(config), self.evaluation(config)
        self.assertLess(evaluation.summary["reference_context_length"], 32768)
        _validate_complete_histories(cohort, evaluation, 32768, 32768)
        with self.assertRaisesRegex(ValueError, "training.*truncation"):
            _validate_complete_histories(cohort, evaluation, 259, 32768)
        with self.assertRaisesRegex(ValueError, "serving.*truncation"):
            _validate_complete_histories(cohort, evaluation, 32768, 261)

    def test_real_training_preserves_raw128_and_accumulates_partial_effective_group(self):
        from rhpr.sasrec.query_prefix_pilot import train_prefix_pass
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model, fixed_prefix_schedule, prefix_variants
        from rhpr.sasrec.long_quality import _optimizer, _scaler
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=2, ffn_multiplier=2, dropout=0.)
        config["compressor"].update(d_inner=8, d_state=2, gate_rank=4)
        cohort = self.cohort(config)
        schedule = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
        for variant in prefix_variants(config)[:2]:
            model = build_prefix_model(config, variant, cohort.num_items)
            optimizer, scaler = _optimizer(model, config["training"]), _scaler(torch.device("cpu"), config["training"])
            before = model.item_embedding.weight.detach().clone()
            row = train_prefix_pass(model, cohort, schedule, config=config, variant=variant,
                physical_batch=1, device=torch.device("cpu"), optimizer=optimizer, scaler=scaler)
            self.assertEqual(model.compressor.tail_size, 128)
            self.assertEqual(row["successful_updates"], 2)
            self.assertEqual(row["windows"], 3)
            self.assertEqual(row["labels"], 384)
            self.assertEqual(row["amp_skipped_update_attempts"], 0)
            self.assertTrue(np.isfinite(row["loss"]))
            self.assertFalse(torch.equal(before, model.item_embedding.weight))

    def test_profile_length_selection_preserves_actual_mixed_batch_order(self):
        from rhpr.sasrec.query_prefix_pilot import _representative_windows
        windows = tuple(SimpleNamespace(user_id=index, end=length)
                        for index, length in enumerate((30, 10, 50, 20, 40)))
        selected = _representative_windows(windows, 3)
        self.assertEqual([window.user_id for window in selected], [0, 1, 2])
        self.assertEqual(min(window.end for window in selected), 10)
        self.assertEqual(max(window.end for window in selected), 50)

    def test_training_rejects_nonfinite_loss_without_optimizer_update(self):
        from rhpr.sasrec.query_prefix_pilot import train_prefix_pass
        from rhpr.sasrec.query_prefix_protocol import build_prefix_model, fixed_prefix_schedule, prefix_variants
        from rhpr.sasrec.long_quality import _optimizer, _scaler
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=2, ffn_multiplier=2, dropout=0.)
        cohort = self.cohort(config)
        variant = prefix_variants(config)[0]
        model = build_prefix_model(config, variant, cohort.num_items)
        optimizer, scaler = _optimizer(model, config["training"]), _scaler(torch.device("cpu"), config["training"])
        schedule = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
        with patch("rhpr.sasrec.query_prefix_pilot._loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite.*loss"):
                train_prefix_pass(model, cohort, schedule, config=config, variant=variant,
                    physical_batch=1, device=torch.device("cpu"), optimizer=optimizer, scaler=scaler)
        self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
