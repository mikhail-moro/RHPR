from __future__ import annotations

import copy
import gzip
import hashlib
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


@unittest.skipIf(torch is None, "Torch and numpy required")
class LongMixedQualityTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_hstu_mixed_budget_daily.json").read_text())

    def small_config(self):
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=1, dropout=0.)
        config["training"].update(raw_context_length=64, raw_tail_size=8, supervised_targets=4,
                                  num_negatives=7, effective_batch_size=4, amp=False,
                                  min_passes=2, max_passes=2, validation_interval=1)
        config["benchmark"].update(warmup=0, iterations=1, batch_sizes=[1, 2],
                                   evaluation_batch_size=2, item_chunk_size=5)
        return config

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser

        users = tuple(LongHistoryUser(i, (np.arange(90 + i) % 11 + 1).astype(np.int32),
                                      np.arange(90 + i, dtype=np.uint32)) for i in range(5))
        return LongHistoryCohort(users, np.arange(11, dtype=np.int64), {})

    def mixed_variant(self, config):
        from rhpr.sasrec.long_mixed_quality import daily_training_variants

        return daily_training_variants(config)[-1]

    def test_registered_daily_configuration_rejects_changed_protocol_and_requires_output(self):
        from rhpr.experiment import run_experiment
        from rhpr.sasrec.long_mixed_quality import (
            daily_training_variants, evaluation_cases, validate_mixed_daily_config,
        )

        config = self.config()
        validate_mixed_daily_config(config)
        variants = daily_training_variants(config)
        self.assertEqual(len(variants), 18)
        self.assertEqual(sum(len(evaluation_cases(v)) for v in variants), 23)
        self.assertEqual([k for _, k in evaluation_cases(variants[-1])], [1, 2, 4, 8, 16, 32])
        self.assertEqual(sum(v["summary_tokens"] == 32 for v in variants[:-1]), 4)
        for path, value in [(("summary_budgets",), [16, 8, 4]),
                            (("mixed_budgets",), [1, 2, 4, 8, 16, 16]),
                            (("mixed_budgets",), [True, 2, 4, 8, 16, 32]),
                            (("training", "min_passes"), 19),
                            (("training", "max_passes"), 21),
                            (("benchmark", "quality_cutoffs"), [10]),
                            (("data", "evaluation_protocol"), "first_period_anchor"),
                            (("data", "evaluation_content_sha256"), "not-pinned"),
                            (("data", "cohort_train_timestamps_sha256"), "F" * 64),
                            (("kaggle", "dataset_sources"), [
                                "carasik/rhpr-yambda500m-train16k-profile",
                                "carasik/rhpr-yambda500m-long-evaluation",
                            ])]:
            bad = copy.deepcopy(config)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                validate_mixed_daily_config(bad)
        for section, field in (("training", "min_passes"),
                               ("data", "evaluation_content_sha256"),
                               ("data", "cohort_train_timestamps_sha256"),
                               ("data", "catalog_sha256")):
            bad = copy.deepcopy(config)
            del bad[section][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_mixed_daily_config(bad)
        with self.assertRaisesRegex(ValueError, "output_dir"):
            run_experiment(config)

    def test_effective_budget_and_partial_update_are_invariant_to_physical_batch(self):
        from rhpr.sasrec.long_mixed_quality import configure_mixed_model
        from rhpr.sasrec.long_quality import build_quality_model, train_quality_pass, _optimizer, _scaler
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule

        config, cohort = self.small_config(), self.cohort()
        variant = self.mixed_variant(config)
        windows = quality_window_schedule(cohort, seed=42, epoch=2, raw_context_length=64)
        device = torch.device("cpu")
        models = []
        for physical in (2, 4):
            model = build_quality_model(config, variant, 11)
            configure_mixed_model(model, variant)
            budgets_seen = []
            hook = model.compressor.register_forward_pre_hook(
                lambda compressor, args: budgets_seen.append((compressor.summary_tokens, len(args[0])))
            )
            result = train_quality_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=physical, device=device, optimizer=_optimizer(model, config["training"]),
                scaler=_scaler(device, config["training"]), budget_sequence=(1, 32))
            hook.remove()
            self.assertEqual(result["windows"], 5)
            self.assertEqual(result["labels"], 20)
            self.assertEqual(result["successful_updates"], 2)
            self.assertEqual(result["amp_skipped_update_attempts"], 0)
            self.assertEqual(result["budget_exposures"], {
                "1": {"updates": 1, "windows": 4, "labels": 16},
                "32": {"updates": 1, "windows": 1, "labels": 4},
            })
            self.assertEqual(budgets_seen, [(1, 2), (1, 2), (32, 1)] if physical == 2
                             else [(1, 4), (32, 1)])
            self.assertTrue(np.isfinite(result["loss"]))
            models.append(model)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], atol=2e-6, rtol=2e-5)

    def test_amp_retry_reuses_effective_budget_without_counting_failed_exposure(self):
        from rhpr.sasrec.long_mixed_quality import configure_mixed_model
        from rhpr.sasrec.long_quality import build_quality_model, train_quality_pass, _optimizer
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule

        config, cohort = self.small_config(), self.cohort()
        variant = self.mixed_variant(config)
        windows = quality_window_schedule(cohort, seed=42, epoch=3, raw_context_length=64)
        model = build_quality_model(config, variant, 11)
        configure_mixed_model(model, variant)
        optimizer = _optimizer(model, config["training"])
        budgets_seen = []
        hook = model.compressor.register_forward_pre_hook(
            lambda compressor, args: budgets_seen.append((compressor.summary_tokens, len(args[0])))
        )
        original_clip = torch.nn.utils.clip_grad_norm_
        clips = 0

        class RecoveryScaler:
            def __init__(self):
                self.steps, self.value, self.reductions = 0, 1024., 0

            def scale(self, value): return value
            def unscale_(self, optimizer): pass
            def is_enabled(self): return True
            def get_scale(self): return self.value

            def update(self, new_scale=None):
                if new_scale is not None:
                    self.value = new_scale
                    self.reductions += 1

            def step(self, optimizer):
                if clips == 1:
                    raise AssertionError("optimizer mutated after nonfinite norm")
                self.steps += 1
                optimizer.step()

        def injected_norm(parameters, max_norm):
            nonlocal clips
            clips += 1
            return torch.tensor(float("inf")) if clips == 1 else original_clip(parameters, max_norm)

        scaler = RecoveryScaler()
        with patch("torch.nn.utils.clip_grad_norm_", side_effect=injected_norm):
            result = train_quality_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=2, device=torch.device("cpu"), optimizer=optimizer, scaler=scaler,
                budget_sequence=(2, 16))
        hook.remove()
        self.assertEqual(budgets_seen, [(2, 2), (2, 2), (2, 2), (2, 2), (16, 1)])
        self.assertEqual(result["successful_updates"], 2)
        self.assertEqual(result["amp_skipped_update_attempts"], 1)
        self.assertEqual((scaler.steps, scaler.reductions), (2, 1))
        self.assertEqual(result["budget_exposures"], {
            "2": {"updates": 1, "windows": 4, "labels": 16},
            "16": {"updates": 1, "windows": 1, "labels": 4},
        })

    def test_actual_training_daily_catalogue_evaluation_and_reloadable_checkpoints(self):
        from rhpr.sasrec.daily_evaluation_data import daily_context_sha256
        from rhpr.sasrec.long_mixed_quality import run_mixed_daily_quality
        from rhpr.sasrec.long_quality import build_quality_model
        from rhpr.sasrec.mixed_budget import select_summary_budget
        from rhpr.sasrec.daily_quality import evaluate_daily_quality

        config, cohort = self.small_config(), self.cohort()
        cohort.summary.update({key: config["data"][key] for key in (
            "train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
            "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256")})
        evaluation = SimpleNamespace(summary={
            "evaluation_content_sha256": config["data"]["evaluation_content_sha256"],
            "validation_users": 5, "test_users": 5, "validation_anchors": 10, "test_anchors": 10,
            "reference_context_length": 64,
            **{key: config["data"][key] for key in (
                "validation_end_timestamp", "test_end_timestamp", "day_seconds")},
        })

        def examples(cohort, evaluation, *, split_name, max_seq_len):
            for user in cohort.users:
                items, timestamps = user.train.copy(), user.timestamps.copy()
                for day in (286, 287, 293, 294):
                    target = 0 if user.user_id == 0 else (user.user_id + day) % 11 + 1
                    if (split_name == "validation") == (day < 293):
                        full_items, full_times = items[-64:], timestamps[-64:]
                        yield SimpleNamespace(user_id=user.user_id, day=day,
                            anchor_id=f"{user.user_id}:{day}", context=items[-max_seq_len:],
                            timestamps=timestamps[-max_seq_len:], target=target,
                            target_timestamp=day * 86400,
                            reference_context_hash=daily_context_sha256(user.user_id, full_items, full_times))
                    items = np.append(items, np.int32(target))
                    timestamps = np.append(timestamps, np.uint32(day * 86400))

        def preflight(config, variant, cohort, evaluation, schedule, device):
            return {**variant, "training_batch_size": 2, "evaluation_batch_size": 2,
                    "estimated_train_pass_seconds": .1,
                    "estimated_validation_seconds": .1, "estimated_test_seconds": .1}

        module = "rhpr.sasrec.long_mixed_quality"
        with tempfile.TemporaryDirectory() as temporary, \
             patch(module + ".validate_mixed_daily_config"), \
             patch(module + ".resolve_dataset_directory", side_effect=Path), \
             patch(module + ".load_prepared_cohort", return_value=cohort), \
             patch(module + ".load_prepared_daily_evaluation", return_value=evaluation), \
             patch(module + ".iter_daily_evaluation_examples", side_effect=examples), \
             patch("rhpr.sasrec.daily_training_profile.profile_daily_variant", side_effect=preflight), \
             patch(module + "._cleanup"), \
             patch(module + "._registered_device", return_value=torch.device("cpu")), \
             patch("builtins.print"):
            output = Path(temporary)
            metrics = run_mixed_daily_quality(config, output)
            report = json.loads((output / "mixed-daily-quality.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertTrue(metrics["quality_complete"])
            self.assertTrue(report["complete"])
            self.assertEqual(metrics["common_training_passes"], 2)
            self.assertEqual(metrics["labels_per_training_model"], 40)
            self.assertEqual((metrics["training_models"], metrics["evaluation_cases"]), (18, 23))
            self.assertEqual((len(report["models"]), len(report["cases"])), (18, 23))
            self.assertEqual(len(manifest["checkpoints"]), 18)
            self.assertEqual(len(list((output / "checkpoints").glob("*.pt"))), 18)
            for split in ("validation", "test"):
                self.assertEqual(len(list(output.glob(f"*-{split}-users.json"))), 23)
                self.assertEqual(len(list(output.glob(f"*-{split}-anchors.json.gz"))), 23)

            checkpoint_records = {record["training_model"]: record for record in manifest["checkpoints"]}
            for name, record in checkpoint_records.items():
                checkpoint = output / record["path"]
                self.assertEqual(record["size_bytes"], checkpoint.stat().st_size)
                self.assertEqual(record["sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(saved["num_items"], 11)
                self.assertEqual(saved["source_code_sha"], manifest["source_code_sha"])
                self.assertEqual(saved["experiment_config"], config)
                self.assertEqual(saved["variant"]["name"], name)
                self.assertEqual(saved["best_epoch"], record["best_epoch"])
                self.assertEqual(saved["summary_budgets"], record["summary_budgets"])
                self.assertEqual(record["precision"], "float32")
                self.assertEqual(saved["hstu_normalization_length"], 43)
                self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all()
                                    for value in saved["state_dict"].values() if value.is_floating_point()))
                torch.testing.assert_close(saved["state_dict"]["item_embedding.weight"][0],
                                           torch.zeros(8), rtol=0, atol=0)
                passes = report["models"][name]["passes"]
                self.assertEqual(len(passes), 2)
                scores = [row["selection_score"] for row in passes]
                self.assertEqual(record["best_epoch"], scores.index(max(scores)) + 1)

            hashes_by_case = {}
            for name, case in report["cases"].items():
                self.assertEqual(case["test"]["user_count"], 5)
                self.assertEqual(case["test"]["anchor_count"], 10)
                self.assertEqual(case["test"]["oov_target_count"], 2)
                self.assertEqual(case["test"]["num_items"], 11)
                self.assertAlmostEqual(case["test"]["hr100"], .8)
                self.assertEqual(case["test"]["warm_hr100"], 1.)
                self.assertEqual(case["trained_inference"]["metadata"]["retained_tail_tokens"], 8)
                users = json.loads((output / f"{name}-test-users.json").read_text())
                self.assertEqual([user["anchor_count"] for user in users], [2] * 5)
                self.assertEqual(users[0]["ndcg100"], 0.)
                self.assertNotIn("rank", users[0])
                with gzip.open(output / f"{name}-test-anchors.json.gz", "rt") as stream:
                    anchors = json.load(stream)
                self.assertEqual(len(anchors), 10)
                hashes_by_case[name] = {row["anchor_id"]: row["reference_context_hash"] for row in anchors}
                for row in anchors:
                    self.assertEqual(row["rank_cutoff"], 100)
                    self.assertTrue({"context", "timestamps", "target", "target_timestamp"}.isdisjoint(row))
                    self.assertEqual(row["target_known"], row["user_id"] != 0)
                    self.assertTrue(1 <= row["rank_at_max_k"] <= 11 if row["target_known"]
                                    else row["rank_at_max_k"] == 0)
            self.assertTrue(all(hashes == hashes_by_case["recent128"] for hashes in hashes_by_case.values()))

            mixed_cases = [report["cases"][f"mixed_pair_mlp_m{k}"] for k in config["mixed_budgets"]]
            self.assertEqual(len({case["checkpoint"] for case in mixed_cases}), 1)
            self.assertEqual(len({case["checkpoint_sha256"] for case in mixed_cases}), 1)
            self.assertEqual(len({case["best_epoch"] for case in mixed_cases}), 1)
            mixed = report["models"]["mixed_pair_mlp"]
            for row in mixed["passes"]:
                self.assertAlmostEqual(row["selection_score"],
                    sum(value["ndcg10"] for value in row["validation"].values()) / 6)
                self.assertEqual(sum(value["windows"] for value in row["budget_exposures"].values()), 5)
                self.assertEqual(sum(value["labels"] for value in row["budget_exposures"].values()), 20)
                self.assertEqual(sum(value["updates"] for value in row["budget_exposures"].values()), 2)
            self.assertEqual(len(report["mixed_adjacent_budget_test"]), 5)
            for comparison in report["mixed_adjacent_budget_test"]:
                for value in comparison["comparisons"].values():
                    self.assertEqual((value["matched_users"], value["matched_anchors"]), (5, 10))

            # Reloaded weights reproduce ranks, not merely the saved scalar schema.
            saved = torch.load(output / mixed_cases[0]["checkpoint"], map_location="cpu", weights_only=True)
            model = build_quality_model(config, saved["variant"], saved["num_items"])
            model.compressor.mixed_budget_choices = tuple(config["mixed_budgets"])
            model.load_state_dict(saved["state_dict"])
            model.compressor.tail_size = 8
            for budget in config["mixed_budgets"]:
                select_summary_budget(model, budget)
                reloaded = evaluate_daily_quality(model, examples(cohort, evaluation,
                    split_name="test", max_seq_len=64), device=torch.device("cpu"), batch_size=2,
                    item_chunk_size=5)
                with gzip.open(output / f"mixed_pair_mlp_m{budget}-test-anchors.json.gz", "rt") as stream:
                    self.assertEqual(reloaded["per_anchor"], json.load(stream))


if __name__ == "__main__":
    unittest.main()
