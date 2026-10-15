from __future__ import annotations

import copy
import gzip
import hashlib
import json
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


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryPilotTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_ssm_block_pilot.json").read_text())

    def small_config(self):
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
        config["compressor"].update(d_inner=4, d_state=3, gate_rank=3, dropout=0.)
        # Retain the real 128-query/159-tail/32-target block. Only shorten old
        # memory and reduce width, so the integration test exercises serving
        # tail resets and query staleness exactly as the registered pilot does.
        config["training"].update(raw_context_length=167, serving_context_length=136,
            old_prefix_length=8, effective_batch_size=2, physical_batch_sizes=[1, 2],
            num_negatives=7, amp=False, min_passes=3, max_passes=3,
            validation_interval=1, cpu_threads=1, budget_seconds=10000)
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
            "validation_users": 3, "test_users": 3, "validation_anchors": 6, "test_anchors": 6,
            **{key: config["data"][key] for key in ("evaluation_content_sha256",
                "validation_end_timestamp", "test_end_timestamp", "day_seconds")}})

    def examples(self, cohort, evaluation, *, split_name, max_seq_len):
        from rhpr.sasrec.daily_evaluation_data import daily_context_sha256

        for user in cohort.users:
            items, timestamps = user.train.copy(), user.timestamps.copy()
            for day in (286, 287, 293, 294):
                target = 0 if user.user_id == 0 else (user.user_id + day) % 41 + 1
                if (split_name == "validation") == (day < 293):
                    yield SimpleNamespace(user_id=user.user_id, day=day,
                        anchor_id=f"{user.user_id}:{day}", context=items[-max_seq_len:],
                        timestamps=timestamps[-max_seq_len:], target=target,
                        target_timestamp=day * 86400,
                        reference_context_hash=daily_context_sha256(user.user_id, items[-167:], timestamps[-167:]))
                items = np.append(items, np.int32(target))
                timestamps = np.append(timestamps, np.uint32(day * 86400))

    def test_registered_config_dispatch_and_protocol_runtime_bounds(self):
        from rhpr.experiment import run_experiment
        from rhpr.sasrec.query_pilot import query_variants, validate_query_pilot_config

        config = self.config()
        validate_query_pilot_config(config)
        self.assertEqual([variant["name"] for variant in query_variants(config)], config["variants"])
        self.assertEqual(len(query_variants(config)), 5)
        for path, value in (
            (("summary_budgets",), [8]),
            (("variants",), config["variants"][::-1]),
            (("seed",), True),
            (("training", "query_context_size"), 159),
            (("compressor", "query_context_size"), 159),
            (("compressor", "dt_min"), 1e-3),
            (("compressor", "dt_max"), .01),
            (("training", "supervised_targets"), 1),
            (("training", "raw_context_length"), 8128),
            (("training", "serving_context_length"), 8159),
            (("training", "old_prefix_length"), 8031),
            (("training", "min_passes"), 29),
            (("training", "max_passes"), 61),
            (("training", "physical_batch_sizes"), [3]),
            (("training", "budget_seconds"), float("inf")),
            (("training", "budget_safety_factor"), .5),
            (("training", "device"), "cpu"),
            (("model", "n_layers"), 4),
            (("model", "use_time_features"), True),
            (("benchmark", "quality_cutoffs"), [10]),
            (("benchmark", "batch_sizes"), [1]),
            (("data", "evaluation_content_sha256"), "invalid"),
            (("data", "test_end_timestamp"), config["data"]["train_end_timestamp"]),
            (("runtime", "target_torch_version"), "2.11.0+cu128"),
            (("runtime", "target_python_minor"), "3.12"),
            (("runtime", "manifest_sha256"), "0" * 64),
            (("kaggle", "dataset_sources"), config["kaggle"]["dataset_sources"][:2]),
        ):
            bad = copy.deepcopy(config)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                validate_query_pilot_config(bad)
        for section, field in (("training", "query_context_size"), ("compressor", "query_context_size"),
                               ("data", "cohort_train_timestamps_sha256")):
            bad = copy.deepcopy(config)
            del bad[section][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_query_pilot_config(bad)
        with self.assertRaisesRegex(ValueError, "output_dir"):
            run_experiment(config)

    def test_build_models_share_backbone_rng_and_recent_query_initialization(self):
        from rhpr.sasrec.query_pilot import build_query_model, query_variants

        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
        config["compressor"].update(d_inner=4, d_state=3, gate_rank=3)
        models, rng = [], []
        for variant in query_variants(config):
            model = build_query_model(config, variant, 41)
            models.append(model)
            rng.append(torch.get_rng_state())
            self.assertEqual(model.max_seq_len, 8159)
            self.assertEqual(model.hstu_normalization_length, 163)
            self.assertEqual(model.compressor.tail_size, 159)
            self.assertEqual(model.position_embedding.num_embeddings, 8159)
        backbone = {key: value for key, value in models[0].state_dict().items()
                    if not key.startswith("compressor.")}
        for model, state in zip(models[1:], rng[1:]):
            self.assertTrue(torch.equal(state, rng[0]))
            for key, expected in backbone.items():
                torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
        queries = models[2].compressor.query_generator.state_dict()
        for model in models[3:]:
            for key, expected in queries.items():
                torch.testing.assert_close(model.compressor.query_generator.state_dict()[key], expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            build_query_model(config, {"name": "mystery", "kind": "mystery", "summary_tokens": 4}, 41)

    def test_block_targets_and_negatives_match_recent_and_full_inputs_without_query_leak(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.long_quality_schedule import quality_batch_arrays, quality_window_schedule
        from rhpr.sasrec.long_training_profile import _loss

        user = LongHistoryUser(9, np.arange(1, 221, dtype=np.int32), np.arange(220, dtype=np.uint32))
        cohort = LongHistoryCohort((user,), np.arange(300, dtype=np.int64), {})
        windows = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=167)
        kwargs = dict(raw_context_length=167, raw_tail_size=128, supervised_targets=32, num_negatives=7)
        full = quality_batch_arrays(cohort, windows, **kwargs)
        recent = quality_batch_arrays(cohort, windows, recent_only=True, **kwargs)
        np.testing.assert_array_equal(full["item_ids"][:, -159:], recent["item_ids"])
        np.testing.assert_array_equal(full["targets"][:, -32:], recent["targets"][:, -32:])
        np.testing.assert_array_equal(full["negative_ids"], recent["negative_ids"])
        query_ids = full["item_ids"][:, 8:136]
        supervised_targets = full["targets"][:, -32:]
        self.assertEqual(query_ids.shape, (1, 128))
        self.assertEqual(supervised_targets.shape, (1, 32))
        self.assertEqual(int(supervised_targets[0, 0]), int(query_ids[0, -1]) + 1)
        self.assertTrue(set(query_ids.flat).isdisjoint(supervised_targets.flat))
        # Candidate negatives can collide with any supervised positive. Correct
        # collision masking remains necessary after packing only 32 raw states.
        torch.manual_seed(19)
        model = SimpleNamespace(num_items=300, item_embedding=torch.nn.Embedding(301, 8))
        states = torch.randn(1, 163, 8, requires_grad=True)
        sources = torch.tensor([[135] * 4 + list(range(8, 167))])
        batch = {key: torch.from_numpy(value) for key, value in full.items()}
        batch["negative_ids"] = torch.tensor([[int(supervised_targets[0, 0]),
            int(supervised_targets[0, 5]), 251, 252, 253, 254, 255]])
        loss = _loss(model, (states, torch.ones((1, 163), dtype=torch.bool), sources),
                     batch, tail_size=32, num_negatives=7)
        selected = states[:, -32:]
        positives = torch.from_numpy(supervised_targets)
        positive_scores = (selected * model.item_embedding(positives)).sum(-1, keepdim=True)
        negative_scores = torch.einsum("bld,bkd->blk", selected, model.item_embedding(batch["negative_ids"]))
        collisions = batch["negative_ids"][:, None, :] == positives[:, :, None]
        self.assertEqual(int(collisions.sum()), 2)
        negative_scores = negative_scores.masked_fill(collisions, torch.finfo(negative_scores.dtype).min)
        expected = torch.nn.functional.cross_entropy(torch.cat((positive_scores, negative_scores), dim=-1).flatten(0, 1),
                                                     torch.zeros(32, dtype=torch.long))
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        loss.backward()
        self.assertEqual(int(states.grad[:, :-32].count_nonzero()), 0)
        self.assertTrue(bool(states.grad[:, -32:].abs().sum(dim=-1).gt(0).all()))

    def test_runner_trains_all_controls_and_tests_once_at_earliest_best_validation_checkpoint(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality
        from rhpr.sasrec.query_pilot import build_query_model, query_variants, run_query_pilot

        config = self.small_config()
        cohort, evaluation = self.cohort(config), self.evaluation(config)
        seen_forward = defaultdict(list)
        validation_weights, test_weights = {}, {}
        eval_counts = defaultdict(lambda: defaultdict(int))
        model_names = {}

        def build(*args):
            model = build_query_model(*args)
            name = args[1]["name"]
            model_names[id(model)] = name
            def observe(compressor, inputs):
                seen_forward[name].append((model.training, compressor.tail_size, inputs[0].shape[1]))
            model.compressor.register_forward_pre_hook(observe)
            return model

        def evaluate(model, examples, **kwargs):
            examples = list(examples)
            split = "validation" if examples[0].day < 293 else "test"
            name = model_names[id(model)]
            eval_counts[name][split] += 1
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            if split == "validation" and eval_counts[name][split] == 1:
                validation_weights[name] = state
            elif split == "test":
                test_weights[name] = state
            self.assertEqual(model.compressor.tail_size, 128)
            self.assertTrue(all(len(example.context) == (128 if name == "recent128" else 136) for example in examples))
            result = evaluate_daily_quality(model, examples, **kwargs)
            if split == "validation":
                # A later tie must keep the first checkpoint. The real evaluator
                # still exercises catalogue scores, anchors, OOV and user means.
                result["metrics"]["ndcg10"] = (.6, .4, .6)[eval_counts[name][split] - 1]
            return result

        def preflight(config, variant, cohort, evaluation, schedule, device):
            return {**variant, "training_batch_size": 2, "evaluation_batch_size": 2,
                "estimated_train_pass_seconds": .1, "estimated_validation_seconds": .1,
                "estimated_test_seconds": .1,
                "evaluation": [{"status": "ok", "batch_size": 2}]}

        def benchmark(model, contexts, **kwargs):
            self.assertEqual(model.compressor.tail_size, 128)
            self.assertTrue(contexts)
            return {"metrics": {"batch32_p50_ms": 1.}, "metadata": {"retained_tail_tokens": 128}}

        module = "rhpr.sasrec.query_pilot."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for name, kwargs in (
                (module + "validate_query_pilot_config", {}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "iter_daily_evaluation_examples", {"side_effect": self.examples}),
                (module + "build_query_model", {"side_effect": build}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "_benchmark_query_serving", {"side_effect": benchmark}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "_cleanup", {}),
                ("rhpr.sasrec.query_profile.audit_native_query_scan", {"return_value": {"passed": True, "test_only": True}}),
                ("rhpr.sasrec.query_profile.profile_query_variant", {"side_effect": preflight}),
                ("torch.cuda.get_device_name", {"return_value": "CPU integration fixture"}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            output = Path(temporary)
            metrics = run_query_pilot(config, output)
            report = json.loads((output / "query-pilot-quality.json").read_text())
            manifest = json.loads((output / "checkpoint-manifest.json").read_text())
            self.assertTrue(metrics["quality_complete"])
            self.assertTrue(report["complete"])
            self.assertEqual(metrics["training_models"], 5)
            self.assertEqual(metrics["common_training_passes"], 3)
            self.assertEqual(metrics["labels_per_training_model"], 3 * 32 * 3)
            self.assertEqual(len(report["schedules"]), 3)
            self.assertEqual(len(manifest["checkpoints"]), 5)
            names = [variant["name"] for variant in query_variants(config)]
            self.assertEqual(list(report["models"]), sorted(names))
            self.assertEqual(set(test_weights), set(names))
            hashes = {}
            for record in manifest["checkpoints"]:
                name = record["training_model"]
                with self.subTest(model=name):
                    self.assertEqual(eval_counts[name], {"validation": 3, "test": 1})
                    self.assertEqual(record["best_epoch"], 1)
                    case = report["cases"][name]
                    self.assertEqual(case["best_epoch"], 1)
                    self.assertFalse(case["best_at_last_pass"])
                    self.assertEqual(case["validation"]["ndcg10"], .6)
                    self.assertEqual(case["test"]["anchor_count"], 6)
                    self.assertEqual(case["test"]["user_count"], 3)
                    self.assertEqual(case["test"]["oov_target_count"], 2)
                    self.assertEqual(case["test"]["num_items"], 41)
                    self.assertAlmostEqual(case["test"]["hr100"], 2 / 3)
                    passes = report["models"][name]["passes"]
                    self.assertEqual([row["windows"] for row in passes], [3] * 3)
                    self.assertEqual([row["labels"] for row in passes], [96] * 3)
                    self.assertEqual([row["successful_updates"] for row in passes], [2] * 3)
                    self.assertEqual([row["amp_skipped_update_attempts"] for row in passes], [0] * 3)
                    for key, expected in validation_weights[name].items():
                        torch.testing.assert_close(test_weights[name][key], expected, rtol=0, atol=0)
                    checkpoint = output / record["path"]
                    self.assertEqual(record["sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                    self.assertEqual(record["size_bytes"], checkpoint.stat().st_size)
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertEqual(saved["best_epoch"], 1)
                    self.assertEqual(saved["num_items"], 41)
                    self.assertEqual(saved["experiment_config"], config)
                    self.assertEqual(saved["source_code_sha"], manifest["source_code_sha"])
                    for key, value in saved["state_dict"].items():
                        torch.testing.assert_close(value, validation_weights[name][key], rtol=0, atol=0)
                        if value.is_floating_point():
                            self.assertEqual(value.dtype, torch.float32)
                            self.assertTrue(bool(torch.isfinite(value).all()))
                    torch.testing.assert_close(saved["state_dict"]["item_embedding.weight"][0], torch.zeros(8), rtol=0, atol=0)
                    train_forwards = [row for row in seen_forward[name] if row[0]]
                    eval_forwards = [row for row in seen_forward[name] if not row[0]]
                    self.assertEqual(len(train_forwards), 6)
                    self.assertTrue(all(tail == 159 for _, tail, _ in train_forwards))
                    self.assertTrue(all(width == (159 if name == "recent128" else 167) for _, _, width in train_forwards))
                    self.assertTrue(eval_forwards)
                    self.assertTrue(all(tail == 128 for _, tail, _ in eval_forwards))
                    self.assertTrue(all(width == (128 if name == "recent128" else 136) for _, _, width in eval_forwards))
                    with gzip.open(output / f"{name}-test-anchors.json.gz", "rt") as stream:
                        rows = json.load(stream)
                    hashes[name] = {row["anchor_id"]: row["reference_context_hash"] for row in rows}
                    self.assertTrue(all(row["target_known"] == (row["user_id"] != 0) for row in rows))
                    self.assertTrue(all({"context", "timestamps", "target", "target_timestamp"}.isdisjoint(row) for row in rows))
            self.assertTrue(all(value == hashes["recent128"] for value in hashes.values()))
            self.assertEqual(len(report["paired_test_contrasts"]), 4)
            with gzip.open(output / "evaluation-input-contexts.json.gz", "rt") as stream:
                contexts = json.load(stream)
            for split in ("validation", "test"):
                self.assertEqual(contexts[split]["anchors"], 6)
                self.assertEqual({row["serving_context_length"] for row in contexts[split]["rows"]}, {136})
                self.assertTrue(all(row["recent_context_sha256"] != row["serving_context_sha256"] for row in contexts[split]["rows"]))

    def test_serving_benchmark_keeps_comparable_batches_and_reports_capacity_skips(self):
        from rhpr.sasrec.query_pilot import _benchmark_query_serving

        config = self.config()
        calls = []
        def measure(model, contexts, **kwargs):
            size = kwargs["batch_size"]
            calls.append(size)
            return {"metrics": {"full_b1_wall_p50_ms": 1., f"full_b{size}_wall_p50_ms": float(size)},
                "cases": {"b1": {"batch_size": 1}, f"b{size}": {"batch_size": size}},
                "metadata": {"retained_tail_tokens": 128}}

        module = "rhpr.sasrec.query_pilot.benchmark_long_quality"
        preflight = {"evaluation": [{"status": "ok", "batch_size": 8},
                                    {"status": "ok", "batch_size": 32}]}
        with patch(module, side_effect=measure):
            result = _benchmark_query_serving(None, [np.arange(128)], config=config,
                preflight=preflight, device=torch.device("cpu"))
        self.assertEqual(calls, [8, 32])
        self.assertEqual(set(result["cases"]), {"b1", "b8", "b32"})
        self.assertEqual(result["metadata"]["measured_batch_sizes"], [1, 8, 32])
        self.assertEqual(result["metadata"]["skipped_batch_sizes"], [])
        calls.clear()
        preflight["evaluation"][1]["status"] = "oom"
        with patch(module, side_effect=measure):
            result = _benchmark_query_serving(None, [np.arange(128)], config=config,
                preflight=preflight, device=torch.device("cpu"))
        self.assertEqual(calls, [8])
        self.assertEqual(result["metadata"]["measured_batch_sizes"], [1, 8])
        self.assertEqual([row["batch_size"] for row in result["metadata"]["skipped_batch_sizes"]], [32])
        preflight["evaluation"][0]["status"] = "oom"
        with patch(module) as benchmark, self.assertRaisesRegex(RuntimeError, "parallel serving batch"):
            _benchmark_query_serving(None, [np.arange(128)], config=config,
                preflight=preflight, device=torch.device("cpu"))
        benchmark.assert_not_called()

    def test_paired_contrasts_cluster_users_and_correct_the_declared_four_tests(self):
        from rhpr.sasrec.query_pilot import _paired_contrasts

        config = self.config()
        rows = {name: [] for name in config["variants"]}
        for user_id, days, baseline, candidate in ((1, (293, 294, 295, 296), .2, .3),
                (2, (293,), .4, .7), (3, (293,), .8, .6)):
            for day in days:
                for name in rows:
                    value = candidate if name == "query_ssm_m4" else baseline
                    rows[name].append({"user_id": user_id, "day": day,
                        "anchor_id": f"{user_id}:{day}", "reference_context_hash": "a" * 64,
                        "target_known": True, "rank_cutoff": 100,
                        "ndcg10": value, "ndcg100": value})
        contrasts = _paired_contrasts(rows, config)
        self.assertEqual(len(contrasts), 4)
        expected_macro_delta = (.1 + .3 - .2) / 3
        for item in contrasts:
            self.assertEqual(item["candidate"], "query_ssm_m4")
            self.assertFalse(item["training_seed_uncertainty_included"])
            for cutoff in ("10", "100"):
                self.assertEqual(item["cutoffs"][cutoff]["matched_users"], 3)
                self.assertEqual(item["cutoffs"][cutoff]["matched_anchors"], 6)
                self.assertAlmostEqual(item["cutoffs"][cutoff]["delta_ndcg"], expected_macro_delta)
        corrected = 0.
        for rank, item in enumerate(sorted(contrasts, key=lambda row: row["sign_flip_two_sided_p10"])):
            corrected = max(corrected, min(1., (4 - rank) * item["sign_flip_two_sided_p10"]))
            self.assertEqual(item["holm_p10_family4"], corrected)
        bad = copy.deepcopy(rows)
        bad["query_cross_attention_m4"][0]["reference_context_hash"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "reference_context_hash"):
            _paired_contrasts(bad, config)


if __name__ == "__main__":
    unittest.main()
