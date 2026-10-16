from __future__ import annotations

import copy
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
from pathlib import Path
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
    class TinyFrozenModel(torch.nn.Module):
        def __init__(self, name="query_ssm_m4"):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float32))
            self.arm_name = name


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryHistoryAblationTest(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_history_ablation.json").read_text())
        config["ablation"].update(bootstrap_draws=99, permutation_draws=99)
        return config

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def records(self, examples, rank=2):
        return [{"user_id": example.user_id, "anchor_id": example.anchor_id,
            "day": example.day, "reference_context_hash": example.reference_context_hash,
            "target_known": True, "rank_cutoff": 100,
            "rank": rank if rank <= 10 else 0, "rank_at_max_k": rank,
            "hr10": float(0 < rank <= 10), "hr100": float(0 < rank <= 100),
            "ndcg10": 1. / math.log2(rank + 1) if 0 < rank <= 10 else 0.,
            "ndcg100": 1. / math.log2(rank + 1) if 0 < rank <= 100 else 0.}
            for example in examples]

    def quality(self, examples, rank=2):
        from rhpr.sasrec.daily_quality import _aggregate
        rows = self.records(examples, rank)
        metrics, users = _aggregate(rows)
        metrics.update(num_items=31, evaluation_wall_seconds=.001)
        return {"metrics": metrics, "per_user": users, "per_anchor": rows, "metadata": {}}

    def setup_parent(self, root, config):
        from rhpr.sasrec.daily_evaluation_data import (
            DailyEvaluationAnchor, DailyEvaluationData, DailyEvaluationExample,
            DailyEvaluationUser, daily_context_sha256,
        )
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        from rhpr.sasrec.query_history_ablation import MODEL_NAMES, PARENT_SOURCE_SHA
        from rhpr.sasrec.query_prefix_protocol import prefix_variants
        root.mkdir(parents=True, exist_ok=True)
        (root / "checkpoints").mkdir(exist_ok=True)
        users = tuple(LongHistoryUser(i, (np.arange(300 + i) % 31 + 1).astype(np.int32),
            np.arange(300 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        cohort = LongHistoryCohort(users, np.arange(31, dtype=np.int64), summary)
        examples = tuple(DailyEvaluationExample(user.user_id, f"{user.user_id}:286", 286,
            user.train, user.timestamps, 2, config["data"]["train_end_timestamp"] + 1, "a" * 64)
            for user in users)
        anchors = tuple(DailyEvaluationAnchor(example.user_id, example.anchor_id, example.day,
            0, example.target, example.target_timestamp, example.reference_context_hash) for example in examples)
        # A test anchor exists in prepared data, but must never be forwarded.
        anchors += (DailyEvaluationAnchor(0, "0:293", 293, 0, 2,
            config["data"]["validation_end_timestamp"] + 1, "b" * 64),)
        evaluation = DailyEvaluationData(tuple(DailyEvaluationUser(user.user_id,
            np.asarray([2], dtype=np.int32), np.asarray([25000000], dtype=np.uint32)) for user in users),
            anchors, {"validation_anchors": 3, "test_anchors": 1,
                **{key: config["data"][key] for key in ("evaluation_content_sha256",
                    "validation_end_timestamp", "test_end_timestamp", "day_seconds")}})
        context_rows = [{"anchor_id": example.anchor_id, "reference_context_hash": example.reference_context_hash,
            "serving_context_length": len(example.context), "serving_context_sha256": daily_context_sha256(
                example.user_id, example.context, example.timestamps)} for example in examples]
        fingerprint = hashlib.sha256()
        for row in context_rows:
            fingerprint.update((row["anchor_id"] + ":" + row["serving_context_sha256"] + "\n").encode("ascii"))
        contexts = {"validation": {"anchors": len(examples), "rows": context_rows,
            "ordered_serving_context_sha256": fingerprint.hexdigest()}}
        path = root / "evaluation-input-contexts.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(contexts, stream)
        config["parent"]["context_manifest_sha256"] = self.sha(path)
        checkpoints = []
        variants = {row["name"]: row for row in prefix_variants(config["parent"]["experiment_config"])}
        score = self.quality(examples)["metrics"]["ndcg10"]
        for name in MODEL_NAMES:
            spec = config["parent"]["checkpoints"][name]
            spec["validation_ndcg10"] = score
            model = TinyFrozenModel(name)
            model.marker.data.fill_(11 if name == "query_ssm_m4" else 22)
            payload = {"format_version": 3, "source_code_sha": PARENT_SOURCE_SHA,
                "experiment_config": config["parent"]["experiment_config"],
                "model_config": config["model"], "variant": variants[name], "num_items": 31,
                "best_epoch": spec["best_epoch"], "selection_score": score, "state_dict": model.state_dict()}
            checkpoint = root / spec["file"]
            torch.save(payload, checkpoint)
            spec["size_bytes"], spec["sha256"] = checkpoint.stat().st_size, self.sha(checkpoint)
            record = {"training_model": name, "path": spec["file"], "size_bytes": spec["size_bytes"],
                "sha256": spec["sha256"], "best_epoch": spec["best_epoch"]}
            checkpoints.append(record)
            anchors_path = root / f"{name}-validation-anchors.json.gz"
            with gzip.open(anchors_path, "wt", encoding="utf-8") as stream:
                json.dump(self.records(examples), stream)
            spec["validation_anchors_sha256"] = self.sha(anchors_path)
        manifest = {"format_version": 3, "source_code_sha": PARENT_SOURCE_SHA,
            "config": config["parent"]["experiment_config"], "checkpoints": checkpoints}
        path = root / "checkpoint-manifest.json"
        path.write_text(json.dumps(manifest))
        config["parent"]["manifest_sha256"] = self.sha(path)
        return cohort, evaluation, examples

    def test_default_config_and_protocol_drift(self):
        from rhpr.sasrec.query_history_ablation import validate_history_ablation_config
        config = self.config()
        validate_history_ablation_config(config)
        for mutate in (
            lambda value: value["model"].update(dropout=.1),
            lambda value: value["kaggle"].update(kernel_sources=[]),
            lambda value: value["ablation"].update(conditions=["full", "query_only_history"]),
            lambda value: value["ablation"].update(budget_seconds=2401),
            lambda value: value["ablation"].update(equivalence_margin_ndcg10=.01),
            lambda value: value["ablation"].update(baseline_metric_tolerance=.01),
            lambda value: value["parent"]["checkpoints"]["query_ssm_m4"].update(file="../bad.pt"),
        ):
            changed = copy.deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_history_ablation_config(changed)

    def test_bounded_mount_layout_and_duplicate_rejection(self):
        from rhpr.sasrec.query_history_ablation import resolve_history_parent
        with tempfile.TemporaryDirectory() as directory:
            base, config = Path(directory), self.config()
            slug = "rhpr-sasrec-prefix-query-pilot"
            root = base / "kernels" / "carasik" / slug
            root.mkdir(parents=True)
            (root / "checkpoint-manifest.json").write_text("{}")
            config["parent"]["checkpoint_root"] = str(base / slug)
            self.assertEqual(resolve_history_parent(config, base), root)
            duplicate = base / slug
            duplicate.mkdir()
            (duplicate / "checkpoint-manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_history_parent(config, base)
            config["parent"]["checkpoint_root"] = "/tmp/outside/" + slug
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_history_parent(config, base)

    def test_checkpoint_strict_loading_and_freezing(self):
        from rhpr.sasrec.query_history_ablation import load_history_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), self.config()
            self.setup_parent(root, config)
            model = TinyFrozenModel()
            result = load_history_checkpoint(config, "query_ssm_m4", num_items=31, model=model, root=root)
            self.assertEqual(model.marker.item(), 11.)
            self.assertFalse(model.training)
            self.assertFalse(model.marker.requires_grad)
            self.assertNotIn("state_dict", result)
            untouched = TinyFrozenModel()
            with self.assertRaisesRegex(ValueError, "num_items"):
                load_history_checkpoint(config, "query_ssm_m4", num_items=32, model=untouched, root=root)
            self.assertEqual(untouched.marker.item(), 0.)
            config["parent"]["checkpoints"]["query_ssm_m4"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_history_checkpoint(config, "query_ssm_m4", num_items=31, model=untouched, root=root)

    def test_corrupt_weight_and_wrong_epoch_rejected_before_mutation(self):
        from rhpr.sasrec.query_history_ablation import load_history_checkpoint
        for corruption in ("nonfinite", "dtype", "shape", "epoch", "source", "config"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                root, config = Path(directory), self.config()
                self.setup_parent(root, config)
                spec = config["parent"]["checkpoints"]["query_ssm_m4"]
                path = root / spec["file"]
                payload = torch.load(path, weights_only=True)
                if corruption == "nonfinite":
                    payload["state_dict"]["marker"][0] = float("nan")
                elif corruption == "dtype":
                    payload["state_dict"]["marker"] = payload["state_dict"]["marker"].half()
                elif corruption == "shape":
                    payload["state_dict"]["marker"] = torch.ones(2)
                elif corruption == "epoch":
                    payload["best_epoch"] += 1
                elif corruption == "source":
                    payload["source_code_sha"] = "0" * 40
                else:
                    payload["experiment_config"]["seed"] += 1
                torch.save(payload, path)
                spec.update(size_bytes=path.stat().st_size, sha256=self.sha(path))
                model = TinyFrozenModel()
                with self.assertRaises(ValueError):
                    load_history_checkpoint(config, "query_ssm_m4", num_items=31, model=model, root=root)
                self.assertEqual(model.marker.item(), 0.)

    def test_full_gate_rejects_rank_and_support_changes_even_if_aggregate_tied(self):
        from rhpr.sasrec.query_history_ablation import verify_full_reproduction
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample
        examples = [DailyEvaluationExample(uid, f"{uid}:286", 286, np.arange(300),
            np.arange(300), 2, 24710401, "a" * 64) for uid in (1, 2)]
        result = self.quality(examples)
        expected = result["metrics"]["ndcg10"]
        gate = verify_full_reproduction(result["per_anchor"], result, expected_ndcg10=expected, tolerance=1e-7)
        self.assertTrue(gate["passed"])
        for field, value in (("rank_at_max_k", 3), ("target_known", False),
                              ("reference_context_hash", "c" * 64)):
            changed = copy.deepcopy(result)
            changed["per_anchor"][0][field] = value
            gate = verify_full_reproduction(result["per_anchor"], changed,
                expected_ndcg10=expected, tolerance=1e-7)
            self.assertFalse(gate["passed"])

    def fixture_run(self, directory, *, mismatch=False, budget_expired=False):
        from rhpr.sasrec.query_history_ablation import run_query_history_ablation
        output, parent = directory / "output", directory / "parent"
        config = self.config()
        cohort, evaluation, examples = self.setup_parent(parent, config)
        originals = {example.anchor_id: example for example in examples}
        events = []

        def iterate(cohort, evaluation, *, split_name, max_seq_len):
            self.assertEqual(split_name, "validation", "test targets must never be evaluated")
            self.assertEqual(max_seq_len, 32768)
            yield from examples

        def evaluate(model, values, **kwargs):
            values = list(values)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(kwargs["batch_size"], 32)
            self.assertEqual(kwargs["item_chunk_size"], 65536)
            if len(values[0].context) == 256:
                condition, rank = "query_only_history", 3
            elif np.array_equal(values[0].context, originals[values[0].anchor_id].context):
                condition, rank = "full", 2
            else:
                condition, rank = "shuffle_remote", 4
            events.append((model.arm_name, condition))
            if mismatch and model.arm_name == "query_cross_attention_m4":
                rank = 3
            return self.quality(values, rank)

        module = "rhpr.sasrec.query_history_ablation"
        with ExitStack() as stack:
            replacements = {"_registered_device": lambda: torch.device("cpu"),
                "audit_native_query_scan": lambda device, **kwargs: {"passed": True, "query_source": kwargs["query_source"]},
                "resolve_dataset_directory": lambda value: Path(value),
                "load_prepared_cohort": lambda path: cohort,
                "load_prepared_daily_evaluation": lambda path, cohort: evaluation,
                "resolve_history_parent": lambda config: parent,
                "build_prefix_model": lambda config, variant, num_items: TinyFrozenModel(variant["name"]),
                "iter_daily_evaluation_examples": iterate, "evaluate_daily_quality": evaluate}
            for name, value in replacements.items():
                stack.enter_context(patch(f"{module}.{name}", value))
            stack.enter_context(patch("torch.optim.AdamW", side_effect=AssertionError("no optimizer may be constructed")))
            if budget_expired:
                stack.enter_context(patch(f"{module}.time.monotonic", side_effect=[0., 3000.]))
            if mismatch:
                with self.assertRaisesRegex(RuntimeError, "reproduction failed"):
                    run_query_history_ablation(config, output)
                return events, json.loads((output / "query-history-ablation.json").read_text())
            if budget_expired:
                with self.assertRaises(TimeoutError):
                    run_query_history_ablation(config, output)
                return events, json.loads((output / "query-history-ablation.json").read_text())
            result = run_query_history_ablation(config, output)
        return events, json.loads((output / "query-history-ablation.json").read_text()), result

    def test_frozen_run_all_full_gates_precede_interventions_and_no_test_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            events, report, metrics = self.fixture_run(Path(directory))
            self.assertEqual(events[:2], [("query_ssm_m4", "full"), ("query_cross_attention_m4", "full")])
            self.assertEqual(len(events), 6)
            self.assertTrue(report["complete"])
            self.assertTrue(metrics["all_full_baselines_reproduced"])
            self.assertEqual(metrics["validation_evaluations"], 6)
            self.assertEqual(metrics["test_evaluations"], 0)
            self.assertEqual(metrics["training_updates"], 0)
            self.assertEqual(len(report["paired_contrasts"]), 4)
            for row in report["paired_contrasts"]:
                self.assertLess(row["cutoffs"]["10"]["delta_ndcg"], 0.)
                self.assertEqual(row["cutoffs"]["10"]["confidence"], .99)
                self.assertFalse(row["cutoffs"]["10"]["fixed_weights_equivalent_within_margin"])
                self.assertIn("holm_p10_family4", row)
            paths = list((Path(directory) / "output").glob("*-validation-anchors.json.gz"))
            self.assertEqual(len(paths), 6)
            with gzip.open(Path(directory) / "output/query_ssm_m4-query_only_history-validation-contexts.json.gz", "rt") as stream:
                contexts = json.load(stream)
            self.assertTrue(all(row["actual_context_length"] == 256 for row in contexts))
            self.assertTrue(all(row["query_and_raw_tail_preserved"] for row in contexts))

    def test_baseline_mismatch_aborts_every_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            events, report = self.fixture_run(Path(directory), mismatch=True)
            self.assertEqual(events, [("query_ssm_m4", "full"), ("query_cross_attention_m4", "full")])
            self.assertFalse(report["complete"])
            self.assertFalse(report["baseline_reproduction"]["query_cross_attention_m4"]["passed"])

    def test_budget_guard_stops_before_dataset_or_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            events, report = self.fixture_run(Path(directory), budget_expired=True)
            self.assertEqual(events, [])
            self.assertIn("budget", report["aborted"])


if __name__ == "__main__":
    unittest.main()
