from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import gzip
import hashlib
import json
import math
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
    class TinyFrozenModel(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float32))
            self.arm_name = name


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryProxyDiagnosticsTest(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / "experiments/yambda_query_prefix_proxy_hpo.json").read_text())
        config["hpo"]["bootstrap_draws"] = 19
        return config

    def example(self, uid=1):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample
        items = (np.arange(4600 + uid) % 31).astype(np.int32)
        return DailyEvaluationExample(uid, f"{uid}:286", 286, items,
            np.arange(len(items), dtype=np.uint32), 0 if uid == 2 else 2, 24710401, "a" * 64)

    def context_row(self, example):
        from rhpr.sasrec.daily_evaluation_data import daily_context_sha256
        return {key: getattr(example, key) for key in
                ("user_id", "anchor_id", "day", "target", "target_timestamp", "reference_context_hash")} | {
            "serving_context_length": len(example.context),
            "serving_context_sha256": daily_context_sha256(example.user_id, example.context, example.timestamps)}

    def quality(self, examples, rank=2):
        from rhpr.sasrec.daily_quality import _aggregate
        rows = []
        for example in examples:
            actual_rank = rank if example.target > 0 else 0
            row = {key: getattr(example, key) for key in ("user_id", "anchor_id", "day", "reference_context_hash")}
            row.update(target_known=example.target > 0, rank_cutoff=100,
                rank=actual_rank if actual_rank <= 10 else 0, rank_at_max_k=actual_rank)
            for k in (10, 100):
                row[f"hr{k}"] = float(0 < actual_rank <= k)
                row[f"ndcg{k}"] = 1 / math.log2(actual_rank + 1) if 0 < actual_rank <= k else 0.
            rows.append(row)
        metrics, users = _aggregate(rows)
        return {"metrics": metrics, "per_anchor": rows, "per_user": users}

    def parent(self, root, config):
        from rhpr.sasrec.query_history_ablation import MODEL_NAMES, PARENT_SOURCE_SHA
        from rhpr.sasrec.query_prefix_protocol import prefix_variants
        root.mkdir()
        (root / "checkpoints").mkdir()
        examples = [self.example(uid) for uid in (1, 2)]
        anchors = [replace(example, context=None, timestamps=None) for example in examples]
        # A held-out anchor exists but no test example may enter the evaluator.
        anchors.append(replace(anchors[0], anchor_id="1:293", day=293, target_timestamp=25315201))
        evaluation = SimpleNamespace(anchors=anchors, summary={
            "validation_anchors": 2, "test_anchors": 1, "train_end_timestamp": 24710400,
            "validation_end_timestamp": 25315200, "test_end_timestamp": 25920000})
        contexts = {"validation": {"rows": [self.context_row(example) for example in examples]},
                    "test": {"rows": [self.context_row(examples[0]) | {
                        "anchor_id": "1:293", "target_timestamp": 25315201,
                        "serving_context_length": 1, "serving_context_sha256": "b" * 64}]}}
        # Only anchor/reference fields are used for held-out metadata here.
        def save(path, value):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(value, stream)
            return hashlib.sha256(path.read_bytes()).hexdigest()
        config["parent"]["context_manifest_sha256"] = save(root / "evaluation-input-contexts.json.gz", contexts)
        quality, records = self.quality(examples), []
        variants = {row["name"]: row for row in prefix_variants(config["parent"]["experiment_config"])}
        config["parent"]["baselines"] = {name: config["parent"]["baselines"][name] for name in MODEL_NAMES}
        for name in MODEL_NAMES:
            spec = config["parent"]["checkpoints"][name]
            spec["validation_ndcg10"] = quality["metrics"]["ndcg10"]
            spec["validation_anchors_sha256"] = save(root / f"{name}-validation-anchors.json.gz", quality["per_anchor"])
            config["parent"]["baselines"][name]["validation"]["sha256"] = spec["validation_anchors_sha256"]
            payload = {"format_version": 3, "source_code_sha": PARENT_SOURCE_SHA,
                "experiment_config": config["parent"]["experiment_config"], "model_config": config["model"],
                "variant": variants[name], "num_items": 31, "best_epoch": spec["best_epoch"],
                "selection_score": spec["validation_ndcg10"], "state_dict": TinyFrozenModel(name).state_dict()}
            torch.save(payload, root / spec["file"])
            path = root / spec["file"]
            spec.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            records.append({"training_model": name, "path": spec["file"], "size_bytes": spec["size_bytes"],
                "sha256": spec["sha256"], "best_epoch": spec["best_epoch"]})
        manifest = {"format_version": 3, "source_code_sha": PARENT_SOURCE_SHA,
                    "config": config["parent"]["experiment_config"], "checkpoints": records}
        path = root / "checkpoint-manifest.json"
        path.write_text(json.dumps(manifest))
        config["parent"]["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return SimpleNamespace(num_items=31), evaluation, examples

    def test_audited_crop_keeps_query_raw_oov_and_separate_input_identity(self):
        from rhpr.sasrec.query_proxy_diagnostics import audited_hpo_examples
        example, rows = self.example(), []
        result = list(audited_hpo_examples([example], 1024, [self.context_row(example)], rows, lambda: None))[0]
        self.assertEqual(len(result.context), 1152)
        np.testing.assert_array_equal(result.context, example.context[-1152:])
        np.testing.assert_array_equal(result.timestamps, example.timestamps[-1152:])
        self.assertIn(0, result.context[-256:])
        self.assertEqual(result.reference_context_hash, example.reference_context_hash)
        self.assertNotEqual(rows[0]["actual_context_sha256"], rows[0]["original_full_context_sha256"])
        self.assertTrue(rows[0]["retained_position_recencies_preserved"])
        full_rows = []
        full = list(audited_hpo_examples([example], None, [self.context_row(example)], full_rows, lambda: None))[0]
        self.assertIs(full, example)
        self.assertEqual(full_rows[0]["actual_context_sha256"], rows[0]["original_full_context_sha256"])

    def test_audit_rejects_original_context_target_support_and_changed_metadata(self):
        from rhpr.sasrec.query_proxy_diagnostics import audited_hpo_examples
        example = self.example()
        for field, value in (("target", 0), ("user_id", 8), ("reference_context_hash", "b" * 64),
                             ("serving_context_sha256", "b" * 64), ("target_timestamp", 24710402)):
            row = self.context_row(example) | {field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "context or target"):
                list(audited_hpo_examples([example], 1024, [row], [], lambda: None))
        with self.assertRaisesRegex(ValueError, "omitted"):
            list(audited_hpo_examples([], None, [self.context_row(example)], [], lambda: None))
        with patch("rhpr.sasrec.query_proxy_diagnostics.transform_hpo_example", return_value=replace(example, target=0)):
            with self.assertRaisesRegex(RuntimeError, "target identity"):
                list(audited_hpo_examples([example], None, [self.context_row(example)], [], lambda: None))

    def run_fixture(self, directory, mismatch=False):
        from rhpr.sasrec.query_proxy_diagnostics import run_proxy_diagnostics
        module, config = "rhpr.sasrec.query_proxy_diagnostics", self.config()
        root = directory / "parent"
        cohort, evaluation, examples = self.parent(root, config)
        events = []
        def iterate(cohort, evaluation, *, split_name, max_seq_len):
            self.assertEqual(split_name, "validation")
            self.assertEqual(max_seq_len, 32768)
            yield from examples
        def evaluate(model, values, **kwargs):
            values = list(values)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(torch.is_inference_mode_enabled())
            length = len(values[0].context)
            events.append((model.arm_name, length))
            return self.quality(values, 3 if length < len(examples[0].context)
                                or mismatch and model.arm_name == "query_cross_attention_m4" else 2)
        with ExitStack() as stack:
            for name, value in {"resolve_history_parent": lambda config: root,
                "build_prefix_model": lambda config, variant, num_items: TinyFrozenModel(variant["name"]),
                "iter_daily_evaluation_examples": iterate, "evaluate_daily_quality": evaluate}.items():
                stack.enter_context(patch(f"{module}.{name}", value))
            stack.enter_context(patch("torch.optim.AdamW", side_effect=AssertionError("no training permitted")))
            if mismatch:
                with self.assertRaisesRegex(RuntimeError, "reproduction failed"):
                    run_proxy_diagnostics(config, cohort, evaluation, torch.device("cpu"), directory / "out", lambda: None)
                report = json.loads((directory / "out/query-proxy-diagnostics.json").read_text())
            else:
                report = run_proxy_diagnostics(config, cohort, evaluation, torch.device("cpu"), directory / "out", lambda: None)
        return events, report

    def test_both_full_gates_precede_six_controls_and_no_test_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            events, report = self.run_fixture(Path(directory))
            self.assertEqual(events[:2], [("query_ssm_m4", 4601), ("query_cross_attention_m4", 4601)])
            self.assertEqual(len(events), 8)
            self.assertTrue(report["complete"])
            self.assertTrue(report["all_full_baselines_reproduced"])
            self.assertEqual(report["test_evaluations"], 0)
            self.assertEqual(len(report["paired_contrasts"]), 6)
            for contrast in report["paired_contrasts"]:
                for k in ("10", "100"):
                    self.assertLess(contrast["cutoffs"][k]["delta_ndcg"], 0)
                    self.assertEqual(contrast["cutoffs"][k]["confidence"], .99)

    def test_failed_second_baseline_stops_before_all_shortened_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            events, report = self.run_fixture(Path(directory), mismatch=True)
            self.assertEqual(len(events), 2)
            self.assertFalse(report["complete"])
            self.assertFalse(report["baseline_reproduction"]["query_cross_attention_m4"]["passed"])

    def test_parent_loader_rejects_rank_target_support_and_context_reference_drift(self):
        from rhpr.sasrec.query_proxy_diagnostics import load_proxy_parent_artifacts
        with tempfile.TemporaryDirectory() as directory:
            config, root = self.config(), Path(directory) / "parent"
            _, evaluation, _ = self.parent(root, config)
            with patch("rhpr.sasrec.query_proxy_diagnostics.resolve_history_parent", return_value=root):
                _, contexts, references, _ = load_proxy_parent_artifacts(config, evaluation)
                self.assertEqual(contexts["validation"]["rows"][1]["target"], 0)
                self.assertFalse(references["query_ssm_m4"][1]["target_known"])
                # There are deliberately no test rank files, so loading them would fail.
                path = root / config["parent"]["baselines"]["query_ssm_m4"]["validation"]["file"]
                with gzip.open(path, "rt") as stream:
                    records = json.load(stream)
                records[1]["target_known"] = True
                with gzip.open(path, "wt") as stream:
                    json.dump(records, stream)
                config["parent"]["baselines"]["query_ssm_m4"]["validation"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "target support"):
                    load_proxy_parent_artifacts(config, evaluation)


if __name__ == "__main__":
    unittest.main()
