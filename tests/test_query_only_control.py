from __future__ import annotations

from contextlib import ExitStack
import copy
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
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor([1.]))
            self.compressor = torch.nn.Module()
            self.compressor.tail_size = 128


@unittest.skipIf(torch is None, "Torch and NumPy are required")
class QueryOnlyControlTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def config(self):
        from rhpr.sasrec.query_history_ablation import PARENT_KERNEL, PARENT_SOURCE_SHA
        original = json.loads((ROOT / "experiments/yambda_sasrec_prefix_query_pilot.json").read_text())
        config = copy.deepcopy(original)
        config.update(kind="yambda_query_only_control", name="query-only-test", variants=["query_only_m4"],
            control={"input_context_length": 256, "training_passes": 30, "bootstrap_draws": 5000})
        config["training"].update(min_passes=30, max_passes=30, physical_batch_sizes=[32], budget_seconds=1200)
        config["kaggle"]["kernel_sources"] = [PARENT_KERNEL]
        config["parent"] = {"kernel": PARENT_KERNEL, "source_code_sha": PARENT_SOURCE_SHA,
            "checkpoint_root": "/kaggle/input/rhpr-sasrec-prefix-query-pilot",
            "experiment_config": original, "manifest_sha256": "a" * 64, "context_manifest_sha256": "b" * 64,
            "baselines": {name: {split: {"file": f"{name}-{split}-anchors.json.gz", "sha256": "c" * 64}
                for split in ("validation", "test")}
                for name in ("recent128", "query_cross_attention_m4", "query_ssm_m4")}}
        return config

    def tiny(self, config):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        config["model"].update(d_model=8, n_layers=1, n_heads=2, ffn_multiplier=2, dropout=0.)
        config["training"].update(effective_batch_size=2, num_negatives=7, amp=False, cpu_threads=1)
        original = config["parent"]["experiment_config"]
        original["model"] = copy.deepcopy(config["model"])
        original["training"].update(effective_batch_size=2, num_negatives=7, amp=False, cpu_threads=1)
        users = tuple(LongHistoryUser(i, (np.arange(260 + i) % 41 + 1).astype(np.int32),
            np.arange(260 + i, dtype=np.uint32)) for i in range(3))
        summary = {key: config["data"][key] for key in ("train_end_timestamp", "cohort_uid_length_sha256",
            "catalog_sha256", "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision")}
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), summary)

    def test_registered_configuration_and_protocol_drift_rejection(self):
        from rhpr.sasrec.query_only_control import validate_query_only_config
        config = self.config()
        validate_query_only_config(config)
        for mutate in (lambda c: c["training"].update(learning_rate=.003),
                       lambda c: c["model"].update(dropout=.1),
                       lambda c: c["training"].update(max_passes=40),
                       lambda c: c["parent"]["baselines"]["recent128"]["test"].update(file="../other.gz"),
                       lambda c: c["parent"].update(manifest_sha256="bad")):
            changed = copy.deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_query_only_config(changed)

    def test_real_training_uses_short_batch_callback_and_original_window_labels(self):
        from rhpr.sasrec.long_quality import _optimizer, _scaler
        from rhpr.sasrec.query_only_protocol import build_query_only_model, query_only_batch_arrays, query_only_variant
        from rhpr.sasrec.query_prefix_pilot import train_prefix_pass
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        config = self.config()
        cohort = self.tiny(config)
        variant = query_only_variant()
        model = build_query_only_model(config, variant, cohort.num_items)
        optimizer, scaler = _optimizer(model, config["training"]), _scaler(torch.device("cpu"), config["training"])
        before = model.item_embedding.weight.detach().clone()
        windows = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
        with patch("rhpr.sasrec.query_prefix_pilot.prefix_batch_arrays", side_effect=AssertionError("legacy batch used")):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=1, device=torch.device("cpu"), optimizer=optimizer, scaler=scaler,
                batch_builder=query_only_batch_arrays)
        self.assertEqual(row["labels"], 384)
        self.assertEqual(row["successful_updates"], 2)
        self.assertEqual(row["amp_skipped_update_attempts"], 0)
        self.assertFalse(torch.equal(before, model.item_embedding.weight))

    def test_fixed30_passes_select_validation_before_single_test_and_save_exact_resume(self):
        from rhpr.sasrec.query_only_control import run_query_only_control
        config = self.config()
        cohort = self.tiny(config)
        summary = {key: config["data"][key] for key in ("evaluation_content_sha256", "validation_end_timestamp",
            "test_end_timestamp", "day_seconds")}
        summary.update(validation_anchors=3, test_anchors=3)
        evaluation = SimpleNamespace(summary=summary)
        events, trained, tested, reads = [], [], [], []

        def rows(score):
            return [{"user_id": i, "day": 286, "anchor_id": f"{i}:286", "reference_context_hash": "c" * 64,
                "target_known": True, "rank_cutoff": 100, "ndcg10": score, "ndcg100": score,
                "hr10": 1., "hr100": 1.} for i in range(3)]

        def baseline(paths, split):
            reads.append((split, len(trained)))
            if split == "test":
                self.assertEqual(len(trained), 30)
                self.assertEqual(len(tested), 1)
            return {name: rows(.4) for name in ("recent128", "query_cross_attention_m4", "query_ssm_m4")}

        def train(model, cohort, windows, *, optimizer, batch_builder, **kwargs):
            self.assertEqual(batch_builder.__name__, "query_only_batch_arrays")
            trained.append([(window.epoch, window.user_id, window.end, window.negative_seed) for window in windows])
            optimizer.zero_grad(set_to_none=True)
            model.marker.grad = torch.ones_like(model.marker)
            optimizer.step()
            events.append("train")
            return {"loss": 1., "windows": 3, "labels": 384, "successful_updates": 1,
                "amp_skipped_update_attempts": 0, "seconds": .01, "windows_per_second": 300.,
                "labels_per_second": 38400., "loss_scale": 1.}

        def evaluate(model, examples, **kwargs):
            split = list(examples)[0]
            events.append(split)
            score = .8 - len(trained) * .01 if split == "validation" else .45
            if split == "test":
                tested.append(float(model.marker.detach()[0]))
                self.assertEqual(len(trained), 30)
            from rhpr.sasrec.daily_quality import _aggregate
            anchors = rows(score)
            metrics, users = _aggregate(anchors)
            return {"metrics": metrics, "per_user": users, "per_anchor": anchors, "metadata": {}}

        module = "rhpr.sasrec.query_only_control."
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for name, kwargs in (
                (module + "validate_query_only_config", {}),
                (module + "resolve_dataset_directory", {"side_effect": Path}),
                (module + "load_prepared_cohort", {"return_value": cohort}),
                (module + "load_prepared_daily_evaluation", {"return_value": evaluation}),
                (module + "_validate_complete_histories", {}),
                (module + "_load_parent_artifacts", {"return_value": (
                    {split: {"rows": []} for split in ("validation", "test")}, {})}),
                (module + "_read_baselines", {"side_effect": baseline}),
                (module + "_preflight", {"return_value": {"estimated_training_pass_seconds": .1,
                    "estimated_validation_seconds": .1, "estimated_test_seconds": .1}}),
                (module + "_registered_device", {"return_value": torch.device("cpu")}),
                (module + "build_query_only_model", {"side_effect": lambda *a: TinyModel()}),
                (module + "train_prefix_pass", {"side_effect": train}),
                (module + "iter_daily_evaluation_examples", {"side_effect": lambda *a, split_name, **kw: iter([split_name])}),
                (module + "_audited_examples", {"side_effect": lambda examples, **kw: examples}),
                (module + "evaluate_daily_quality", {"side_effect": evaluate}),
                (module + "_contrasts", {"return_value": []}),
                ("rhpr.sasrec.query_prefix_benchmark.benchmark_prefix_quality", {"return_value": {
                    "metrics": {"full_b1_wall_p50_ms": .1}, "metadata": {}, "cases": {}}}),
                ("builtins.print", {}),
            ):
                stack.enter_context(patch(name, **kwargs))
            output = Path(temporary)
            metrics = run_query_only_control(config, output)
            report = json.loads((output / "query-only-control.json").read_text())
            best = torch.load(output / "checkpoints/query_only_m4.pt", weights_only=True)
            resume = torch.load(output / "checkpoints/query_only_m4-resume.pt", weights_only=True)
        self.assertTrue(metrics["quality_complete"])
        self.assertEqual(metrics["test_evaluations"], 1)
        self.assertEqual(report["best_epoch"], 10)
        self.assertEqual([i + 1 for i, row in enumerate(report["passes"]) if "validation" in row], [10, 20, 30])
        self.assertEqual(float(best["state_dict"]["marker"][0]), tested[0])
        self.assertEqual(resume["epoch"], 30)
        self.assertEqual(resume["next_epoch"], 31)
        self.assertTrue(resume["optimizer_state_dict"]["state"])
        self.assertEqual(reads, [("validation", 0), ("test", 30)])
        self.assertEqual(events.count("test"), 1)
        self.assertTrue(all(window[0] == epoch for epoch, group in enumerate(trained, 1) for window in group))


if __name__ == "__main__":
    unittest.main()
