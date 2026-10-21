from __future__ import annotations

import copy
from dataclasses import replace
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
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class QueryBaselineProtocolTest(unittest.TestCase):
    def config(self, *, tiny=False):
        config = json.loads((ROOT / "experiments/yambda_query_baseline_screen.json").read_text())
        if tiny:
            parent = config["parent"]["experiment_config"]
            parent["model"].update(d_model=8, n_layers=1, n_heads=2, dropout=0.)
            parent["compressor"].update(d_inner=4, d_state=3, gate_rank=3, dropout=0.)
            parent["training"].update(num_negatives=7, amp=False)
        return config

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(uid, (np.arange(length) % 41 + 1).astype(np.int32),
                      np.arange(length, dtype=np.uint32)) for uid, length in ((17, 279), (4, 1533)))
        return LongHistoryCohort(users, np.arange(41, dtype=np.int64), {})

    def test_registered_six_models_have_four_fresh_trials_and_two_pinned_controls(self):
        from rhpr.sasrec.query_baseline_protocol import (
            baseline_trials, baseline_variants, validate_baseline_config,
        )
        config = self.config()
        original = copy.deepcopy(config)
        validate_baseline_config(config)
        variants = baseline_variants(config)
        self.assertEqual([v["name"] for v in variants], config["variants"])
        self.assertEqual(set(config["reference_search"]["models"]),
                         {"query_cross_attention_m4", "query_ssm_m4"})
        self.assertEqual(config["reference_search"]["experiment_config"], json.loads(
            (ROOT / "experiments/yambda_query_compressor_architecture.json").read_text()))
        self.assertEqual(config["reference_search"]["source_code_sha"],
                         "a8f6f0dc760624e83d1f8261bd19333cfbd4dc02")
        for variant in variants:
            trials = baseline_trials(config, variant)
            self.assertEqual(len(trials), 1)
            self.assertEqual((trials[0]["backbone_lr"], trials[0]["compressor_lr"]), (.001, .001))
            if variant["name"] in config["reference_search"]["models"]:
                self.assertEqual(trials[0], config["reference_search"]["models"][variant["name"]]["trial"])
            else:
                self.assertEqual(trials[0]["name"], "fixed_lr")
            trials[0]["backbone_lr"] = 99
            self.assertEqual(baseline_trials(config, variant)[0]["backbone_lr"], .001)
        self.assertEqual(config, original)

    def test_registration_rejects_data_policy_architecture_and_reference_pin_drift(self):
        from rhpr.sasrec.query_baseline_protocol import validate_baseline_config
        mutations = [
            ("hpo", "min_passes", 10), ("hpo", "patience", 1),
            ("hpo", "proxy_old_length", 2048), ("hpo", "test_evaluations", 1),
            ("training", "learning_rate", .003), ("training", "budget_seconds", 999999),
            ("training", "effective_batch_size", 64), ("model", "n_layers", 3),
            ("compressor", "d_inner", 64), ("data", "catalog_sha256", "0" * 64),
            ("runtime", "target_torch_version", "2.5.1"),
            ("parent", "source_code_sha", "0" * 40),
            ("reference_search", "source_code_sha", "0" * 40),
            ("reference_search", "kernel", "carasik/other-kernel"),
            ("kaggle", "kernel_sources", []),
        ]
        for section, key, value in mutations:
            bad = self.config()
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_baseline_config(bad)
        for change in (lambda c: c.update(variants=c["variants"][::-1]),
                       lambda c: c["baseline_batches"]["mamba4rec"].update(full=32),
                       lambda c: c["baseline_batches"]["fuxi_linear"].update(training=32),
                       lambda c: c["standalone_models"]["mamba4rec"].update(d_state=16),
                       lambda c: c["standalone_models"]["fuxi_linear"].update(temporal_heads=4),
                       lambda c: c["reference_search"]["models"]["query_ssm_m4"]["checkpoint"].update(sha256="0" * 64),
                       lambda c: c.update(unregistered=True), lambda c: c.pop("reference_search")):
            bad = self.config()
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_baseline_config(bad)

    def test_native_caps_and_full_evaluation_batches_are_registered_per_model(self):
        from rhpr.sasrec.query_baseline_protocol import (
            baseline_context_cap, baseline_evaluation_batch, baseline_trials, baseline_variants,
        )
        config = self.config()
        for variant in baseline_variants(config):
            name = variant["name"]
            native = 128 if name == "recent128" else 256 if name == "query_only_m4" else 1152
            self.assertEqual(baseline_context_cap(config, variant, 1024), native)
            self.assertEqual(baseline_context_cap(config, variant, None),
                             native if native < 1152 else 32768)
            self.assertEqual(baseline_evaluation_batch(config, variant, 1024), 32)
            self.assertEqual(baseline_evaluation_batch(config, variant, None),
                             8 if name == "mamba4rec" else 4 if name == "fuxi_linear" else 32)
            for invalid in (True, 0, 128, 1024., 2048):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    baseline_context_cap(config, variant, invalid)
        unknown = {"name": "other", "kind": "mamba4rec", "summary_tokens": 0}
        for method, args in ((baseline_context_cap, (1024,)),
                             (baseline_evaluation_batch, (1024,)), (baseline_trials, ())):
            with self.assertRaises(ValueError):
                method(config, unknown, *args)

    def test_all_native_training_inputs_keep_final128_targets_and_identical_negatives(self):
        from rhpr.sasrec.query_baseline_protocol import baseline_batch_arrays, baseline_variants
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_batch_arrays
        from rhpr.sasrec.query_proxy_protocol import hpo_variants
        config, cohort = self.config(tiny=True), self.cohort()
        # Zero IDs are occupied OOV events, including both Q and R positions.
        for user in cohort.users:
            user.train[-200] = user.train[-20] = 0
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=7)
        schedule_before = tuple(windows)
        full = prefix_batch_arrays(cohort, windows, config["parent"]["experiment_config"],
                                   hpo_variants(config)[0])
        for variant in baseline_variants(config):
            name = variant["name"]
            for length in (1024, None):
                batch = baseline_batch_arrays(cohort, windows, config, variant, old_length=length)
                np.testing.assert_array_equal(batch["negative_ids"], full["negative_ids"])
                for index, window in enumerate(windows):
                    user = cohort.users[window.user_index]
                    limit = 128 if name == "recent128" else 256 if name == "query_only_m4" else (
                        window.end if length is None else 1152)
                    count = min(window.end, limit)
                    self.assertEqual(int(batch["valid_mask"][index].sum()), count)
                    np.testing.assert_array_equal(batch["item_ids"][index, :count], user.train[window.end-count:window.end])
                    np.testing.assert_array_equal(batch["targets"][index, count-128:count], user.train[-128:])
                    self.assertFalse(bool(batch["valid_mask"][index, count:].any()))
                    occupied_zero = (batch["item_ids"][index, :count] == 0) & batch["valid_mask"][index, :count]
                    self.assertTrue(bool(occupied_zero.any()))
        self.assertEqual(windows, schedule_before)
        later = fixed_prefix_schedule(cohort, seed=42, epoch=8)
        self.assertEqual({(w.user_id, w.end) for w in windows}, {(w.user_id, w.end) for w in later})
        self.assertNotEqual({(w.user_id, w.negative_seed) for w in windows},
                            {(w.user_id, w.negative_seed) for w in later})

    def test_every_native_batch_rejects_changed_endpoints_and_original_context_overflow(self):
        from rhpr.sasrec.query_baseline_protocol import baseline_batch_arrays, baseline_variants
        from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
        config, cohort = self.config(tiny=True), self.cohort()
        windows = fixed_prefix_schedule(cohort, seed=42, epoch=1)
        for variant in baseline_variants(config):
            with self.subTest(variant=variant["name"]):
                with self.assertRaises(ValueError):
                    baseline_batch_arrays(cohort, [replace(windows[0], end=windows[0].end-1)],
                                          config, variant, old_length=1024)
                with self.assertRaises(ValueError):
                    baseline_batch_arrays(cohort, [], config, variant, old_length=1024)
                bad = copy.deepcopy(config)
                bad["parent"]["experiment_config"]["training"]["raw_context_length"] = 256
                with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
                    baseline_batch_arrays(cohort, windows, bad, variant, old_length=1024)

    def evaluation_fixture(self):
        from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample, daily_context_sha256
        items = (np.arange(1500) % 41 + 1).astype(np.int32)
        items[-200] = items[-20] = 0
        times = np.arange(1500, dtype=np.uint32) * 7
        digest = daily_context_sha256(17, items, times)
        example = DailyEvaluationExample(17, "17:3", 3, items, times, 41, 20000, "1" * 64)
        reference = {key: getattr(example, key) for key in
            ("user_id", "anchor_id", "day", "target", "target_timestamp", "reference_context_hash")}
        reference.update(serving_context_length=len(items), serving_context_sha256=digest)
        return example, reference

    def test_evaluation_audits_full_input_then_crops_native_suffix_and_records_actual_hash(self):
        from rhpr.sasrec.daily_evaluation_data import daily_context_sha256
        from rhpr.sasrec.query_baseline_protocol import baseline_evaluation_examples, baseline_variants
        original, reference = self.evaluation_fixture()
        for variant in baseline_variants(self.config()):
            for length in (1024, None):
                rows, guarded = [], []
                changed, = baseline_evaluation_examples([original], length, variant,
                    [reference], rows, lambda: guarded.append(True))
                name = variant["name"]
                count = 128 if name == "recent128" else 256 if name == "query_only_m4" else (
                    len(original.context) if length is None else 1152)
                self.assertEqual(len(guarded), 1)
                self.assertEqual(len(rows), 1)
                np.testing.assert_array_equal(changed.context, original.context[-count:])
                np.testing.assert_array_equal(changed.timestamps, original.timestamps[-count:])
                for key in ("user_id", "anchor_id", "day", "target", "target_timestamp", "reference_context_hash"):
                    self.assertEqual(getattr(changed, key), getattr(original, key))
                self.assertEqual(rows[0]["original_context_length"], 1500)
                self.assertEqual(rows[0]["actual_context_length"], count)
                self.assertEqual(rows[0]["actual_context_sha256"],
                                 daily_context_sha256(17, changed.context, changed.timestamps))
                self.assertEqual(rows[0]["original_full_context_sha256"], reference["serving_context_sha256"])
                self.assertEqual(rows[0]["query_input_retained"], name != "recent128")
                self.assertEqual(rows[0]["query_and_raw_tail_preserved"], name != "recent128")
                self.assertTrue(rows[0]["raw_tail_preserved"])
                self.assertEqual(rows[0]["query_context_sha256"], daily_context_sha256(
                    17, original.context[-256:-128], original.timestamps[-256:-128]))
                self.assertEqual(rows[0]["raw_context_sha256"], daily_context_sha256(
                    17, original.context[-128:], original.timestamps[-128:]))
                self.assertTrue(bool((changed.context == 0).any()))
                if name in ("recent128", "query_only_m4") or length is not None:
                    self.assertFalse(np.shares_memory(changed.context, original.context))

    def test_evaluation_rejects_remote_full_context_tamper_before_native_crop_and_missing_support(self):
        from rhpr.sasrec.query_baseline_protocol import baseline_evaluation_examples, baseline_variants
        original, reference = self.evaluation_fixture()
        remote = original.context.copy()
        remote[0] += 1  # Outside every short model's actual input; original audit must still fail.
        for variant in baseline_variants(self.config()):
            cases = ([replace(original, context=remote)], [replace(original, target=40)], [])
            for examples in cases:
                rows = []
                with self.subTest(name=variant["name"], count=len(examples)), self.assertRaises(ValueError):
                    list(baseline_evaluation_examples(examples, 1024, variant, [reference], rows, lambda: None))
                self.assertEqual(rows, [])

    def fixture(self, directory):
        from rhpr.sasrec.daily_quality import _aggregate
        config = self.config()
        mount, output = Path(directory) / "mount", Path(directory) / "output"
        roots = {kernel: mount / kernel.split("/")[1] for kernel in config["kaggle"]["kernel_sources"]}
        for root in roots.values():
            (root / "checkpoints").mkdir(parents=True)
        reference = config["reference_search"]
        root = roots[reference["kernel"]]
        original = reference["experiment_config"]
        contexts = [{"user_id": 7, "anchor_id": "7:286", "day": 286, "target": 1,
            "reference_context_hash": "1" * 64, "serving_context_length": 2000,
            "serving_context_sha256": "2" * 64}]
        row = {"user_id": 7, "anchor_id": "7:286", "day": 286,
            "reference_context_hash": "1" * 64, "target_known": True,
            "rank": 1, "rank_at_max_k": 1, "rank_cutoff": 100,
            "hr10": 1., "hr100": 1., "ndcg10": 1., "ndcg100": 1.}
        metrics, _ = _aggregate([row])
        metrics["num_items"] = 10
        report = {"complete": True, "source_code_sha": reference["source_code_sha"],
                  "config": copy.deepcopy(original), "trials": {}, "preflight": {}}
        manifest = {"source_code_sha": reference["source_code_sha"], "config": copy.deepcopy(original),
                    "format_version": 3, "checkpoints": []}
        for family, spec in reference["models"].items():
            trial = spec["trial"]
            weight = roots[spec["weight_kernel"]] / spec["checkpoint"]["file"]
            weight.write_bytes((family + "-finite-fixture-weights").encode())
            spec["checkpoint"].update(size_bytes=weight.stat().st_size,
                                      sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
            expected = spec["checkpoint"]
            # Reused CA addresses mounted 1K weights; freshly trained QSSM is relative.
            saved_path = str(weight) if family == "query_cross_attention_m4" else expected["file"]
            saved = {"path": saved_path, "sha256": expected["sha256"],
                "size_bytes": expected["size_bytes"], "best_epoch": 50,
                "active_old_length": 1024, "precision": "float32", "training_model": family}
            if saved_path != expected["file"]:
                saved["original_path"] = expected["file"]
            result = {"status": "success", "complete": True, "trial": copy.deepcopy(trial),
                "trial_name": trial["name"], "best_checkpoint": saved, "best_epoch": 50,
                "best_validation": copy.deepcopy(metrics), "best_full_validation": copy.deepcopy(metrics),
                "passes_completed": 60, "old_length": 1024, "physical_batch_size": 32,
                "test_evaluations": 0, "from_scratch": True, "full_length_training_performed": False,
                "full_validation_is_frozen_1k_checkpoint": True, "resume_checkpoint_retained": True,
                "resumable_checkpoint": {"path": "nonexistent-overwritten-resume.pt"},
                "passes": [{"epoch": epoch} for epoch in range(1, 61)]}
            for epoch, score in zip(range(10, 61, 10), (.1, .3, .5, .7, 1., .9)):
                result["passes"][epoch - 1]["validation"] = {"ndcg10": score}
            for field, full in (("best_validation_rank_file", False), ("full_validation_rank_file", True)):
                rank_spec = spec["artifacts"][field]
                path = root / rank_spec["file"]
                with gzip.open(path, "wt") as stream:
                    json.dump([row], stream)
                rank_spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                result[field] = path.name
                context = {key: contexts[0][key] for key in ("user_id", "anchor_id", "reference_context_hash")}
                context.update(original_full_context_sha256="2" * 64, original_context_length=2000,
                    actual_context_length=2000 if full else 1152,
                    actual_context_sha256="2" * 64 if full else "3" * 64,
                    query_context_sha256="4" * 64, raw_context_sha256="5" * 64,
                    query_and_raw_tail_preserved=True, retained_position_recencies_preserved=True)
                context_spec = spec["artifacts"][field + "_contexts"]
                path = root / context_spec["file"]
                with gzip.open(path, "wt") as stream:
                    json.dump([context], stream)
                context_spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["trials"][family] = {trial["name"]: result}
            report["preflight"][family + "-" + trial["name"]] = {"fixture_cost": 1}
            manifest["checkpoints"].append(copy.deepcopy(saved))
        fixture = SimpleNamespace(config=config, root=root, roots=roots, output=output,
                                  contexts=contexts, report=report, manifest=manifest)
        for name, document in (("query-compressor-architecture.json", report),
                               ("checkpoint-manifest.json", manifest), ("experiment.json", original)):
            self.rewrite_document(fixture, name, document)
        return fixture

    def rewrite_document(self, fixture, name, document):
        path = fixture.root / name
        path.write_text(json.dumps(document))
        fixture.config["reference_search"]["documents"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def rewrite_gzip(self, fixture, family, field, mutate):
        spec = fixture.config["reference_search"]["models"][family]["artifacts"][field]
        path = fixture.root / spec["file"]
        with gzip.open(path, "rt") as stream:
            rows = json.load(stream)
        mutate(rows)
        with gzip.open(path, "wt") as stream:
            json.dump(rows, stream)
        spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, fixture):
        from rhpr.sasrec import query_baseline_protocol as protocol
        # Synthetic bytes get synthetic hash pins; the entire real source config
        # and the real strict registration/policy/rank validators stay active.
        with patch.object(protocol, "_REFERENCE_DIGEST", protocol._canonical(fixture.config["reference_search"])), \
                patch.object(protocol, "_resolve_kernel_root", side_effect=lambda kernel, configured: fixture.roots[kernel]), \
                patch.object(protocol, "_load_reference_contexts", return_value=fixture.contexts):
            return protocol.load_baseline_controls(fixture.config, object(), fixture.output)

    def test_reuse_verifies_two_generations_and_copies_portable_weights_without_test_or_resume(self):
        from rhpr.sasrec import query_history_ablation
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(torch, "load", side_effect=AssertionError("must not deserialize weights")), \
                    patch.object(query_history_ablation, "_verified_file", wraps=query_history_ablation._verified_file) as verified:
                controls = self.load(fixture)
            self.assertEqual(set(controls), {"query_cross_attention_m4", "query_ssm_m4"})
            self.assertTrue(all("test" not in call.args[1] for call in verified.call_args_list))
            for family, result in controls.items():
                spec = fixture.config["reference_search"]["models"][family]
                record = result["best_checkpoint"]
                self.assertFalse(Path(record["path"]).is_absolute())
                self.assertTrue(record["path"].startswith("checkpoints/"))
                self.assertEqual((fixture.output / record["path"]).read_bytes(),
                                 (fixture.roots[spec["weight_kernel"]] / spec["checkpoint"]["file"]).read_bytes())
                self.assertEqual(record["sha256"], spec["checkpoint"]["sha256"])
                self.assertEqual(record["source_code_sha"], spec["weight_source_code_sha"])
                self.assertTrue(result["reused"])
                self.assertFalse(result["training_performed_in_this_run"])
                self.assertFalse(result["resume_checkpoint_retained"])
                self.assertNotIn("resumable_checkpoint", result)
                self.assertFalse(result["reused_provenance"]["test_ranks_read"])
                self.assertTrue(result["reused_provenance"]["weights_copied"])
                self.assertEqual(result["stopping_policy_replay"]["best_epoch"], 50)
                self.assertEqual(result["stopping_policy_replay"]["end_pass"], 60)
                self.assertEqual(result["source_preflight"], {"fixture_cost": 1})
                self.assertTrue((fixture.output / result["best_validation_rank_file"]).is_file())
                self.assertTrue((fixture.output / result["full_validation_rank_file"]).is_file())
            self.assertEqual(len(list(fixture.output.rglob("*.pt"))), 2)
            self.assertFalse(any("test" in path.name for path in fixture.output.rglob("*")))

    def test_reuse_rejects_wrong_source_source_config_or_document_hash(self):
        mutations = [("query-compressor-architecture.json", lambda d: d.update(source_code_sha="0" * 40)),
            ("query-compressor-architecture.json", lambda d: d.update(complete=False)),
            ("checkpoint-manifest.json", lambda d: d["config"]["model"].update(n_heads=8)),
            ("checkpoint-manifest.json", lambda d: d.update(source_code_sha="0" * 40)),
            ("checkpoint-manifest.json", lambda d: d.update(format_version=2)),
            ("experiment.json", lambda d: d["hpo"].update(min_passes=10))]
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                document = json.loads((fixture.root / name).read_text())
                mutate(document)
                self.rewrite_document(fixture, name, document)
                with self.assertRaisesRegex(ValueError, "source, configuration"):
                    self.load(fixture)
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            (fixture.root / "query-compressor-architecture.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.load(fixture)

    def test_reuse_rejects_changed_checkpoint_bytes_size_epoch_or_manifest(self):
        for mutation in ("bytes", "size", "epoch", "manifest", "original_path"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                family = "query_cross_attention_m4"
                spec = fixture.config["reference_search"]["models"][family]
                weights = fixture.roots[spec["weight_kernel"]] / spec["checkpoint"]["file"]
                if mutation == "bytes":
                    weights.write_bytes(b"x" * weights.stat().st_size)
                elif mutation == "size":
                    weights.write_bytes(b"x")
                elif mutation == "manifest":
                    document = copy.deepcopy(fixture.manifest)
                    document["checkpoints"][0]["best_epoch"] = 40
                    self.rewrite_document(fixture, "checkpoint-manifest.json", document)
                else:
                    document = copy.deepcopy(fixture.report)
                    result = document["trials"][family][spec["trial"]["name"]]
                    result["best_checkpoint"]["best_epoch" if mutation == "epoch" else "original_path"] = (
                        40 if mutation == "epoch" else "checkpoints/other.pt")
                    self.rewrite_document(fixture, "query-compressor-architecture.json", document)
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_reuse_rejects_rank_identity_support_metric_hash_and_changed_qr(self):
        for mutation in ("hash", "identity", "support", "metric", "aggregate", "context", "query", "raw"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                family = "query_ssm_m4"
                spec = fixture.config["reference_search"]["models"][family]
                if mutation == "hash":
                    (fixture.root / spec["artifacts"]["full_validation_rank_file"]["file"]).write_bytes(b"wrong")
                elif mutation == "aggregate":
                    document = copy.deepcopy(fixture.report)
                    document["trials"][family][spec["trial"]["name"]]["best_full_validation"]["ndcg10"] = .2
                    self.rewrite_document(fixture, "query-compressor-architecture.json", document)
                elif mutation in ("context", "query", "raw"):
                    key = {"context": "actual_context_length", "query": "query_context_sha256", "raw": "raw_context_sha256"}[mutation]
                    self.rewrite_gzip(fixture, family, "full_validation_rank_file_contexts",
                        lambda rows: rows[0].update({key: 128 if mutation == "context" else "9" * 64}))
                else:
                    changes = {"identity": {"anchor_id": "7:287"}, "support": {"target_known": False},
                               "metric": {"ndcg10": .2}}
                    self.rewrite_gzip(fixture, family, "full_validation_rank_file",
                                     lambda rows: rows[0].update(changes[mutation]))
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_reuse_rejects_nonmatched_trial_or_non_equivalent_early_stopping(self):
        for mutation in ("trial", "trial_name", "test", "full_training", "not_frozen", "not_fresh",
                         "batch", "status", "earlier_stop", "best", "missing_check", "missing_pass"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                family = "query_ssm_m4"
                spec = fixture.config["reference_search"]["models"][family]
                document = copy.deepcopy(fixture.report)
                result = document["trials"][family][spec["trial"]["name"]]
                if mutation == "trial":
                    result["trial"]["compressor_lr"] = .003
                elif mutation == "trial_name":
                    result["trial_name"] = "other"
                elif mutation == "test":
                    result["test_evaluations"] = 1
                elif mutation == "full_training":
                    result["full_length_training_performed"] = True
                elif mutation == "not_frozen":
                    result["full_validation_is_frozen_1k_checkpoint"] = False
                elif mutation == "not_fresh":
                    result["from_scratch"] = False
                elif mutation == "batch":
                    result["physical_batch_size"] = 16
                elif mutation == "status":
                    result["status"] = "numerical_failed"
                elif mutation == "earlier_stop":
                    result["passes"][29]["validation"]["ndcg10"] = .2
                    result["passes"][39]["validation"]["ndcg10"] = .1
                elif mutation == "best":
                    result["passes"][59]["validation"]["ndcg10"] = 1.1
                elif mutation == "missing_check":
                    del result["passes"][19]["validation"]
                else:
                    result["passes"].pop(10)
                self.rewrite_document(fixture, "query-compressor-architecture.json", document)
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_reference_mount_resolution_is_unique_and_cannot_escape_input_root(self):
        from rhpr.sasrec.query_baseline_protocol import _resolve_kernel_root
        kernel = self.config()["reference_search"]["kernel"]
        slug = kernel.split("/")[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "input"
            root = base / "kernels" / "carasik" / slug
            root.mkdir(parents=True)
            self.assertEqual(_resolve_kernel_root(kernel, str(base / slug), base), root)
            duplicate = base / slug
            duplicate.mkdir()
            with self.assertRaisesRegex(ValueError, "expected one"):
                _resolve_kernel_root(kernel, str(base / slug), base)
            duplicate.rmdir()
            outside = Path(directory) / "outside" / slug
            outside.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                _resolve_kernel_root(kernel, str(outside), base)


if __name__ == "__main__":
    unittest.main()
