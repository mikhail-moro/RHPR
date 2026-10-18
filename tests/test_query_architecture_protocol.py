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
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class ArchitectureProtocolTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_query_compressor_architecture.json").read_text())

    def test_exact_registration_preserves_source_and_returns_independent_trials(self):
        from rhpr.sasrec.query_architecture_protocol import (
            _source_config, architecture_trials, validate_architecture_config,
        )
        config = self.config()
        original = copy.deepcopy(config)
        validate_architecture_config(config)
        self.assertEqual(config, original)
        self.assertEqual(_source_config(config), json.loads(
            (ROOT / "experiments/yambda_query_prefix_1k_search.json").read_text()))
        trial = architecture_trials(config, {"name": "query_ssm_m4"})
        trial[0]["d_inner"] = 17
        self.assertEqual(architecture_trials(config, "query_ssm_m4")[0]["d_inner"], 128)
        with self.assertRaises(ValueError):
            architecture_trials(config, "recent128")

    def test_registration_rejects_lr_policy_shapes_pins_and_common_protocol_drift(self):
        from rhpr.sasrec.query_architecture_protocol import validate_architecture_config
        config = self.config()
        mutations = [
            ("hpo", "min_passes", 10), ("hpo", "patience", 1), ("hpo", "test_evaluations", 1),
            ("hpo", "proxy_old_length", 2048), ("training", "learning_rate", .003),
            ("training", "early_stopping_patience", 3), ("model", "n_heads", 8),
            ("compressor", "d_state", 32), ("parent", "source_code_sha", "0" * 40),
            ("data", "catalog_sha256", "0" * 64), ("runtime", "target_torch_version", "2.5.1"),
            ("reused_controls", "source_code_sha", "0" * 40), ("kaggle", "kernel_sources", []),
        ]
        for section, key, value in mutations:
            bad = copy.deepcopy(config)
            bad[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_architecture_config(bad)
        for key, value in (("compressor_lr", .003), ("cross_attention_heads", 16), ("readout_depth", 3)):
            bad = copy.deepcopy(config)
            bad["architecture_trials"]["query_cross_attention_m4"][0][key] = value
            with self.subTest(trial_key=key), self.assertRaises(ValueError):
                validate_architecture_config(bad)
        bad = copy.deepcopy(config)
        bad["unexpected"] = True
        with self.assertRaises(ValueError):
            validate_architecture_config(bad)
        bad = copy.deepcopy(config)
        del bad["reused_controls"]
        with self.assertRaises(ValueError):
            validate_architecture_config(bad)

    def fixture(self, directory):
        from rhpr.sasrec import query_architecture_protocol as protocol
        from rhpr.sasrec.daily_quality import _aggregate
        config = self.config()
        root, output = Path(directory) / "mount", Path(directory) / "output"
        root.mkdir()
        (root / "checkpoints").mkdir()
        reference = {"user_id": 7, "anchor_id": "7:286", "day": 286, "target": 1,
            "reference_context_hash": "1" * 64, "serving_context_length": 2000,
            "serving_context_sha256": "2" * 64}
        row = {"user_id": 7, "anchor_id": "7:286", "day": 286,
            "reference_context_hash": "1" * 64, "target_known": True,
            "rank": 1, "rank_at_max_k": 1, "rank_cutoff": 100,
            "hr10": 1.0, "hr100": 1.0, "ndcg10": 1.0, "ndcg100": 1.0}
        metrics, _ = _aggregate([row])
        metrics["num_items"] = 10
        source = protocol._source_config(config)
        report = {"complete": True, "source_code_sha": config["reused_controls"]["source_code_sha"],
                  "config": source, "trials": {}, "preflight": {}}
        manifest = {"source_code_sha": report["source_code_sha"], "config": source,
                    "format_version": 3, "checkpoints": []}
        controls = copy.deepcopy(config["reused_controls"])
        for family, spec in controls["models"].items():
            weight = root / spec["checkpoint"]["file"]
            weight.write_bytes((family + "-finite-fixture-weights").encode())
            spec["checkpoint"].update(size_bytes=weight.stat().st_size,
                sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
            checkpoint = {"path": spec["checkpoint"]["file"],
                "sha256": spec["checkpoint"]["sha256"], "size_bytes": weight.stat().st_size,
                "best_epoch": 50, "active_old_length": 1024, "precision": "float32",
                "training_model": family, "trial": dict(protocol._FIXED_LR, name="current_lr")}
            trial = {"status": "success", "complete": True, "trial": checkpoint["trial"],
                "trial_name": "current_lr", "best_checkpoint": checkpoint, "best_epoch": 50,
                "best_validation": metrics, "best_full_validation": metrics,
                "passes_completed": 60, "old_length": 1024, "physical_batch_size": 32,
                "test_evaluations": 0, "from_scratch": True, "full_length_training_performed": False,
                "full_validation_is_frozen_1k_checkpoint": True, "resume_checkpoint_retained": True,
                "resumable_checkpoint": {"path": "nonexistent-overwritten-resume.pt"},
                "passes": [{"epoch": epoch} for epoch in range(1, 61)]}
            scores = [.1, .3, .5, .7, 1., .9]
            for epoch, score in zip(range(10, 61, 10), scores):
                trial["passes"][epoch - 1]["validation"] = {"ndcg10": score}
            for field, full in (("best_validation", False), ("full_validation", True)):
                path = root / spec[field]["file"]
                with gzip.open(path, "wt") as stream:
                    json.dump([row], stream)
                spec[field]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                trial["best_validation_rank_file" if not full else "full_validation_rank_file"] = path.name
                context = {key: reference[key] for key in ("user_id", "anchor_id", "reference_context_hash")}
                context.update(original_full_context_sha256="2" * 64, original_context_length=2000,
                    actual_context_length=2000 if full else 1152,
                    actual_context_sha256="2" * 64 if full else "3" * 64,
                    query_context_sha256="4" * 64, raw_context_sha256="5" * 64,
                    query_and_raw_tail_preserved=True, retained_position_recencies_preserved=True)
                path = root / spec[field + "_contexts"]["file"]
                with gzip.open(path, "wt") as stream:
                    json.dump([context], stream)
                spec[field + "_contexts"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["trials"][family] = {"current_lr": trial}
            manifest["checkpoints"].append(checkpoint)
        config["reused_controls"] = controls
        for field, content in (("report", report), ("manifest", manifest), ("experiment", source)):
            path = root / controls[field]["file"]
            path.write_text(json.dumps(content))
            controls[field]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return SimpleNamespace(config=config, root=root, output=output, references=[reference],
                               report=report, manifest=manifest)

    def load(self, fixture):
        from rhpr.sasrec import query_architecture_protocol as protocol
        with patch.object(protocol, "_REUSED_CONTROLS", fixture.config["reused_controls"]), \
                patch.object(protocol, "_resolve_control_root", return_value=fixture.root), \
                patch.object(protocol, "_load_reference_contexts", return_value=fixture.references):
            return protocol.load_architecture_controls(fixture.config, object(), fixture.output)

    def rewrite(self, fixture, field, document):
        spec = fixture.config["reused_controls"][field]
        path = fixture.root / spec["file"]
        path.write_text(json.dumps(document))
        spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def rewrite_gzip(self, fixture, family, field, mutate):
        spec = fixture.config["reused_controls"]["models"][family][field]
        path = fixture.root / spec["file"]
        with gzip.open(path, "rt") as stream:
            records = json.load(stream)
        mutate(records)
        with gzip.open(path, "wt") as stream:
            json.dump(records, stream)
        spec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_verified_controls_copy_only_small_validation_artifacts_and_strip_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(torch, "load", side_effect=AssertionError("must not deserialize weights")):
                controls = self.load(fixture)
            self.assertEqual(len(controls), 2)
            for family, result in controls.items():
                self.assertTrue(result["reused"])
                self.assertEqual(result["trial_name"], "ca_base" if "cross_attention" in family else "q_base")
                self.assertEqual(result["stopping_policy_replay"]["best_epoch"], 50)
                self.assertEqual(result["stopping_policy_replay"]["end_pass"], 60)
                self.assertEqual(result["best_checkpoint"]["source_code_sha"],
                                 fixture.config["reused_controls"]["source_code_sha"])
                self.assertTrue(Path(result["best_checkpoint"]["path"]).is_absolute())
                self.assertNotIn("resumable_checkpoint", result)
                self.assertFalse(result["resume_checkpoint_retained"])
                self.assertTrue((fixture.output / result["best_validation_rank_file"]).is_file())
                self.assertTrue((fixture.output / result["full_validation_rank_file"]).is_file())
            self.assertFalse(list(fixture.output.rglob("*.pt")))
            self.assertEqual(len(list(fixture.output.iterdir())), 8)

    def test_loader_rejects_wrong_source_and_semantically_changed_configuration(self):
        for field, mutation in (("report", lambda doc: doc.update(source_code_sha="0" * 40)),
                ("manifest", lambda doc: doc["config"]["model"].update(n_heads=8)),
                ("experiment", lambda doc: doc["hpo"].update(min_passes=20))):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                spec = fixture.config["reused_controls"][field]
                document = json.loads((fixture.root / spec["file"]).read_text())
                mutation(document)
                self.rewrite(fixture, field, document)
                with self.assertRaisesRegex(ValueError, "source, configuration"):
                    self.load(fixture)

    def test_loader_rejects_changed_weight_bytes_size_and_manifest_checkpoint(self):
        for mutation in ("bytes", "size", "manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                spec = fixture.config["reused_controls"]["models"]["query_ssm_m4"]["checkpoint"]
                if mutation == "bytes":
                    path = fixture.root / spec["file"]
                    path.write_bytes(b"x" * spec["size_bytes"])
                elif mutation == "size":
                    (fixture.root / spec["file"]).write_bytes(b"x")
                else:
                    document = copy.deepcopy(fixture.manifest)
                    document["checkpoints"][1]["best_epoch"] = 40
                    self.rewrite(fixture, "manifest", document)
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_loader_rejects_rank_hash_identity_support_and_forged_metric(self):
        for mutation in ("hash", "identity", "support", "metric", "aggregate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                family = "query_cross_attention_m4"
                if mutation == "hash":
                    spec = fixture.config["reused_controls"]["models"][family]["full_validation"]
                    (fixture.root / spec["file"]).write_bytes(b"wrong")
                elif mutation == "aggregate":
                    document = copy.deepcopy(fixture.report)
                    document["trials"][family]["current_lr"]["best_full_validation"]["ndcg10"] = .2
                    self.rewrite(fixture, "report", document)
                else:
                    changes = {"identity": {"anchor_id": "7:287"}, "support": {"target_known": False},
                               "metric": {"ndcg10": .2}}
                    self.rewrite_gzip(fixture, family, "full_validation", lambda rows: rows[0].update(changes[mutation]))
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_loader_rejects_non_equivalent_stopping_policy_and_missing_checks(self):
        for mutation in ("earlier_stop", "different_best", "missing_check", "missing_pass"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                document = copy.deepcopy(fixture.report)
                trial = document["trials"]["query_ssm_m4"]["current_lr"]
                if mutation == "earlier_stop":
                    trial["passes"][29]["validation"]["ndcg10"] = .2
                    trial["passes"][39]["validation"]["ndcg10"] = .1
                elif mutation == "different_best":
                    trial["passes"][59]["validation"]["ndcg10"] = 1.1
                elif mutation == "missing_check":
                    del trial["passes"][19]["validation"]
                else:
                    trial["passes"].pop(10)
                self.rewrite(fixture, "report", document)
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_loader_rejects_changed_actual_context_or_query(self):
        for changes in ({"actual_context_length": 128}, {"original_full_context_sha256": "9" * 64},
                        {"query_context_sha256": "9" * 64}, {"raw_context_sha256": "9" * 64}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                self.rewrite_gzip(fixture, "query_ssm_m4", "full_validation_contexts",
                                  lambda rows: rows[0].update(changes))
                with self.assertRaises(ValueError):
                    self.load(fixture)

    def test_mount_resolution_accepts_only_one_bounded_known_kernel(self):
        from rhpr.sasrec.query_architecture_protocol import _resolve_control_root
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = self.config()
            slug = config["reused_controls"]["kernel"].split("/")[1]
            root = base / "kernels" / "carasik" / slug
            root.mkdir(parents=True)
            (root / "checkpoint-manifest.json").write_text("{}")
            config["reused_controls"]["checkpoint_root"] = str(base / slug)
            self.assertEqual(_resolve_control_root(config, base), root)
            duplicate = base / slug
            duplicate.mkdir()
            (duplicate / "checkpoint-manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _resolve_control_root(config, base)
            config["reused_controls"]["checkpoint_root"] = "/outside/" + slug
            with self.assertRaisesRegex(ValueError, "escapes"):
                _resolve_control_root(config, base)

    def test_reference_loader_matches_original_contexts_to_prepared_validation_anchors(self):
        from rhpr.sasrec.query_architecture_protocol import _load_reference_contexts
        from rhpr.sasrec import query_history_ablation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.json.gz"
            context = {"anchor_id": "7:286", "reference_context_hash": "1" * 64,
                       "serving_context_length": 2000, "serving_context_sha256": "2" * 64}
            with gzip.open(path, "wt") as stream:
                json.dump({"validation": {"rows": [context]}}, stream)
            anchor = SimpleNamespace(anchor_id="7:286", reference_context_hash="1" * 64,
                                     user_id=7, day=286, target=1, target_timestamp=20)
            evaluation = SimpleNamespace(anchors=[anchor], summary={"train_end_timestamp": 10,
                "validation_end_timestamp": 30, "validation_anchors": 1})
            with patch.object(query_history_ablation, "resolve_history_parent", return_value=Path(directory)), \
                    patch.object(query_history_ablation, "_verified_file", return_value=path):
                rows = _load_reference_contexts(self.config(), evaluation)
                self.assertEqual(rows[0]["target"], 1)
                anchor.reference_context_hash = "9" * 64
                with self.assertRaisesRegex(ValueError, "reference"):
                    _load_reference_contexts(self.config(), evaluation)


if __name__ == "__main__":
    unittest.main()
