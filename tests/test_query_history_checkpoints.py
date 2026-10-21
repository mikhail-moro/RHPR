from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "Torch is required")
class QueryHistoryCheckpointTest(unittest.TestCase):
    def fixture(self, directory, family="qssm"):
        from rhpr.sasrec.daily_quality import _aggregate
        from rhpr.sasrec.query_history_checkpoints import _canonical
        from rhpr.sasrec.query_history_protocol import history_variant

        config = json.loads((ROOT / f"experiments/yambda_history_continuation_{family}.json").read_text())
        source = config["selected_source"]
        mount, output = Path(directory) / "mount", Path(directory) / "output"
        mount.mkdir()
        model = torch.nn.Linear(3, 2)
        trial = source["trial"]
        variant = history_variant(config)
        source_config = {"model": {"d_model": 3}, "hpo": {"min_passes": 20,
            "max_passes": 60, "patience": 2, "validation_interval": 10}}
        row = {"user_id": 7, "anchor_id": "7:286", "day": 286,
            "reference_context_hash": "1" * 64, "target_known": True,
            "rank": 1, "rank_at_max_k": 1, "rank_cutoff": 100,
            "hr10": 1., "hr100": 1., "ndcg10": 1., "ndcg100": 1.}
        metrics, _ = _aggregate([row])
        metrics["num_items"] = 11
        references = [{"user_id": 7, "anchor_id": "7:286", "day": 286, "target": 1,
            "reference_context_hash": "1" * 64, "serving_context_length": 2000,
            "serving_context_sha256": "2" * 64}]
        contexts = [{"user_id": 7, "anchor_id": "7:286", "reference_context_hash": "1" * 64,
            "original_full_context_sha256": "2" * 64, "original_context_length": 2000,
            "actual_context_length": 1152, "actual_context_sha256": "3" * 64,
            "query_context_sha256": "4" * 64, "raw_context_sha256": "5" * 64,
            "query_and_raw_tail_preserved": True, "retained_position_recencies_preserved": True}]
        best, end = (20, 40) if family == "qssm" else (60, 60)
        source["checkpoint"]["best_epoch"] = best
        checkpoint_config = dict(source_config, active_trial=copy.deepcopy(trial),
            active_stage="qssm_summary_memory_search" if family == "qssm" else "matched_baseline_screen",
            active_old_length=1024, active_pass_cap=60)
        payload = {"format_version": 3, "source_code_sha": source["source_code_sha"],
            "experiment_config": checkpoint_config, "model_config": source_config["model"],
            "variant": variant, "num_items": 11, "best_epoch": best, "selection_score": 1.,
            "state_dict": copy.deepcopy(model.state_dict())}
        weight = mount / source["checkpoint"]["file"]
        weight.parent.mkdir(parents=True)
        torch.save(payload, weight)
        source["checkpoint"].update(sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                                     size_bytes=weight.stat().st_size)
        saved = {"path": source["checkpoint"]["file"], "sha256": source["checkpoint"]["sha256"],
            "size_bytes": weight.stat().st_size, "best_epoch": best, "active_old_length": 1024,
            "precision": "float32", "training_model": source["model"], "trial": copy.deepcopy(trial)}
        result = {"status": "success" if family == "qssm" else "full_transfer_failed",
            "complete": True, "trial": copy.deepcopy(trial), "trial_name": trial["name"],
            "old_length": 1024, "from_scratch": True, "full_length_training_performed": False,
            "test_evaluations": 0, "best_epoch": best, "best_checkpoint": saved,
            "best_validation": metrics, "passes_completed": end,
            "resumable_checkpoint": {"path": "missing-optimizer-resume.pt"},
            "passes": [{"epoch": epoch} for epoch in range(1, end + 1)]}
        if family == "fuxi":
            result["failure"] = {"stage": "full_length_validation"}
        checks = (.5, 1., .9, .8) if family == "qssm" else (.1, .2, .3, .4, .5, 1.)
        for epoch, score in zip(range(10, end + 1, 10), checks):
            result["passes"][epoch - 1]["validation"] = {"ndcg10": score}
        for key, data in (("validation_ranks", [row]), ("validation_contexts", contexts)):
            pin = source["artifacts"][key]
            path = mount / pin["file"]
            with gzip.open(path, "wt") as stream:
                json.dump(data, stream)
            pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["best_validation_rank_file"] = source["artifacts"]["validation_ranks"]["file"]
        report = {"complete": True, "source_code_sha": source["source_code_sha"], "config": source_config,
            "selected_trials": {source["model"]: copy.deepcopy(trial)},
            "trials": {source["model"]: {trial["name"]: result}}}
        if family == "qssm":
            report["global_selected_configuration"] = {"variant": variant, "trial": copy.deepcopy(trial)}
        manifest = {"format_version": 3, "config": source_config,
            "source_code_sha": source["source_code_sha"], "best_checkpoint_optimizer_state_saved": False,
            "checkpoints": [copy.deepcopy(saved)], "resume_checkpoints": [{"path": "missing-optimizer-resume.pt"}]}
        report_name = "query-qssm-search.json" if family == "qssm" else "query-baseline-screen.json"
        documents = {report_name: report, "experiment.json": source_config, "checkpoint-manifest.json": manifest}
        source["experiment_config_sha256"] = _canonical(source_config)

        def persist():
            for name, value in documents.items():
                path = mount / name
                path.write_text(json.dumps(value))
                source["documents"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

        persist()
        return config, mount, output, model, payload, references, report, manifest, result, persist

    def prepare(self, fixture):
        from rhpr.sasrec.query_history_checkpoints import prepare_history_source
        config, mount, output, _, _, references, *_ = fixture
        with patch("rhpr.sasrec.query_history_protocol.validate_history_config"), \
                patch("rhpr.sasrec.query_baseline_protocol._resolve_kernel_root", return_value=mount), \
                patch("rhpr.sasrec.query_architecture_protocol._load_reference_contexts", return_value=references):
            return prepare_history_source(config, object(), output)

    def test_both_selected_families_copy_only_best_weights_and_validation_artifacts(self):
        for family in ("qssm", "fuxi"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory, family)
                metadata = self.prepare(fixture)
                config, mount, output, _, _, _, *_ = fixture
                source = config["selected_source"]
                self.assertEqual(Path(metadata["checkpoint_path"]).read_bytes(),
                                 (mount / source["checkpoint"]["file"]).read_bytes())
                self.assertFalse(Path(metadata["checkpoint_record"]["path"]).is_absolute())
                self.assertEqual(len(list((output / "checkpoints").iterdir())), 1)
                self.assertEqual(len(list((output / "source-validation").iterdir())), 2)
                self.assertFalse(metadata["provenance"]["test_ranks_read"])
                self.assertFalse(metadata["provenance"]["optimizer_state_inherited"])
                self.assertEqual(metadata["source_best_epoch"], 20 if family == "qssm" else 60)
                json.dumps(metadata)  # Run-report provenance must remain serializable.

    def test_actual_document_and_weight_digest_drift_is_rejected_before_copy(self):
        for artifact in ("experiment.json", "query-qssm-search.json", "checkpoint-manifest.json", "weight", "rank", "context"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                config, mount, output, *_ = fixture
                source = config["selected_source"]
                name = (source["checkpoint"]["file"] if artifact == "weight" else
                    source["artifacts"]["validation_ranks"]["file"] if artifact == "rank" else
                    source["artifacts"]["validation_contexts"]["file"] if artifact == "context" else artifact)
                path = mount / name
                path.write_bytes(path.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA256|size mismatch"):
                    self.prepare(fixture)
                self.assertFalse((output / "checkpoints").exists())

    def test_report_manifest_selection_and_completion_drift_is_rejected_even_when_rehashed(self):
        changes = (
            lambda r, m, t: r.update(complete=False),
            lambda r, m, t: r.update(source_code_sha="0" * 40),
            lambda r, m, t: m.update(source_code_sha="0" * 40),
            lambda r, m, t: m.update(best_checkpoint_optimizer_state_saved=True),
            lambda r, m, t: m.update(checkpoints=[]),
            lambda r, m, t: t.update(best_epoch=30),
            lambda r, m, t: t.update(trial_name="other"),
            lambda r, m, t: t.update(test_evaluations=1),
            lambda r, m, t: t.update(full_length_training_performed=True),
            lambda r, m, t: t.update(best_validation_rank_file="other.json.gz"),
            lambda r, m, t: r.update(selected_trials={}),
            lambda r, m, t: r.update(global_selected_configuration={}),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                change(fixture[6], fixture[7], fixture[8])
                fixture[9]()
                with self.assertRaises(ValueError):
                    self.prepare(fixture)

    def test_rank_support_aggregate_and_input_context_checks_use_actual_records(self):
        mutations = (
            ("validation_ranks", lambda rows: rows[0].update(anchor_id="7:287")),
            ("validation_ranks", lambda rows: rows[0].update(ndcg10=.1)),
            ("validation_ranks", lambda rows: rows.clear()),
            ("validation_contexts", lambda rows: rows[0].update(original_full_context_sha256="9" * 64)),
            ("validation_contexts", lambda rows: rows[0].update(actual_context_length=2000)),
        )
        for artifact, change in mutations:
            with self.subTest(artifact=artifact, change=change), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                config, mount, *_ = fixture
                pin = config["selected_source"]["artifacts"][artifact]
                path = mount / pin["file"]
                with gzip.open(path, "rt") as stream:
                    rows = json.load(stream)
                change(rows)
                with gzip.open(path, "wt") as stream:
                    json.dump(rows, stream)
                pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.prepare(fixture)

    def test_failed_fuxi_training_is_rejected_while_optional_transfer_failure_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory, "fuxi")
            fixture[8]["failure"]["stage"] = "training"
            fixture[9]()
            with self.assertRaisesRegex(ValueError, "optional full-history"):
                self.prepare(fixture)

    def test_two_models_receive_identical_trainable_weights_without_inheriting_optimizer_or_rng(self):
        from rhpr.sasrec.query_history_checkpoints import load_history_source_weights
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            metadata = self.prepare(fixture)
            models = [torch.nn.Linear(3, 2).train(), torch.nn.Linear(3, 2).eval()]
            rng_before = torch.get_rng_state().clone()
            for model in models:
                proof = load_history_source_weights(metadata, model, num_items=11)
                self.assertTrue(proof["model_parameters_loaded_strictly"])
                self.assertFalse(proof["optimizer_state_inherited"])
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                self.assertEqual(torch.optim.AdamW(model.parameters()).state, {})
                for key, value in fixture[3].state_dict().items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value))
            self.assertTrue(models[0].training)
            self.assertFalse(models[1].training)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

    def test_payload_metadata_model_shape_nonfinite_and_resume_state_are_rejected(self):
        from rhpr.sasrec.query_history_checkpoints import load_history_source_weights
        changes = (
            lambda p: p.update(source_code_sha="0" * 40),
            lambda p: p.update(variant={}),
            lambda p: p.update(num_items=True),
            lambda p: p.update(best_epoch=30),
            lambda p: p.update(selection_score=float("nan")),
            lambda p: p.update(optimizer_state_dict={}),
            lambda p: p.update(torch_rng_state=torch.get_rng_state()),
            lambda p: p["experiment_config"].update(active_old_length=2048),
            lambda p: p["state_dict"].pop("bias"),
            lambda p: p["state_dict"].update(weight=torch.zeros(3, 2)),
            lambda p: p["state_dict"].update(weight=p["state_dict"]["weight"].half()),
            lambda p: p["state_dict"]["weight"].fill_(float("nan")),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                metadata = self.prepare(fixture)
                payload = copy.deepcopy(fixture[4])
                change(payload)
                path = Path(metadata["checkpoint_path"])
                torch.save(payload, path)
                metadata["checkpoint_record"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                      size_bytes=path.stat().st_size)
                with self.assertRaises(ValueError):
                    load_history_source_weights(metadata, torch.nn.Linear(3, 2), num_items=11)

    def test_copied_checkpoint_is_rehashed_at_every_model_load(self):
        from rhpr.sasrec.query_history_checkpoints import load_history_source_weights
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            metadata = self.prepare(fixture)
            with Path(metadata["checkpoint_path"]).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                load_history_source_weights(metadata, torch.nn.Linear(3, 2), num_items=11)


if __name__ == "__main__":
    unittest.main()
