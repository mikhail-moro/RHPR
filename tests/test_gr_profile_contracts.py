"""Precision, stopping cadence, and immutable parent inputs for the GR profile."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rhpr.gr.data import UserSequence
from rhpr.gr.experiment import _autocast, _train_generator
from rhpr.gr.resources import verify_parent_artifacts


_HAS_TORCH = importlib.util.find_spec("torch") is not None


class ParentArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.sha = "a" * 40

    def write_parent(self, directory: Path, source_sha: str | None = None):
        directory.mkdir(parents=True)
        (directory / "run-manifest.json").write_text(json.dumps({
            "git_sha": self.sha if source_sha is None else source_sha,
            "config": {"seed": 42},
        }))
        (directory / "item-embeddings.npy").write_bytes(b"fixture-embedding-bytes")
        return {
            "kernel": "owner/notebook",
            "source_code_sha": self.sha,
            "files": {path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            } for path in directory.iterdir()},
        }

    def test_known_mount_layouts_and_versioned_namespace_resolve_exact_parent(self):
        layouts = ["notebook", "owner/notebook", "notebooks/owner/notebook",
                   "kernels/owner/notebook", "custom/owner/notebook/2"]
        for layout in layouts:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                input_root = Path(tmp)
                directory = input_root / layout
                parent = self.write_parent(directory)
                self.assertEqual(verify_parent_artifacts(parent, input_root), directory)

    def test_explicit_local_fixture_requires_the_same_registered_bytes(self):
        directory = self.root / "local-parent"
        parent = self.write_parent(directory)
        parent["path"] = str(directory)
        self.assertEqual(verify_parent_artifacts(parent, self.inputs), directory)

    def test_known_mount_does_not_scan_unrelated_remote_inputs(self):
        directory = self.inputs / 'notebooks/owner/notebook'
        parent = self.write_parent(directory)
        with patch.object(Path,'rglob',side_effect=AssertionError('unnecessary remote traversal')):
            self.assertEqual(verify_parent_artifacts(parent,self.inputs),directory)

    def test_equal_size_byte_substitution_is_rejected_by_hash(self):
        directory = self.inputs / "notebook"
        parent = self.write_parent(directory)
        path = directory / "item-embeddings.npy"
        before = path.read_bytes()
        path.write_bytes(bytes([before[0] ^ 1]) + before[1:])
        self.assertEqual(path.stat().st_size, parent["files"][path.name]["size_bytes"])
        with self.assertRaisesRegex(ValueError, "differs from the registered run"):
            verify_parent_artifacts(parent, self.inputs)

    def test_artifact_size_and_missing_file_fail_before_use(self):
        for mutation in ("truncate", "remove"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                input_root = Path(tmp)
                directory = input_root / "notebook"
                parent = self.write_parent(directory)
                path = directory / "item-embeddings.npy"
                if mutation == "truncate":
                    path.write_bytes(b"short")
                    error = ValueError
                else:
                    path.unlink()
                    error = FileNotFoundError
                with self.assertRaises(error):
                    verify_parent_artifacts(parent, input_root)

    def test_matching_file_hash_does_not_override_source_revision_guard(self):
        directory = self.inputs / "notebook"
        # Even deliberately re-pinning a valid manifest's bytes cannot turn
        # a checkpoint from another code revision into the registered parent.
        parent = self.write_parent(directory, source_sha="b" * 40)
        with self.assertRaisesRegex(ValueError, "source revision differs"):
            verify_parent_artifacts(parent, self.inputs)

    def test_provenance_manifest_and_safe_artifact_names_are_required(self):
        parent = self.write_parent(self.inputs / "notebook")
        missing_manifest = deepcopy(parent)
        missing_manifest["files"].pop("run-manifest.json")
        with self.assertRaisesRegex(ValueError, "provenance manifest must be pinned"):
            verify_parent_artifacts(missing_manifest, self.inputs)
        for name in ("../outside.npy", "/outside.npy", "nested/file.npy", "bad\\file.npy", ""):
            with self.subTest(name=name):
                malformed = deepcopy(parent)
                malformed["files"][name] = {"size_bytes": 1, "sha256": "c" * 64}
                with self.assertRaisesRegex(ValueError, "safe basenames"):
                    verify_parent_artifacts(malformed, self.inputs)
        for kernel in ("notebook", "owner/notebook/2", "../notebook", "owner/..", "/notebook"):
            with self.subTest(kernel=kernel):
                malformed = dict(parent, kernel=kernel)
                with self.assertRaisesRegex(ValueError, "owner/slug form"):
                    verify_parent_artifacts(malformed, self.inputs)


@unittest.skipUnless(_HAS_TORCH, "PyTorch is required for GR precision and training contracts")
class PrecisionAndStoppingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np = np
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def tiny_model(self):
        torch = self.torch

        class TinyGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.signal = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.1, -0.4]))
                self.decoder = torch.nn.Module()
                self.decoder.decoder_mlp = torch.nn.ModuleList([
                    torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(4, 2, bias=False),
                ])

            def forward(self, input_ids, future_ids, **kwargs):
                return self.signal[None, None, :].expand(input_ids.shape[0], future_ids.shape[1] + 1, -1)

        torch.manual_seed(42)
        return TinyGenerator()

    def fixtures(self):
        from rhpr.gr.sid import SidIndex
        users = [UserSequence(0, (0, 1, 0, 1), 0, 1),
                 UserSequence(1, (1, 0, 1, 0), 1, 0)]
        index = SidIndex(self.np.array([[0], [1]], dtype="int64"), width=2)
        config = {"seed": 42, "data": {"history_items": 2}, "training": {
            "learning_rate": 0.01, "weight_decay": 0.0,
            "precision": "fp32", "evaluation_precision": "fp32", "amp": False,
            "batch_size": 4, "physical_batch_size": 2, "max_epochs": 14,
            "min_epochs": 1, "max_seconds": 1000,
            "grad_clip_norm": 1.0, "validation_interval": 2,
            "evaluation_batch_size": 2, "patience": 2,
        }}
        return users, index, config

    def test_explicit_cpu_precision_preserves_fp32_and_allows_bf16(self):
        torch = self.torch
        matrix = torch.ones((4, 4), dtype=torch.float32)
        with _autocast("fp32", "cpu"):
            self.assertEqual((matrix @ matrix).dtype, torch.float32)
        with _autocast("bf16", "cpu"):
            self.assertEqual((matrix @ matrix).dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            _autocast("fp16", "cpu")
        with self.assertRaisesRegex(ValueError, "must be fp32, fp16, or bf16"):
            _autocast("float64", "cpu")

    def test_native_bf16_is_rejected_on_t4_before_a_cuda_context_is_entered(self):
        with patch.object(self.torch.cuda, "get_device_capability", return_value=(7, 5)):
            with self.assertRaisesRegex(ValueError, "native BF16"):
                _autocast("bf16", "cuda:0")

    def test_invalid_training_or_evaluation_precision_cannot_take_an_update(self):
        for field in ("precision", "evaluation_precision"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                users, index, config = self.fixtures()
                config["training"][field] = "fp16"
                model = self.tiny_model()
                before = {key: value.detach().clone() for key, value in model.state_dict().items()}
                with self.assertRaisesRegex(ValueError, "requires CUDA"):
                    _train_generator(model, users, index, config, Path(tmp), "cpu")
                for key, value in model.state_dict().items():
                    self.assertTrue(self.torch.equal(before[key], value), key)
                self.assertFalse((Path(tmp) / "generator-last.pt").exists())

    def run_stopping_fixture(self, cadence, patience_epochs=None):
        users, index, config = self.fixtures()
        config["training"]["validation_interval"] = cadence
        if patience_epochs is not None:
            config["training"]["patience_epochs"] = patience_epochs
        model = self.tiny_model()
        selected_state = {}
        checks = []

        def validation(active_model, examples, sid_index, batch_size, device, precision):
            self.assertEqual(precision, "fp32")
            checks.append(len(checks) + 1)
            if len(checks) == 1:
                selected_state.update({key: value.detach().clone() for key, value in active_model.state_dict().items()})
            return {"ndcg@10": 1.0 if len(checks) == 1 else 0.5}, None, None

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch("rhpr.gr.experiment._evaluate", side_effect=validation), patch("rhpr.gr.experiment._event"):
                values = _train_generator(model, users, index, config, output, "cpu")
            curves = json.loads((output / "training-curves.json").read_text())
            last = self.torch.load(output / "generator-last.pt", map_location="cpu", weights_only=False)
            for key, value in model.state_dict().items():
                self.assertTrue(self.torch.equal(selected_state[key], value), key)
            self.assertTrue(any(not self.torch.equal(selected_state[key], value)
                                for key, value in last["model"].items()))
            self.assertEqual(last["best_epoch"], 1)
            self.assertEqual(last["best_metric"], 1.0)
        return values, curves

    def test_epoch_patience_waits_for_a_validation_boundary_without_multiplying_checks(self):
        for cadence, expected_stop in ((2, 6), (5, 5)):
            with self.subTest(cadence=cadence):
                values, curves = self.run_stopping_fixture(cadence, patience_epochs=4)
                self.assertEqual(values["training_epochs"], expected_stop)
                self.assertEqual(values["best_epoch"], 1)
                self.assertEqual(values["stopping_reason"], "early_stopping")
                self.assertEqual(values["optimizer_updates"], expected_stop * 2)
                self.assertEqual(values["training_precision"], "fp32")
                self.assertEqual(values["evaluation_precision"], "fp32")
                self.assertIn("validation", curves[-1])
                self.assertGreaterEqual(expected_stop - 1, 4)
                self.assertLess(expected_stop - 1, 4 + cadence)

    def test_legacy_check_based_patience_is_still_supported(self):
        values, curves = self.run_stopping_fixture(cadence=2)
        self.assertEqual(values["training_epochs"], 4)
        self.assertEqual([row["epoch"] for row in curves if "validation" in row], [1, 2, 4])

    def test_rqvae_prefix_epochs_cover_all_targets_and_validate_only_at_10_and_20(self):
        from rhpr.gr.rqvae_experiment import _train_rqvae_generator
        users, index, config = self.fixtures()
        config["training"].update(sampling="all_train_prefixes", max_epochs=20,
            min_epochs=20, validation_interval=10, validate_first_epoch=False,
            patience_epochs=20)
        model = self.tiny_model()
        selected_state = {}
        checks = []

        def validation(active_model, examples, sid_index, batch_size, device, precision):
            checks.append(len(checks) + 1)
            if len(checks) == 1:
                selected_state.update({key: value.detach().clone() for key, value in active_model.state_dict().items()})
            return {"ndcg@10": 1.0 if len(checks) == 1 else 0.5}, None, None

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch("rhpr.gr.experiment._evaluate", side_effect=validation), patch("rhpr.gr.experiment._event"):
                values = _train_rqvae_generator(model, users, index, config, output, "cpu")
            curves = json.loads((output / "training-curves.json").read_text())
            last = self.torch.load(output / "generator-last.pt", map_location="cpu", weights_only=False)
            stream = json.loads((output / "training-stream.json").read_text())
            self.assertEqual([row["epoch"] for row in curves if "validation" in row], [10, 20])
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(row["train_examples"] == 6 for row in curves))
            self.assertEqual(values["epochs"], 20)
            self.assertEqual(values["updates"], 40)
            self.assertEqual(values["model_examples_consumed"], 120)
            self.assertEqual(values["selected_epoch"], 10)
            self.assertEqual(values["validation_checks"], 2)
            self.assertEqual(values["stop_reason"], "epoch_limit")
            self.assertEqual(last["epoch"], 20)
            self.assertEqual(stream["examples_per_epoch"], 6)
            self.assertAlmostEqual(values["training_seconds"], sum(row["train_seconds"] for row in curves))
            for key, value in model.state_dict().items():
                self.assertTrue(self.torch.equal(selected_state[key], value), key)


if __name__ == "__main__":
    unittest.main()
