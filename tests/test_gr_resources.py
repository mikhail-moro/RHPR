"""Fail closed on changed inputs and archive escapes before running GR code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.gr import grid
from rhpr.gr.resources import unpack_grid, verify_resources
from rhpr.gr.runtime import install_gr_runtime
from rhpr.offline_runtime import install_offline_runtime
from scripts.prepare_gr_resources import build


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> str:
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return digest(data)


class ResourceIntegrityTests(unittest.TestCase):
    def make_bundle(self, root: Path, name: str = "data/catalog.json") -> tuple[dict, str]:
        payload = root / name
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"catalog A")
        manifest = {"files": [{"path": name, "size_bytes": 9, "sha256": digest(b"catalog A")} ]}
        return manifest, write_json(root / "resources-manifest.json", manifest)

    def test_same_size_payload_tamper_and_manifest_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, checksum = self.make_bundle(root)
            self.assertEqual(verify_resources(root, checksum), manifest)
            (root / "data/catalog.json").write_bytes(b"catalog B")
            with self.assertRaisesRegex(ValueError, "input digest mismatch"):
                verify_resources(root, checksum)
            write_json(root / "resources-manifest.json", {"files": []})
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_resources(root, checksum)

    def test_manifest_paths_cannot_escape_or_alias_via_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            outside = Path(temporary) / "outside"
            outside.write_bytes(b"catalog A")
            (root / "alias").symlink_to(outside)
            for name in ("../outside", str(outside), "alias", "..\\outside", "."):
                with self.subTest(name=name):
                    manifest = {"files": [{"path": name, "size_bytes": 9, "sha256": digest(b"catalog A")} ]}
                    checksum = write_json(root / "resources-manifest.json", manifest)
                    with self.assertRaisesRegex(ValueError, "unsafe resource path"):
                        verify_resources(root, checksum)

    def test_duplicate_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            manifest["files"] *= 2
            checksum = write_json(root / "resources-manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "unsafe resource path"):
                verify_resources(root, checksum)

    def test_zip_rejects_escape_before_writing_any_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"  # Kaggle input keeps this ZIP compressed.
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("safe.py", b"pass\n")
                archive.writestr("../escaped.py", b"do not write")
            target = root / "destination"
            with self.assertRaisesRegex(ValueError, "unsafe GRID archive entry"):
                unpack_grid(source, target)
            self.assertEqual(list(target.iterdir()), [])
            self.assertFalse((root / "escaped.py").exists())

    def test_zip_rejects_symlinks_and_existing_parent_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            with zipfile.ZipFile(source, "w") as archive:
                entry = zipfile.ZipInfo("link")
                entry.create_system = 3
                entry.external_attr = 0o120777 << 16
                archive.writestr(entry, "../outside")
            with self.assertRaisesRegex(ValueError, "unsafe GRID archive entry"):
                unpack_grid(source, root / "target")
            outside = root / "outside"
            outside.mkdir()
            (root / "target/link").symlink_to(outside, target_is_directory=True)
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("link/escaped.py", b"do not write")
            with self.assertRaisesRegex(ValueError, "unsafe GRID archive entry"):
                unpack_grid(source, root / "target")
            self.assertEqual(list(outside.iterdir()), [])

    def test_zip_accepts_regular_files_without_zip_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("src/", b"")
                archive.writestr("src/model.py", b"pass\n")
            target = unpack_grid(source, root / "target")
            self.assertEqual((target / "src/model.py").read_bytes(), b"pass\n")

    def test_resource_builder_cannot_write_downloads_into_checkout(self):
        checkout = Path(__file__).resolve().parents[1]
        with patch("scripts.prepare_gr_resources._verify_source") as verify:
            with self.assertRaisesRegex(ValueError, "outside the checkout"):
                build(Path("unused"), Path("unused"), Path("unused"), Path("unused"), checkout / "data")
            verify.assert_not_called()

    def test_source_preparation_preserves_original_and_defers_unused_iterator_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            interface = original / "src/data/loading/components/interfaces.py"
            interface.parent.mkdir(parents=True)
            before = (b"from typing import Any, Dict, List, Optional, Union\n"
                      b"from src.data.loading.components.iterators import RawDataIterator\n"
                      b"class Example:\n    iterator: RawDataIterator\n")
            after = (b"from __future__ import annotations\n\n"
                     b"from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union\n"
                     b"if TYPE_CHECKING:\n    from src.data.loading.components.iterators import RawDataIterator\n"
                     b"class Example:\n    iterator: RawDataIterator\n")
            interface.write_bytes(before)
            expected_fixture = root / "expected"
            expected_interface = expected_fixture / interface.relative_to(original)
            expected_interface.parent.mkdir(parents=True)
            expected_interface.write_bytes(after)
            with patch.object(grid, "GRID_SOURCE_TREE_SHA256", grid._source_digest(original)), \
                    patch.object(grid, "GRID_PREPARED_TREE_SHA256", grid._source_digest(expected_fixture)):
                prepared = grid.prepare_grid_source(original, root / "prepared")
                self.assertEqual(interface.read_bytes(), before)
                self.assertEqual((prepared / interface.relative_to(original)).read_bytes(), after)
                # This succeeds without an installed iterator module or TensorFlow.
                loaded = runpy.run_path(str(prepared / interface.relative_to(original)))
                self.assertEqual(loaded["Example"].__annotations__["iterator"], "RawDataIterator")
                self.assertEqual(grid.prepare_grid_source(original, prepared), prepared)
                (prepared / interface.relative_to(original)).write_bytes(after + b"\n# changed\n")
                with self.assertRaisesRegex(ValueError, "does not match pinned revision"):
                    grid.prepare_grid_source(original, prepared)


class GRRuntimePrecheckTests(unittest.TestCase):
    def fixture(self, root: Path, package: str = "safe-package") -> tuple[dict, dict]:
        wheel = root / "safe_package-1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel bytes")
        lock = root / "requirements-lock.txt"
        lock.write_text(f"safe-package==1.0 --hash=sha256:{digest(wheel.read_bytes())}\n")
        manifest = {
            "target_python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "expected_torch_version": "2.10.0",
            "wheels": [{"name": package, "filename": wheel.name,
                        "size_bytes": wheel.stat().st_size, "sha256": digest(wheel.read_bytes())}],
        }
        runtime = {"kind": "gr", "target_python_minor": manifest["target_python_minor"],
                   "wheel_path": str(root), "target_site": str(root / "site"),
                   "requirements_sha256": digest(lock.read_bytes()),
                   "manifest_sha256": write_json(root / "runtime-manifest.json", manifest)}
        return runtime, manifest

    def test_python_mismatch_and_loaded_torch_stop_before_pip(self):
        with patch("rhpr.gr.runtime.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "require Python"):
                install_gr_runtime({"target_python_minor": "0.0"}, Path("/unused"))
            with patch.dict(sys.modules, {"torch": object()}):
                with self.assertRaisesRegex(RuntimeError, "before importing torch"):
                    install_gr_runtime({"target_python_minor": f"{sys.version_info.major}.{sys.version_info.minor}"}, Path("/unused"))
            run.assert_not_called()

    def test_changed_lock_or_manifest_stops_before_pip(self):
        for changed in ("requirements-lock.txt", "runtime-manifest.json"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
                sys.modules.pop("torch", None)
                root = Path(temporary)
                runtime, _ = self.fixture(root)
                (root / changed).write_bytes((root / changed).read_bytes() + b" ")
                with patch("rhpr.gr.runtime.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "digest mismatch"):
                        install_gr_runtime(runtime, root)
                    run.assert_not_called()

    def test_bundled_torch_stack_and_wrong_manifest_target_stop_before_pip(self):
        for package in ("torch", "torchvision", "torchaudio", "triton", "nvidia-cublas-cu12"):
            with self.subTest(package=package), tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
                sys.modules.pop("torch", None)
                root = Path(temporary)
                runtime, _ = self.fixture(root, package)
                with patch("rhpr.gr.runtime.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "preserve the platform PyTorch"):
                        install_gr_runtime(runtime, root)
                    run.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            root = Path(temporary)
            runtime, manifest = self.fixture(root)
            manifest["target_python_minor"] = "0.0"
            runtime["manifest_sha256"] = write_json(root / "runtime-manifest.json", manifest)
            with patch("rhpr.gr.runtime.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "different Python version"):
                    install_gr_runtime(runtime, root)
                run.assert_not_called()

    def test_missing_or_wrong_size_wheel_stops_before_pip(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            sys.modules.pop("torch", None)
            root = Path(temporary)
            runtime, manifest = self.fixture(root)
            (root / manifest["wheels"][0]["filename"]).write_bytes(b"short")
            with patch("rhpr.gr.runtime.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "wrong size"):
                    install_gr_runtime(runtime, root)
                run.assert_not_called()

    def test_dispatch_uses_locked_wheels_and_reuses_cuda_torch_without_real_install(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules), patch.dict(os.environ), \
                patch.object(sys, "path", list(sys.path)):
            sys.modules.pop("torch", None)
            root = Path(temporary)
            runtime, _ = self.fixture(root)
            modules = {"torch": SimpleNamespace(__version__="2.10.0+cu128", version=SimpleNamespace(cuda="12.8"))}
            for name in ("transformers", "sentence_transformers", "lightning", "torchmetrics", "hydra"):
                modules[name] = SimpleNamespace(__version__="test")
            def pretend_install(*args, **kwargs):
                sys.modules.update(modules)
            with patch("rhpr.gr.runtime.subprocess.run", side_effect=pretend_install) as run:
                report = install_offline_runtime(runtime, root)
            command = run.call_args.args[0]
            for flag in ("--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:"):
                self.assertIn(flag, command)
            self.assertEqual(report["torch"], "2.10.0+cu128")
            self.assertTrue(report["platform_torch_reused"])
            self.assertEqual(json.loads((root / "runtime-bootstrap.json").read_text()), report)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_wrong_platform_torch_is_rejected_after_fake_install(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules), patch.dict(os.environ), \
                patch.object(sys, "path", list(sys.path)):
            sys.modules.pop("torch", None)
            root = Path(temporary)
            runtime, _ = self.fixture(root)
            def pretend_install(*args, **kwargs):
                sys.modules["torch"] = SimpleNamespace(__version__="2.9.0+cu128")
            with patch("rhpr.gr.runtime.subprocess.run", side_effect=pretend_install):
                with self.assertRaisesRegex(RuntimeError, "platform PyTorch version differs"):
                    install_gr_runtime(runtime, root)


if __name__ == "__main__":
    unittest.main()
