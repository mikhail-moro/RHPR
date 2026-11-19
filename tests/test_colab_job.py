import io
import json
import re
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import colab_job


class ColabJobTests(unittest.TestCase):
    def test_last8_config_keeps_protocol_and_uses_one_gpu(self):
        from rhpr.gr.ml1m_devices import assignments_for, devices_for
        from rhpr.gr.ml1m_last8_pairs import validate

        original = json.loads((colab_job.ROOT / "experiments/ml1m_last8_mixed64_thinned_2t4.json").read_text())
        config = colab_job.colab_config(original, "L4")
        validate(config)
        self.assertEqual(config["training"], original["training"])
        self.assertEqual(config["frozen_parent"], original["frozen_parent"])
        self.assertEqual(original["execution"]["worker_devices"], ["cuda:0", "cuda:1"])
        with (patch("torch.cuda.device_count", return_value=1),
              patch("torch.cuda.get_device_name", return_value="NVIDIA L4") as actual):
            devices = devices_for(config)
            self.assertEqual(assignments_for(config, devices),
                             [{"slot": 0, "device": "cuda:0", "trials": original["trials"]}])
            actual.return_value = "Tesla T4"
            with self.assertRaisesRegex(RuntimeError, "exactly one L4"):
                devices_for(config)

    def test_parent_assets_are_verified_and_logs_are_not_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "notebooks" / "owner" / "model"
            payload = b"{}"
            expected = {"sha256": colab_job.hashlib.sha256(payload).hexdigest(), "size_bytes": 2}
            config = {"kaggle": {"kernel_sources": ["owner/model"]},
                      "frozen_parent": {"kernel": "owner/model", "files": {"run-manifest.json": expected}}}

            def download(args, **kwargs):
                pattern = args[args.index("--file-pattern") + 1]
                self.assertIsNotNone(re.fullmatch(pattern, "run-manifest.json"))
                self.assertIsNone(re.fullmatch(pattern, "model.log"))
                (target / "run-manifest.json").write_bytes(payload)
                (target / "model.log").write_text("Kaggle log")

            with patch.object(colab_job.subprocess, "run", side_effect=download):
                colab_job.stage_inputs(root, config, io.BytesIO())
            self.assertEqual([path.name for path in target.iterdir()], ["run-manifest.json"])
            with self.assertRaisesRegex(ValueError, "parent notebook"):
                colab_job.input_plan({"kaggle": config["kaggle"]})

    def test_dataset_staging_downloads_pinned_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"wheel bytes"
            digest = colab_job.hashlib.sha256(data).hexdigest()
            manifest = json.dumps({"files": [{"path": "wheels/dependency.whl", "sha256": digest}]}).encode()
            config = {"resources": {"path": "/kaggle/input/datasets/owner/resource",
                                    "manifest_sha256": colab_job.hashlib.sha256(manifest).hexdigest()}}
            files = {"resources-manifest.json": manifest, "wheels/dependency.whl": data}

            def download(args, **kwargs):
                name = args[args.index("-f") + 1]
                directory = Path(args[args.index("--path") + 1])
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(files[name])
                return subprocess.CompletedProcess(args, 0, stdout=b"")

            with patch.object(colab_job.subprocess, "run", side_effect=download):
                colab_job.stage_datasets(root, config, ["owner/resource"], io.BytesIO())
            self.assertEqual((root / "datasets/owner/resource/wheels/dependency.whl").read_bytes(), data)

    def test_working_directory_is_collected_and_runtime_is_released_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state.json"
            output = root / "output"
            calls = []

            def command(args, **kwargs):
                calls.append(args)
                if "new" in args:
                    state.write_text("{}")
                if args[-1].startswith("mkdir"):
                    with tarfile.open(fileobj=kwargs["stdin"], mode="r") as archive:
                        self.assertIn("job/colab_runtime.py", archive.getnames())
                        self.assertIn("kaggle/working", archive.getnames())
                if args[-1].startswith("timeout"):
                    return subprocess.CompletedProcess(args, 3)
                if args[-1].startswith("tar -C"):
                    kwargs["stdout"].write(b"partial run outputs")
                return subprocess.CompletedProcess(args, 0)

            with (patch.object(colab_job.subprocess, "run", side_effect=command),
                  patch.object(colab_job.subprocess, "check_output", return_value="sha")):
                with self.assertRaises(subprocess.CalledProcessError):
                    colab_job.run_job("experiments/baseline.json", "cpu", 1, output, state, "rhpr-test")
            self.assertEqual((output / "results.tar").read_bytes(), b"partial run outputs")
            self.assertIn("stop", calls[-1])
            self.assertNotIn("--gpu", next(args for args in calls if "new" in args))
