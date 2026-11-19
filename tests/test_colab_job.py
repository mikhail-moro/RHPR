import io
from contextlib import redirect_stdout
import json
import re
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import colab_job, colab_runtime


class ColabJobTests(unittest.TestCase):
    def test_runtime_relocates_gpu_paths_and_finds_pinned_parent(self):
        from rhpr.gr.pair_test import locate_parent

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, kaggle = root / "job", root / "kaggle"
            job.mkdir()
            config = {"resources": {"path": "/kaggle/input/datasets/owner/resources",
                                    "scratch_path": "/kaggle/temp/trial", "manifest_sha256": "pinned"},
                      "runtime": {"wheel_path": "/kaggle/input/datasets/owner/resources/wheels",
                                  "target_site": "/kaggle/temp/site"}}
            (job / "experiment.json").write_text(json.dumps(config))
            effective, path = colab_runtime.mount_config(job, kaggle)
            self.assertEqual(effective, json.loads(path.read_text()))
            self.assertEqual(effective["resources"]["path"], str(kaggle / "input/datasets/owner/resources"))
            self.assertEqual(effective["resources"]["scratch_path"], str(kaggle / "temp/trial"))
            self.assertEqual(effective["runtime"]["target_site"], str(kaggle / "temp/site"))
            self.assertEqual(effective["resources"]["manifest_sha256"], "pinned")
            self.assertEqual(json.loads((job / "experiment.json").read_text()), config)

            parent = kaggle / "input/notebooks/owner/model"
            parent.mkdir(parents=True)
            manifest = parent / "run-manifest.json"
            manifest.write_text('{"git_sha":"revision"}')
            spec = {"kernel": "owner/model", "git_sha": "revision", "files": {
                "run-manifest.json": {"size_bytes": manifest.stat().st_size, "sha256": colab_job.sha256(manifest)}}}
            with patch.dict("os.environ", {"RHPR_INPUT_ROOT": str(kaggle / "input")}):
                self.assertEqual(locate_parent(spec), parent)

    def test_colab_failure_reports_public_phase_without_service_details(self):
        raw = b"Forbidden (HTTP 403): private-url\n"
        process = subprocess.CompletedProcess(["colab"], 255, stderr=raw)
        public, private = io.StringIO(), io.BytesIO()
        with (patch.object(colab_job.subprocess, "run", return_value=process), redirect_stdout(public),
              self.assertRaises(subprocess.CalledProcessError)):
            colab_job.run_colab_phase(["colab"], "upload", private, timeout=1)
        self.assertEqual(private.getvalue(), raw)
        self.assertIn("upload: exit 255, undetermined, HTTP 403", public.getvalue())
        self.assertNotIn("private-url", public.getvalue())

    def test_allocation_failure_reports_only_known_backend_diagnostics(self):
        stderr = (b"Colab allocation returned HTTP 503; retrying in 60s\n"
                  b"Colab allocation exception: ValidationError\nTraceback: private-credential\n")
        process = subprocess.CompletedProcess(["colab"], 1, stderr=stderr)
        public, private = io.StringIO(), io.BytesIO()
        with (patch.object(colab_job.subprocess, "run", return_value=process), redirect_stdout(public),
              self.assertRaises(subprocess.CalledProcessError)):
            colab_job.run_colab_phase(["colab"], "allocation", private, timeout=1)
        self.assertEqual(public.getvalue(),
                         "Colab allocation finished (retry count 1, exit 1)\n"
                         "Colab phase allocation: exit 1, invalid-backend-response\n")
        self.assertEqual(private.getvalue(), stderr)

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

    def test_depth_trials_can_be_deployed_individually_on_colab_gpus(self):
        from rhpr.gr.ml1m_devices import devices_for
        from rhpr.gr.ml1m_last8_pairs import single_colab_depth_trial, validate

        for depth in (4, 8):
            original = json.loads((colab_job.ROOT / f"experiments/ml1m_last8_depth{depth}_capacity_colab.json").read_text())
            for trial in original["trials"]:
                for gpu in colab_job.GPU_TYPES:
                    config = colab_job.colab_config(original, gpu, trial["id"])
                    validate(config)
                    self.assertTrue(single_colab_depth_trial(config))
                    self.assertEqual(config["trials"], [trial])
                    self.assertEqual(config["training"], original["training"])
                    self.assertEqual(config["frozen_parent"], original["frozen_parent"])
                    with (patch("torch.cuda.device_count", return_value=1),
                          patch("torch.cuda.get_device_name", return_value=f"NVIDIA {gpu}-test")):
                        self.assertEqual(devices_for(config), ["cuda:0"])
                    config["execution"].pop("colab_gpu")
                    with self.assertRaisesRegex(ValueError, "invalid depth-by-history"):
                        validate(config)
            with self.assertRaisesRegex(ValueError, "registered depth pilot trial ID"):
                colab_job.colab_config(original, "T4", "unknown-trial")

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
                return subprocess.CompletedProcess(args, 0, stdout=b"")

            with patch.object(colab_job.subprocess, "run", side_effect=download):
                colab_job.stage_inputs(root, config, io.BytesIO())
            self.assertEqual([path.name for path in target.iterdir()], ["run-manifest.json"])
            with self.assertRaisesRegex(ValueError, "parent notebook"):
                colab_job.input_plan({"kaggle": config["kaggle"]})

    def test_kaggle_staging_retries_transient_errors_and_logs_only_status(self):
        args = ["kaggle"]
        rate_limit = b"429 Client Error: private location\n"
        denied = b"HTTP 403: private location\n"
        results = [subprocess.CompletedProcess(args, 1, stdout=message) for message in (rate_limit, denied)]
        public, private = io.StringIO(), io.BytesIO()
        with (patch.object(colab_job.subprocess, "run", side_effect=results),
              patch.object(colab_job.time, "sleep") as sleep, redirect_stdout(public),
              self.assertRaises(subprocess.CalledProcessError)):
            colab_job.download_kaggle_input(args, "dataset", private)
        sleep.assert_called_once_with(5)
        self.assertEqual(private.getvalue(), rate_limit + denied)
        self.assertEqual(public.getvalue(), "Kaggle dataset download failed: exit 1, HTTP 403\n")

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
                    self.assertEqual(args[-1], f"mkdir -p {colab_job.REMOTE} && tar -xf - -C {colab_job.REMOTE}")
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

    def test_latest_colab_checkpoint_survives_remote_disconnection(self):
        from rhpr.gr.ml1m_last8_pairs import snapshot_colab_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trial, remote, output = (root / name for name in ("trial", "remote", "output"))
            for path in (trial, remote, output):
                path.mkdir()
            for name in ("generator-last.pt", "generator-best.pt", "training-curves.json"):
                (trial / name).write_text(name)
            snapshot_colab_checkpoint(trial, remote, 10)
            self.assertEqual((remote / "checkpoint-epoch").read_text(), "10\n")
            probes = iter([b"-1\n10\n", *([None] * 5)])

            def transfer(args, phase, log, **kwargs):
                if phase == "execution-status":
                    status = next(probes)
                    return subprocess.CompletedProcess(args, 0 if status else 255, stdout=status)
                if phase == "checkpoint-download":
                    kwargs["stdout"].write((remote / "checkpoint.tar").read_bytes())
                return subprocess.CompletedProcess(args, 0)

            with (patch.object(colab_job, "run_colab_phase", side_effect=transfer),
                  patch.object(colab_job.time, "sleep"),
                  self.assertRaisesRegex(RuntimeError, "Lost contact")):
                colab_job.run_checkpointed_training(["colab"], "/remote", 5, output, io.BytesIO())
            self.assertFalse((output / "checkpoint.partial").exists())
            with tarfile.open(output / "checkpoint.tar") as archive:
                self.assertEqual(set(archive.getnames()), {"generator-best.pt", "generator-last.pt", "training-curves.json"})
