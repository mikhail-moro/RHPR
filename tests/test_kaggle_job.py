from __future__ import annotations

import unittest
import tempfile
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_job import copy_result_file, execute, parse_kernel_status, write_collection_manifest


class KaggleJobTest(unittest.TestCase):
    def test_parses_plain_status(self) -> None:
        self.assertEqual(parse_kernel_status('kernel has status "complete"'), "complete")

    def test_parses_namespaced_status(self) -> None:
        output = 'kernel has status "KernelWorkerStatus.ERROR"'
        self.assertEqual(parse_kernel_status(output), "error")

    def test_rejects_missing_status(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_kernel_status("unexpected output")

    def test_suite_prefers_root_metrics_over_variant_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded, destination = root / "downloaded", root / "artifacts"
            (downloaded / "runs" / "full").mkdir(parents=True)
            destination.mkdir()
            (downloaded / "metrics.json").write_text('{"pilot_complete": true}')
            (downloaded / "runs" / "full" / "metrics.json").write_text('{"variant": "full"}')
            copy_result_file(downloaded, "metrics.json", destination)
            self.assertEqual((destination / "metrics.json").read_text(), '{"pilot_complete": true}')

    def test_output_pattern_is_forwarded_without_changing_default_download(self) -> None:
        for pattern in (None, "", r".*\.(json|json\.gz|md|log|zip)$"):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "artifacts"
                commands = []

                def fake_run(command, *, check=True):
                    commands.append(command)
                    if command[1:3] == ["kernels", "output"]:
                        downloaded = output / "downloaded"
                        (downloaded / "metrics.json").write_text('{"pilot_complete": true}')
                        (downloaded / "summary.md").write_text("Completed\n")
                        (downloaded / "run-manifest.json").write_text('{"git_sha": "source"}')
                    stdout = "Kernel version 7 successfully pushed. Please check progress" if command[2] == "push" else ""
                    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

                with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                    "scripts.kaggle_job.kernel_status", return_value=("complete", "complete")
                ), patch("scripts.kaggle_job.time.sleep"):
                    execute(root / "bundle", "owner/kernel", output, 1, output_file_pattern=pattern)

                expected_command = [
                    "kaggle", "kernels", "output", "owner/kernel", "-p",
                    str(output / "downloaded"), "--force",
                ]
                if pattern:
                    expected_command.extend(["--file-pattern", pattern])
                self.assertEqual(commands[-1], expected_command)
                manifest = json.loads((output / "output-collection-manifest.json").read_text())
                self.assertEqual(manifest["filtered_download"], bool(pattern))
                self.assertEqual(manifest["output_file_pattern"], pattern or None)
                self.assertEqual(manifest["kernel_version"], 7)
                self.assertEqual(manifest["versioned_kernel_id"], "owner/kernel/7")

    def test_invalid_output_pattern_does_not_submit_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.kaggle_job.run") as run:
            with self.assertRaises(re.error):
                execute(Path(temporary), "owner/kernel", Path(temporary) / "outputs", 1,
                        output_file_pattern="[")
            run.assert_not_called()

    def test_failed_kernel_retains_partial_outputs_and_reports_missing_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []
            pattern = r".*\.(json|json\.gz|md|log|zip)$"

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    downloaded = output / "downloaded"
                    (downloaded / "trials" / "finished").mkdir(parents=True)
                    (downloaded / "trials" / "finished" / "metrics.json").write_text('{"ndcg": 0.1}')
                    (downloaded / "proxy-hpo-progress.json").write_text('{"finished_trials": 6}')
                    (downloaded / "run-manifest.json").write_text('{"git_sha": "source"}')
                    (downloaded / "checkpoint-manifest.json").write_text('{"checkpoints": [')
                    (downloaded / "kernel-slug.log").write_text("Training failed\n")
                return subprocess.CompletedProcess(command, 0, stdout="Kernel version 4 successfully pushed.", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("error", "error")
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "finished with status error"):
                    execute(root / "bundle", "owner/kernel-slug", output, 1, output_file_pattern=pattern)

            self.assertEqual(sum(command[2] == "push" for command in commands), 1)
            self.assertEqual(commands[-1][-2:], ["--file-pattern", pattern])
            self.assertFalse((output / "metrics.json").exists())
            self.assertEqual(json.loads((output / "run-manifest.json").read_text()), {"git_sha": "source"})
            self.assertEqual((output / "downloaded" / "kernel.log").read_text(), "Training failed\n")
            manifest = json.loads((output / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_manifest_error_type"], "JSONDecodeError")
            self.assertIn("proxy-hpo-progress.json", [entry["path"] for entry in manifest["downloaded_files"]])
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertFalse(report["completed_successfully"])
            self.assertEqual(report["terminal_status"], "error")
            self.assertEqual(report["kernel_version"], 4)
            self.assertEqual(report["missing_root_files"], ["metrics.json", "summary.md"])
            self.assertEqual(report["copied_root_files"], ["run-manifest.json"])
            self.assertTrue(report["kernel_log_downloaded"])

    def test_polling_failure_is_preserved_when_both_diagnostic_downloads_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []
            primary = RuntimeError("status API failed with secret-primary")

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    raise subprocess.CalledProcessError(1, command, stderr="secret-secondary")
                return subprocess.CompletedProcess(command, 0, stdout="Kernel version 2 successfully pushed.", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", side_effect=primary
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaises(RuntimeError) as caught:
                    execute(root / "bundle", "owner/kernel", output, 1)

            self.assertIs(caught.exception, primary)
            self.assertEqual(sum(command[2] == "push" for command in commands), 1)
            self.assertEqual(sum(command[2] == "output" for command in commands), 2)
            self.assertEqual(commands[-1][-2:], ["--file-pattern", r"(?!)"])
            manifest = json.loads((output / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["downloaded_file_count"], 0)
            report_text = (output / "terminal-status.json").read_text()
            self.assertNotIn("secret-", report_text)
            report = json.loads(report_text)
            self.assertEqual(report["terminal_status"], "status_poll_error")
            self.assertEqual([entry["stage"] for entry in report["collection_errors"]],
                             ["download_outputs", "download_kernel_log"])

    def test_failed_output_download_still_collects_kernel_log_and_available_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []

            def fake_run(command, *, check=True):
                commands.append(command)
                if command[2] == "output":
                    downloaded = output / "downloaded"
                    if command[-1] != r"(?!)":
                        (downloaded / "metrics.json").write_text('{"completed_trials": 5}')
                        raise subprocess.CalledProcessError(1, command)
                    (downloaded / "kernel.log").write_text("Failure details\n")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("cancelled", "cancelled")
            ), patch("scripts.kaggle_job.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "finished with status cancelled"):
                    execute(root / "bundle", "owner/kernel", output, 1, output_file_pattern=r".*\.json$")

            self.assertEqual(json.loads((output / "metrics.json").read_text()), {"completed_trials": 5})
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertEqual(report["terminal_status"], "cancelled")
            self.assertTrue(report["kernel_log_downloaded"])
            self.assertEqual(report["collection_errors"],
                             [{"stage": "download_outputs", "error_type": "CalledProcessError"}])

    def test_timeout_preserves_timeout_after_cancel_and_download_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifacts"
            commands = []

            def fake_run(command, *, check=True):
                commands.append((command, check))
                if command[2] == "cancel":
                    self.assertFalse(check)
                    raise OSError("cancel transport failed")
                if command[2] == "output":
                    raise OSError("download transport failed")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                "scripts.kaggle_job.kernel_status", return_value=("running", "running")
            ), patch("scripts.kaggle_job.time.sleep"), patch(
                "scripts.kaggle_job.time.monotonic", side_effect=[0, 61]
            ):
                with self.assertRaisesRegex(TimeoutError, "exceeded 1 minutes"):
                    execute(root / "bundle", "owner/kernel", output, 1)

            self.assertEqual(sum(command[2] == "push" for command, _ in commands), 1)
            self.assertEqual([command[2] for command, _ in commands], ["push", "cancel", "output", "output"])
            report = json.loads((output / "terminal-status.json").read_text())
            self.assertEqual(report["terminal_status"], "timeout")
            self.assertEqual(report["last_observed_kernel_status"], "running")
            self.assertEqual(report["primary_error_type"], "TimeoutError")
            self.assertEqual(report["collection_errors"][0], {"stage": "cancel", "error_type": "OSError"})

    def test_failed_submission_does_not_poll_or_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = subprocess.CalledProcessError(1, ["kaggle", "kernels", "push"])
            with patch("scripts.kaggle_job.run", side_effect=primary) as run, patch(
                "scripts.kaggle_job.kernel_status"
            ) as status:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    execute(root / "bundle", "owner/kernel", root / "outputs", 1)
            self.assertIs(caught.exception, primary)
            run.assert_called_once()
            status.assert_not_called()

    def test_success_still_requires_each_aggregate_output(self) -> None:
        for omitted in ("metrics.json", "summary.md", "run-manifest.json"):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "artifacts"

                def fake_run(command, *, check=True):
                    if command[2] == "output":
                        for name, contents in {"metrics.json": "{}", "summary.md": "Done",
                                               "run-manifest.json": "{}"}.items():
                            if name != omitted:
                                (output / "downloaded" / name).write_text(contents)
                    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

                with patch("scripts.kaggle_job.run", side_effect=fake_run), patch(
                    "scripts.kaggle_job.kernel_status", return_value=("complete", "complete")
                ), patch("scripts.kaggle_job.time.sleep"):
                    with self.assertRaisesRegex(RuntimeError, f"expected one {re.escape(omitted)}, found 0"):
                        execute(root / "bundle", "owner/kernel", output, 1)

    def test_collection_manifest_distinguishes_remote_checkpoints_from_downloaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / "downloaded"
            (downloaded / "checkpoints").mkdir(parents=True)
            (downloaded / "checkpoints" / "small.pt").write_bytes(b"weights")
            (downloaded / "checkpoint-manifest.json").write_text(json.dumps({
                "checkpoints": [
                    {"path": "checkpoints/small.pt", "size_bytes": 7, "sha256": "small-hash"},
                    {"path": "checkpoints/large.pt", "size_bytes": 536_870_912, "sha256": "large-hash"},
                ]
            }))
            write_collection_manifest(downloaded, root, "owner/kernel/3", r".*\.json$")
            manifest = json.loads((root / "output-collection-manifest.json").read_text())
            self.assertEqual(manifest["kernel_id"], "owner/kernel/3")
            self.assertTrue(manifest["checkpoint_manifest_downloaded"])
            collected, retained = manifest["checkpoints"]
            self.assertEqual(collected["storage"], "downloaded")
            self.assertEqual(collected["downloaded_size_bytes"], 7)
            self.assertEqual(retained["storage"], "retained_on_private_kaggle")
            self.assertFalse(retained["downloaded"])
            self.assertIsNone(retained["downloaded_size_bytes"])
            self.assertEqual(retained["kaggle_size_bytes"], 536_870_912)
            self.assertEqual(manifest["downloaded_file_count"], 2)
            self.assertEqual(manifest["downloaded_size_bytes"],
                             sum(path.stat().st_size for path in downloaded.rglob("*") if path.is_file()))

    def test_collection_manifest_rejects_checkpoint_paths_outside_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoint-manifest.json").write_text(json.dumps({
                "checkpoints": [{"path": "../outside.pt", "size_bytes": 10}],
            }))
            with self.assertRaisesRegex(ValueError, "within kernel outputs"):
                write_collection_manifest(root, root, "owner/kernel", None)


if __name__ == "__main__":
    unittest.main()
