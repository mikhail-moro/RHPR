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
