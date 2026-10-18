from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.kaggle_job import execute
from scripts.write_job_summary import main as write_job_summary


class KaggleSubmitOnlyTests(unittest.TestCase):
    def bundle(self, root: Path, kernel_id: str = "owner/kernel") -> Path:
        bundle = root / "bundle"
        bundle.mkdir()
        (bundle / "kernel-metadata.json").write_text(json.dumps({"id": kernel_id, "is_private": True}))
        (bundle / "run.py").write_text("print('training starts on Kaggle')\n")
        return bundle

    def test_submission_pins_bundle_and_returns_without_collecting_or_claiming_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, output = self.bundle(root), root / "artifacts"
            pushed = subprocess.CompletedProcess([], 0, stdout="Kernel version 17 successfully pushed.\n", stderr="")
            with patch("scripts.kaggle_job.run", return_value=pushed) as run, patch(
                "scripts.kaggle_job.kernel_status"
            ) as status, patch("scripts.kaggle_job.time.sleep") as sleep, patch(
                "scripts.kaggle_job.download_outputs"
            ) as download, patch.dict("os.environ", {"GITHUB_SHA": "a" * 40}):
                execute(bundle, "owner/kernel", output, 720, submit_only=True)

            run.assert_called_once_with(["kaggle", "kernels", "push", "-p", str(bundle)])
            status.assert_not_called()
            sleep.assert_not_called()
            download.assert_not_called()
            self.assertEqual([path.name for path in output.iterdir()], ["submission.json"])
            report = json.loads((output / "submission.json").read_text())
            self.assertEqual(report["status"], "submitted")
            self.assertIs(report["quality_complete"], False)
            self.assertEqual(report["kernel_id"], "owner/kernel")
            self.assertEqual(report["kernel_version"], 17)
            self.assertEqual(report["versioned_kernel_id"], "owner/kernel/17")
            self.assertEqual(report["source_code_sha"], "a" * 40)
            self.assertEqual(report["kernel_metadata_sha256"], hashlib.sha256(
                (bundle / "kernel-metadata.json").read_bytes()).hexdigest())
            self.assertEqual(report["runner_sha256"], hashlib.sha256((bundle / "run.py").read_bytes()).hexdigest())
            self.assertNotIn("metrics", report)
            self.assertNotIn("completed_successfully", report)

    def test_unknown_version_and_source_are_explicitly_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, output = self.bundle(root), root / "artifacts"
            pushed = subprocess.CompletedProcess([], 0, stdout="Submission accepted.\n", stderr="")
            with patch("scripts.kaggle_job.run", return_value=pushed), patch.dict(
                "os.environ", {"GITHUB_SHA": "not-a-commit"}
            ):
                execute(bundle, "owner/kernel", output, 1, submit_only=True)
            report = json.loads((output / "submission.json").read_text())
            self.assertIsNone(report["kernel_version"])
            self.assertIsNone(report["versioned_kernel_id"])
            self.assertIsNone(report["source_code_sha"])

    def test_failed_push_has_no_submission_or_completed_result_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, output = self.bundle(root), root / "artifacts"
            error = subprocess.CalledProcessError(1, ["kaggle", "kernels", "push"])
            with patch("scripts.kaggle_job.run", side_effect=error) as run, patch(
                "scripts.kaggle_job.kernel_status"
            ) as status:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    execute(bundle, "owner/kernel", output, 1, submit_only=True)
            self.assertIs(caught.exception, error)
            run.assert_called_once()
            status.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])

    def test_mismatched_bundle_id_is_rejected_before_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.bundle(root, "different/kernel")
            with patch("scripts.kaggle_job.run") as run:
                with self.assertRaisesRegex(ValueError, "kernel id does not match"):
                    execute(bundle, "owner/kernel", root / "artifacts", 1, submit_only=True)
            run.assert_not_called()

    def test_workflow_summary_distinguishes_submission_from_training_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission, summary, metrics = root / "submission.json", root / "summary.md", root / "metrics.json"
            submission.write_text(json.dumps({
                "status": "submitted", "quality_complete": False,
                "kernel_id": "owner/kernel", "kernel_version": 17, "source_code_sha": "a" * 40,
            }))
            # A stale result must not be presented as quality from this submission.
            metrics.write_text('{"old_quality": 1.0}')
            with patch("sys.argv", ["write_job_summary.py", "--metrics", str(metrics), "--summary", str(summary),
                                    "--experiment", "experiments/long.json", "--submission", str(submission)]):
                write_job_summary()
            text = summary.read_text()
            self.assertIn("Submitted to Kaggle", text)
            self.assertIn("quality evaluation are pending", text)
            self.assertIn("did not wait for completion or collect results", text)
            self.assertIn("owner/kernel", text)
            self.assertIn("17", text)
            self.assertNotIn("old_quality", text)
            self.assertNotIn("No `metrics.json` was produced", text)


if __name__ == "__main__":
    unittest.main()
