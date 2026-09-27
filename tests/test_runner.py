from __future__ import annotations

import ast
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.runner import execute  # noqa: E402
from scripts.build_kaggle_bundle import build  # noqa: E402
from scripts.notify_telegram import build_message  # noqa: E402


class RunnerTest(unittest.TestCase):
    def test_baseline_is_deterministic_and_writes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            result_a = execute(ROOT / "experiments" / "baseline.json", Path(first))
            result_b = execute(ROOT / "experiments" / "baseline.json", Path(second))
            self.assertEqual(result_a["metrics"], result_b["metrics"])
            for name in ("metrics.json", "summary.md", "run-manifest.json"):
                self.assertTrue((Path(first) / name).is_file())

    def test_bundle_is_private_and_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            with patch.dict(os.environ, {"GITHUB_SHA": "test-revision"}):
                metadata_path = build(
                    ROOT / "experiments" / "baseline.json",
                    "owner/rhpr-experiments",
                    "gpu",
                    output,
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["title"], "Rhpr Experiments")
            self.assertFalse((output / "rhpr").exists())
            self.assertFalse((output / "experiment.json").exists())

            runner = (output / "run.py").read_text(encoding="utf-8")
            assignments = {
                node.targets[0].id: ast.literal_eval(node.value)
                for node in ast.parse(runner).body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id
                in {"EMBEDDED_PACKAGE", "EXPERIMENT_CONFIG", "GIT_SHA"}
            }
            self.assertEqual(json.loads(assignments["EXPERIMENT_CONFIG"])["seed"], 42)
            self.assertEqual(assignments["GIT_SHA"], "test-revision")
            package = base64.b64decode(assignments["EMBEDDED_PACKAGE"])
            with zipfile.ZipFile(io.BytesIO(package)) as archive:
                self.assertIn("rhpr/runner.py", archive.namelist())

            working = Path(directory) / "working"
            local_runner = output / "local-run.py"
            local_runner.write_text(
                runner.replace('Path("/kaggle/working")', f"Path({str(working)!r})"),
                encoding="utf-8",
            )
            subprocess.run([sys.executable, str(local_runner)], check=True, capture_output=True)
            for name in ("metrics.json", "summary.md", "run-manifest.json"):
                self.assertTrue((working / name).is_file())

    def test_telegram_message_includes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.json"
            metrics.write_text('{"accuracy": 0.9}', encoding="utf-8")
            message = build_message("success", "baseline", "https://example.test/run", metrics)
            self.assertIn("accuracy: 0.9", message)
            self.assertIn("https://example.test/run", message)


if __name__ == "__main__":
    unittest.main()
