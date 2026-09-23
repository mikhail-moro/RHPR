from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


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
            metadata_path = build(
                ROOT / "experiments" / "baseline.json",
                "owner/rhpr-experiments",
                "gpu",
                output,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertTrue((output / "rhpr" / "runner.py").is_file())

    def test_telegram_message_includes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.json"
            metrics.write_text('{"accuracy": 0.9}', encoding="utf-8")
            message = build_message("success", "baseline", "https://example.test/run", metrics)
            self.assertIn("accuracy: 0.9", message)
            self.assertIn("https://example.test/run", message)


if __name__ == "__main__":
    unittest.main()
