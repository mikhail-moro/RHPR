from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.metrics import validate_metrics  # noqa: E402


class MetricsTest(unittest.TestCase):
    def test_accepts_flat_finite_scalars(self) -> None:
        metrics = {"accuracy": 0.91, "samples": 10, "name": "baseline", "ok": True}
        self.assertEqual(validate_metrics(metrics), metrics)

    def test_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            validate_metrics({"loss": math.nan})

    def test_rejects_nested_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_metrics({"folds": [0.1, 0.2]})


if __name__ == "__main__":
    unittest.main()
