from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class QueryRotatingDispatchTest(unittest.TestCase):
    def test_output_is_required_before_heavy_runner_import(self):
        from rhpr.experiment import run_experiment

        with self.assertRaisesRegex(ValueError, "output_dir is required"):
            run_experiment({"kind": "yambda_query_rotating_windows"})

    def test_registered_kind_dispatches_config_and_output(self):
        from rhpr.experiment import run_experiment

        config = {"kind": "yambda_query_rotating_windows"}
        output = Path("rotating-output")
        run = Mock(return_value={"quality_complete": False})
        module = Mock(run_query_rotating_windows=run)
        with patch.dict(sys.modules, {"rhpr.sasrec.query_rotating_windows": module}):
            self.assertEqual(run_experiment(config, output), {"quality_complete": False})
        run.assert_called_once_with(config, output)


if __name__ == "__main__":
    unittest.main()
