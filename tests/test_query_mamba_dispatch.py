from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class QueryMambaDispatchTest(unittest.TestCase):
    def test_requires_output_before_heavy_import(self):
        from rhpr.experiment import run_experiment

        with self.assertRaisesRegex(ValueError, "output_dir is required"):
            run_experiment({"kind": "yambda_query_mamba_ablation"})

    def test_dispatches_to_registered_mamba_runner(self):
        from rhpr.experiment import run_experiment

        config = {"kind": "yambda_query_mamba_ablation"}
        output = Path("mamba-output")
        runner = Mock(return_value={"quality_complete": False})
        module = Mock(run_query_mamba_ablation=runner)
        with patch.dict(sys.modules, {"rhpr.sasrec.query_mamba_ablation": module}):
            self.assertEqual(run_experiment(config, output), {"quality_complete": False})
        runner.assert_called_once_with(config, output)


if __name__ == "__main__":
    unittest.main()
