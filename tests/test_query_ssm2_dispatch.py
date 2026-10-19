from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class QuerySSM2DispatchTest(unittest.TestCase):
    def test_output_required_before_heavy_runner_import(self):
        from rhpr.experiment import run_experiment

        with self.assertRaisesRegex(ValueError, "output_dir is required"):
            run_experiment({"kind": "yambda_query_ssm2_benchmark"})

    def test_config_and_output_are_forwarded(self):
        from rhpr.experiment import run_experiment

        config = {"kind": "yambda_query_ssm2_benchmark"}
        output = Path("ssm2-output")
        run = Mock(return_value={"benchmark_complete": True})
        with patch.dict(sys.modules, {"rhpr.sasrec.query_ssm2_benchmark":
                                     Mock(run_query_ssm2_benchmark=run)}):
            self.assertEqual(run_experiment(config, output), {"benchmark_complete": True})
        run.assert_called_once_with(config, output)


if __name__ == "__main__":
    unittest.main()
