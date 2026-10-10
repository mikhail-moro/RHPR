from __future__ import annotations

import unittest

from scripts.kaggle_job import parse_kernel_status


class KaggleJobTest(unittest.TestCase):
    def test_parses_plain_status(self) -> None:
        self.assertEqual(parse_kernel_status('kernel has status "complete"'), "complete")

    def test_parses_namespaced_status(self) -> None:
        output = 'kernel has status "KernelWorkerStatus.ERROR"'
        self.assertEqual(parse_kernel_status(output), "error")

    def test_rejects_missing_status(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_kernel_status("unexpected output")


if __name__ == "__main__":
    unittest.main()
