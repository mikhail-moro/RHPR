from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rhpr.gr.yambda_ntp import single_target_cases


class YambdaNtpCasesTest(unittest.TestCase):
    def test_single_target_examples_see_only_their_own_predecessors(self):
        cases = (
            np.array([[-1, 2, 3], [7, 8, 9]], dtype=np.int32),
            np.array([11, 12]),
            np.array([[4, 5, 6], [10, 13, 14]], dtype=np.int32),
        )
        contexts, users, targets = single_target_cases(cases, 5)
        np.testing.assert_array_equal(contexts, [
            [-1, -1, -1, 2, 3], [-1, -1, 2, 3, 4], [-1, 2, 3, 4, 5],
            [-1, -1, 7, 8, 9], [-1, 7, 8, 9, 10], [7, 8, 9, 10, 13],
        ])
        np.testing.assert_array_equal(users, [11, 11, 11, 12, 12, 12])
        np.testing.assert_array_equal(targets, [4, 5, 6, 10, 13, 14])
