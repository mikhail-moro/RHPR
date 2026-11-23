from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rhpr.gr.sid import SidIndex
from rhpr.gr.yambda_window_tiger import reassign_overflow


class YambdaSidOverflowTest(unittest.TestCase):
    def test_excess_uses_closest_available_third_center_and_keeps_other_assignments(self):
        codes = np.array([[0, 0, 0]] * 4 + [[0, 0, 1], [2, 1, 0]])
        vectors = np.tile([1., 0.], (len(codes), 1))
        centers = np.zeros((3, 3, 2))
        centers[2] = [[1., 0.], [-1., 0.], [0., 1.]]
        repaired = reassign_overflow(codes, vectors, centers, width=3)
        expect = codes.copy()
        expect[3, 2] = 2
        np.testing.assert_array_equal(repaired, expect)
        np.testing.assert_array_equal(reassign_overflow(repaired, vectors, centers, width=3), expect)
        self.assertEqual(SidIndex(repaired, width=3).diagnostics()['max_collision_group'], 3)

    def test_full_second_prefix_spills_without_displacing_other_second_prefixes(self):
        codes = np.array([[1, 0, 0]] * 5 + [[1, 1, 1]] * 3)
        vectors = np.tile([1., 0.], (len(codes), 1))
        centers = np.zeros((3, 2, 2))
        centers[2] = [[1., 0.], [0., 1.]]
        repaired = reassign_overflow(codes, vectors, centers, width=2)
        np.testing.assert_array_equal(repaired, [[1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1],
                                                 [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]])
        self.assertEqual(SidIndex(repaired, width=2).diagnostics()['resolved_unique_ids'], len(codes))
