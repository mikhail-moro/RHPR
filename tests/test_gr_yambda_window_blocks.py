from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from rhpr.gr.yambda_window_tiger import split_training_blocks


class YambdaBlockSplitTest(unittest.TestCase):
    def test_split_keeps_every_target_and_its_exact_causal_prefix(self):
        contexts = np.array([[-1, -1, 5, 6], [9, 10, 11, 12]], dtype=np.int32)
        users = np.array([10, 11])
        targets = np.arange(100, 116).reshape(2, 8)
        new_contexts, new_users, new_targets = split_training_blocks((contexts, users, targets), 4)

        self.assertEqual(new_contexts.shape, (4, 8))
        np.testing.assert_array_equal(new_users, [10, 10, 11, 11])
        np.testing.assert_array_equal(new_targets.reshape(2, 8), targets)
        for row in range(2):
            for position in range(8):
                block = 2 * row + position // 4
                old_prefix = np.concatenate((contexts[row], targets[row, :position]))
                new_prefix = np.concatenate((new_contexts[block], new_targets[block, :position % 4]))
                np.testing.assert_array_equal(new_prefix[new_prefix >= 0], old_prefix[old_prefix >= 0])
