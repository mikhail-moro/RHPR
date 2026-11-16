import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.history_windows import encode_window, prepare_cases, validate


class HistoryWindowsTests(unittest.TestCase):
    def test_window_changes_only_visible_history_and_preserves_suffix(self):
        sid = np.column_stack([np.arange(64), np.ones((64, 2), dtype=int), np.arange(64) % 7])
        h = np.full((31, 30), -1, dtype=np.int32)
        for n in range(1, 31): h[n, -n:] = np.arange(n)
        for w in (10, 20, 30):
            x = encode_window(h, sid, w)
            self.assertEqual(x.shape, (31, 4*w))
            for n, row in enumerate(x):
                values = row[row >= 0].reshape(-1, 4)-np.arange(4)*256
                np.testing.assert_array_equal(values, sid[max(0, n-w):n])
        for bad_width in (0, 15, 32, 64):
            with self.assertRaises(ValueError): encode_window(h, sid, bad_width)
        bad = h.copy(); bad[10, 0] = 0
        with self.assertRaises(ValueError): encode_window(bad, sid, 10)

    def test_all_targets_and_temporal_holdouts_are_shared_across_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); source = p/'source'; source.mkdir()
            (source/'catalog.json').write_text(json.dumps([dict(item_id=i, text=str(i)) for i in range(64)]))
            (source/'sequences.json').write_text(json.dumps([dict(user_id=1, train=list(range(30)), validation=30, test=31)]))
            _, report = prepare_cases(source, dict(users=1, items=64, train_examples=29), p/'cases')
            self.assertEqual(report['train_examples'], 29)
            sid = np.column_stack([np.arange(64), np.zeros((64, 3), dtype=int)])
            for split, expected_targets in [('train', list(range(1, 30))), ('validation', [30]), ('test', [31])]:
                with np.load(p/'cases'/(split+'-cases.npz')) as z:
                    self.assertEqual(z['targets'].tolist(), expected_targets)
                    for width in (10, 20, 30):
                        x = encode_window(z['contexts'], sid, width)
                        for row, target in zip(x, expected_targets):
                            actual = (row[row >= 0].reshape(-1, 4)-np.arange(4)*256)[:, 0].tolist()
                            self.assertEqual(actual, list(range(max(0, target-width), target)))
            seq = [dict(user_id=1, train=list(range(29)), validation=29, test=30)]
            (source/'sequences.json').write_text(json.dumps(seq))
            with self.assertRaises(ValueError): prepare_cases(source, {}, p/'short')

    def test_only_three_authorized_fits_and_previous_training_policy(self):
        c = json.loads(Path('experiments/steam_history_windows_2t4.json').read_text()); validate(c)
        self.assertEqual({v['history_items'] for v in c['trials']}, {10, 20, 30})
        self.assertEqual(len(c['trials']), 3)
        for field, value in [('max_epochs', 30), ('evaluation_precision', 'fp16')]:
            bad = copy.deepcopy(c); bad['training'][field] = value
            with self.assertRaises(ValueError): validate(bad)
        bad = copy.deepcopy(c); bad['cohorts']['kindle2014'] = {}
        with self.assertRaises(ValueError): validate(bad)
