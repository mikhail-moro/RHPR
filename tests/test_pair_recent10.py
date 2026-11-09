import copy, json, tempfile, unittest
from pathlib import Path
import numpy as np
from rhpr.gr.pair_recent10 import TRIALS, recent_preencode, resolve_parent, validate

ROOT = Path(__file__).resolve().parents[1]

class Recent10Test(unittest.TestCase):
    def test_padding_and_order(self):
        sid = np.arange(25*4).reshape(25,4) % 256
        contexts = np.array([list(range(20)), [-1]*17 + [2,3,4], [1,2,3] + [-1]*17, [-1]*20])
        got = recent_preencode(contexts, sid)
        self.assertEqual(got.shape, (4,40))
        offsets = np.arange(4)*256
        np.testing.assert_array_equal(got[0], (sid[10:20] + offsets).reshape(-1))
        for row in (1,2):
            np.testing.assert_array_equal(got[row,:28], -1)
            np.testing.assert_array_equal(got[row,28:], (sid[[2,3,4] if row == 1 else [1,2,3]] + offsets).reshape(-1))
        self.assertTrue((got[3] == -1).all())

    def test_matched_protocol(self):
        c = json.loads((ROOT/'experiments/ml1m_pair_recent10_2t4.json').read_text())
        validate(c)
        self.assertEqual([x['seed'] for x in TRIALS], [42,101,202])
        for key, value in [('recent_items', 11), ('kind', 'other')]:
            bad = copy.deepcopy(c); bad[key] = value
            with self.assertRaises(ValueError): validate(bad)
        for key, value in [('max_epochs', 49), ('validation_interval', 10)]:
            bad = copy.deepcopy(c); bad['training'][key] = value
            with self.assertRaises(ValueError): validate(bad)

    def test_parent_sha_required(self):
        c = json.loads((ROOT/'experiments/ml1m_pair_recent10_2t4.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError): resolve_parent(c, Path(folder))

if __name__ == '__main__': unittest.main()
