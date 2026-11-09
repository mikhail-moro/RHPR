import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from rhpr.gr.pair_sid import NITEMS
from rhpr.gr.pair_test import TRIALS,cache_plan,validate

class PairTestProtocol(unittest.TestCase):
    def test_test_pair_cache_keeps_old_pairs_and_detects_missing(self):
        h=np.array([list(range(20))]); needed=np.array([i*NITEMS+i+1 for i in range(0,10,2)])
        old=np.array([needed[0], needed[2], 100*NITEMS+101]); expected_missing=needed[[1,3,4]]
        observed,missing,merged=cache_plan(old,h)
        np.testing.assert_array_equal(observed,needed)
        np.testing.assert_array_equal(missing,expected_missing)
        np.testing.assert_array_equal(merged,np.union1d(old,needed))
        with self.assertRaises(ValueError):cache_plan(old[::-1],h)

    def test_protocol_requires_only_fixed_checkpoints(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_test_2t4.json').read_text());validate(c)
        self.assertEqual(len(TRIALS),18)
        for key,value in [('test_training_forbidden',False),('selection_policy','use_test'),('seeds',[42])]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)

if __name__=='__main__':unittest.main()
