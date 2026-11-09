import unittest
import numpy as np
from rhpr.gr.pair_sid import NITEMS, collect_pairs, pairs_for_context, preencode

class PairInputs(unittest.TestCase):
    def test_pairing_and_unpaired_old_item(self):
        item_sids=np.stack([np.arange(4)+i for i in range(30)]).astype(np.int64)
        h=np.array([[-1]*1+list(range(19)),list(range(20))],dtype=np.int64)
        pairs=collect_pairs(h); keys=np.array(sorted(pairs),dtype=np.int64)
        codes=np.tile(np.array([[10,11,12]]),(len(keys),1))
        result=preencode(h,item_sids,keys,codes)
        self.assertEqual(result.shape,(2,56))
        self.assertEqual(len(pairs_for_context(h[0])),4)
        self.assertEqual(int((result[0]>=0).sum()),56)  # 19*4 - 4*(8-3)
        self.assertEqual(int((result[1]>=0).sum()),55)
        self.assertEqual(list(result[1,1:4]),[1034,1291,1548])
        self.assertEqual(list(result[1,-4:]),[19,276,533,790])
        plain=preencode(h,item_sids)
        self.assertEqual(int((plain[0]>=0).sum()),76)
        self.assertEqual(int((plain[1]>=0).sum()),80)
    def test_order_and_missing_pair(self):
        r=np.array([[-1]*16+[1,2,3,4],[-1]*16+[2,1,3,4]])
        self.assertEqual(collect_pairs(r),{NITEMS+2,2*NITEMS+1})
        with self.assertRaises(ValueError):preencode(r,np.zeros((8,4),dtype='i'),np.array([NITEMS+2]),np.zeros((1,3),dtype='i'))

if __name__=='__main__':unittest.main()

class RecoveryProtocol(unittest.TestCase):
    def test_recovery_runs_only_missing_seed_and_keeps_original_schedule(self):
        import copy,json
        from pathlib import Path
        from rhpr.gr.pair_sid_recovery import validate
        path=Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_sid_recovery_2t4.json'
        c=json.loads(path.read_text());validate(c)
        bad=copy.deepcopy(c);bad['trials'].append(dict(id='pair_101',mode='pair',seed=101))
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['training']['max_epochs']=20
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['recovery_input']['source_commit']='different'
        with self.assertRaises(ValueError):validate(bad)
