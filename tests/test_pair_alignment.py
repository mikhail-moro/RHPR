import copy,json,unittest
from pathlib import Path
import numpy as np
from rhpr.gr.pair_alignment import VARIANTS,segments,encode_layout,recover22,validate
from rhpr.gr.pair_sid import preencode

class PairAlignmentTests(unittest.TestCase):
    def test_each_variant_covers_exactly_the_original_items_in_order(self):
        for variant in VARIANTS:
            length=int(variant[:2]);parts=segments(np.arange(length),variant)
            expanded=[x for p in parts for x in p[1:]]
            self.assertEqual(expanded,list(range(length)),variant)
            old=length//2
            self.assertEqual([p[0] for p in parts].count('p'),{'20_reference':5,'20_edges':4,'18_first':4,'18_last':4,'22_first':5,'22_last':5}[variant])
            if variant.endswith('_first') or variant=='20_edges':self.assertEqual(parts[0],('i',0))
            if variant.endswith('_last') or variant=='20_edges':self.assertIn(('i',old-1),parts)

    def test_baseline_encoding_exactly_matches_previous_preencode(self):
        nitems=30;sids=np.arange(nitems*4).reshape(nitems,4)%256;h=np.arange(20).reshape(1,20);ks=np.array([i*2874+i+1 for i in range(0,10,2)]);codes=np.array([[i,i+1,i+2] for i in range(5)])
        np.testing.assert_array_equal(encode_layout(h,'20_reference',sids,ks,codes),preencode(h,sids,ks,codes))

    def test_twenty_edges_preserves_four_pairs_after_one_step(self):
        previous=segments(np.arange(20),'20_reference')
        shifted=segments(np.arange(1,21),'20_edges')
        previous_pairs=[part for part in previous if part[0]=='p']
        shifted_pairs=[part for part in shifted if part[0]=='p']
        self.assertEqual(shifted_pairs,previous_pairs[1:])

    def test_recover22_requires_unique_and_verified_overlap(self):
        history=np.arange(2,22);older=np.arange(0,20);train=(np.array([older]),np.array([8]),np.array([20]));validation=(history[None,:],np.array([8]),np.array([22]));rows,h22=recover22(train,validation)
        np.testing.assert_array_equal(rows,[0]);np.testing.assert_array_equal(h22[0],np.arange(22))
        bad=(train[0],np.array([9]),train[2]);self.assertEqual(len(recover22(bad,validation)[0]),0)

    def test_recovery_does_not_assume_training_target_is_adjacent(self):
        # The target may belong to a tied timestamp group and skip an event.
        history=np.arange(9,29);train=(np.arange(20)[None,:],np.array([8]),np.array([20]))
        validation=(history[None,:],np.array([8]),np.array([29]))
        self.assertEqual(len(recover22(train,validation)[0]),0)

    def test_validation_only(self):
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_alignment_2t4.json').read_text());validate(c)
        for key,value in [('evaluation_split','test'),('training_forbidden',False),('variants',['20_reference'])]:
            d=copy.deepcopy(c);d[key]=value
            with self.assertRaises(ValueError):validate(d)

if __name__=='__main__':unittest.main()
