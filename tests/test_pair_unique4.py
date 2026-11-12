import copy,json,unittest,importlib.util
from pathlib import Path
import numpy as np
from rhpr.gr.pair_unique4 import TRIALS,validate,unique_pair_sids,pair_preencode
class UniquePairTest(unittest.TestCase):
 def test_suffix_unique_and_no_future_reassignment(self):
  keys=np.array([1,2,3,4]);codes=np.array([[5,6,7],[5,6,7],[5,6,7],[1,2,3]])
  a=unique_pair_sids(keys[:2],codes[:2],keys[:1],keys[:2]);b=unique_pair_sids(keys,codes,keys[:1],keys[:2]);np.testing.assert_array_equal(a,b[:2]);self.assertEqual(len(np.unique(b,axis=0)),4);self.assertEqual(b[:,3].tolist(),[0,1,2,0])
 def test_suffix_overflow_fails(self):
  with self.assertRaises(ValueError):unique_pair_sids(np.arange(257),np.zeros((257,3),dtype=int),np.arange(257),np.arange(257))
 def test_20_items_have_five_four_token_pairs_and_ten_items(self):
  sids=np.zeros((20,4),dtype=int);h=np.arange(20).reshape(1,20);keys=np.array([i*2874+i+1 for i in range(0,10,2)]);p=np.arange(20).reshape(5,4);out=pair_preencode(h,sids,keys,p)
  self.assertEqual(out.shape,(1,60));self.assertTrue((out>=0).all());np.testing.assert_array_equal(out[0,:4],p[0]+4*256+np.arange(4)*256)
 def test_protocol(self):
  c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_pair_unique4_2t4.json').read_text());validate(c);self.assertEqual(len(TRIALS),9)
  d=copy.deepcopy(c);d['suffix_policy']='untrusted';self.assertRaises(ValueError,validate,d)
  d=copy.deepcopy(c);d['parents']['recent']=d['parents'].pop('test_cache');self.assertRaises(ValueError,validate,d)
 @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
 def test_fourth_pair_level_has_its_own_embedding(self):
  import torch
  from types import SimpleNamespace
  from rhpr.gr.pair_unique4 import _encode4
  item=torch.nn.Embedding(1024,1);pair=torch.nn.Embedding(1024,1)
  with torch.no_grad():item.weight[:,0]=torch.arange(1024);pair.weight[:,0]=torch.arange(1024)
  model=SimpleNamespace(pair_input_embedding=pair,user_embedding=None,get_embedding_table=lambda **_:item,encoder=lambda sequence_embedding,attention_mask:sequence_embedding)
  encoded,_=_encode4(model,torch.ones((1,2),dtype=torch.long),torch.tensor([[3,1792+17]]),None)
  self.assertEqual(encoded.flatten().tolist(),[3.,785.])
if __name__=='__main__':unittest.main()
