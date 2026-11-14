import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_prefix_mask import mask_old,validate


class PrefixMaskTest(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((Path(__file__).resolve().parents[1]/'experiments/kindle_prefix_mask_2t4.json').read_text())

    def test_only_two_frozen_models_eight_inferences_no_training_parameters(self):
        validate(self.c)
        self.assertNotIn('learning_rate',self.c['training'])
        self.assertNotIn('max_epochs',self.c['training'])
        for field,value in [('selected_epoch',60),('training_forbidden',False),('variants',['full','masked_old','random_prefix'])]:
            c=copy.deepcopy(self.c);c['evaluation'][field]=value
            with self.assertRaises(ValueError):validate(c)

    def test_mask_preserves_tail_positions_shape_and_original_array(self):
        for name,width,prefix in [('pair32',96,32),('recent32',128,64)]:
            x=np.arange(2*width,dtype=np.int32).reshape(2,width);before=x.copy()
            masked=mask_old(x,name)
            np.testing.assert_array_equal(x,before)
            np.testing.assert_array_equal(masked[:,prefix:],before[:,prefix:])
            self.assertTrue(np.all(masked[:,:prefix]==-1))
            self.assertEqual(masked.shape,x.shape)
            self.assertTrue(np.all((masked>=0).sum(1)==64))
        for x,name in [(np.zeros((1,95)), 'pair32'),(np.full((1,96),-1),'pair32'),(np.zeros((1,96)),'pair64')]:
            with self.assertRaises(ValueError):mask_old(x,name)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch')
    def test_real_batch_mask_hides_only_prefix_and_keeps_user_tail_target(self):
        from rhpr.gr.pair_unique4 import InputIndex4
        from rhpr.gr.sid import SidIndex
        wrapped=InputIndex4(SidIndex(np.array([[1,2,3],[4,5,6]]),width=256))
        for model,width,prefix in [('pair32',96,32),('recent32',128,64)]:
            x=np.zeros((1,width),dtype=np.int32);x[:,prefix:]=np.tile(np.arange(4)*256,16)
            if model=='pair32':x[:,:prefix]=np.tile((np.arange(4)+4)*256,8)
            full=wrapped.model_batch(x,np.array([7]),np.array([1]),'cpu')
            masked=wrapped.model_batch(mask_old(x,model),np.array([7]),np.array([1]),'cpu')
            self.assertEqual(int(masked['attention_mask_encoder'].sum()),64)
            for key in ('user_id','future_ids'):np.testing.assert_array_equal(masked[key],full[key])
            np.testing.assert_array_equal(masked['input_ids'][:,prefix:],full['input_ids'][:,prefix:])
            self.assertTrue(np.all(masked['attention_mask_encoder'][:,:prefix].numpy()==0))


if __name__=='__main__':unittest.main()
