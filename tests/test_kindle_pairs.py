import copy
import json
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_pairs import TRIALS, check_parent, collect_keys, encode, validate
from rhpr.gr.pair_unique4 import unique_pair_sids


class KindlePairsTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]
        self.c=json.loads((root/'experiments/kindle_long_pairs_2t4.json').read_text())
        self.parent=json.loads((root/'experiments/kindle_capacity_2t4.json').read_text())

    def test_two_requested_fits_and_read_only_selected_control_tests(self):
        validate(self.c);check_parent(self.c,self.parent)
        self.assertEqual(self.c['trials'],TRIALS)
        self.assertEqual(self.c['control_tests'],[20,32])
        for width in (20,32):
            self.assertIn(f'trials/recent{width}_wide_42/generator-best.pt',self.c['parent']['files'])
        for group,key,value in [('training','learning_rate',.001),('generator','d_model',64),('data','items',100)]:
            changed=copy.deepcopy(self.c);changed[group][key]=value
            with self.assertRaises(ValueError):check_parent(changed,self.parent)
        changed=copy.deepcopy(self.c);changed['control_tests'].append(64)
        with self.assertRaises(ValueError):validate(changed)

    def test_pairing_every_short_and_full_length_against_literal_event_walk(self):
        nitems=70
        items=np.column_stack([np.arange(nitems)%256,np.zeros((nitems,3),dtype=int)])
        for width in (32,64):
            contexts=np.full((width+1,64),-1,dtype=np.int32)
            for n in range(width+1):
                if n:contexts[n,-n:]=np.arange(n)
            keys=collect_keys(contexts,width,nitems)
            codes=np.column_stack([np.arange(len(keys)),np.zeros((len(keys),2),dtype=int)])
            pairs=unique_pair_sids(keys,codes,keys,keys)
            actual=encode(contexts,items,keys,pairs,width)
            self.assertEqual(actual.shape,(width+1,3*width))
            for n in range(width+1):
                expected=[];j=0
                while j+1<n//2:
                    k=j*nitems+j+1;pos=np.searchsorted(keys,k)
                    expected.extend((pairs[pos]+np.arange(4)*256+1024).tolist());j+=2
                while j<n:
                    expected.extend((items[j]+np.arange(4)*256).tolist());j+=1
                np.testing.assert_array_equal(actual[n][actual[n]>=0],expected)
                self.assertEqual(np.sum(actual[n]<0),3*width-len(expected))
            # Full-window suffix stays SINGLE; a pair uses all FOUR codes.
            np.testing.assert_array_equal(actual[-1,-4:],items[width-1]+np.arange(4)*256)

    def test_window_crop_order_and_no_pairs_across_padding(self):
        h=np.full((3,64),-1,dtype=int)
        h[0,-3:]=[4,5,6];h[1,-4:]=[6,5,4,3];h[2]=np.arange(64)
        keys=collect_keys(h,32,100)
        self.assertNotIn(4*100+5,keys) # length3: older half has one unpaired item
        self.assertIn(6*100+5,keys)
        self.assertIn(32*100+33,keys)
        self.assertNotIn(0*100+1,keys)
        self.assertNotIn(5*100+6,keys)

    def test_eval_only_pairs_do_not_rewrite_training_or_validation_suffixes(self):
        keys=np.array([2,4,6,8]);codes=np.zeros((4,3),dtype=int)
        train=np.array([4,8]);old=np.array([2,4,8])
        full=unique_pair_sids(keys,codes,train,old)
        np.testing.assert_array_equal(full[:,3],[2,0,3,1])
        partial=unique_pair_sids(old,np.zeros((3,3),dtype=int),train,old)
        np.testing.assert_array_equal(full[np.searchsorted(keys,old)],partial)

    def test_missing_pair_is_an_error_not_silently_dropped(self):
        h=np.full((1,64),-1,dtype=int);h[0,-4:]=[0,1,2,3]
        with self.assertRaises(ValueError):
            encode(h,np.zeros((4,4),dtype=int),np.array([],dtype=int),np.zeros((0,4),dtype=int),32)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
    def test_test_reads_only_declared_val_checkpoint_and_saves_predictions(self):
        import tempfile
        from unittest.mock import Mock, patch
        import torch
        from rhpr.gr.kindle_pairs import test_selected
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'best.pt'
            torch.save(dict(model={'fixture':torch.tensor([1])},epoch=55),checkpoint)
            model=Mock();examples=(np.zeros((2,4)),np.array([1,2]),np.array([3,4]))
            with patch('rhpr.gr.kindle_pairs._evaluate',return_value=({'ndcg@10':.1},np.zeros((2,20,4)),np.zeros((2,20)))) as evaluate:
                result=test_selected(model,examples,None,root,checkpoint,55)
            self.assertEqual(result['epoch'],55)
            self.assertEqual(evaluate.call_args.args[-3:],(64,'cuda:0','fp32'))
            model.load_state_dict.assert_called_once()
            self.assertTrue((root/'test-best-predictions.npz').is_file())
            with self.assertRaises(ValueError):test_selected(model,examples,None,root,checkpoint,60)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'requires torch/HF')
    def test_worker_events_match_real_launcher_contract(self):
        import contextlib,io,tempfile
        from unittest.mock import patch
        from rhpr.gr.kindle_pairs import worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'workers').mkdir()
            plan=root/'plan.json';plan.write_text(json.dumps(dict(config=self.c,output=tmp,assignments=[dict(trials=TRIALS[:1])])))
            stream=io.StringIO();row=dict(trial=TRIALS[0]['id'],seed=42)
            with patch('torch.cuda.device_count',return_value=1),patch('torch.cuda.get_device_capability',return_value=(7,5)),patch('torch.cuda.empty_cache'),patch('torch._dynamo.reset'),patch('transformers.__version__','5.22.1'),patch('rhpr.gr.kindle_pairs.run_trial',return_value=row),contextlib.redirect_stdout(stream):
                worker(plan,0)
            events=[json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(events[0]['phase'],'kindle_pairs_trial_start')
            self.assertEqual(events[0]['trial'],TRIALS[0]['id'])
            status=json.loads((root/'workers/worker-0.json').read_text())
            self.assertEqual(status,dict(results={row['trial']:row},failures=[]))


if __name__=='__main__':unittest.main()
