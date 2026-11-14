import itertools
import unittest
import numpy as np
from rhpr.gr.kindle_cohort import core_mask, candidate_users, prefix_cases
from rhpr.gr.kindle_windows import encode_window, WINDOWS


class LongCohortTest(unittest.TestCase):
    def test_core_is_the_maximal_feasible_subgraph_not_one_pass(self):
        u=np.array([0,0,1,1,2,2,3]);i=np.array([0,1,0,1,1,2,2])
        selected,steps=core_mask(u,i,min_events=2,min_item_users=2)
        # Exhaust all retained vertex choices as an independent definition.
        feasible=[]
        for us in itertools.product([False,True],repeat=4):
            for bs in itertools.product([False,True],repeat=3):
                mask=np.array([us[a]and bs[b]for a,b in zip(u,i)])
                if any(sum(mask&(u==a))<2 for a in set(u[mask])):continue
                if any(len(set(u[mask&(i==b)]))<2 for b in set(i[mask])):continue
                feasible.append(mask)
        np.testing.assert_array_equal(selected,np.logical_or.reduce(feasible))
        np.testing.assert_array_equal(selected,[True,True,True,True,False,False,False])
        self.assertGreater(steps,1)
        permutation=np.array([6,4,0,3,2,5,1]);changed,_=core_mask(u[permutation],i[permutation],min_events=2,min_item_users=2)
        np.testing.assert_array_equal(changed,selected[permutation])

    def test_repeated_events_do_not_count_as_multiple_item_users(self):
        keep,_=core_mask(np.array([0,0,0]),np.array([1,1,1]),min_events=2,min_item_users=2)
        self.assertFalse(keep.any())

    def test_title_filter_then_core_cannot_resurrect_pruned_vertices(self):
        u=np.array([0,0,1,1,2,2,3]);i=np.array([0,1,0,1,1,2,2]);initial,_=core_mask(u,i,min_events=2,min_item_users=2)
        title=i!=0
        direct,_=core_mask(u[title],i[title],min_events=2,min_item_users=2)
        partial=initial&title;after,_=core_mask(u[partial],i[partial],min_events=2,min_item_users=2)
        self.assertEqual(int(direct.sum()),int(after.sum()))

    def test_whole_user_sampling_does_not_trim_histories(self):
        u=np.repeat(np.arange(4),[66,80,100,120]);a=candidate_users(u,event_budget=175,seed=42)
        np.testing.assert_array_equal(a,candidate_users(u,event_budget=175,seed=42))
        for uid in range(4):self.assertIn(int(a[u==uid].sum()),[0,int((u==uid).sum())])

    def test_prefix_targets_and_window_cut_are_causal_and_matched(self):
        c=prefix_cases([np.arange(66),np.arange(100,170)])
        train,val,test=(c[s]for s in ('train','validation','test'))
        np.testing.assert_array_equal(val[2],[64,168]);np.testing.assert_array_equal(test[2],[65,169])
        np.testing.assert_array_equal(val[0][0],np.arange(64));np.testing.assert_array_equal(test[0][0],np.arange(1,65))
        self.assertEqual(len(train[2]),63+67)
        self.assertTrue(set(train[2]).isdisjoint(set(val[2])|set(test[2])))
        sids=np.column_stack([np.arange(170)%256,np.zeros((170,3),dtype=int)])
        for w in WINDOWS:
            x=encode_window(val[0],sids,w)
            self.assertEqual(x.shape,(2,w*4))
            expected=(sids[val[0][:,-w:]]+np.arange(4)*256).reshape(2,-1)
            np.testing.assert_array_equal(x,expected)
            early=encode_window(train[0][:1],sids,w)
            self.assertEqual(int((early>=0).sum()),4)
            np.testing.assert_array_equal(early[0,-4:],sids[0]+np.arange(4)*256)

    def test_registered_config_and_reject_protocol_drift(self):
        import copy,json
        from pathlib import Path
        from rhpr.gr.kindle_windows import validate
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/kindle_long_windows_2t4.json').read_text());validate(c)
        for key,value in [('learning_rate',.001),('max_epochs',120),('evaluation_batch_size',256)]:
            changed=copy.deepcopy(c);changed['training'][key]=value
            with self.assertRaises(ValueError):validate(changed)
        changed=copy.deepcopy(c);changed['evaluation']['test_after_selection']=True
        with self.assertRaises(ValueError):validate(changed)
