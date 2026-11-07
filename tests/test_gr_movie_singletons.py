import json,tempfile,unittest,copy
from pathlib import Path
import numpy as np
from rhpr.gr.movie_singleton import singleton_rows
from rhpr.gr.movie_data import load_temporal_users,temporal_examples

class SingletonTest(unittest.TestCase):
    def fixture(self):
        s=[dict(user_id=7,train=[[10,1],[11,1],[12,2],[13,2],[14,3],[15,4],[16,4]],validation=[17,5],test=[18,6])]
        e=[dict(user_id=7,movie_id=19,timestamp=6,split='test')]
        return s,e
    def test_latest_singletons_and_full_strictly_past_test_history(self):
        seq,emb=self.fixture();r,excluded,a=singleton_rows(seq,emb)
        self.assertEqual(excluded,[]);self.assertEqual(r[0]['validation'],[14,3]);self.assertEqual(r[0]['test'],[17,5]);self.assertEqual(a[0]['discarded_after_test'],2)
        self.assertEqual(r[0]['train'],[[10,1],[11,1],[12,2],[13,2]])
        self.assertEqual(r[0]['test_between'],[[15,4],[16,4]])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'seq.json';p.write_text(json.dumps(r));u=load_temporal_users(p,list(range(10,20)))
            h,uid,t=temporal_examples(u,20,'test');np.testing.assert_array_equal(h[0,-7:],[0,1,2,3,4,5,6]);self.assertEqual(t[0],7)
            h,uid,t=temporal_examples(u,20,'validation');np.testing.assert_array_equal(h[0,-4:],[0,1,2,3]);self.assertEqual(t[0],4)
            h,uid,t=temporal_examples(u,20,'train');self.assertEqual(len(t),2)
            self.assertTrue(all(set(row[row>=0])=={0,1} for row in h));self.assertEqual(set(t),{2,3})
    def test_source_permutation_does_not_change_split(self):
        s,e=self.fixture();r=singleton_rows(s,e);s[0]['train'].reverse();self.assertEqual(r,singleton_rows(s,e))
    def test_missing_singletons_and_short_train_are_excluded(self):
        s=[dict(user_id=1,train=[[1,1]],validation=[2,2],test=[3,3])]
        r,e,a=singleton_rows(s,[]);self.assertEqual(r,[]);self.assertEqual(e[0]['reason'],'fewer_than_two_training_groups')
        s=[dict(user_id=1,train=[[1,1],[2,1]],validation=[3,2],test=[4,2])]
        self.assertEqual(singleton_rows(s,[])[1][0]['reason'],'fewer_than_two_singleton_groups')
    def test_intervening_future_and_duplicate_movies_rejected(self):
        s,e=self.fixture();r=singleton_rows(s,e)[0]
        for change in ['future','duplicate']:
            bad=copy.deepcopy(r);bad[0]['test_between']=[[15,5]] if change=='future' else [[10,4]]
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'seq.json';p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_temporal_users(p,list(range(10,20)))

    def test_three_seed_protocol_and_balanced_worker_plan(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/movielens100k_singleton_mtl_3seeds_2t4.json').read_text())
        validate(c);jobs=movie_assignments(c,2)
        self.assertEqual([len(j['trials'])for j in jobs],[6,6])
        for badkey in ['users','train_examples']:
            b=copy.deepcopy(c);b['data'][badkey]+=1
            with self.assertRaises(ValueError):validate(b)
