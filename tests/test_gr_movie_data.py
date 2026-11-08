import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rhpr.gr.movie_data import split_ratings, load_temporal_users, temporal_examples


class MovieTimeTests(unittest.TestCase):
    def rows(self):
        return np.array([[7,10,5,1],[7,11,1,1],[7,12,4,2],[7,13,5,2],
                         [7,14,4,3],[7,15,1,3],[7,16,5,4],[7,17,2,4],
                         [8,10,3,1],[8,11,4,2]])

    def load(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sequences.json';p.write_text(json.dumps(rows))
            return load_temporal_users(p,list(range(10,18)))

    def test_no_same_timestamp_peeking_and_test_gets_only_selected_validation(self):
        rows,excluded,embargo=split_ratings(self.rows())
        self.assertEqual([e['user_id'] for e in excluded],[8])
        self.assertEqual(len(embargo),2)
        users=self.load(rows)
        h,u,y=temporal_examples(users,3,'train')
        np.testing.assert_array_equal(h,[[-1,0,1],[-1,0,1]])
        np.testing.assert_array_equal(y,[2,3])
        vh,_,vy=temporal_examples(users,3,'validation')
        th,_,ty=temporal_examples(users,3,'test')
        np.testing.assert_array_equal(vh,[[1,2,3]])
        np.testing.assert_array_equal(th,[[2,3,vy[0]]])
        self.assertNotEqual(vy[0],ty[0])
        self.assertNotIn(ty[0],th[0])

    def test_target_selection_independent_of_rating_and_input_order(self):
        a=split_ratings(self.rows())
        changed=self.rows()[::-1].copy();changed[:,2]=6-changed[:,2]
        self.assertEqual(a,split_ratings(changed))

    def test_refuse_duplicate_movie_or_temporal_boundary_overlap(self):
        a=self.rows();a[1,1]=a[0,1]
        with self.assertRaises(ValueError):split_ratings(a)
        rows,_,_=split_ratings(self.rows());rows[0]['validation'][1]=2
        with self.assertRaises(ValueError):self.load(rows)

    def test_reject_invalid_history_or_split(self):
        users=self.load(split_ratings(self.rows())[0])
        for h,s in [(0,'train'),(3,'unknown')]:
            with self.assertRaises(ValueError):temporal_examples(users,h,s)


if __name__=='__main__':unittest.main()
