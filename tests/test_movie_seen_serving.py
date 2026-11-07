import unittest
from types import SimpleNamespace
import numpy as np
from rhpr.gr.movie_seen_serving import known_sets,eligible_topk,decode_paths,retrieval_metrics


class MovieSeenServingTest(unittest.TestCase):
    def test_full_item_filter_preserves_unseen_sibling_invalid_and_duplicate_handling(self):
        sids=np.array([[1,2,3,0],[1,2,3,1],[4,5,6,0]])
        paths=np.array([[sids[0],sids[1],[9,9,9,9],sids[1],sids[2]]])
        scores=np.array([[.5,.3,.2,.1,.05]])
        decoded=decode_paths(paths,sids)
        items,values=eligible_topk(decoded,scores,[{0}],k=3)
        np.testing.assert_array_equal(items,[[1,2,-1]])
        np.testing.assert_allclose(values,[[.3,.05,-1]])
        np.testing.assert_array_equal(decoded,[[0,1,-1,1,2]])

    def test_known_mask_uses_only_available_past_in_user_order(self):
        users=[SimpleNamespace(user_id=7,train=(1,2),validation=3,test=4),SimpleNamespace(user_id=8,train=(6,),validation=7,test=8)]
        self.assertEqual(known_sets(users,[8,7],'validation'),[{6},{1,2}])
        self.assertEqual(known_sets(users,[8,7],'test'),[{6,7},{1,2,3}])
        with self.assertRaises(ValueError):known_sets(users,[7],'train')

    def test_target_ranks_use_first_occurrence_and_do_not_count_padding(self):
        items=np.array([[3,2,1],[2,3,-1],[-1,-1,-1]])
        scores=retrieval_metrics(items,np.array([3,3,3]))
        self.assertAlmostEqual(scores['recall@10'],2/3)
        self.assertAlmostEqual(scores['ndcg@10'],(1+1/np.log2(3))/3)


if __name__=='__main__':unittest.main()
