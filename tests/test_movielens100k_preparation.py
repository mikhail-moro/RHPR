import unittest

import numpy as np

from scripts.prepare_movielens100k_text_ladder import build_bridge,canonical_ratings,normalized_title


class Movie100KMapping(unittest.TestCase):
    def test_title_bridge_never_treats_numeric_ids_as_shared(self):
        audit={10:dict(title='GoldenEye (1995)',metadata_present=True,tmdb_title='GoldenEye',tmdb_year=1995,imdb_id='tt0113189')}
        rows=[['2','GoldenEye (1995)','','','http://us.imdb.com/M/title-exact?GoldenEye%20(1995)']]
        match,missing,ambiguous=build_bridge(rows,audit)
        self.assertEqual(match[0]['ml100k_id'],2)
        self.assertEqual(match[0]['ml1m_id'],10)
        self.assertFalse(missing or ambiguous)

    def test_article_normalization_and_ambiguous_alias_rejection(self):
        self.assertEqual(normalized_title('Empire Strikes Back, The'),normalized_title('The Empire Strikes Back'))
        audit={i:dict(title='A Film (1995)',metadata_present=False) for i in [1,2]}
        match,missing,ambiguous=build_bridge([['50','A Film (1995)','','','']],audit)
        self.assertFalse(match or missing)
        self.assertEqual(ambiguous[0]['candidate_ml1m_ids'],[1,2])

    def test_same_movie_aliases_merge_and_latest_rating_wins(self):
        a=np.array([[1,2,5,10],[1,9,1,20],[2,9,4,10],[2,5,3,20]])
        clean,ids,lost=canonical_ratings(a,{2:100,9:100,5:200})
        self.assertEqual(ids,{2:2,9:2,5:5})
        self.assertEqual(lost,1)
        self.assertEqual(clean[clean[:,0]==1].tolist(),[[1,2,1,20]])
        b,_,_=canonical_ratings(a[::-1],{2:100,9:100,5:200})
        np.testing.assert_array_equal(clean,b)


if __name__=='__main__':unittest.main()
