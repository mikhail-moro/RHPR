"""Exact local moves, graph sampling, and complete-path conservation."""
import unittest
from types import SimpleNamespace
import numpy as np
from rhpr.gr.sid_graph_swap import (NeighborGraph, PairTerm, ContrastiveTerm,
    SwapSearch, couser_neighbors, cosine_neighbors, lcp_similarity, heldout_gain)


class GraphSwapTests(unittest.TestCase):
    def setUp(self):
        self.sids = np.array([[i//8, (i//2)%4, i%2, 0] for i in range(32)])
        self.graph = NeighborGraph(np.repeat(np.arange(32), 2),
            np.array([[(i+1)%32, (i+7)%32] for i in range(32)]).ravel(),
            np.tile([.9, .4], 32), 32)

    def test_incoming_edges_and_negative_incidence_match_exhaustive_delta(self):
        terms = [PairTerm(self.graph), ContrastiveTerm(self.graph, 17, negatives=4)]
        search = SwapSearch(self.sids, terms, [1., .3]); rng = np.random.default_rng(77)
        accepted = 0
        for i in range(100):
            a, b = rng.choice(32, 2, replace=False)
            before = search.codes.copy(); swapped = before.copy(); swapped[[a,b]] = swapped[[b,a]]
            old = sum(c*t.values(before).sum() for c,t in zip(search.coefficients,terms))
            new = sum(c*t.values(swapped).sum() for c,t in zip(search.coefficients,terms))
            ok, delta = search.try_swap(a,b,force=i%7==0)
            self.assertAlmostEqual(delta, new-old, places=12)
            self.assertEqual(ok, i%7==0 or new-old > 1e-14)
            np.testing.assert_array_equal(search.codes, swapped if ok else before)
            search.audit(self.sids); accepted += ok
        self.assertGreater(accepted, 0)

    def test_prefix_not_hamming_suffix_excluded_and_shape_preserved(self):
        a = np.array([[1,2,3,0],[1,8,3,0],[1,2,3,1],[7,2,3,0]])
        np.testing.assert_allclose(lcp_similarity(a,[0,0,0],[1,2,3]),[1/3,1,0])
        s = SwapSearch(self.sids,[PairTerm(self.graph)],[1])
        for i in range(31): s.try_swap(i,i+1,force=True)
        for d in range(1,5):
            x,c = np.unique(self.sids[:,:d],axis=0,return_counts=True)
            y,e = np.unique(s.codes[:,:d],axis=0,return_counts=True)
            np.testing.assert_array_equal(x,y); np.testing.assert_array_equal(c,e)

    def test_graph_only_reads_train_and_ignores_duplicates(self):
        class User:
            train = (0,0,1,2)
            @property
            def validation(self): raise AssertionError('validation accessed')
            @property
            def test(self): raise AssertionError('test accessed')
        g,f = couser_neighbors([User(),User()],4,1)
        np.testing.assert_array_equal(f,[2,2,2,0]); self.assertTrue(np.allclose(g.weight,1))
        self.assertEqual(len(g.source),3)
        with self.assertRaises(ValueError):
            NeighborGraph(np.array([0]),np.array([1]),np.array([-1.]),4)

    def test_positive_cosine_clamp_and_contrastive_samples_are_fixed(self):
        z=np.array([[1,0],[.8,.2],[-1,0],[-.8,-.2]],dtype=float)
        g=cosine_neighbors(z,2,block=2)
        self.assertEqual(len(g.source),4); self.assertTrue(np.all(g.weight>0))
        a=ContrastiveTerm(self.graph,12,negatives=4); b=ContrastiveTerm(self.graph,12,negatives=4)
        np.testing.assert_array_equal(a.candidates,b.candidates)
        for anchor,row in zip(a.anchor,a.candidates):
            positive=set(self.graph.target[self.graph.source==anchor])
            self.assertIn(row[0],positive);self.assertFalse(positive.intersection(row[1:]))
            self.assertNotIn(anchor,row);self.assertEqual(len(np.unique(row)),len(row))
        # All equal logits -> uniform classification, zero contrastive skill.
        collapsed=np.zeros_like(self.sids)
        self.assertAlmostEqual(a.values(collapsed).sum(),0,places=12)

    def test_heldout_count_model_matches_manual_probability(self):
        fit=SimpleNamespace(src=np.array([0,0,1]),dst=np.array([1,1,0]))
        held=SimpleNamespace(src=np.array([0]),dst=np.array([1]))
        r=heldout_gain(np.array([0,1]),fit,held)
        prior=2.5/4; prob=(2+10*prior)/12
        self.assertAlmostEqual(r['heldout_gain_bits'],np.log2(prob/prior))


if __name__=='__main__': unittest.main()
