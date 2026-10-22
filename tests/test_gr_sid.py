from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required")
class SidContractTest(unittest.TestCase):
    def test_suffix_resolves_every_item_and_follows_artifact_row_order(self):
        from rhpr.gr.sid import SidIndex
        semantic = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 0], [1, 2, 3]])
        index = SidIndex(semantic, width=4)
        self.assertEqual(index.sids.tolist(), [[1, 2, 3, 0], [1, 2, 3, 1],
                                              [1, 2, 0, 0], [1, 2, 3, 2]])
        self.assertEqual(dict(index.sid_to_item), {
            (1, 2, 3, 0): 0, (1, 2, 3, 1): 1, (1, 2, 0, 0): 2, (1, 2, 3, 2): 3,
        })
        self.assertTrue(np.array_equal(index.sids, SidIndex(semantic, width=4).sids))
        # Mutating the caller's source cannot silently change a fitted index.
        semantic[0, 0] = 0
        self.assertEqual(index.semantic_codes[0, 0], 1)
        with self.assertRaises(ValueError):
            index.sids[0, 0] = 0
        with self.assertRaises(TypeError):
            index.sid_to_item[(0, 0, 0, 0)] = 99

    def test_explicit_catalogue_permutation_preserves_identity(self):
        from rhpr.gr.sid import SidIndex
        index = SidIndex([[1], [1], [0]], width=3, item_ids=[2, 0, 1])
        self.assertEqual(index.sid_to_item[(1, 0)], 2)
        self.assertEqual(index.sid_to_item[(1, 1)], 0)
        self.assertEqual(index.sid_to_item[(0, 0)], 1)

    def test_wrong_item_in_same_semantic_group_is_not_a_hit(self):
        from rhpr.gr.sid import SidIndex, evaluate_ranked_sids
        index = SidIndex([[1, 1, 1], [1, 1, 1]], width=2)
        result = evaluate_ranked_sids(index, [[[1, 1, 1, 0]]], [1], cutoffs=(1, 5))
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["ndcg@5"], 0.0)
        self.assertEqual(result["invalid_predictions"], 0)
        hit = evaluate_ranked_sids(index, [[[1, 1, 1, 1]]], [1], cutoffs=(1,))
        self.assertEqual(hit["recall@1"], 1.0)

    def test_invalid_paths_are_misses_and_keep_rank_positions(self):
        from rhpr.gr.sid import SidIndex, evaluate_ranked_sids
        index = SidIndex([[0, 0], [1, 1]], width=2)
        predictions = np.array([
            [[-1, -1, -1], [1, 1, 0], [9, 9, 9]],
            [[0, 0, 1], [1, 0, 0], [0, 0, 0]],
        ])
        result = evaluate_ranked_sids(index, predictions, [1, 0], cutoffs=(1, 2, 3))
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["recall@2"], .5)
        self.assertEqual(result["recall@3"], 1.0)
        self.assertAlmostEqual(result["ndcg@2"], .5 / math.log2(3))
        self.assertAlmostEqual(result["ndcg@3"], (.5 + 1 / math.log2(3)) / 2)
        self.assertEqual(result["invalid_predictions"], 4)
        self.assertAlmostEqual(result["invalid_prediction_fraction"], 4 / 6)

    def test_duplicates_never_add_relevance_or_shift_later_items(self):
        from rhpr.gr.sid import SidIndex, evaluate_ranked_sids
        index = SidIndex([[0], [1]], width=2)
        result = evaluate_ranked_sids(index, [[[0, 0], [0, 0], [1, 0]]], [1], cutoffs=(2, 3))
        self.assertEqual(result["recall@2"], 0.0)
        self.assertEqual(result["ndcg@3"], .5)
        self.assertEqual(result["duplicate_predictions"], 1)
        repeat_hit = evaluate_ranked_sids(index, [[[1, 0], [1, 0]]], [1], cutoffs=(2,))
        self.assertEqual(repeat_hit["recall@2"], 1.0)
        self.assertEqual(repeat_hit["ndcg@2"], 1.0)
        self.assertEqual(repeat_hit["duplicate_prediction_fraction"], .5)

    def test_no_candidates_produces_finite_zero_metrics(self):
        from rhpr.gr.sid import SidIndex, evaluate_ranked_sids
        index = SidIndex([[0]], width=1)
        result = evaluate_ranked_sids(index, np.empty((1, 0, 2), dtype=np.int64), [0], cutoffs=(1,))
        self.assertEqual(result["recall@1"], 0.0)
        self.assertEqual(result["ndcg@1"], 0.0)
        self.assertEqual(result["prediction_count"], 0)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_collision_and_codebook_diagnostics_are_hand_calculated(self):
        from rhpr.gr.sid import SidIndex
        stats = SidIndex([[0, 0], [0, 0], [1, 1], [1, 0]], width=4).diagnostics()
        self.assertEqual(stats["semantic_unique_ids"], 3)
        self.assertEqual(stats["collision_groups"], 1)
        self.assertEqual(stats["collided_items"], 2)
        self.assertEqual(stats["collision_excess_items"], 1)
        self.assertEqual(stats["max_collision_group"], 2)
        self.assertEqual(stats["resolved_unique_ids"], 4)
        first, second, suffix = stats["level_statistics"]
        self.assertEqual(first["coverage"], .5)
        self.assertEqual(first["entropy_bits"], 1.0)
        self.assertEqual(first["entropy_normalized"], .5)
        expected = -.75 * math.log2(.75) - .25 * math.log2(.25)
        self.assertAlmostEqual(second["entropy_bits"], expected)
        self.assertAlmostEqual(suffix["entropy_bits"], expected)
        self.assertTrue(suffix["is_collision_suffix"])
        # Artifact diagnostics have no numpy scalars, NaN or Infinity.
        json.dumps(stats, allow_nan=False)
        single = SidIndex([[0]], width=1).diagnostics()
        self.assertEqual(single["level_statistics"][0]["entropy_normalized"], 0.0)

    def test_collision_suffix_capacity_has_no_silent_truncation(self):
        from rhpr.gr.sid import SidIndex
        SidIndex([[0], [0]], width=2)
        with self.assertRaisesRegex(ValueError, "collision group exceeds"):
            SidIndex([[0], [0], [0]], width=2)

    def test_trie_allows_only_catalogue_paths_including_disambiguator(self):
        from rhpr.gr.sid import PrefixTrie, SidIndex
        index = SidIndex([[0, 1], [0, 1], [1, 0]], width=2)
        trie = index.trie
        self.assertEqual(trie.allowed_next(), (0, 1))
        self.assertEqual(trie.allowed_next([0]), (1,))
        self.assertEqual(trie.allowed_next([0, 1]), (0, 1))
        self.assertEqual(trie.allowed_next([1, 0]), (0,))
        self.assertEqual(trie.allowed_next([1, 1]), ())
        self.assertEqual(trie.allowed_next([0, 1, 0]), ())
        self.assertTrue(trie.contains([0, 1, 0]))
        self.assertFalse(trie.contains([0, 1]))
        self.assertFalse(trie.contains([1, 1, 0]))
        with self.assertRaises(ValueError):
            trie.allowed_next([True])
        with self.assertRaises(ValueError):
            trie.contains([0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            PrefixTrie([[-1, 0]])

    def test_semantic_matrix_width_and_item_identity_validation(self):
        from rhpr.gr.sid import SidIndex
        invalid = ([], [[]], [1, 2], [[[0]]], [[0.0]], [[True]], [[-1]], [[2]],
                   np.array([[2**63]], dtype=np.uint64))
        for codes in invalid:
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                SidIndex(codes, width=2)
        for width in (0, -1, True, 2.0):
            with self.subTest(width=width), self.assertRaises(ValueError):
                SidIndex([[0]], width=width)
        for ids in ([0, 0], [0, 2], [-1, 1], [0.0, 1.0], [[0, 1]], [True, False]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                SidIndex([[0], [1]], width=2, item_ids=ids)

    def test_embedding_artifact_validation_preserves_geometry(self):
        from rhpr.gr.sid import validate_item_embeddings
        embeddings = np.array([[3, 4], [5, 12]], dtype=np.float64)
        result = validate_item_embeddings(embeddings, num_items=2)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.array_equal(result, embeddings))
        embeddings[0, 0] = 999
        self.assertEqual(result[0, 0], 3)
        for values in ([], [[]], [1, 2], [[float("nan")]], [[float("inf")]],
                       [[1e300]], [[1 + 2j]], [[True]], [["1"]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_item_embeddings(values)
        with self.assertRaises(ValueError):
            validate_item_embeddings([[1, 2]], num_items=2)
        with self.assertRaises(ValueError):
            validate_item_embeddings([[1, 2]], num_items=True)

    def test_evaluation_rejects_partial_sids_and_invalid_targets(self):
        from rhpr.gr.sid import SidIndex, evaluate_ranked_sids
        index = SidIndex([[0, 0], [1, 1]], width=2)
        for predictions in ([[[0, 0]]], [[[0.0, 0.0, 0.0]]], [[[True, False, False]]],
                            [[0, 0, 0]], np.empty((0, 1, 3), dtype=np.int64),
                            np.array([[[2**63, 0, 0]]], dtype=np.uint64)):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_ranked_sids(index, predictions, [0])
        for target in ([], [2], [-1], [[0]], [0.0], [True]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                evaluate_ranked_sids(index, [[[0, 0, 0]]], target)
        for cutoffs in ((), (0,), (-1,), (True,), (1.0,), (1, 1)):
            with self.subTest(cutoffs=cutoffs), self.assertRaises(ValueError):
                evaluate_ranked_sids(index, [[[0, 0, 0]]], [0], cutoffs=cutoffs)


if __name__ == "__main__":
    unittest.main()
