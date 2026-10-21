from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class LongQualityBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def ranking_model(self, num_items=11):
        class RankingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_items, self.max_seq_len = num_items, 8
                self.item_embedding = torch.nn.Embedding(num_items + 1, 1, padding_idx=0)
                self.item_embedding.weight.data[:, 0] = torch.arange(num_items + 1)
                self.masks = []

            def last_state(self, ids, mask):
                self.masks.append(mask.cpu().clone())
                return torch.ones((len(ids), 1), device=ids.device)

        return RankingModel()

    def test_all_anchor_oov_denominator_and_observed_zero_slot_with_exact_chunks(self):
        from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality

        model = self.ranking_model().train()
        examples = [SimpleNamespace(user_id=uid, context=np.asarray(context), target=target)
                    for uid, context, target in ((1, [4, 0, 5], 11), (2, [3], 10), (3, [2, 1], 0))]
        result = evaluate_long_quality(model, iter(examples), device=torch.device("cpu"),
                                       batch_size=2, k=3, item_chunk_size=2)
        self.assertTrue(model.training)
        self.assertEqual(model.masks[0].tolist(), [[True, True, True], [True, False, False]])
        metrics = result["metrics"]
        self.assertEqual(metrics["anchor_count"], 3)
        self.assertEqual(metrics["warm_anchor_count"], 2)
        self.assertAlmostEqual(metrics["hr3"], 2 / 3)
        self.assertAlmostEqual(metrics["ndcg3"], (1 + 1 / math.log2(3)) / 3)
        self.assertAlmostEqual(metrics["warm_hr3"], 1)
        self.assertEqual([record["rank"] for record in result["per_user"]], [1, 2, 0])
        self.assertNotIn("target", result["per_user"][0])
        json.dumps(result, allow_nan=False)

    def test_top100_reuses_catalogue_scoring_and_preserves_top10_with_boundary_ranks(self):
        from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality
        from rhpr.sasrec.benchmark import score_catalogue
        model = self.ranking_model(num_items=200)
        examples = [SimpleNamespace(user_id=i, context=np.asarray([1]), target=target)
                    for i, target in enumerate((200, 190, 101, 100, 0))]
        with patch('rhpr.sasrec.long_quality_benchmark.score_catalogue', wraps=score_catalogue) as scoring:
            result = evaluate_long_quality(model, examples, device=torch.device('cpu'), batch_size=5,
                                           additional_cutoffs=(100,), item_chunk_size=31)
        self.assertEqual(scoring.call_count, 1)
        self.assertEqual(scoring.call_args.kwargs['k'], 100)
        rows = result['per_user']
        self.assertEqual([r['rank_at_max_k'] for r in rows], [1, 11, 100, 0, 0])
        self.assertEqual([r['rank'] for r in rows], [1, 0, 0, 0, 0])
        self.assertTrue(all(r['rank_cutoff'] == 100 for r in rows))
        self.assertEqual(result['metadata']['cutoffs'], [10, 100])
        self.assertEqual(result['metrics']['hr10'], .2)
        self.assertEqual(result['metrics']['hr100'], .6)
        self.assertEqual(result['metrics']['warm_hr100'], .75)
        self.assertAlmostEqual(rows[1]['ndcg100'], 1 / math.log2(12))
        self.assertAlmostEqual(rows[2]['ndcg100'], 1 / math.log2(101))
        self.assertEqual(rows[3]['ndcg100'], 0)
        legacy = evaluate_long_quality(model, examples, device=torch.device('cpu'), batch_size=5,
                                       item_chunk_size=31)
        self.assertEqual([r['ndcg10'] for r in rows], [r['ndcg10'] for r in legacy['per_user']])
        with self.assertRaises(ValueError):
            evaluate_long_quality(model, examples, device=torch.device('cpu'), additional_cutoffs=(True,))

    def test_invalid_duplicate_anchor_and_empty_iterator_are_rejected(self):
        from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality

        example = SimpleNamespace(user_id=1, context=np.asarray([1]), target=1)
        for values in ([], [example, example]):
            with self.assertRaises(ValueError):
                evaluate_long_quality(self.ranking_model(), values, device=torch.device("cpu"))

    def test_paired_bootstrap_matches_user_identity_is_deterministic_and_keeps_oov(self):
        from rhpr.sasrec.long_quality_benchmark import paired_ndcg_bootstrap

        base = [{"user_id": 1, "target_known": True, "ndcg10": .2},
                {"user_id": 2, "target_known": False, "ndcg10": 0}]
        candidate = [{"user_id": 2, "target_known": False, "ndcg10": 0},
                     {"user_id": 1, "target_known": True, "ndcg10": .4}]
        result = paired_ndcg_bootstrap(base, candidate, draws=100, seed=19)
        self.assertEqual(result, paired_ndcg_bootstrap(base, candidate, draws=100, seed=19))
        self.assertAlmostEqual(result["delta_ndcg"], .1)
        self.assertAlmostEqual(result["relative_delta_percent"], 100)
        self.assertEqual(result["matched_users"], 2)
        self.assertLessEqual(result["ci_lower"], result["delta_ndcg"])
        self.assertGreaterEqual(result["ci_upper"], result["delta_ndcg"])
        with self.assertRaisesRegex(ValueError, "same users"):
            paired_ndcg_bootstrap(base, candidate[:1])
        with self.assertRaisesRegex(ValueError, "distinct"):
            paired_ndcg_bootstrap(base, candidate + candidate)

    def hstu(self, kind):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.long_compression import make_prefix_compressor

        compressor = make_prefix_compressor(kind=kind, d_model=8, summary_tokens=2,
                                             tail_size=3, n_heads=2)
        return HSTUModel(num_items=11, max_seq_len=17, d_model=8, n_heads=2,
                         n_layers=1, ffn_multiplier=4, dropout=0,
                         custom_sequence_compressor=compressor, normalization_length=5)

    def test_query_flops_include_kv_projections_and_both_attentions_without_double_counting(self):
        from rhpr.sasrec.long_quality_benchmark import _batch, _observe_flops

        model = self.hstu("qformer").eval()
        batch = _batch([np.arange(17) % 12], model, torch.device("cpu"))
        _, flops = _observe_flops(model, batch, torch.device("cpu"), False)
        # Q=2, old K=14, d=8. Two attentions plus the 4d FFN.
        expected = (4 * (2 + 2) * 8**2 + 4 * 2 * 2 * 8
                    + 4 * (2 + 14) * 8**2 + 4 * 2 * 14 * 8
                    + 2 * 2 * (8 * 32 + 32 * 8))
        self.assertEqual(flops["compressor_flops"], expected)
        self.assertEqual(flops["encoder_length"], 5)
        self.assertEqual(flops["scoring_flops"], 2 * 11 * 8)

    def test_hierarchy_hooks_count_each_reused_level_and_mean_has_no_matmul(self):
        from rhpr.sasrec.long_quality_benchmark import _batch, _observe_flops

        # Two regions of seven events: padded 8->2->1, i.e. 4+2 projection rows.
        for kind, expected in (("pair_mlp", 2 * (4 + 2) * 32 * 8), ("mean", 0)):
            model = self.hstu(kind).eval()
            batch = _batch([np.arange(17) % 12], model, torch.device("cpu"))
            _, flops = _observe_flops(model, batch, torch.device("cpu"), False)
            self.assertEqual(flops["compressor_flops"], expected)

    def test_cpu_latency_reports_real_compression_shapes_and_preserves_model_mode(self):
        from rhpr.sasrec.long_quality_benchmark import benchmark_long_quality

        model = self.hstu("qformer").train()
        contexts = [np.arange(17) % 12, np.arange(12) % 12]
        result = benchmark_long_quality(model, contexts, device=torch.device("cpu"),
                                        batch_size=2, warmup=0, iterations=2,
                                        item_chunk_size=3)
        self.assertTrue(model.training)
        self.assertEqual(set(result["cases"]), {"b1", "b2"})
        self.assertEqual(result["cases"]["b2"]["raw_padded_length"], 17)
        self.assertEqual(result["cases"]["b2"]["estimated_flops"]["encoder_length"], 5)
        self.assertFalse(result["metadata"]["cache"])
        self.assertEqual(result["metadata"]["summary_token_budget"], 2)
        for value in result["metrics"].values():
            self.assertTrue(math.isfinite(value))
        self.assertGreater(result["metrics"]["full_b2_users_per_second"], 0)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
