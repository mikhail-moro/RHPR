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
class DailyQualityTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def ranking_model(self):
        class RankingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_items, self.max_seq_len = 200, 8
                self.item_embedding = torch.nn.Embedding(201, 1, padding_idx=0)
                self.item_embedding.weight.data[:, 0] = torch.arange(201)
                self.inputs = []

            def last_state(self, ids, mask):
                self.inputs.append((ids.cpu().clone(), mask.cpu().clone()))
                return torch.ones((len(ids), 1), device=ids.device)

        return RankingModel()

    def example(self, user, day, target, context=(4, 0, 5)):
        return SimpleNamespace(user_id=user, day=day, anchor_id=f"{user}:{day}",
                               context=np.asarray(context, dtype=np.int32),
                               timestamps=np.arange(day * 86400 - len(context), day * 86400,
                                                    dtype=np.uint32),
                               target=target, target_timestamp=day * 86400,
                               reference_context_hash="a" * 64)

    def test_macro_user_weights_differ_from_micro_anchor_weights_and_oov_stays(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality
        from rhpr.sasrec.benchmark import score_catalogue

        model = self.ranking_model().train()
        examples = [self.example(10, 286, 200), self.example(10, 287, 200),
                    self.example(10, 288, 0), self.example(20, 286, 190)]
        with patch("rhpr.sasrec.long_quality_benchmark.score_catalogue", wraps=score_catalogue) as scoring:
            result = evaluate_daily_quality(model, iter(examples), device="cpu", batch_size=4,
                                            item_chunk_size=31)
        self.assertTrue(model.training)
        self.assertEqual(scoring.call_count, 1)
        self.assertEqual(scoring.call_args.kwargs["k"], 100)
        metrics = result["metrics"]
        self.assertEqual((metrics["user_count"], metrics["anchor_count"]), (2, 4))
        self.assertEqual((metrics["warm_user_count"], metrics["warm_anchor_count"]), (2, 3))
        self.assertEqual(metrics["oov_target_count"], 1)
        self.assertAlmostEqual(metrics["ndcg10"], 1 / 3)
        self.assertAlmostEqual(metrics["micro_anchor_ndcg10"], .5)
        self.assertAlmostEqual(metrics["warm_ndcg10"], .5)
        d11 = 1 / math.log2(12)
        self.assertAlmostEqual(metrics["ndcg100"], (2 / 3 + d11) / 2)
        self.assertAlmostEqual(metrics["micro_anchor_ndcg100"], (2 + d11) / 4)
        self.assertAlmostEqual(metrics["warm_ndcg100"], (1 + d11) / 2)
        self.assertAlmostEqual(metrics["first_anchor_ndcg10"], .5)
        self.assertAlmostEqual(metrics["first_anchor_ndcg100"], (1 + d11) / 2)
        self.assertEqual([record["rank"] for record in result["per_anchor"]], [1, 1, 0, 0])
        self.assertEqual([record["rank_at_max_k"] for record in result["per_anchor"]], [1, 1, 0, 11])
        self.assertEqual([record["anchor_count"] for record in result["per_user"]], [3, 1])
        self.assertNotIn("rank", result["per_user"][0])
        self.assertNotIn("target_known", result["per_user"][0])
        self.assertEqual(result["metadata"]["daily_quality_format_version"], 1)
        json.dumps(result, allow_nan=False)

    def test_warm_macro_excludes_only_users_without_any_supported_anchor(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality

        result = evaluate_daily_quality(self.ranking_model(),
                                        [self.example(1, 286, 200), self.example(1, 287, 0),
                                         self.example(2, 286, 0)], device="cpu")
        self.assertAlmostEqual(result["metrics"]["ndcg10"], .25)
        self.assertEqual(result["metrics"]["warm_ndcg10"], 1)
        self.assertEqual(result["metrics"]["warm_user_count"], 1)
        all_oov = evaluate_daily_quality(self.ranking_model(), [self.example(1, 286, 0)], device="cpu")
        self.assertEqual(all_oov["metrics"]["warm_user_count"], 0)
        self.assertEqual(all_oov["metrics"]["warm_ndcg10"], 0)

    def test_metadata_mapping_survives_evaluator_record_reordering(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality
        from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality

        def reordered(*args, **kwargs):
            result = evaluate_long_quality(*args, **kwargs)
            result["per_user"].reverse()
            return result

        examples = [self.example(5, 288, 190), self.example(2, 286, 200), self.example(5, 286, 0)]
        with patch("rhpr.sasrec.daily_quality.evaluate_long_quality", side_effect=reordered):
            result = evaluate_daily_quality(self.ranking_model(), iter(examples), device="cpu", batch_size=2)
        self.assertEqual([(row["anchor_id"], row["rank_at_max_k"]) for row in result["per_anchor"]],
                         [("5:288", 11), ("2:286", 1), ("5:286", 0)])
        # Earliest day, not iterator order, defines the compatibility diagnostic.
        self.assertEqual(result["metrics"]["first_anchor_ndcg10"], .5)

    def test_future_context_and_duplicate_anchor_are_rejected_with_training_mode_restored(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality

        example = self.example(1, 286, 200)
        model = self.ranking_model().train()
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_daily_quality(model, [example, example], device="cpu")
        self.assertTrue(model.training)
        future = self.example(1, 286, 200)
        future.timestamps[-1] = future.target_timestamp + 1
        with self.assertRaisesRegex(ValueError, "strictly before"):
            evaluate_daily_quality(model, [future], device="cpu")
        self.assertEqual(model.inputs, [])
        future.timestamps[-1] = future.target_timestamp
        with self.assertRaisesRegex(ValueError, "strictly before"):
            evaluate_daily_quality(model, [future], device="cpu")
        unordered = self.example(1, 286, 200)
        unordered.timestamps[:] = unordered.timestamps[::-1]
        with self.assertRaisesRegex(ValueError, "chronological"):
            evaluate_daily_quality(model, [unordered], device="cpu")
        with self.assertRaises(ValueError):
            evaluate_daily_quality(model, [], device="cpu")

    def test_diagnostic_fields_never_reach_model_or_persisted_records(self):
        from rhpr.sasrec.daily_quality import evaluate_daily_quality

        class GuardedExample:
            def __init__(self, fields):
                self.__dict__.update(fields)

            @property
            def target_external_id(self):
                raise AssertionError("external target must not be consumed")

            @property
            def features(self):
                raise AssertionError("history features must not be passed to the predictor")

        example = GuardedExample(vars(self.example(1, 286, 200)))
        model = self.ranking_model()
        result = evaluate_daily_quality(model, [example], device="cpu")
        ids, mask = model.inputs[0]
        self.assertEqual(ids.tolist(), [[4, 0, 5]])
        self.assertEqual(mask.tolist(), [[True, True, True]])
        for record in result["per_anchor"] + result["per_user"]:
            self.assertTrue({"context", "target", "timestamps", "target_timestamp", "target_external_id"}.isdisjoint(record))
        self.assertEqual(result["per_anchor"][0]["reference_context_hash"], "a" * 64)

    def records(self, user, values, *, known=True):
        return [{"user_id": user, "day": 286 + offset, "anchor_id": f"{user}:{286 + offset}",
                 "target_known": known, "reference_context_hash": "a" * 64,
                 "rank_cutoff": 100, "ndcg10": value, "ndcg100": value}
                for offset, value in enumerate(values)]

    def test_paired_bootstrap_clusters_users_and_preserves_unequal_anchor_weights(self):
        from rhpr.sasrec.daily_quality import paired_daily_ndcg_bootstrap

        reference = self.records(1, [0, 0, 0]) + self.records(2, [.5]) + self.records(3, [0], known=False)
        candidate = self.records(1, [1, 1, 1]) + self.records(2, [0]) + self.records(3, [0], known=False)
        result = paired_daily_ndcg_bootstrap(reference, list(reversed(candidate)), draws=128, seed=19)
        self.assertEqual(result, paired_daily_ndcg_bootstrap(reference, candidate, draws=128, seed=19))
        self.assertEqual((result["matched_users"], result["matched_anchors"]), (3, 5))
        self.assertAlmostEqual(result["delta_ndcg"], (1 - .5) / 3)
        self.assertAlmostEqual(result["reference_ndcg"], .5 / 3)
        self.assertAlmostEqual(result["candidate_ndcg"], 1 / 3)
        self.assertEqual((result["user_win_count"], result["user_loss_count"], result["user_tie_count"]), (1, 1, 1))
        # Independently reconstruct the user-cluster bootstrap. Sampling the
        # five anchors instead would yield a different distribution and mean.
        rng = np.random.default_rng(19)
        differences = np.asarray([1., -.5, 0.])
        samples = differences[rng.integers(0, 3, size=(128, 3))].mean(axis=1)
        self.assertAlmostEqual(result["ci_lower"], np.quantile(samples, .025))
        self.assertAlmostEqual(result["ci_upper"], np.quantile(samples, .975))
        json.dumps(result, allow_nan=False)

    def test_pairing_rejects_different_anchor_sets_context_hashes_support_and_bad_scores(self):
        from rhpr.sasrec.daily_quality import paired_daily_ndcg_bootstrap

        reference = self.records(1, [.2, .3])
        with self.assertRaisesRegex(ValueError, "same anchors"):
            paired_daily_ndcg_bootstrap(reference, reference[:1])
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_daily_ndcg_bootstrap(reference, reference + reference)
        for key, value, message in (("reference_context_hash", "b" * 64, "reference_context_hash"),
                                    ("target_known", False, "unsupported"),
                                    ("ndcg10", float("nan"), "finite"),
                                    ("rank_cutoff", 9, "cutoff")):
            changed = [dict(record) for record in reference]
            changed[0][key] = value
            with self.assertRaisesRegex(ValueError, message):
                paired_daily_ndcg_bootstrap(reference, changed)
        changed = self.records(1, [0, .3])
        changed[0]["target_known"] = False
        with self.assertRaisesRegex(ValueError, "target_known"):
            paired_daily_ndcg_bootstrap(reference, changed)
        with self.assertRaises(ValueError):
            paired_daily_ndcg_bootstrap([], [])
        with self.assertRaises(ValueError):
            paired_daily_ndcg_bootstrap(reference, reference, draws=True)


if __name__ == "__main__":
    unittest.main()
