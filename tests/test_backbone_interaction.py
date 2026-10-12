from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BackboneInteractionTest(unittest.TestCase):
    def _rows(self, architecture, short, long):
        return [{"architecture": architecture, "user_id": user, "split": "validation",
                 "context_hash": str(user), "target": 7, "seed": 42, "time_enabled": False,
                 "observed_length_raw": 1500, "window_length": 1028, "old_length": 900,
                 "tail_length": 128, "old_unique_fraction": .5, "target_seen_tail": False,
                 "target_seen_old": True, "target_seen_only_old": True, "target_not_in_window": False,
                 "short_input_context_hash": "tail" + str(user), "long_input_context_hash": str(user),
                 "short_ndcg_at_10": short, "long_ndcg_at_10": long,
                 "short_recall_at_10": 0, "long_recall_at_10": 1}
                for user in (1, 2, 3)]

    def test_interaction_resamples_all_four_models_jointly(self):
        from rhpr.sasrec.backbone_interaction import analyze_interaction

        result = analyze_interaction(self._rows("sasrec", .1, .2), self._rows("hstu", .2, .5), samples=50)
        stats = result["cohorts"][0]["metrics"]["ndcg_at_10"]
        self.assertAlmostEqual(stats["interaction_absolute"], .2)
        for endpoint in stats["interaction_absolute_ci95"]:
            self.assertAlmostEqual(endpoint, .2)
        self.assertAlmostEqual(stats["interaction_relative_gain_difference_pp"], 50)

    def test_rejects_mismatched_contexts_and_duplicate_users(self):
        from rhpr.sasrec.backbone_interaction import analyze_interaction

        sasrec, hstu = self._rows("sasrec", .1, .2), self._rows("hstu", .2, .5)
        changed = copy.deepcopy(hstu)
        changed[0]["context_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "evaluation identity"):
            analyze_interaction(sasrec, changed, samples=2)
        with self.assertRaisesRegex(ValueError, "unique users"):
            analyze_interaction(sasrec, hstu + [hstu[0]], samples=2)


if __name__ == "__main__":
    unittest.main()
