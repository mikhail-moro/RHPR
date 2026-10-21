from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    np = pa = pq = None


@unittest.skipIf(pa is None, "Numpy and pyarrow required")
class PrepareHistoryFeaturesTest(unittest.TestCase):
    def test_verified_fixture_extraction_preserves_unknowns_and_rejects_run_mismatch(self):
        from rhpr.sasrec.long_evaluation_data import prepare_long_evaluation_data, write_prepared_long_evaluation
        from rhpr.sasrec.long_history_data import load_long_history_cohort, write_prepared_cohort
        spec = importlib.util.spec_from_file_location("prepare_history_features", ROOT / "scripts/prepare_history_features.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "listens.parquet"
            train_items = [1, 2] * 65
            pq.write_table(pa.table({
                "uid": [1, 2],
                "item_id": [train_items + [999, 1, 2], train_items + [999]],
                "timestamp": [list(range(130)) + [220, 240, 310], list(range(130)) + [310]],
            }), source)
            cohort = load_long_history_cohort(source, train_end_timestamp=200, max_seq_len=129)
            evaluation = prepare_long_evaluation_data(source, cohort,
                validation_end_timestamp=300, test_end_timestamp=400)
            train_path, eval_path, run_path, output_path = [root / name for name in ("train", "evaluation", "run", "features")]
            write_prepared_cohort(cohort, train_path)
            write_prepared_long_evaluation(evaluation, cohort, eval_path)
            run_path.mkdir()
            config = {"data": {"evaluation_content_sha256": evaluation.summary["evaluation_content_sha256"]},
                      "training": {"raw_context_length": 129, "raw_tail_size": 128}}
            for filename, value in (("experiment.json", config), ("cohort-summary.json", cohort.summary),
                                    ("evaluation-summary.json", evaluation.summary),
                                    ("recent128-test-users.json", [dict(user_id=1, target_known=True),
                                                                   dict(user_id=2, target_known=False)])):
                (run_path / filename).write_text(json.dumps(value))
            manifest = module.prepare_history_features(cohort_directory=train_path,
                evaluation_directory=eval_path, run_output=run_path, output=output_path)
            self.assertEqual(manifest["users"], 2)
            self.assertEqual(manifest["warm_targets_checked"], 1)
            self.assertEqual(manifest["provenance"]["catalogue_train_event_count"], 260)
            self.assertFalse(manifest["targets_or_target_timestamps_in_features"])
            rows = pq.read_table(output_path / "history-features.parquet").to_pylist()
            self.assertEqual([row["user_id"] for row in rows], [1, 2])
            self.assertEqual(rows[0]["observed_event_count_before_cropping"], 132)
            self.assertEqual(rows[0]["train_event_count"], 130)
            self.assertEqual(rows[0]["context_oov_fraction"], 1 / 129)
            self.assertAlmostEqual(rows[0]["recent_mean_log_train_frequency"], np.log1p(130))
            self.assertNotIn("target_known", rows[0])
            self.assertNotIn("target_timestamp", rows[0])
            self.assertEqual(rows[1]["context_oov_fraction"], 0.)
            self.assertTrue(all(row["observed_context_length"] == 129 and row["old_prefix_length"] == 1 for row in rows))
            self.assertEqual(set(path.name for path in output_path.iterdir()),
                             {"history-features.parquet", "history-features-manifest.json"})
            changed = {**cohort.summary, "catalog_sha256": "0" * 64}
            (run_path / "cohort-summary.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "run training-cohort mismatch: catalog_sha256"):
                module.prepare_history_features(cohort_directory=train_path,
                    evaluation_directory=eval_path, run_output=run_path, output=output_path)


if __name__ == "__main__":
    unittest.main()
