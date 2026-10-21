from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.sasrec.query_search_retention import prune_trial_resumes


class QuerySearchRetentionTest(unittest.TestCase):
    def fixture(self, root, trials, records, family, name, score, **updates):
        path = root / "checkpoints" / f"{family}-{name}-resume.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"resume-{family}-{name}".encode())
        record = {"path": path.relative_to(root).as_posix(), "training_model": family,
            "trial": {"name": name}, "size_bytes": path.stat().st_size,
            "optimizer_state_saved": True}
        result = {"status": "training_completed", "complete": True, "from_scratch": True,
            "best_validation": {"ndcg10": score}, "resumable_checkpoint": dict(record)}
        result.update(updates)
        trials.setdefault(family, {})[name] = result
        records.append(record)
        return path, result

    def test_top_two_across_families_replace_earlier_resumes_without_losing_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            first, a = self.fixture(root, trials, records, "q4", "default", .10)
            second, b = self.fixture(root, trials, records, "q8", "default", .20,
                                     status="success")
            historical = copy.deepcopy(a["resumable_checkpoint"])
            records = prune_trial_resumes(root, trials, records, 2)
            self.assertTrue(first.exists() and second.exists())
            third, c = self.fixture(root, trials, records, "q16", "slow", .30,
                                    status="full_transfer_failed")
            records = prune_trial_resumes(root, trials, records, 2)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists() and third.exists())
            self.assertEqual([row["training_model"] for row in records], ["q8", "q16"])
            self.assertFalse(a["resume_checkpoint_retained"])
            self.assertTrue(b["resume_checkpoint_retained"] and c["resume_checkpoint_retained"])
            self.assertEqual(a["resumable_checkpoint"], historical)

    def test_ties_use_report_registration_order_even_if_manifest_order_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            paths = [self.fixture(root, trials, records, family, "trial", .2)[0]
                     for family in ("first", "second", "third")]
            records.reverse()
            kept = prune_trial_resumes(root, trials, records, 2)
            self.assertEqual([path.exists() for path in paths], [True, True, False])
            self.assertEqual([row["training_model"] for row in kept], ["second", "first"])

    def test_failed_incomplete_reused_and_nonfinite_trials_are_not_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            good, _ = self.fixture(root, trials, records, "q", "good", .1)
            cases = ({"status": "numerical_failed"}, {"status": "out_of_memory"},
                {"complete": False}, {"from_scratch": False},
                {"best_validation": {"ndcg10": float("nan")}},
                {"best_validation": {"ndcg10": float("inf")}},
                {"best_validation": {"ndcg10": True}})
            rejected = [self.fixture(root, trials, records, "q", f"bad{i}", 99., **updates)[0]
                        for i, updates in enumerate(cases)]
            kept = prune_trial_resumes(root, trials, records, 2)
            self.assertTrue(good.exists())
            self.assertTrue(all(not path.exists() for path in rejected))
            self.assertEqual(len(kept), 1)

    def test_zero_limit_removes_only_registered_resumes_and_preserves_best_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            resume, result = self.fixture(root, trials, records, "q", "good", .1)
            best = root / "checkpoints/q-good.pt"
            best.write_bytes(b"best weights")
            result["best_checkpoint"] = {"path": "checkpoints/q-good.pt"}
            unregistered = root / "checkpoints/unregistered-resume.pt"
            unregistered.write_bytes(b"not registered")
            self.assertEqual(prune_trial_resumes(root, trials, records, 0), [])
            self.assertFalse(resume.exists())
            self.assertTrue(best.exists() and unregistered.exists())

    def test_unsafe_and_unrecognized_paths_are_never_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, trials, records = Path(temporary), {}, []
            root = base / "output"
            original, result = self.fixture(root, trials, records, "q", "unsafe", .1)
            outside = base / "outside-resume.pt"
            outside.write_bytes(original.read_bytes())
            best = root / "checkpoints/q-unsafe.pt"
            best.write_bytes(original.read_bytes())
            result["best_checkpoint"] = {"path": "checkpoints/q-unsafe.pt"}
            for path in ("../outside-resume.pt", str(outside), "checkpoints/../outside-resume.pt",
                         "checkpoints/q-unsafe.pt"):
                with self.subTest(path=path):
                    record = {**records[0], "path": path}
                    result["resumable_checkpoint"] = dict(record)
                    self.assertEqual(prune_trial_resumes(root, trials, [record], 0), [])
                    self.assertTrue(original.exists() and outside.exists() and best.exists())
                    self.assertFalse(result["resume_checkpoint_retained"])

    def test_best_weight_path_is_protected_even_if_named_like_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            path, result = self.fixture(root, trials, records, "q", "same", .1)
            result["best_checkpoint"] = {"path": records[0]["path"]}
            self.assertEqual(prune_trial_resumes(root, trials, records, 0), [])
            self.assertTrue(path.exists())

    def test_ambiguous_family_and_trial_names_do_not_delete_shared_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            path, _ = self.fixture(root, trials, records, "q-inner", "slow", .1)
            same_path, _ = self.fixture(root, trials, records, "q", "inner-slow", .2)
            self.assertEqual(path, same_path)
            self.assertEqual(prune_trial_resumes(root, trials, records, 0), [])
            self.assertTrue(path.exists())

    def test_symlink_resume_and_symlink_checkpoint_directory_are_not_followed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, trials, records = Path(temporary), {}, []
            root = base / "output"
            path, result = self.fixture(root, trials, records, "q", "link", .1)
            outside = base / "outside.pt"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
            self.assertEqual(prune_trial_resumes(root, trials, records, 0), [])
            self.assertTrue(outside.exists() and path.is_symlink())
            self.assertFalse(result["resume_checkpoint_retained"])
            path.unlink()
            (root / "checkpoints").rmdir()
            external_dir = base / "external-checkpoints"
            external_dir.mkdir()
            external_file = external_dir / path.name
            external_file.write_bytes(outside.read_bytes())
            (root / "checkpoints").symlink_to(external_dir, target_is_directory=True)
            self.assertEqual(prune_trial_resumes(root, trials, records, 0), [])
            self.assertTrue(external_file.exists())

    def test_wrong_owner_optimizer_marker_size_and_missing_files_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, trials, records = Path(temporary), {}, []
            path, result = self.fixture(root, trials, records, "q", "good", .1)
            for updates in ({"training_model": "other"}, {"trial": {"name": "other"}},
                            {"optimizer_state_saved": False}, {"size_bytes": 123456}):
                with self.subTest(updates=updates):
                    record = {**records[0], **updates}
                    result["resumable_checkpoint"] = dict(record)
                    self.assertEqual(prune_trial_resumes(root, trials, [record], 0), [])
                    self.assertTrue(path.exists())
            result["resumable_checkpoint"] = dict(records[0])
            path.unlink()
            self.assertEqual(prune_trial_resumes(root, trials, records, 2), [])
            self.assertFalse(result["resume_checkpoint_retained"])

    def test_invalid_limits_fail_before_touching_files(self):
        for limit in (-1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                prune_trial_resumes(Path("unused"), {}, [], limit)


if __name__ == "__main__":
    unittest.main()
