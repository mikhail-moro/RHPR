from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy required")
class GrDataContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # Deliberately unordered source IDs; source ID zero is a real item.
        self.catalog = [
            {"item_id": item_id, "text": f"description for source item {item_id}"}
            for item_id in (55, 0, 77, 12, 90, 4, 8, 100)
        ]
        self.sequences = [
            {"user_id": 11, "train": [55, 0, 77, 12, 90], "validation": 4, "test": 8},
            {"user_id": 22, "train": [90, 0, 90, 100], "validation": 77, "test": 55},
        ]
        self.write_data()

    def write_data(self, catalog=None, sequences=None):
        (self.directory / "catalog.json").write_text(json.dumps(
            self.catalog if catalog is None else catalog), encoding="utf-8")
        (self.directory / "sequences.json").write_text(json.dumps(
            self.sequences if sequences is None else sequences), encoding="utf-8")

    def load(self, expected=None):
        from rhpr.gr.data import load_beauty
        return load_beauty(self.directory, expected)

    def test_catalogue_row_mapping_and_counts_are_explicit(self):
        from rhpr.gr.data import UserSequence
        texts, users, source_ids, counts = self.load()
        self.assertEqual(source_ids, [55, 0, 77, 12, 90, 4, 8, 100])
        self.assertEqual(texts, [row["text"] for row in self.catalog])
        self.assertEqual(users, [UserSequence(11, (0, 1, 2, 3, 4), 5, 6),
                                 UserSequence(22, (4, 1, 4, 7), 2, 0)])
        self.assertEqual(counts, {"items": 8, "users": 2, "train_events": 9, "train_examples": 7})

    def test_all_training_event_positions_appear_once_per_complete_pass(self):
        from rhpr.gr.data import build_examples
        _, users, _, _ = self.load()
        contexts, user_ids, targets = build_examples(users, history_items=3, split="train")
        self.assertEqual(contexts.tolist(), [
            [-1, -1, 0], [-1, 0, 1], [0, 1, 2], [1, 2, 3],
            [-1, -1, 4], [-1, 4, 1], [4, 1, 4],
        ])
        self.assertEqual(user_ids.tolist(), [11, 11, 11, 11, 22, 22, 22])
        self.assertEqual(targets.tolist(), [1, 2, 3, 4, 1, 4, 7])
        self.assertEqual(contexts.dtype, np.int64)
        self.assertEqual(targets.dtype, np.int64)
        self.assertEqual(user_ids.dtype, np.int64)
        # Repeated target identities are distinct event positions, not deduplicated
        # or replaced with repeated training of the final user suffix.
        self.assertEqual(int((targets[user_ids == 22] == 4).sum()), 1)
        for again, first in zip(build_examples(users, 3, "train"),
                                (contexts, user_ids, targets), strict=True):
            self.assertTrue(np.array_equal(again, first))

    def test_history_truncation_keeps_the_immediate_causal_prefix(self):
        from rhpr.gr.data import build_examples
        _, users, _, _ = self.load()
        short_contexts, user_ids, short_targets = build_examples(users, 1, "train")
        self.assertEqual(short_contexts.tolist(), [[0], [1], [2], [3], [4], [1], [4]])
        self.assertEqual(short_targets.tolist(), [1, 2, 3, 4, 1, 4, 7])
        self.assertEqual(user_ids.tolist(), [11, 11, 11, 11, 22, 22, 22])
        long_contexts, _, long_targets = build_examples(users, 9, "train")
        self.assertEqual(long_contexts[3].tolist(), [-1] * 5 + [0, 1, 2, 3])
        self.assertTrue(np.array_equal(short_targets, long_targets))

    def test_validation_uses_train_and_test_adds_only_observed_validation(self):
        from rhpr.gr.data import build_examples
        _, users, _, _ = self.load()
        validation, val_users, val_targets = build_examples(users, 3, "validation")
        test, test_users, test_targets = build_examples(users, 3, "test")
        self.assertEqual(validation.tolist(), [[2, 3, 4], [1, 4, 7]])
        self.assertEqual(val_targets.tolist(), [5, 2])
        self.assertEqual(test.tolist(), [[3, 4, 5], [4, 7, 2]])
        self.assertEqual(test_targets.tolist(), [6, 0])
        self.assertEqual(val_users.tolist(), [11, 22])
        self.assertTrue(np.array_equal(val_users, test_users))
        # User 11's held-out identities are distinct from all training identities.
        # Neither the train inputs nor train labels may contain those holdouts.
        train, train_users, train_targets = build_examples(users, 9, "train")
        self.assertFalse(np.isin(train[train_users == 11], [5, 6]).any())
        self.assertFalse(np.isin(train_targets[train_users == 11], [5, 6]).any())
        self.assertNotIn(6, test[0])
        self.assertNotIn(5, validation[0])

    def test_zero_item_and_user_ids_are_valid_and_padding_is_negative_one(self):
        from rhpr.gr.data import build_examples
        sequences = [{"user_id": 0, "train": [55, 0], "validation": 77, "test": 12}]
        self.write_data(sequences=sequences)
        _, users, _, _ = self.load()
        train, user_ids, targets = build_examples(users, 4, "train")
        self.assertEqual(train.tolist(), [[-1, -1, -1, 0]])
        self.assertEqual(targets.tolist(), [1])
        self.assertEqual(user_ids.tolist(), [0])
        self.assertEqual((train >= 0).tolist(), [[False, False, False, True]])
        validation, _, _ = build_examples(users, 4, "validation")
        test, _, _ = build_examples(users, 4, "test")
        self.assertEqual(validation.tolist(), [[-1, -1, 0, 1]])
        self.assertEqual(test.tolist(), [[-1, 0, 1, 2]])

    def test_registered_digests_check_bytes_even_when_counts_are_unchanged(self):
        from rhpr.gr.data import file_sha256
        expected = {"sha256": {
            name: file_sha256(self.directory / name)
            for name in ("catalog.json", "sequences.json")
        }, "items": 8, "users": 2, "train_events": 9, "train_examples": 7}
        self.load(expected)
        changed = copy.deepcopy(self.sequences)
        changed[0]["train"][0:2] = reversed(changed[0]["train"][0:2])
        self.write_data(sequences=changed)
        with self.assertRaisesRegex(ValueError, "digest mismatch: sequences.json"):
            self.load(expected)
        self.write_data()
        catalog = copy.deepcopy(self.catalog)
        catalog[0]["text"] += " edited"
        self.write_data(catalog=catalog)
        with self.assertRaisesRegex(ValueError, "digest mismatch: catalog.json"):
            self.load(expected)

    def test_digest_paths_cannot_escape_the_dataset_directory(self):
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.load({"sha256": {"../catalog.json": "0" * 64}})
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.load({"sha256": {str(self.directory / "catalog.json"): "0" * 64}})

    def test_hash_function_matches_known_bytes_and_count_mismatches_fail(self):
        from rhpr.gr.data import file_sha256
        path = self.directory / "known.bin"
        path.write_bytes(b"abc")
        self.assertEqual(file_sha256(path), hashlib.sha256(b"abc").hexdigest())
        for key in ("items", "users", "train_events", "train_examples"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"count mismatch: {key}"):
                self.load({key: 999})

    def test_duplicate_and_unknown_catalogue_identities_are_rejected(self):
        for bad_id in (55, 0.0, True):
            catalog = copy.deepcopy(self.catalog)
            catalog[1]["item_id"] = bad_id
            self.write_data(catalog=catalog)
            with self.subTest(bad_id=bad_id), self.assertRaises(ValueError):
                self.load()
        self.write_data(catalog=[])
        with self.assertRaises(ValueError):
            self.load()
        for location in ("train", "validation", "test"):
            sequences = copy.deepcopy(self.sequences)
            if location == "train":
                sequences[0][location][0] = 999
            else:
                sequences[0][location] = 999
            self.write_data(sequences=sequences)
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "unknown items"):
                self.load()

    def test_empty_text_duplicate_users_and_noninteger_sequences_are_rejected(self):
        for bad_text in ("", "  ", None, 1):
            catalog = copy.deepcopy(self.catalog)
            catalog[0]["text"] = bad_text
            self.write_data(catalog=catalog)
            with self.subTest(text=bad_text), self.assertRaisesRegex(ValueError, "nonempty text"):
                self.load()
        sequences = copy.deepcopy(self.sequences)
        sequences[1]["user_id"] = sequences[0]["user_id"]
        self.write_data(sequences=sequences)
        with self.assertRaisesRegex(ValueError, "user IDs must be unique"):
            self.load()
        self.write_data(sequences=[])
        with self.assertRaises(ValueError):
            self.load()
        for location in ("user_id", "train", "validation", "test"):
            for bad_id in (False, 55.0, "55"):
                sequences = copy.deepcopy(self.sequences)
                if location == "train":
                    sequences[0][location][0] = bad_id
                else:
                    sequences[0][location] = bad_id
                self.write_data(sequences=sequences)
                with self.subTest(location=location, value=bad_id), self.assertRaisesRegex(ValueError, "integers"):
                    self.load()
        for train in ([], [55]):
            sequences = copy.deepcopy(self.sequences)
            sequences[0]["train"] = train
            self.write_data(sequences=sequences)
            with self.subTest(train=train), self.assertRaisesRegex(ValueError, "insufficient"):
                self.load()

    def test_invalid_history_length_and_unknown_split_fail_before_allocation(self):
        from rhpr.gr.data import build_examples
        _, users, _, _ = self.load()
        for history_items in (0, -1, True, 1.0, "3"):
            with self.subTest(history_items=history_items), self.assertRaises(ValueError):
                build_examples(users, history_items, "train")
        with self.assertRaisesRegex(ValueError, "unknown split"):
            build_examples(users, 3, "holdout")


if __name__ == "__main__":
    unittest.main()
