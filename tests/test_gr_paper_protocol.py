"""Author-collator parity and fixed example-counted paper batches."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from rhpr.gr.data import UserSequence
from rhpr.gr.paper_protocol import PaperBatchStream, collate_paper_rows, move_batch

_SOURCE = os.environ.get("RHPR_GRID_SOURCE")
_REQUIRED = ("torch", "numpy", "transformers", "lightning", "pytorch_lightning", "torchmetrics", "pyarrow", "hydra")
_CAN_INTEGRATE = bool(_SOURCE) and all(importlib.util.find_spec(name) is not None for name in _REQUIRED)


@unittest.skipUnless(_CAN_INTEGRATE, "Set RHPR_GRID_SOURCE and install GR dependencies for actual GRID integration")
class PaperProtocolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from rhpr.gr.grid import load_grid
        load_grid(Path(_SOURCE))
        from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate
        from src.data.loading.components.label_function import NextKTokenMasking
        cls.np, cls.torch = np, torch
        cls.collator, cls.label_type = staticmethod(collate_with_sid_causal_duplicate), NextKTokenMasking
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.sids = np.array([[item % 4, item // 4, 0, 0] for item in range(12)], dtype=np.int64)
        cls.users = [UserSequence(i, tuple((i + j) % 12 for j in range(4 + i)), 0, 0)
                     for i in range(5)]

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.old_threads)

    def direct(self, users, indices, length=120, maximum=512):
        torch = self.torch
        rows = [{"sequence_data": torch.tensor(self.sids[list(users[index].train)].reshape(-1)),
                 "user_id": torch.tensor([users[index].user_id])} for index in indices]
        inputs, labels = self.collator(
            batch=rows, sequence_field_name="sequence_data", sid_hierarchy=4,
            labels={"sequence_data": SimpleNamespace(transform=self.label_type(next_k=4))},
            sequence_length=length, masking_token=-1, padding_token=-1,
            oov_token=None, max_batch_size=maximum,
        )
        return {
            "input_ids": inputs.transformed_sequences["sequence_data"],
            "attention_mask_encoder": inputs.mask,
            "user_id": inputs.transformed_sequences["user_id"],
            "future_ids": labels.labels["sequence_data"].reshape(len(inputs.mask), 4),
        }

    def assert_same(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            self.assertTrue(self.torch.equal(actual[name], expected[name]), name)

    def test_exact_rng_parity_with_actual_collator_including_duplicate_samples(self):
        # The first source user alone has 3,160 windows. Fifty random indices
        # contain replacements; upstream membership emits each selected index
        # once in enumeration order, which is deliberately preserved here.
        users = [UserSequence(17, tuple(i % 12 for i in range(80)), 0, 0)]
        self.torch.manual_seed(9)
        before = self.torch.get_rng_state()
        expected = self.direct(users, [0], maximum=50)
        after = self.torch.get_rng_state()
        self.torch.set_rng_state(before)
        actual = collate_paper_rows(users, self.sids, [0], max_augmented_batch=50)
        self.assert_same(actual, expected)
        self.assertTrue(self.torch.equal(after, self.torch.get_rng_state()))
        self.assertLessEqual(len(actual["future_ids"]), 50)

    def test_labels_mask_and_item_zero_match_all_contiguous_windows(self):
        user = UserSequence(9, (0, 1, 2), 0, 0)
        batch = collate_paper_rows([user], self.sids, [0], sequence_length=12)
        self.assertEqual(tuple(batch["input_ids"].shape), (3, 12))
        expected_targets = self.torch.tensor(self.sids[[1, 2, 2]])
        self.assertTrue(self.torch.equal(batch["future_ids"], expected_targets))
        self.assertTrue(self.torch.equal(batch["input_ids"][0, :4], self.torch.zeros(4, dtype=self.torch.long)))
        self.assertTrue(self.torch.equal(batch["attention_mask_encoder"][0, :4], self.torch.ones(4, dtype=self.torch.long)))
        self.assertEqual(batch["attention_mask_encoder"].sum(1).tolist(), [4, 8, 4])
        self.assertTrue((batch["input_ids"][0, 4:] == -1).all())
        self.assertTrue((batch["input_ids"][:, -4:] == -1).all())

    def test_long_full_row_is_sampled_before_suffix_truncation(self):
        history = tuple(i % 12 for i in range(40))
        batch = collate_paper_rows([UserSequence(7, history, 0, 0)], self.sids, [0],
                                   sequence_length=120, max_augmented_batch=1000)
        self.assertEqual(len(batch["future_ids"]), 40 * 39 // 2)
        # Windows ending at source position40 start after windows ending2..39.
        row = 39 * 38 // 2
        self.assertTrue(self.torch.equal(batch["input_ids"][row, :116],
                                       self.torch.tensor(self.sids[list(history[10:39])].reshape(-1))))
        self.assertTrue((batch["input_ids"][row, 116:] == -1).all())
        self.assertEqual(int(batch["attention_mask_encoder"][row].sum()), 116)
        self.assertTrue(self.torch.equal(batch["future_ids"][row], self.torch.tensor(self.sids[history[39]])))

    def test_buffering_retains_upstream_order_without_dropping_surplus_examples(self):
        torch = self.torch
        users = [UserSequence(1, (0, 1, 2, 3), 0, 0)]
        # Six windows from a single source row are split2+2+2, not resampled.
        expected = self.direct(users, [0])
        stream = PaperBatchStream(users, self.sids, source_batch_size=1,
                                 effective_batch_size=2, seed=22)
        batches = [stream.next_batch() for _ in range(3)]
        joined = {name: torch.cat([batch[name] for batch in batches]) for name in expected}
        self.assert_same(joined, expected)
        report = stream.report()
        self.assertEqual(report["source_batches"], 1)
        self.assertEqual(report["augmented_examples_generated"], 6)
        self.assertEqual(report["model_examples_yielded"], 6)
        self.assertEqual(report["buffered_examples"], 0)
        self.assertEqual(report["max_context_items"], 29)

    def test_stream_combines_short_batches_cycles_sources_and_isolates_rng(self):
        torch = self.torch
        # One supervised window per source user: each model batch needs2source
        # batches, and carries its partial buffer across the next call.
        users = [UserSequence(i, (i, i + 1), 0, 0) for i in range(3)]
        left = PaperBatchStream(users, self.sids, source_batch_size=2,
                               effective_batch_size=3, seed=42)
        right = PaperBatchStream(users, self.sids, source_batch_size=2,
                                effective_batch_size=3, seed=42)
        before = torch.get_rng_state()
        a = [next(left) for _ in range(3)]
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = [next(right) for _ in range(3)]
        for aa, bb in zip(a, b):
            self.assert_same(aa, bb)
            self.assertEqual(len(aa["future_ids"]), 3)
        report = left.report()
        self.assertEqual(report["source_rows_consumed"], 10)
        self.assertEqual(report["model_examples_yielded"], 9)
        self.assertEqual(report["buffered_examples"], 1)
        self.assertGreater(report["source_cycles_started"], 1)
        self.assertGreater(report["source_collate_seconds"], 0)

    def test_held_out_targets_are_never_read_and_move_preserves_every_tensor(self):
        class TrainOnly:
            train = (0, 1, 2)
            user_id = 90
            @property
            def validation(self):
                raise AssertionError("validation accessed")
            @property
            def test(self):
                raise AssertionError("test accessed")
        stream = PaperBatchStream([TrainOnly()], self.sids, source_batch_size=1,
                                 effective_batch_size=2)
        batch = stream.next_batch()
        self.assert_same(move_batch(batch, "cpu"), batch)
        self.assertIs(stream.report()["test_targets_read"], False)

    def test_model_loss_matches_actual_grid_model_step_and_backward(self):
        from rhpr.gr.experiment import _loss
        from rhpr.gr.grid import build_grid_generator
        from src.data.loading.components.interfaces import SequentialModelInputData, SequentialModuleLabelData
        torch = self.torch
        batch = collate_paper_rows(self.users, self.sids, [0, 1], sequence_length=20)
        model = build_grid_generator(self.sids, {
            "codebook_size": 4, "beam_width": 2, "d_model": 16, "d_ff": 32,
            "num_heads": 2, "d_kv": 8, "encoder_layers": 1, "decoder_layers": 1,
            "num_user_bins": None, "dropout": 0.0, "mlp_layers": 2, "separator_token": True,
        })
        model.train()
        actual = _loss(model, batch)
        upstream_inputs = SequentialModelInputData(
            transformed_sequences={"sequence_data": batch["input_ids"], "user_id": batch["user_id"]},
            mask=batch["attention_mask_encoder"])
        labels = SequentialModuleLabelData(labels={"sequence_data": batch["future_ids"].reshape(-1)})
        _, expected = model.model_step(upstream_inputs, labels)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_invalid_history_indices_sid_shapes_and_budgets_fail(self):
        for kwargs in ({"effective_batch_size": 0}, {"source_batch_size": True},
                       {"sequence_length": 119}, {"sequence_length": 4},
                       {"max_augmented_batch": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PaperBatchStream(self.users, self.sids, **kwargs)
        for indices in ([], [-1], [999]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                collate_paper_rows(self.users, self.sids, indices)
        with self.assertRaises(ValueError):
            PaperBatchStream([UserSequence(1, (0,), 0, 0)], self.sids)
        with self.assertRaises(ValueError):
            PaperBatchStream(self.users, self.sids.astype("float32"))


if __name__ == "__main__":
    unittest.main()
