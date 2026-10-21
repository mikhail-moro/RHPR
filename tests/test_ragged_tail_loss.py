from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy are required")
class RaggedTailLossTest(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(814)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def model(self):
        class Standalone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_items, self.tail_size = 31, 128
                self.item_embedding = torch.nn.Embedding(32, 8)

            def encode(self, ids, mask):
                sources = torch.arange(ids.shape[1])[None].expand_as(ids)
                return self.item_embedding(ids), mask, sources.masked_fill(~mask, -1)

        return Standalone()

    def batch(self, lengths, width=None):
        width = max(lengths) if width is None else width
        positions = torch.arange(width)[None].expand(len(lengths), -1)
        mask = positions < torch.tensor(lengths)[:, None]
        targets = ((positions + 5) % 19 + 1).masked_fill(~mask, 0)
        return {"item_ids": ((positions + 1) % 19 + 1).masked_fill(~mask, 0),
                "valid_mask": mask, "targets": targets,
                "negative_ids": torch.tensor([[21, 22, 23, 24]]).expand(len(lengths), -1).clone()}

    def test_ragged_batch_loss_and_gradients_match_target_weighted_solo_losses(self):
        from rhpr.sasrec.long_training_profile import _loss

        model = self.model()
        lengths, tail = [167, 140, 47], 128
        batch = self.batch(lengths)
        # OOV events occupy timeline slots: neither input nor target ID zero
        # is permission to change the history length or the target schedule.
        batch["item_ids"][1, 130] = 0
        batch["targets"][1, 131] = 0
        states = torch.randn(3, 167, 8, requires_grad=True)
        mask = batch["valid_mask"]
        sources = torch.arange(167)[None].expand_as(mask).masked_fill(~mask, -1)
        actual = _loss(model, (states, mask, sources), batch, tail, 4)
        weights = [min(length, tail) for length in lengths]
        expected = sum(_loss(model, (states[i:i + 1, :length], mask[i:i + 1, :length],
                                      sources[i:i + 1, :length]),
                             {key: value[i:i + 1] for key, value in batch.items()}, tail, 4) * weight
                       for i, (length, weight) in enumerate(zip(lengths, weights))) / sum(weights)
        parameters = (states, model.item_embedding.weight)
        actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, parameters)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        for got, want in zip(actual_grad, expected_grad):
            torch.testing.assert_close(got, want, rtol=2e-6, atol=1e-7)

    def test_every_eligible_row_selects_exactly_128_original_target_sources(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.losses import build_loss_mask, next_item_loss

        batch = self.batch([300, 179, 67])
        states = torch.randn(3, 300, 8)
        mask = batch["valid_mask"]
        sources = torch.arange(300)[None].expand_as(mask).masked_fill(~mask, -1)
        seen = []

        def inspect(**kwargs):
            selected = build_loss_mask(kwargs["valid_mask"], kwargs["source_positions"],
                                       kwargs["original_lengths"], kwargs["loss_tail_size"])
            seen.append([kwargs["source_positions"][row][selected[row]].tolist() for row in range(3)])
            return next_item_loss(**kwargs)

        with patch("rhpr.sasrec.long_training_profile.next_item_loss", side_effect=inspect):
            _loss(self.model(), (states, mask, sources), batch, 128, 4)
        self.assertEqual(seen, [[list(range(172, 300)), list(range(51, 179)), list(range(67))]])

    def test_additional_nan_padding_changes_neither_loss_nor_real_state_gradients(self):
        from rhpr.sasrec.long_training_profile import _loss

        model = self.model()
        lengths = [150, 53]
        batch = self.batch(lengths)
        states = torch.randn(2, 150, 8, requires_grad=True)
        sources = torch.arange(150)[None].expand_as(batch["valid_mask"]).masked_fill(~batch["valid_mask"], -1)
        loss = _loss(model, (states, batch["valid_mask"], sources), batch, 128, 4)
        gradient = torch.autograd.grad(loss, states)[0]
        wide_batch = self.batch(lengths, width=500)
        wide_states = torch.cat((states.detach(), torch.full((2, 350, 8), float("nan"))), dim=1)
        wide_states = wide_states.masked_fill(~wide_batch["valid_mask"].unsqueeze(-1), float("nan")).requires_grad_()
        wide_sources = torch.arange(500)[None].expand_as(wide_batch["valid_mask"])
        wide_loss = _loss(model, (wide_states, wide_batch["valid_mask"], wide_sources), wide_batch, 128, 4)
        wide_gradient = torch.autograd.grad(wide_loss, wide_states)[0]
        torch.testing.assert_close(loss, wide_loss, rtol=0, atol=0)
        torch.testing.assert_close(gradient, wide_gradient[:, :150], rtol=0, atol=0)
        self.assertEqual(int(wide_gradient[~wide_batch["valid_mask"]].count_nonzero()), 0)

    def test_uniform_compressed_tail_matches_previous_loss_and_gradients_exactly(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.losses import next_item_loss

        model = self.model()
        batch = self.batch([900, 800])
        states = torch.randn(2, 132, 8, requires_grad=True)
        mask = torch.ones(2, 132, dtype=torch.bool)
        sources = torch.stack([torch.cat((torch.full((4,), length - 129),
                                         torch.arange(length - 128, length))) for length in (900, 800)])
        actual = _loss(model, (states, mask, sources), batch, 128, 4)
        previous = next_item_loss(name="sampled_softmax", model=model, encoded=states[:, -128:].float(),
            valid_mask=mask[:, -128:], source_positions=sources[:, -128:], targets=batch["targets"],
            original_lengths=batch["valid_mask"].sum(1), loss_tail_size=128,
            num_negatives=4, negative_ids=batch["negative_ids"])
        torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(actual, (states, model.item_embedding.weight), retain_graph=True)
        previous_grad = torch.autograd.grad(previous, (states, model.item_embedding.weight))
        for got, want in zip(actual_grad, previous_grad):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_only_gathered_tail_is_cast_to_fp32(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.losses import next_item_loss

        batch = self.batch([300, 159])
        states = torch.randn(2, 300, 8, dtype=torch.float16, requires_grad=True)
        sources = torch.arange(300)[None].expand_as(batch["valid_mask"])

        def inspect(**kwargs):
            self.assertEqual(kwargs["encoded"].shape, (2, 128, 8))
            self.assertEqual(kwargs["encoded"].dtype, torch.float32)
            return next_item_loss(**kwargs)

        with patch("rhpr.sasrec.long_training_profile.next_item_loss", side_effect=inspect):
            loss = _loss(self.model(), (states, batch["valid_mask"], sources), batch, 128, 4)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(states.grad).all()))

    def test_standalone_diagnostic_uses_model_tail_without_dummy_compressor(self):
        from rhpr.sasrec.query_numerics import diagnose_nonfinite_batch

        model = self.model()
        arrays = {key: value.numpy() for key, value in self.batch([140, 137]).items()}
        before = copy.deepcopy(model.state_dict())
        report = diagnose_nonfinite_batch(model, arrays, torch.device("cpu"), amp=False)
        for mode in ("training_precision", "fp32"):
            self.assertTrue(report[mode]["loss_finite"])
            self.assertNotIn("error_type", report[mode])
        self.assertFalse(hasattr(model, "compressor"))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_standalone_training_step_requires_no_dummy_compressor(self):
        from rhpr.sasrec.query_prefix_pilot import train_prefix_pass

        model = self.model()
        config = {"training": {"raw_tail_size": 128, "effective_batch_size": 2,
            "supervised_targets": 128, "num_negatives": 4, "amp": False, "grad_clip_norm": 1.0}}
        arrays = {key: value.numpy() for key, value in self.batch([140, 137]).items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        result = train_prefix_pass(model, None, (0, 1), config=config, variant={}, physical_batch=2,
            device=torch.device("cpu"), optimizer=optimizer, scaler=scaler,
            batch_builder=lambda *_args: arrays)
        self.assertEqual(result["successful_updates"], 1)
        self.assertEqual(result["labels"], 256)
        self.assertFalse(hasattr(model, "compressor"))
        self.assertEqual(model.tail_size, 128)


if __name__ == "__main__":
    unittest.main()
