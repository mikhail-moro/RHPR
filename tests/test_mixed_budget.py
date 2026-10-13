from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class MixedBudgetTest(unittest.TestCase):
    def model(self, *, tail=159, normalization=191, budgets=(1, 2, 4, 8, 16, 32)):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.mixed_budget import make_mixed_budget_compressor

        compressor = make_mixed_budget_compressor(8, budgets=budgets, tail_size=tail)
        return HSTUModel(
            num_items=200, max_seq_len=16384, d_model=8, n_heads=2,
            n_layers=1, ffn_multiplier=4, dropout=0,
            custom_sequence_compressor=compressor,
            normalization_length=normalization,
        ).eval()

    def test_all_16k_budgets_preserve_raw_tail_in_training_and_evaluation(self):
        from rhpr.sasrec.mixed_budget import DEFAULT_SUMMARY_BUDGETS, select_summary_budget

        model = self.model()
        values = torch.randn(1, 16384, 8)
        mask = torch.ones((1, 16384), dtype=torch.bool)
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        compressor_id = id(model.compressor)
        snapshots = {name: value.clone() for name, value in model.state_dict().items()}
        for tail in (159, 128):
            model.compressor.tail_size = tail
            model.tail_size = tail
            for budget in DEFAULT_SUMMARY_BUDGETS:
                with self.subTest(tail=tail, budget=budget), torch.no_grad():
                    select_summary_budget(model, budget)
                    compressed, output_mask, sources = model.compressor(values, mask)
                    self.assertEqual(compressed.shape, (1, tail + budget, 8))
                    self.assertTrue(bool(output_mask.all()))
                    torch.testing.assert_close(compressed[:, -tail:], values[:, -tail:], rtol=0, atol=0)
                    self.assertEqual(sources[0, -tail:].tolist(), list(range(16384 - tail, 16384)))
                    self.assertTrue(bool((sources[:, :budget] < 16384 - tail).all()))
                    self.assertEqual(model.hstu_context_len, tail + budget)
                    self.assertEqual(model.hstu_normalization_length, 191)
                    self.assertEqual(model.position_embedding.num_embeddings, 16384)
                    self.assertEqual(model.encoder.layers[0]._normalization_length, 191)
                    self.assertEqual(model.encoder.layers[0]._rel_attn_bias._w.numel(), 381)
                    self.assertEqual(id(model.compressor), compressor_id)
                    self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), parameter_ids)
                    for name, parameter in model.state_dict().items():
                        torch.testing.assert_close(parameter, snapshots[name], rtol=0, atol=0)

    def test_every_budget_keeps_the_first_training_target_causal(self):
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors
        from rhpr.sasrec.mixed_budget import DEFAULT_SUMMARY_BUDGETS, select_summary_budget

        torch.manual_seed(207)
        model = self.model()
        items = (torch.arange(16384) % 70 + 1)[None, :]
        future_changed = items.clone()
        future_changed[:, -31:] += 70
        mask = torch.ones_like(items, dtype=torch.bool)
        for budget in DEFAULT_SUMMARY_BUDGETS:
            with self.subTest(budget=budget), torch.no_grad():
                select_summary_budget(model, budget)
                before, output_mask, sources = model.encode(items, mask)
                after, after_mask, after_sources = model.encode(future_changed, mask)
                self.assertEqual(before.shape, (1, 159 + budget, 8))
                self.assertTrue(torch.equal(output_mask, after_mask))
                self.assertTrue(torch.equal(sources, after_sources))
                first_supervised = budget + 127
                self.assertEqual(int(sources[0, first_supervised]), 16352)
                torch.testing.assert_close(before[:, first_supervised], after[:, first_supervised], rtol=0, atol=1e-6)
                self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))
                loss_mask = build_loss_mask(output_mask, sources, mask.sum(dim=1), 32)
                selected_states, targets, selected = supervised_state_tensors(
                    before, loss_mask, sources, items + 1,
                )
                self.assertEqual(selected_states.shape, (1, 32, 8))
                self.assertTrue(bool(selected.all()))
                torch.testing.assert_close(targets, (items + 1)[:, -32:], rtol=0, atol=0)

    def test_all_old_events_and_shared_merge_receive_gradients_at_every_budget(self):
        from rhpr.sasrec.mixed_budget import DEFAULT_SUMMARY_BUDGETS, select_summary_budget

        model = self.model(tail=128)
        original = torch.randn(1, 16384, 8)
        mask = torch.ones((1, 16384), dtype=torch.bool)
        projection = model.compressor.projection
        for budget in DEFAULT_SUMMARY_BUDGETS:
            with self.subTest(budget=budget):
                model.zero_grad(set_to_none=True)
                select_summary_budget(model, budget)
                inputs = original.clone().requires_grad_()
                compressed, _, _ = model.compressor(inputs, mask)
                compressed[:, :budget].square().sum().backward()
                self.assertTrue(bool(inputs.grad[:, :16256].abs().sum(dim=-1).gt(0).all()))
                self.assertEqual(int(inputs.grad[:, 16256:].count_nonzero()), 0)
                self.assertIs(model.compressor.projection, projection)
                self.assertTrue(bool(torch.isfinite(projection.weight.grad).all()))
                self.assertGreater(float(projection.weight.grad.abs().sum()), 0)

    def test_budget_schedules_are_balanced_reproducible_and_rng_independent(self):
        from rhpr.sasrec.mixed_budget import DEFAULT_SUMMARY_BUDGETS, balanced_budget_schedule

        torch.manual_seed(2026)
        random.seed(2026)
        torch_state = torch.get_rng_state().clone()
        python_state = random.getstate()
        schedules = []
        for epoch in range(20):
            schedule = balanced_budget_schedule(seed=42, epoch=epoch, effective_updates=35)
            self.assertEqual(len(schedule), 35)
            counts = Counter(schedule)
            self.assertEqual(set(counts), set(DEFAULT_SUMMARY_BUDGETS))
            self.assertEqual(sorted(counts.values()), [5, 6, 6, 6, 6, 6])
            schedules.append(schedule)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertGreater(len(set(schedules)), 1)
        random.random()
        torch.randn(30)
        self.assertEqual(
            schedules[0], balanced_budget_schedule(seed=42, epoch=0, effective_updates=35),
        )
        self.assertNotEqual(
            schedules[0], balanced_budget_schedule(seed=43, epoch=0, effective_updates=35),
        )
        self.assertEqual(
            schedules[0], balanced_budget_schedule(seed=42, epoch=0, effective_updates=35,
                                                 budgets=(32, 16, 8, 4, 2, 1)),
        )

    def test_incomplete_cycles_randomize_extra_updates_instead_of_favoring_small_budgets(self):
        from rhpr.sasrec.mixed_budget import DEFAULT_SUMMARY_BUDGETS, balanced_budget_schedule

        extra_budgets = set()
        for epoch in range(40):
            schedule = balanced_budget_schedule(seed=52, epoch=epoch, effective_updates=11)
            counts = Counter(schedule)
            self.assertEqual(set(counts), set(DEFAULT_SUMMARY_BUDGETS))
            self.assertEqual(sorted(counts.values()), [1, 2, 2, 2, 2, 2])
            extra_budgets.update(budget for budget, count in counts.items() if count == 2)
        self.assertEqual(extra_budgets, set(DEFAULT_SUMMARY_BUDGETS))

    def test_physical_microbatch_size_cannot_change_effective_budget_or_sampling_rng(self):
        from rhpr.sasrec.mixed_budget import balanced_budget_schedule

        # A budget is indexed once per optimizer update.  Both accumulation
        # shapes therefore retain the same choice for all windows in that update.
        schedule = balanced_budget_schedule(seed=28, epoch=7, effective_updates=35)
        reference = tuple(budget for budget in schedule for _ in range(128))[:4427]
        for physical_batch in (8, 16, 32):
            windows = []
            for budget in schedule:
                for _ in range(128 // physical_batch):
                    windows.extend([budget] * physical_batch)
            self.assertEqual(tuple(windows[:4427]), reference)
        torch.manual_seed(13)
        negatives_before = torch.randint(1, 10000, (6, 512))
        torch.manual_seed(13)
        balanced_budget_schedule(seed=28, epoch=7, effective_updates=35)
        negatives_after = torch.randint(1, 10000, (6, 512))
        self.assertTrue(torch.equal(negatives_before, negatives_after))

    def test_factory_preserves_backbone_rng_and_initializes_only_one_merge_projection(self):
        from rhpr.sasrec.mixed_budget import make_mixed_budget_compressor

        torch.manual_seed(11)
        expected_state = torch.get_rng_state().clone()
        compressor = make_mixed_budget_compressor(8)
        self.assertTrue(torch.equal(expected_state, torch.get_rng_state()))
        self.assertEqual(compressor.summary_tokens, 32)
        self.assertEqual(compressor.mixed_budget_choices, (1, 2, 4, 8, 16, 32))
        self.assertEqual(set(compressor.state_dict()), {"projection.weight", "projection.bias"})
        self.assertEqual(compressor.projection.weight.shape, (8, 32))
        self.assertEqual(int(compressor.projection.weight.count_nonzero()), 0)

    def test_invalid_budgets_fail_before_mutating_model_state(self):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.mixed_budget import (
            balanced_budget_schedule, make_mixed_budget_compressor, select_summary_budget,
        )

        for budgets in ((), (0, 1), (True, 2), (1.5, 2), (1, 1)):
            with self.subTest(budgets=budgets), self.assertRaisesRegex(ValueError, "budgets"):
                make_mixed_budget_compressor(8, budgets=budgets)
        model = self.model()
        for budget in (0, 3, 17, 33, True, 1.0):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "declared"):
                select_summary_budget(model, budget)
            self.assertEqual(model.compressor.summary_tokens, 32)
            self.assertEqual(model.hstu_context_len, 191)
        insufficient_table = self.model(normalization=175)
        with self.assertRaisesRegex(ValueError, "largest declared"):
            select_summary_budget(insufficient_table, 1)
        self.assertEqual(insufficient_table.compressor.summary_tokens, 32)
        identity = HSTUModel(num_items=20, max_seq_len=32, d_model=8, n_heads=2,
                             n_layers=1, ffn_multiplier=4, dropout=0)
        with self.assertRaisesRegex(ValueError, "mixed-budget"):
            select_summary_budget(identity, 1)
        for kwargs in ({"seed": True}, {"seed": 1.2}, {"epoch": -1},
                       {"epoch": True}, {"effective_updates": -1},
                       {"effective_updates": 1.5}):
            arguments = dict(seed=1, epoch=1, effective_updates=35)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                balanced_budget_schedule(**arguments)
        self.assertEqual(balanced_budget_schedule(seed=1, epoch=0, effective_updates=0), ())


if __name__ == "__main__":
    unittest.main()
