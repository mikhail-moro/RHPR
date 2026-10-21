from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class LongCompressionTest(unittest.TestCase):
    def compressor(self, kind, *, slots=4, tail=3, dimensions=8, arity=4):
        from rhpr.sasrec.long_compression import make_prefix_compressor

        return make_prefix_compressor(kind=kind, d_model=dimensions, summary_tokens=slots,
                                      tail_size=tail, group_arity=arity, n_heads=2).eval()

    def batch(self):
        lengths = torch.tensor([65, 17, 8, 4, 3, 1])
        mask = torch.arange(68)[None, :] < lengths[:, None]
        values = torch.randn(6, 68, 8)
        return values, mask, lengths

    def test_16k_exact_summary_budgets_and_untouched_128_tail(self):
        values = torch.randn(1, 16384, 8)
        mask = torch.ones((1, 16384), dtype=torch.bool)
        for slots in (16, 8, 4):
            for kind in ("mean", "uniform", "pair_mlp", "qformer"):
                with self.subTest(slots=slots, kind=kind), torch.no_grad():
                    compressed, output_mask, sources = self.compressor(kind, slots=slots, tail=128)(values, mask)
                    self.assertEqual(compressed.shape, (1, 128 + slots, 8))
                    self.assertTrue(bool(output_mask.all()))
                    self.assertTrue(torch.equal(compressed[:, -128:], values[:, -128:]))
                    self.assertTrue(torch.equal(sources[:, -128:], torch.arange(16256, 16384)[None, :]))
                    self.assertTrue(bool((sources[:, :slots] < 16256).all()))
                    if kind == "qformer":
                        self.assertEqual(sources[:, :slots].tolist(), [[16255] * slots])

    def test_ragged_masks_pack_summaries_directly_before_raw_tail(self):
        values, mask, lengths = self.batch()
        for kind in ("mean", "uniform", "pair_mlp", "qformer"):
            with self.subTest(kind=kind), torch.no_grad():
                compressor = self.compressor(kind)
                compressed, output_mask, sources = compressor(values, mask)
                old = (lengths - 3).clamp(min=0)
                expected_lengths = old.clamp(max=4) + lengths.clamp(max=3)
                self.assertTrue(torch.equal(output_mask.sum(dim=1), expected_lengths))
                self.assertTrue(torch.equal(output_mask, torch.arange(compressed.shape[1])[None, :] < expected_lengths[:, None]))
                self.assertEqual(int(compressed[~output_mask].count_nonzero()), 0)
                self.assertTrue(bool((sources[~output_mask] == -1).all()))
                for row, length in enumerate(lengths.tolist()):
                    tail = min(length, 3)
                    end = int(expected_lengths[row])
                    self.assertTrue(torch.equal(compressed[row, end - tail:end], values[row, length - tail:length]))
                    self.assertEqual(sources[row, end - tail:end].tolist(), list(range(length - tail, length)))
                    solo = compressor(values[row:row + 1, :length], mask[row:row + 1, :length])
                    torch.testing.assert_close(compressed[row, :end], solo[0][0], rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.equal(sources[row, :end], solo[2][0]))

    def test_zero_initialized_binary_and_four_way_hierarchies_equal_regional_mean(self):
        values, mask, _ = self.batch()
        for slots in (1, 2, 4, 8, 16):
            expected = self.compressor("mean", slots=slots)(values, mask)
            for arity in (2, 4):
                with self.subTest(slots=slots, arity=arity):
                    actual = self.compressor("pair_mlp", slots=slots, arity=arity)(values, mask)
                    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
                    self.assertTrue(torch.equal(actual[1], expected[1]))
                    self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_learned_merge_receives_every_old_event_and_never_recent_tail_or_padding(self):
        values, mask, lengths = self.batch()
        for kind in ("pair_mlp", "qformer"):
            with self.subTest(kind=kind):
                inputs = values.detach().clone().requires_grad_()
                compressor = self.compressor(kind)
                outputs, output_mask, sources = compressor(inputs, mask)
                old_lengths = (lengths - 3).clamp(min=0)
                summary_mask = output_mask & (sources < old_lengths[:, None])
                weights = torch.linspace(0.7, 1.3, outputs.shape[-1])
                loss = (outputs.square() * weights[None, None, :])[summary_mask].sum()
                loss.backward()
                for row, old in enumerate(old_lengths.tolist()):
                    if old:
                        self.assertTrue(bool(inputs.grad[row, :old].abs().sum(dim=-1).gt(0).all()))
                    self.assertEqual(int(inputs.grad[row, old:].count_nonzero()), 0)
                for name, parameter in compressor.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                if kind == "pair_mlp":
                    self.assertGreater(float(compressor.projection.weight.grad.abs().sum()), 0)
                else:
                    self.assertGreater(float(compressor.cross_attention.in_proj_weight.grad.abs().sum()), 0)

    def test_summaries_cannot_see_recent_tail_and_padding_is_ignored(self):
        values, mask, lengths = self.batch()
        old_lengths = (lengths - 3).clamp(min=0)
        changed = values.clone()
        changed[torch.arange(values.shape[1])[None, :] >= old_lengths[:, None]] += 123
        for kind in ("mean", "uniform", "pair_mlp", "qformer"):
            with self.subTest(kind=kind), torch.no_grad():
                compressor = self.compressor(kind)
                before = compressor(values, mask)
                after = compressor(changed, mask)
                selected = before[1] & (before[2] < old_lengths[:, None])
                torch.testing.assert_close(before[0][selected], after[0][selected], rtol=0, atol=0)
                self.assertTrue(torch.equal(before[2], after[2]))

    def test_qformer_position_features_make_old_event_order_observable(self):
        torch.manual_seed(102)
        compressor = self.compressor("qformer")
        values = torch.randn(1, 35, 8)
        mask = torch.ones((1, 35), dtype=torch.bool)
        permuted = values.clone()
        permuted[:, :32] = permuted[:, :32].flip(dims=(1,))
        with torch.no_grad():
            original = compressor(values, mask)[0][:, :4]
            changed = compressor(permuted, mask)[0][:, :4]
            self.assertFalse(torch.allclose(original, changed, rtol=1e-6, atol=1e-7))
            # Without positional K/V features, cross-attention is invariant to a
            # permutation of the same set of prefix embeddings.
            compressor.positional_scale = 0
            original = compressor(values, mask)[0][:, :4]
            changed = compressor(permuted, mask)[0][:, :4]
            torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-6)

    def test_hstu_tail_queries_remain_causal_and_select_identical_targets(self):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors

        items = torch.arange(1, 33)[None, :]
        changed = items.clone()
        changed[:, -2:] += 40
        mask = torch.ones_like(items, dtype=torch.bool)
        for kind in ("mean", "uniform", "pair_mlp", "qformer"):
            with self.subTest(kind=kind):
                torch.manual_seed(54)
                model = HSTUModel(num_items=100, max_seq_len=32, d_model=8,
                                  n_heads=2, n_layers=2, ffn_multiplier=4, dropout=0)
                model.compressor = self.compressor(kind, slots=4, tail=4)
                model.eval()
                with torch.no_grad():
                    before, output_mask, sources = model.encode(items, mask)
                    after, _, changed_sources = model.encode(changed, mask)
                first_tail = int(torch.where(sources[0] == 28)[0][0])
                torch.testing.assert_close(before[:, first_tail], after[:, first_tail], rtol=0, atol=1e-6)
                self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))
                self.assertTrue(torch.equal(sources, changed_sources))
                loss_mask = build_loss_mask(output_mask, sources, mask.sum(dim=1), 4)
                states, targets, selected = supervised_state_tensors(before, loss_mask, sources, items + 1)
                self.assertEqual(states.shape, (1, 4, 8))
                self.assertEqual(targets.tolist(), [[30, 31, 32, 33]])
                self.assertTrue(bool(selected.all()))

    def test_factory_preserves_backbone_rng_and_invalid_inputs_fail(self):
        from rhpr.sasrec.long_compression import make_prefix_compressor

        torch.manual_seed(42)
        expected = torch.get_rng_state()
        for kind in ("mean", "uniform", "pair_mlp", "qformer"):
            self.compressor(kind)
            self.assertTrue(torch.equal(expected, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.compressor("unknown")
        with self.assertRaisesRegex(ValueError, "positive integers"):
            self.compressor("mean", slots=0)
        with self.assertRaisesRegex(ValueError, "group_arity"):
            self.compressor("pair_mlp", arity=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_prefix_compressor(kind="qformer", d_model=7, summary_tokens=4, n_heads=2)
        values = torch.randn(1, 5, 8)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.compressor("mean")(values, torch.tensor([[True, False, True, False, False]]))
        with self.assertRaisesRegex(ValueError, "nonempty histories"):
            self.compressor("mean")(values, torch.zeros((1, 5), dtype=torch.bool))

    def test_nondivisible_regions_have_ordered_exact_endpoints_and_event_means(self):
        # Eleven old events split into widths 2,3,3,3, followed by three untouched
        # recent events.  Inspect arithmetic independently of the implementation.
        values = torch.arange(14, dtype=torch.float32)[None, :, None].expand(-1, -1, 8)
        mask = torch.ones((1, 14), dtype=torch.bool)
        expected_endpoints = [[1, 4, 7, 10, 11, 12, 13]]
        expected_means = torch.tensor([0.5, 3., 6., 9., 11., 12., 13.])
        for kind in ("mean", "pair_mlp"):
            with self.subTest(kind=kind):
                output, _, sources = self.compressor(kind)(values, mask)
                self.assertEqual(sources.tolist(), expected_endpoints)
                torch.testing.assert_close(output[0, :, 0], expected_means, rtol=0, atol=0)
        output, _, sources = self.compressor("uniform")(values, mask)
        self.assertEqual(sources.tolist(), expected_endpoints)
        self.assertEqual(output[0, :, 0].tolist(), [1., 4., 7., 10., 11., 12., 13.])

    def test_observed_zero_embedding_is_counted_instead_of_treated_as_padding(self):
        # Catalogue-unknown observed events may use embedding ID zero, while their
        # event mask is true.  Compression lengths must follow that mask alone.
        values = torch.tensor([1., 0., 5., 7.])[None, :, None].expand(-1, -1, 8)
        mask = torch.ones((1, 4), dtype=torch.bool)
        for kind in ("mean", "pair_mlp"):
            with self.subTest(kind=kind):
                output, output_mask, sources = self.compressor(kind, slots=1, tail=1)(values, mask)
                self.assertEqual(output[0, :, 0].tolist(), [2., 7.])
                self.assertEqual(sources.tolist(), [[2, 3]])
                self.assertTrue(bool(output_mask.all()))

    def test_hstu_default_matches_explicit_original_normalization_and_seed_state(self):
        from rhpr.sasrec.hstu import HSTUModel

        kwargs = dict(num_items=100, max_seq_len=32, d_model=8,
                      n_heads=2, n_layers=2, ffn_multiplier=4, dropout=0)
        torch.manual_seed(89)
        original = HSTUModel(**kwargs).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(89)
        explicit = HSTUModel(**kwargs, normalization_length=32).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, parameter in original.state_dict().items():
            self.assertTrue(torch.equal(parameter, explicit.state_dict()[name]), name)
        inputs = torch.arange(1, 33)[None, :]
        mask = torch.ones_like(inputs, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(original.encode(inputs, mask)[0], explicit.encode(inputs, mask)[0], rtol=0, atol=0)

    def test_training159_and_evaluation128_keep_common175_normalization_and_tail_causality(self):
        from rhpr.sasrec.hstu import HSTUModel

        items = ((torch.arange(16384) % 80) + 1)[None, :]
        mask = torch.ones_like(items, dtype=torch.bool)
        future_changed = items.clone()
        future_changed[:, -31:] += 100
        for kind in ("mean", "uniform", "pair_mlp", "qformer"):
            with self.subTest(kind=kind):
                compressor = self.compressor(kind, slots=16, tail=159)
                torch.manual_seed(90)
                model = HSTUModel(num_items=300, max_seq_len=16384, d_model=8,
                                  n_heads=2, n_layers=2, ffn_multiplier=4, dropout=0,
                                  custom_sequence_compressor=compressor,
                                  normalization_length=175).eval()
                self.assertEqual(model.compressor_type, "custom")
                self.assertEqual(model.position_embedding.num_embeddings, 16384)
                self.assertEqual(model.hstu_normalization_length, 175)
                self.assertEqual(model.hstu_context_len, 175)
                self.assertEqual(model.compressor.tail_size, 159)
                for block in model.encoder.layers:
                    self.assertEqual(block._normalization_length, 175)
                    self.assertEqual(block._rel_attn_bias._w.numel(), 349)
                with torch.no_grad():
                    before, output_mask, sources = model.encode(items, mask)
                    after, _, _ = model.encode(future_changed, mask)
                self.assertEqual(before.shape, (1, 175, 8))
                self.assertTrue(bool(output_mask.all()))
                self.assertEqual(sources[0, -159:].tolist(), list(range(16225, 16384)))
                # The first of G=32 training states sees exactly 128 raw recent
                # events after compression; later target events remain hidden.
                first_supervised = int(torch.where(sources[0] == 16352)[0][0])
                self.assertEqual(first_supervised - 16 + 1, 128)
                torch.testing.assert_close(before[:, first_supervised], after[:, first_supervised], rtol=0, atol=1e-6)
                # The evaluation change is only the raw-tail boundary; both the
                # attention divisor and parameter tables remain the same.
                model.compressor.tail_size = 128
                model.tail_size = 128
                with torch.no_grad():
                    evaluated, evaluated_mask, evaluated_sources = model.encode(items, mask)
                self.assertEqual(evaluated.shape, (1, 144, 8))
                self.assertTrue(bool(evaluated_mask.all()))
                self.assertEqual(evaluated_sources[0, -128:].tolist(), list(range(16256, 16384)))
                self.assertEqual(model.hstu_normalization_length, 175)
                self.assertEqual(model.encoder.layers[0]._normalization_length, 175)

    def test_custom_hstu_rejects_incompatible_options_and_attention_width(self):
        from rhpr.sasrec.hstu import HSTUModel

        kwargs = dict(num_items=100, max_seq_len=32, d_model=8,
                      n_heads=2, n_layers=1, ffn_multiplier=4, dropout=0)
        for value in (0, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "normalization_length"):
                HSTUModel(**kwargs, normalization_length=value)
        with self.assertRaisesRegex(ValueError, "choose a custom"):
            HSTUModel(**kwargs, compressor_type="tail_only",
                      custom_sequence_compressor=self.compressor("mean"))
        with self.assertRaisesRegex(ValueError, "nn.Module"):
            HSTUModel(**kwargs, custom_sequence_compressor=object())
        model = HSTUModel(**kwargs, normalization_length=8)
        inputs = torch.arange(1, 33)[None, :]
        with self.assertRaisesRegex(ValueError, "compressed sequence exceeds"):
            model.encode(inputs, torch.ones_like(inputs, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
