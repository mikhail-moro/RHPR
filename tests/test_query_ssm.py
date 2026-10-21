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


KINDS = ("query_cross_attention", "query_ssm", "query_ssm_readout")


@unittest.skipIf(torch is None, "torch is not installed")
class QueryCompressorTest(unittest.TestCase):
    """Tiny four-event queries reproduce the 128 + 31 training-block contract."""

    def compressor(self, kind="query_ssm", *, tail=6, context=4, slots=2):
        from rhpr.sasrec.query_ssm import make_query_compressor

        return make_query_compressor(
            kind=kind, d_model=8, summary_tokens=slots, tail_size=tail,
            query_context_size=context, n_heads=2, d_state=3,
            d_inner=4, gate_rank=3, dropout=0.0,
        ).eval()

    def test_summaries_read_initial_query_context_but_not_later_targets(self):
        torch.manual_seed(37)
        values = torch.randn(1, 17, 8)
        mask = torch.ones((1, 17), dtype=torch.bool)
        old = 11
        # The query observes tail positions 0..3. Positions 4..5 are future
        # inputs relative to the first target and must not influence summaries.
        later_changed = values.clone()
        later_changed[:, old + 4:] = torch.randn_like(later_changed[:, old + 4:]) * 10
        recent_changed = values.clone()
        recent_changed[:, old:old + 4] = torch.randn_like(recent_changed[:, old:old + 4]) * 10
        old_changed = values.clone()
        old_changed[:, :old] = torch.randn_like(old_changed[:, :old]) * 10
        for kind in KINDS:
            with self.subTest(kind=kind), torch.no_grad():
                compressor = self.compressor(kind)
                before, output_mask, sources = compressor(values, mask)
                later = compressor(later_changed, mask)[0]
                recent = compressor(recent_changed, mask)[0]
                history = compressor(old_changed, mask)[0]
                torch.testing.assert_close(before[:, :2], later[:, :2], rtol=0, atol=0)
                self.assertFalse(torch.allclose(before[:, :2], recent[:, :2], rtol=1e-5, atol=1e-7))
                self.assertFalse(torch.allclose(before[:, :2], history[:, :2], rtol=1e-5, atol=1e-7))
                self.assertTrue(bool(output_mask.all()))
                # Source provenance includes query inputs, not only old memory.
                self.assertEqual(sources[:, :2].tolist(), [[old + 3, old + 3]])
                self.assertEqual(sources[:, 2:].tolist(), [list(range(old, 17))])
                torch.testing.assert_close(before[:, 2:], values[:, -6:], rtol=0, atol=0)

    def test_summary_gradient_cannot_reach_future_targets_or_padding(self):
        torch.manual_seed(85)
        original = torch.randn(1, 20, 8)
        mask = torch.arange(20)[None, :] < 17
        for kind in KINDS:
            with self.subTest(kind=kind):
                values = original.clone().requires_grad_()
                compressor = self.compressor(kind)
                summaries = compressor(values, mask)[0][:, :2]
                weights = torch.linspace(0.3, 1.7, 8)
                (summaries.square() * weights).sum().backward()
                self.assertGreater(float(values.grad[:, :11].abs().sum()), 0)
                self.assertGreater(float(values.grad[:, 11:15].abs().sum()), 0)
                self.assertEqual(int(values.grad[:, 15:].count_nonzero()), 0)
                self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_ragged_packing_matches_single_histories_and_padding_is_ignored(self):
        torch.manual_seed(23)
        lengths = torch.tensor([17, 11, 6, 4, 1])
        values = torch.randn(5, 21, 8)
        mask = torch.arange(21)[None, :] < lengths[:, None]
        changed_padding = values.clone()
        changed_padding[~mask] = torch.randn_like(changed_padding[~mask]) * 1000
        for kind in KINDS:
            with self.subTest(kind=kind), torch.no_grad():
                compressor = self.compressor(kind)
                compressed, output_mask, sources = compressor(values, mask)
                padded = compressor(changed_padding, mask)
                torch.testing.assert_close(compressed, padded[0], rtol=0, atol=0)
                old_lengths = (lengths - 6).clamp(min=0)
                expected_lengths = old_lengths.clamp(max=2) + lengths.clamp(max=6)
                self.assertTrue(torch.equal(output_mask.sum(dim=1), expected_lengths))
                self.assertTrue(torch.equal(output_mask, torch.arange(compressed.shape[1])[None, :] < expected_lengths[:, None]))
                self.assertEqual(int(compressed[~output_mask].count_nonzero()), 0)
                self.assertTrue(bool((sources[~output_mask] == -1).all()))
                for row, length in enumerate(lengths.tolist()):
                    with self.subTest(length=length):
                        end = int(expected_lengths[row])
                        solo = compressor(values[row:row + 1, :length], mask[row:row + 1, :length])
                        torch.testing.assert_close(compressed[row, :end], solo[0][0], rtol=2e-5, atol=2e-6)
                        self.assertTrue(torch.equal(sources[row, :end], solo[2][0]))
                        tail = min(length, 6)
                        torch.testing.assert_close(compressed[row, end-tail:end], values[row, length-tail:length], rtol=0, atol=0)

    def test_observed_zero_event_keeps_its_slot_and_can_receive_gradient(self):
        torch.manual_seed(19)
        values = torch.randn(1, 10, 8)
        values[:, 2] = 0  # An observed OOV event, not right padding.
        mask = torch.ones((1, 10), dtype=torch.bool)
        for kind in KINDS:
            with self.subTest(kind=kind):
                inputs = values.clone().requires_grad_()
                compressor = self.compressor(kind)
                output, output_mask, sources = compressor(inputs, mask)
                self.assertEqual(int(output_mask.sum()), 8)
                self.assertEqual(sources[:, 2:].tolist(), [list(range(4, 10))])
                (output[:, :2].square() * torch.arange(1., 9.)).sum().backward()
                self.assertGreater(float(inputs.grad[:, 2].abs().sum()), 0)

    def test_query_conditioned_writing_is_distinct_from_query_only_readout(self):
        torch.manual_seed(41)
        memory = torch.randn(2, 7, 8)
        queries = torch.randn(2, 2, 8)
        changed_queries = torch.randn_like(queries) * 3
        recent = self.compressor("query_ssm")
        readout = self.compressor("query_ssm_readout")
        for kind, compressor in (("recent", recent), ("readout", readout)):
            with self.subTest(kind=kind):
                before = compressor.build_scan_inputs(memory, queries)
                after = compressor.build_scan_inputs(memory, changed_queries)
                torch.testing.assert_close(before["u"], after["u"], rtol=0, atol=0)
                torch.testing.assert_close(before["A"], after["A"], rtol=0, atol=0)
                self.assertFalse(torch.allclose(before["C"], after["C"]))
                if kind == "readout":
                    # All recurrence coefficients are independent of recent q.
                    for name in ("delta", "B", "delta_bias"):
                        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
                else:
                    self.assertFalse(torch.allclose(before["delta"], after["delta"]))
                    self.assertFalse(torch.allclose(before["B"], after["B"]))

    def test_last_output_backpropagates_through_early_history_and_scan_parameters(self):
        torch.manual_seed(73)
        compressor = self.compressor("query_ssm")
        values = torch.randn(1, 13, 8, requires_grad=True)
        output = compressor(values, torch.ones((1, 13), dtype=torch.bool))[0]
        (output[:, :2] * torch.linspace(-0.8, 1.5, 8)).square().sum().backward()
        self.assertGreater(float(values.grad[:, 0].abs().sum()), 0)
        for name in ("A_log", "input_gate", "query_gate", "delta_projection", "write_projection", "read_projection", "query_generator"):
            parameters = [(key, value) for key, value in compressor.named_parameters()
                          if key == name or key.startswith(name + ".")]
            self.assertTrue(parameters, name)
            gradients = [parameter.grad for _, parameter in parameters]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), name)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0, name)

    def test_factory_preserves_rng_and_rejects_invalid_contracts(self):
        from rhpr.sasrec.query_ssm import make_query_compressor

        torch.manual_seed(71)
        before = torch.get_rng_state()
        for kind in KINDS:
            self.compressor(kind)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for kind in KINDS:
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    self.compressor(kind, context=7, tail=6)
                with self.assertRaises(ValueError):
                    self.compressor(kind, context=0)
                with self.assertRaises(ValueError):
                    self.compressor(kind, slots=0)
                compressor = self.compressor(kind)
                values = torch.randn(1, 4, 8)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    compressor(values, torch.tensor([[True, False, True, False]]))
                with self.assertRaises(ValueError):
                    compressor(values, torch.zeros((1, 4), dtype=torch.bool))
        with self.assertRaises(ValueError):
            make_query_compressor(kind="unknown", d_model=8, summary_tokens=2, tail_size=6)


@unittest.skipIf(torch is None, "torch is not installed")
class SelectiveScanReferenceTest(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(101)
        return {
            "u": torch.randn(2, 4, 5),
            "delta": torch.randn(2, 4, 5) * .2,
            "A": -torch.rand(4, 3) - .1,
            "B": torch.randn(2, 3, 5) * .3,
            "C": torch.randn(2, 3, 5) * .3,
            "delta_bias": torch.randn(4) * .1,
        }

    def test_reference_matches_independent_affine_recurrence(self):
        from rhpr.sasrec.query_ssm import selective_scan_reference

        # One channel and one state; expansion gives y2 = C2 *
        # (exp(dt2*A) * dt1*B1*u1 + dt2*B2*u2).
        u = torch.tensor([[[2., 3.]]])
        delta = torch.tensor([[[.2, .4]]])
        A = torch.tensor([[-.7]])
        B = torch.tensor([[[.5, -.6]]])
        C = torch.tensor([[[.8, 1.1]]])
        output = selective_scan_reference(u, delta, A, B, C, delta_softplus=False)
        first_state = .2 * .5 * 2
        second_state = torch.exp(torch.tensor(-.7 * .4)) * first_state + .4 * -.6 * 3
        expected = torch.tensor([[[.8 * first_state, 1.1 * second_state]]])
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
        inputs = self.inputs()
        with_bias = selective_scan_reference(**inputs)
        inputs["delta"] = inputs["delta"] + inputs.pop("delta_bias")[None, :, None]
        without_bias = selective_scan_reference(**inputs)
        torch.testing.assert_close(with_bias, without_bias, rtol=0, atol=0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_native_matches_reference_output_and_final_output_gradients(self):
        from rhpr.sasrec.query_ssm import make_query_compressor, selective_scan_reference

        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        except (ImportError, OSError) as error:  # pragma: no cover
            self.skipTest(f"native selective scan is not installed: {error}")
        reference_inputs = {key: value.clone().requires_grad_() for key, value in self.inputs().items()}
        native_inputs = {key: value.detach().cuda().requires_grad_() for key, value in reference_inputs.items()}
        reference = selective_scan_reference(**reference_inputs)
        compressor = make_query_compressor(kind="query_ssm", d_model=8, summary_tokens=2,
            tail_size=6, query_context_size=4, n_heads=2, d_inner=4, d_state=3, gate_rank=3).cuda()
        native = compressor.scan_output(native_inputs, backend="native")
        torch.testing.assert_close(native.cpu(), reference, rtol=3e-4, atol=3e-5)
        weights = torch.tensor([[.2, -.7, 1.3, .5], [-.3, .9, .4, -.8]])
        # A loss on the last *output*, never the returned last_state. This must
        # train even the first input event through the complete recurrence.
        (reference[:, :, -1] * weights).square().sum().backward()
        (native[:, :, -1] * weights.cuda()).square().sum().backward()
        self.assertGreater(float(native_inputs["u"].grad[:, :, 0].abs().sum()), 0)
        for name in reference_inputs:
            with self.subTest(parameter=name):
                self.assertIsNotNone(native_inputs[name].grad)
                torch.testing.assert_close(native_inputs[name].grad.cpu(), reference_inputs[name].grad,
                                           rtol=1e-3, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
