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
class OldPrefixQueryTest(unittest.TestCase):
    def compressor(self, kind="query_ssm", *, tail=6, context=4,
                   query_source="old_suffix"):
        from rhpr.sasrec.query_ssm import make_query_compressor

        return make_query_compressor(kind=kind, d_model=8, summary_tokens=2,
            tail_size=tail, query_context_size=context, n_heads=2, d_state=3,
            d_inner=4, gate_rank=3, dropout=0, query_source=query_source).eval()

    def test_query_context_is_the_occupied_suffix_of_each_old_prefix(self):
        torch.manual_seed(112)
        values = torch.randn(4, 19, 8)
        lengths = torch.tensor([17, 10, 6, 2])
        mask = torch.arange(19)[None, :] < lengths[:, None]
        for kind in KINDS:
            with self.subTest(kind=kind):
                compressor = self.compressor(kind)
                captured = []
                handle = compressor.query_generator.register_forward_pre_hook(
                    lambda module, arguments: captured.append(arguments))
                output, output_mask, sources = compressor(values, mask)
                handle.remove()
                context, context_mask = captured[0]
                torch.testing.assert_close(context[0], values[0, 7:11], rtol=0, atol=0)
                torch.testing.assert_close(context[1], values[1, :4], rtol=0, atol=0)
                self.assertEqual(context_mask.tolist(), [[True] * 4, [True] * 4,
                                                         [False] * 4, [False] * 4])
                self.assertEqual(int(context[2:].count_nonzero()), 0)
                self.assertEqual(sources[0, :2].tolist(), [10, 10])
                self.assertEqual(sources[1, :2].tolist(), [3, 3])
                self.assertEqual(sources[0, 2:].tolist(), list(range(11, 17)))
                self.assertEqual(output_mask.sum(dim=1).tolist(), [8, 8, 6, 2])
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertEqual(compressor.query_source, "old_suffix")
                self.assertEqual(compressor.source_position_policy,
                                 "latest_input_of_global_old_prefix")

    def test_all_raw_tail_and_padding_have_zero_summary_influence_and_gradient(self):
        torch.manual_seed(117)
        original = torch.randn(1, 21, 8)
        mask = torch.arange(21)[None, :] < 17
        changed = original.clone()
        changed[:, 11:] = torch.randn_like(changed[:, 11:]) * 100
        for kind in KINDS:
            with self.subTest(kind=kind):
                compressor = self.compressor(kind)
                inputs = original.clone().requires_grad_()
                summaries = compressor(inputs, mask)[0][:, :2]
                with torch.no_grad():
                    altered = compressor(changed, mask)[0][:, :2]
                torch.testing.assert_close(summaries, altered, rtol=0, atol=0)
                (summaries.square() * torch.linspace(.3, 1.7, 8)).sum().backward()
                self.assertGreater(float(inputs.grad[:, :7].abs().sum()), 0)
                self.assertGreater(float(inputs.grad[:, 7:11].abs().sum()), 0)
                self.assertEqual(int(inputs.grad[:, 11:].count_nonzero()), 0)
                self.assertTrue(bool(torch.isfinite(inputs.grad).all()))

    def test_fixed_prefix_summaries_can_be_reused_when_raw_tail_grows(self):
        torch.manual_seed(121)
        values = torch.randn(1, 19, 8)
        for kind in KINDS:
            with self.subTest(kind=kind), torch.no_grad():
                compressor = self.compressor(kind, tail=6)
                before = compressor(values[:, :17], torch.ones((1, 17), dtype=torch.bool))
                # Append two R events while keeping the same eleven-event H.
                # Sliding H itself would require recomputing its summaries.
                compressor.tail_size = 8
                after = compressor(values, torch.ones((1, 19), dtype=torch.bool))
                torch.testing.assert_close(before[0][:, :2], after[0][:, :2], rtol=0, atol=0)
                self.assertEqual(after[2][:, :2].tolist(), [[10, 10]])
                torch.testing.assert_close(after[0][:, 2:], values[:, 11:], rtol=0, atol=0)

    def test_ragged_short_and_observed_zero_histories_preserve_event_masks(self):
        torch.manual_seed(125)
        lengths = torch.tensor([17, 11, 7, 6, 4, 1])
        values = torch.randn(6, 21, 8)
        values[0, 8] = 0  # Observed OOV event inside old query support.
        mask = torch.arange(21)[None, :] < lengths[:, None]
        changed = values.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1000
        for kind in KINDS:
            with self.subTest(kind=kind):
                compressor = self.compressor(kind)
                inputs = values.clone().requires_grad_()
                output, output_mask, sources = compressor(inputs, mask)
                self.assertTrue(bool(torch.isfinite(output).all()))
                with torch.no_grad():
                    altered = compressor(changed, mask)
                    torch.testing.assert_close(output, altered[0], rtol=0, atol=0)
                    for row, length in enumerate(lengths.tolist()):
                        solo = compressor(values[row:row + 1, :length], mask[row:row + 1, :length])
                        end = int(output_mask[row].sum())
                        torch.testing.assert_close(output[row, :end], solo[0][0], rtol=2e-5, atol=2e-6)
                        self.assertTrue(torch.equal(sources[row, :end], solo[2][0]))
                        if length <= 6:
                            torch.testing.assert_close(output[row, :end], values[row, :length],
                                                       rtol=0, atol=0)
                self.assertEqual(int(output[~output_mask].count_nonzero()), 0)
                self.assertTrue(bool((sources[~output_mask] == -1).all()))
                output[0, :2].square().sum().backward()
                self.assertGreater(float(inputs.grad[0, 8].abs().sum()), 0)

    def test_old_query_context_may_exceed_raw_tail_and_short_old_prefix(self):
        torch.manual_seed(129)
        values = torch.randn(1, 7, 8)
        mask = torch.ones((1, 7), dtype=torch.bool)
        for kind in KINDS:
            with self.subTest(kind=kind):
                compressor = self.compressor(kind, tail=2, context=128)
                captured = []
                handle = compressor.query_generator.register_forward_pre_hook(
                    lambda module, arguments: captured.append(arguments))
                output, _, sources = compressor(values, mask)
                handle.remove()
                context, context_mask = captured[0]
                self.assertEqual(context_mask.sum().item(), 5)
                torch.testing.assert_close(context[:, :5], values[:, :5], rtol=0, atol=0)
                self.assertEqual(int(context[:, 5:].count_nonzero()), 0)
                self.assertEqual(sources.tolist(), [[4, 4, 5, 6]])
                self.assertTrue(bool(torch.isfinite(output).all()))
                with self.assertRaises(ValueError):
                    self.compressor(kind, tail=2, context=128, query_source="recent_start")
                with self.assertRaisesRegex(ValueError, "query_source"):
                    self.compressor(kind, query_source="unknown")

    def test_default_recent_source_preserves_state_keys_rng_and_outputs(self):
        from rhpr.sasrec.query_ssm import make_query_compressor

        torch.manual_seed(133)
        values = torch.randn(1, 17, 8)
        mask = torch.ones((1, 17), dtype=torch.bool)
        before = torch.get_rng_state().clone()
        for kind in KINDS:
            with self.subTest(kind=kind):
                arguments = dict(kind=kind, d_model=8, summary_tokens=2, tail_size=6,
                    query_context_size=4, n_heads=2, d_state=3, d_inner=4, gate_rank=3,
                    dropout=0)
                default = make_query_compressor(**arguments).eval()
                explicit = make_query_compressor(**arguments, query_source="recent_start").eval()
                old = make_query_compressor(**arguments, query_source="old_suffix").eval()
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(default.state_dict().keys(), old.state_dict().keys())
                for name, value in default.state_dict().items():
                    torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)
                with torch.no_grad():
                    for actual, expected in zip(default(values, mask), explicit(values, mask)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(default.source_position_policy,
                                 "latest_input_of_query_and_global_old_prefix")

    def test_every_raw_tail_state_is_causal_in_sasrec_and_hstu(self):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.model import SASRecModel

        items = torch.arange(1, 18)[None, :]
        mask = torch.ones_like(items, dtype=torch.bool)
        # Tiny analogue: H has eleven events, Q=H[-4:], R has six events.
        # Every R state predicts its next raw event after one shared compression.
        for architecture in ("sasrec", "hstu"):
            for kind in KINDS:
                with self.subTest(architecture=architecture, kind=kind):
                    torch.manual_seed(137)
                    common = dict(num_items=100, max_seq_len=17, d_model=8,
                                  n_heads=2, n_layers=2, ffn_multiplier=4, dropout=0)
                    if architecture == "hstu":
                        model = HSTUModel(**common, custom_sequence_compressor=self.compressor(kind),
                                          normalization_length=8).eval()
                    else:
                        model = SASRecModel(**common).eval()
                        # Attach after generic initialization to preserve SSM dt.
                        model.compressor = self.compressor(kind)
                    with torch.no_grad():
                        before, output_mask, sources = model.encode(items, mask)
                        self.assertEqual(sources[:, :2].tolist(), [[10, 10]])
                        self.assertEqual(sources[:, 2:].tolist(), [list(range(11, 17))])
                        for source in range(11, 16):
                            changed = items.clone()
                            changed[:, source + 1:] += 40
                            after, after_mask, after_sources = model.encode(changed, mask)
                            self.assertTrue(torch.equal(output_mask, after_mask))
                            self.assertTrue(torch.equal(sources, after_sources))
                            # Summary columns and every observed raw state remain
                            # unchanged when only its future raw inputs change.
                            end = 2 + source - 11 + 1
                            torch.testing.assert_close(before[:, :end], after[:, :end],
                                                       rtol=0, atol=1e-6)
                        changed = items.clone()
                        changed[:, -1] += 40
                        self.assertFalse(torch.allclose(before[:, -1], model.encode(changed, mask)[0][:, -1],
                                                        rtol=1e-5, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
