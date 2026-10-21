from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class QueryBlockProtocolTest(unittest.TestCase):
    def test_truthful_summary_sources_are_excluded_before_next_item_loss(self):
        from rhpr.sasrec.long_training_profile import _loss
        from rhpr.sasrec.losses import build_loss_mask

        torch.manual_seed(33)
        # Analogue of K summaries + raw tail159, query first128, last32 loss.
        # Here tail6/query4/loss3. Both summaries share the first scored raw
        # source, so source-based selection alone would incorrectly score them.
        states = torch.randn(1, 8, 8, requires_grad=True)
        sources = torch.tensor([[11, 11, 8, 9, 10, 11, 12, 13]])
        output_mask = torch.ones((1, 8), dtype=torch.bool)
        batch = {
            "targets": torch.arange(2, 16)[None, :],
            "valid_mask": torch.ones((1, 14), dtype=torch.bool),
            "negative_ids": torch.tensor([[21, 22, 23, 24]]),
        }
        self.assertEqual(int(build_loss_mask(output_mask, sources, torch.tensor([14]), 3).sum()), 5)
        model = SimpleNamespace(num_items=30, item_embedding=torch.nn.Embedding(31, 8))
        loss = _loss(model, (states, output_mask, sources), batch, tail_size=3, num_negatives=4)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(int(states.grad[:, :-3].count_nonzero()), 0)
        self.assertTrue(bool(states.grad[:, -3:].abs().sum(dim=-1).gt(0).all()))

    def test_every_scored_hstu_state_is_causal_with_shared_early_query(self):
        from rhpr.sasrec.hstu import HSTUModel
        from rhpr.sasrec.losses import supervised_state_tensors
        from rhpr.sasrec.query_ssm import make_query_compressor

        items = torch.arange(1, 18)[None, :]
        mask = torch.ones_like(items, dtype=torch.bool)
        # Old prefix11 + raw tail6; query sees raw positions11..14, and the
        # scored states14,15,16 predict positions15,16,17 respectively.
        for kind in ("query_cross_attention", "query_ssm", "query_ssm_readout"):
            with self.subTest(kind=kind):
                torch.manual_seed(93)
                compressor = make_query_compressor(kind=kind, d_model=8, summary_tokens=2,
                    tail_size=6, query_context_size=4, n_heads=2, d_inner=4, d_state=3, gate_rank=3)
                model = HSTUModel(num_items=100, max_seq_len=17, d_model=8,
                    n_heads=2, n_layers=2, ffn_multiplier=4, dropout=0,
                    custom_sequence_compressor=compressor, normalization_length=8).eval()
                with torch.no_grad():
                    before, output_mask, sources = model.encode(items, mask)
                    selected = torch.zeros_like(output_mask)
                    selected[:, -3:] = True
                    states, targets, target_mask = supervised_state_tensors(before, selected, sources, items + 1)
                    self.assertEqual(targets.tolist(), [[16, 17, 18]])
                    self.assertTrue(bool(target_mask.all()))
                    self.assertEqual(states.shape, (1, 3, 8))
                    for source in (14, 15):
                        with self.subTest(source=source):
                            changed = items.clone()
                            changed[:, source + 1:] += 40
                            after, after_mask, after_sources = model.encode(changed, mask)
                            self.assertTrue(torch.equal(sources, after_sources))
                            self.assertTrue(torch.equal(output_mask, after_mask))
                            output_column = 2 + source - 11
                            torch.testing.assert_close(before[:, output_column], after[:, output_column],
                                                       rtol=0, atol=1e-6)
                    # Changing the currently observed event must remain visible.
                    changed = items.clone()
                    changed[:, -1] += 40
                    after = model.encode(changed, mask)[0]
                    self.assertFalse(torch.allclose(before[:, -1], after[:, -1], rtol=1e-5, atol=1e-7))
                    # No causality assertion is made for unscored burn-in states:
                    # those can legitimately see the later query through Z.


if __name__ == "__main__":
    unittest.main()
