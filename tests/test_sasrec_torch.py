from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local developer image
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class SASRecTorchTest(unittest.TestCase):
    def test_model_and_losses_are_finite(self) -> None:
        from rhpr.sasrec.losses import next_item_loss
        from rhpr.sasrec.model import SASRecModel

        model = SASRecModel(
            num_items=20,
            max_seq_len=5,
            d_model=32,
            n_heads=1,
            n_layers=1,
            ffn_multiplier=1,
            dropout=0.0,
        )
        item_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
        valid_mask = item_ids != 0
        targets = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
        encoded, compressed_mask, source_positions = model.encode(item_ids, valid_mask)
        self.assertEqual(tuple(encoded.shape), (2, 4, 32))
        for name, negatives in (("bce", 1), ("sampled_softmax", 8)):
            loss = next_item_loss(
                name=name,
                model=model,
                encoded=encoded,
                valid_mask=compressed_mask,
                source_positions=source_positions,
                targets=targets,
                num_negatives=negatives,
            )
            self.assertTrue(bool(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()
