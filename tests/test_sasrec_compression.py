from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local developer image
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class SASRecCompressionTest(unittest.TestCase):
    def _model(self, kind: str = "identity", *, time: bool = False, dropout: float = 0.0):
        from rhpr.sasrec.model import SASRecModel

        return SASRecModel(
            num_items=50,
            max_seq_len=16,
            d_model=16,
            n_heads=2,
            n_layers=2,
            ffn_multiplier=2,
            dropout=dropout,
            compressor_type=kind,
            tail_size=3,
            use_time_features=time,
        )

    def _batch(self):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 0, 0, 0, 0, 0]])
        targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0], [2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 0, 0, 0, 0, 0]])
        return ids, ids != 0, targets

    def test_pair_endpoints_singletons_and_physical_packing(self):
        from rhpr.sasrec.model import MeanSequenceCompressor, PairMLPSequenceCompressor, RecentSequenceCompressor

        ids, mask, _ = self._batch()
        embeddings = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
        mean = MeanSequenceCompressor(tail_size=3)
        learned = PairMLPSequenceCompressor(d_model=4, tail_size=3)
        output, compressed_mask, sources = mean(embeddings, mask)
        self.assertEqual(tuple(output.shape), (3, 6, 4))
        self.assertEqual(sources.tolist(), [[1, 3, 4, 5, 6, -1], [1, 3, 4, 5, 6, 7], [0, 1, 2, -1, -1, -1]])
        self.assertEqual(compressed_mask.sum(dim=1).tolist(), [5, 6, 3])
        torch.testing.assert_close(output[0, 0], (embeddings[0, 0] + embeddings[0, 1]) / 2)
        torch.testing.assert_close(output[1, 2], embeddings[1, 4])  # Odd old singleton.
        torch.testing.assert_close(output[1, 3:], embeddings[1, 5:])
        torch.testing.assert_close(output[2, :3], embeddings[2, :3])
        self.assertEqual(int(output[~compressed_mask].count_nonzero()), 0)
        learned_output, learned_mask, learned_sources = learned(embeddings, mask)
        self.assertTrue(torch.equal(output, learned_output))
        self.assertTrue(torch.equal(compressed_mask, learned_mask))
        self.assertTrue(torch.equal(sources, learned_sources))
        recent_output, recent_mask, recent_sources = RecentSequenceCompressor(3)(embeddings, mask)
        self.assertEqual(recent_sources.tolist(), [[2, 3, 4, 5, 6, -1], [2, 3, 4, 5, 6, 7], [0, 1, 2, -1, -1, -1]])
        self.assertTrue(torch.equal(recent_mask, compressed_mask))
        torch.testing.assert_close(recent_output[0, :5], embeddings[0, 2:7])

    def test_default_length_budget_and_projection_counts(self):
        from rhpr.sasrec.model import PairMLPSequenceCompressor

        compressor = PairMLPSequenceCompressor(d_model=4)
        embeddings = torch.randn(2, 1028, 4)
        mask = torch.arange(1028)[None, :] < torch.tensor([1028, 128])[:, None]
        with mock.patch.object(compressor.projection, "forward", wraps=compressor.projection.forward) as projection:
            output, output_mask, sources = compressor(embeddings, mask)
        self.assertEqual(tuple(output.shape), (2, 578, 4))
        self.assertEqual(output_mask.sum(dim=1).tolist(), [578, 128])
        self.assertEqual(tuple(projection.call_args.args[0].shape), (450, 8))
        self.assertEqual(compressor.pair_count(mask.sum(dim=1)), 450)
        self.assertEqual(sources[0, 449:452].tolist(), [899, 900, 901])
        torch.testing.assert_close(output[1, :128], embeddings[1, :128])

    def test_shared_initialization_rng_and_time_zero_start(self):
        torch.manual_seed(314)
        full = self._model().eval()
        backbone = {name: value for name, value in full.state_dict().items() if not name.startswith(("compressor.", "gap_embedding."))}
        expected_rng = torch.get_rng_state()
        for kind in ("identity", "recent", "mean", "pair_mlp"):
            for use_time in (False, True):
                with self.subTest(kind=kind, time=use_time):
                    torch.manual_seed(314)
                    model = self._model(kind, time=use_time).eval()
                    for name, value in backbone.items():
                        self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    if use_time:
                        self.assertEqual(int(model.gap_embedding.weight.count_nonzero()), 0)
                    if kind == "pair_mlp":
                        self.assertEqual(int(model.compressor.projection.weight.count_nonzero()), 0)
                        self.assertEqual(int(model.compressor.projection.bias.count_nonzero()), 0)
        ids, mask, _ = self._batch()
        torch.manual_seed(314)
        time_full = self._model(time=True).eval()
        self.assertTrue(torch.equal(full.encode(ids, mask)[0], time_full.encode(ids, mask, torch.zeros_like(ids))[0]))
        torch.manual_seed(314)
        mean = self._model("mean").eval()
        torch.manual_seed(314)
        learned = self._model("pair_mlp").eval()
        self.assertTrue(torch.equal(mean.encode(ids, mask)[0], learned.encode(ids, mask)[0]))

    def test_original_recency_positions_and_ragged_last_state(self):
        ids, mask, _ = self._batch()
        model = self._model("mean").eval()
        positions = []
        handle = model.position_embedding.register_forward_pre_hook(lambda module, args: positions.append(args[0].clone()))
        try:
            encoded, compressed_mask, _ = model.encode(ids, mask)
        finally:
            handle.remove()
        self.assertEqual(positions[0].tolist(), [[5, 3, 2, 1, 0, 0], [6, 4, 3, 2, 1, 0], [2, 1, 0, 0, 0, 0]])
        expected = encoded[torch.arange(3), compressed_mask.sum(dim=1) - 1]
        torch.testing.assert_close(model.last_state(ids, mask), expected)

    def test_same_tail_targets_and_compact_head_across_variants(self):
        from rhpr.sasrec.losses import build_loss_mask, next_item_loss, supervised_state_tensors

        ids, mask, targets = self._batch()
        expected_targets = torch.tensor([[6, 7, 8], [7, 8, 9], [2, 3, 4]])
        for kind in ("identity", "recent", "mean", "pair_mlp"):
            with self.subTest(kind=kind):
                model = self._model(kind)
                encoded, compressed_mask, sources = model.encode(ids, mask)
                loss_mask = build_loss_mask(compressed_mask, sources, mask.sum(dim=1), 3)
                states, selected_targets, selected_mask = supervised_state_tensors(encoded, loss_mask, sources, targets)
                self.assertEqual(tuple(states.shape), (3, 3, 16))
                self.assertTrue(bool(selected_mask.all()))
                self.assertTrue(torch.equal(selected_targets, expected_targets))
                with mock.patch("rhpr.sasrec.losses.torch.einsum", wraps=torch.einsum) as einsum:
                    loss = next_item_loss(
                        name="sampled_softmax", model=model, encoded=encoded, valid_mask=compressed_mask,
                        source_positions=sources, targets=targets, num_negatives=512, loss_tail_size=3,
                        original_lengths=mask.sum(dim=1), negative_generator=torch.Generator().manual_seed(17),
                    )
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertEqual(tuple(einsum.call_args.args[1].shape), (3, 3, 16))
                self.assertEqual(tuple(einsum.call_args.args[2].shape), (3, 512, 16))

    def test_only_selected_targets_drive_loss_and_explicit_sampling_rng(self):
        from rhpr.sasrec.losses import next_item_loss

        ids, mask, targets = self._batch()
        model = self._model("pair_mlp")
        encoded, compressed_mask, sources = model.encode(ids, mask)
        kwargs = dict(name="sampled_softmax", model=model, encoded=encoded, valid_mask=compressed_mask,
                      source_positions=sources, num_negatives=512, loss_tail_size=3, original_lengths=mask.sum(dim=1))
        first = next_item_loss(**kwargs, targets=targets, negative_generator=torch.Generator().manual_seed(91))
        torch.rand(999)  # Dropout and other consumers must not change this sampler.
        changed = targets.clone()
        changed[0, :4] = 35
        changed[1, :5] = 36
        second = next_item_loss(**kwargs, targets=changed, negative_generator=torch.Generator().manual_seed(91))
        self.assertTrue(torch.equal(first, second))
        with self.assertRaisesRegex(ValueError, "original_lengths"):
            next_item_loss(**{key: value for key, value in kwargs.items() if key != "original_lengths"}, targets=targets)

    def test_tail_loss_backpropagates_into_old_pairs(self):
        from rhpr.sasrec.losses import next_item_loss

        ids, mask, targets = self._batch()
        model = self._model("pair_mlp", time=True)
        gaps = torch.zeros_like(ids)
        input_embeddings = []

        def retain_input_grad(module, args, output):
            output.retain_grad()
            input_embeddings.append(output)

        handle = model.item_embedding.register_forward_hook(retain_input_grad)
        try:
            encoded, compressed_mask, sources = model.encode(ids, mask, gaps)
        finally:
            handle.remove()
        loss = next_item_loss(
            name="sampled_softmax", model=model, encoded=encoded, valid_mask=compressed_mask,
            source_positions=sources, targets=targets, num_negatives=512, loss_tail_size=3,
            original_lengths=mask.sum(dim=1), negative_generator=torch.Generator().manual_seed(51),
        )
        loss.backward()
        self.assertGreater(float(model.compressor.projection.weight.grad.abs().sum()), 0)
        # Inspect input activations: catalogue negatives also update embedding weights.
        self.assertGreater(float(input_embeddings[0].grad[0, :4].abs().sum()), 0)
        self.assertGreater(float(model.gap_embedding.weight.grad.abs().sum()), 0)

    def test_future_items_and_timestamps_do_not_change_earlier_tail_query(self):
        from rhpr.sasrec.data import _gap_ids

        torch.manual_seed(11)
        model = self._model("pair_mlp", time=True).eval()
        with torch.no_grad():
            model.gap_embedding.weight.normal_(std=0.3)
            model.compressor.projection.weight.normal_(std=0.1)
        ids = torch.arange(1, 9).unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        times = np.asarray([0, 1, 3, 8, 16, 32, 64, 128], dtype=np.int64)
        gaps = torch.from_numpy(_gap_ids(times, num_buckets=32, time_unit=1.0)).unsqueeze(0)
        original, _, sources = model.encode(ids, mask, gaps)
        changed_ids = ids.clone()
        changed_ids[:, 6:] = torch.tensor([20, 21])
        changed_times = times.copy()
        changed_times[6:] = [256, 1024]
        changed_gaps = torch.from_numpy(_gap_ids(changed_times, num_buckets=32, time_unit=1.0)).unsqueeze(0)
        changed, _, changed_sources = model.encode(changed_ids, mask, changed_gaps)
        self.assertTrue(torch.equal(sources, changed_sources))
        first_tail_column = int(torch.where(sources[0] == 5)[0][0])
        torch.testing.assert_close(original[:, first_tail_column], changed[:, first_tail_column], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(original[:, -1], changed[:, -1]))


if __name__ == "__main__":
    unittest.main()
