from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local developer image
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "torch is not installed")
class SASRecDeepModelTest(unittest.TestCase):
    def model(self, kind="recursive_pair_mlp", depth=6):
        from rhpr.sasrec.model import SASRecModel

        return SASRecModel(
            num_items=1200, max_seq_len=1028, d_model=8, n_heads=2,
            n_layers=1, ffn_multiplier=2, dropout=0.0,
            compressor_type=kind, compression_depth=depth, tail_size=128,
        )

    def batch(self, lengths=(1028, 257, 64)):
        lengths = torch.tensor(lengths)
        raw = torch.arange(1028)[None, :].expand(len(lengths), -1)
        mask = raw < lengths[:, None]
        items = (raw + 1).masked_fill(~mask, 0)
        targets = (raw + 2).masked_fill(~mask, 0)
        return items, mask, targets

    def test_deep_budgets_count_every_projection_and_preserve_raw_tail(self):
        from rhpr.sasrec.model import RecursivePairMLPSequenceCompressor

        compressor = RecursivePairMLPSequenceCompressor(4, depth=6)
        embeddings = torch.randn(2, 1028, 4)
        lengths = torch.tensor([1028, 127])
        mask = torch.arange(1028)[None, :] < lengths[:, None]
        parameters = tuple(id(parameter) for parameter in compressor.parameters())
        self.assertEqual(set(dict(compressor.named_parameters())), {"projection.weight", "projection.bias"})
        pair_counts = [450, 225, 112, 56, 28, 14]
        for depth, width in ((4, 185), (5, 157), (6, 143)):
            with self.subTest(depth=depth):
                compressor.depth = depth
                with mock.patch.object(compressor.projection, "forward", wraps=compressor.projection.forward) as projection:
                    output, output_mask, sources, counts = compressor.forward_with_metadata(embeddings, mask)
                self.assertEqual(output.shape, (2, width, 4))
                self.assertEqual(output_mask.sum(dim=1).tolist(), [width, 127])
                self.assertEqual([call.args[0].shape[0] for call in projection.call_args_list], pair_counts[:depth])
                self.assertEqual(compressor.pair_count(lengths), sum(pair_counts[:depth]))
                self.assertEqual(tuple(id(parameter) for parameter in compressor.parameters()), parameters)
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertTrue(torch.equal(output[0, -128:], embeddings[0, -128:]))
                self.assertTrue(torch.equal(output[1, :127], embeddings[1, :127]))
                self.assertTrue(torch.equal(sources[0, -128:], torch.arange(900, 1028)))
                self.assertTrue(torch.equal(counts[0, -128:], torch.ones(128, dtype=counts.dtype)))
                self.assertEqual(counts.sum(dim=1).tolist(), lengths.tolist())
                self.assertEqual(int(output[~output_mask].count_nonzero()), 0)

    def test_deep_ragged_spans_cover_every_source_once_including_odd_carries(self):
        from rhpr.sasrec.model import RecursivePairMLPSequenceCompressor

        lengths = [1028, 1027, 901, 133, 131, 129, 128, 1]
        embeddings = torch.randn(len(lengths), 1028, 3)
        mask = torch.arange(1028)[None, :] < torch.tensor(lengths)[:, None]
        compressor = RecursivePairMLPSequenceCompressor(3)
        for depth in (4, 5, 6):
            _, output_mask, sources, counts = compressor.forward_with_metadata(embeddings, mask, depth)
            for row, length in enumerate(lengths):
                with self.subTest(depth=depth, length=length):
                    old = max(length - 128, 0)
                    spans = [[index] for index in range(old)]
                    for _ in range(depth):
                        spans = [sum(spans[start:start + 2], []) for start in range(0, len(spans), 2)]
                    spans += [[index] for index in range(old, length)]
                    width = int(output_mask[row].sum())
                    self.assertEqual(width, len(spans))
                    self.assertEqual(sources[row, :width].tolist(), [span[-1] for span in spans])
                    self.assertEqual(counts[row, :width].tolist(), [len(span) for span in spans])
                    reconstructed = []
                    for endpoint, count in zip(sources[row, :width].tolist(), counts[row, :width].tolist()):
                        reconstructed.extend(range(endpoint - count + 1, endpoint + 1))
                    self.assertEqual(reconstructed, list(range(length)))
                    self.assertTrue(bool((sources[row, width:] == -1).all()))
                    self.assertTrue(bool((counts[row, width:] == 0).all()))

    def test_deep_and_recent_training_select_identical_original_tail_targets_and_pe(self):
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors

        items, mask, targets = self.batch()
        lengths = mask.sum(dim=1)
        expected_targets = torch.zeros((3, 128), dtype=targets.dtype)
        for row, length in enumerate(lengths.tolist()):
            retained = min(length, 128)
            expected_targets[row, :retained] = targets[row, length - retained:length]
        for kind, depth in (("recursive_pair_mlp", 4), ("recursive_pair_mlp", 5),
                            ("recursive_pair_mlp", 6), ("tail_only", 0)):
            with self.subTest(kind=kind, depth=depth):
                model = self.model(kind, depth).eval()
                positions = []
                hook = model.position_embedding.register_forward_pre_hook(
                    lambda module, args: positions.append(args[0].clone()))
                with torch.no_grad():
                    encoded, output_mask, sources = model.encode(items, mask)
                hook.remove()
                loss_mask = build_loss_mask(output_mask, sources, lengths, 128)
                states, selected_targets, selected_mask = supervised_state_tensors(encoded, loss_mask, sources, targets)
                self.assertEqual(states.shape, (3, 128, 8))
                self.assertTrue(torch.equal(selected_targets, expected_targets))
                self.assertEqual(selected_mask.sum(dim=1).tolist(), [128, 128, 64])
                for row, length in enumerate(lengths.tolist()):
                    retained = min(length, 128)
                    self.assertTrue(torch.equal(positions[0][row, loss_mask[row]], torch.arange(retained - 1, -1, -1)))

    def test_future_tail_changes_cannot_reach_earlier_tail_queries_at_deep_exits(self):
        torch.manual_seed(8)
        items, mask, _ = self.batch((1028,))
        changed_items = items.clone()
        changed_items[:, 901:] += 37
        model = self.model().eval()
        with torch.no_grad():
            model.compressor.projection.weight.normal_(std=0.07)
        for kind, depth in (("recursive_pair_mlp", 4), ("recursive_pair_mlp", 5),
                            ("recursive_pair_mlp", 6), ("tail_only", 0)):
            with self.subTest(kind=kind, depth=depth):
                current = model if kind == "recursive_pair_mlp" else self.model(kind, depth).eval()
                if kind == "recursive_pair_mlp":
                    current.set_compression_depth(depth)
                with torch.no_grad():
                    original, _, sources = current.encode(items, mask)
                    changed, _, changed_sources = current.encode(changed_items, mask)
                self.assertTrue(torch.equal(sources, changed_sources))
                first_tail = int(torch.where(sources[0] == 900)[0][0])
                torch.testing.assert_close(original[:, first_tail], changed[:, first_tail], rtol=0, atol=1e-6)
                self.assertFalse(torch.allclose(original[:, -1], changed[:, -1]))

    def test_tail_loss_backpropagates_through_all_six_shared_passes(self):
        from rhpr.sasrec.losses import next_item_loss

        torch.manual_seed(17)
        model = self.model(depth=6)
        items, mask, targets = self.batch((1028,))
        retained_inputs, retained_projections = [], []

        def retain_activations(storage):
            def hook(module, args, output):
                output.retain_grad()
                storage.append(output)
            return hook

        input_hook = model.item_embedding.register_forward_hook(retain_activations(retained_inputs))
        projection_hook = model.compressor.projection.register_forward_hook(retain_activations(retained_projections))
        encoded, output_mask, sources = model.encode(items, mask)
        input_hook.remove()
        projection_hook.remove()
        loss = next_item_loss(
            name="sampled_softmax", model=model, encoded=encoded, valid_mask=output_mask,
            source_positions=sources, targets=targets, num_negatives=32, loss_tail_size=128,
            original_lengths=mask.sum(dim=1), negative_generator=torch.Generator().manual_seed(19),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(len(retained_projections), 6)
        for output in retained_projections:
            self.assertTrue(bool(torch.isfinite(output.grad).all()))
            self.assertGreater(float(output.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(model.compressor.projection.weight.grad).all()))
        self.assertGreater(float(model.compressor.projection.weight.grad.abs().sum()), 0)
        self.assertTrue(bool((retained_inputs[0].grad[0, :900].abs().sum(dim=-1) > 0).all()))

    def test_recent128_matches_cropped_backbone_without_shifting_initialization_or_rng(self):
        from rhpr.sasrec.model import TailOnlySequenceCompressor
        from rhpr.sasrec.tail_ablation import TailOnlySequenceCompressor as AblationTailOnly

        self.assertIs(TailOnlySequenceCompressor, AblationTailOnly)
        torch.manual_seed(314)
        identity = self.model("identity", 0).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(314)
        recent = self.model("tail_only", 0).eval()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(identity.state_dict().keys(), recent.state_dict().keys())
        for name, parameter in identity.state_dict().items():
            self.assertTrue(torch.equal(parameter, recent.state_dict()[name]), name)
        items, mask, _ = self.batch((1028, 64))
        cropped_items = torch.zeros((2, 128), dtype=items.dtype)
        cropped_items[0] = items[0, -128:]
        cropped_items[1, :64] = items[1, :64]
        cropped_mask = cropped_items != 0
        with torch.no_grad():
            tail_encoded, tail_mask, raw_sources = recent.encode(items, mask)
            cropped_encoded, _, _ = identity.encode(cropped_items, cropped_mask)
        self.assertTrue(torch.equal(tail_mask, cropped_mask))
        self.assertTrue(torch.equal(raw_sources[0], torch.arange(900, 1028)))
        self.assertTrue(torch.equal(raw_sources[1, :64], torch.arange(64)))
        torch.testing.assert_close(tail_encoded, cropped_encoded, rtol=0, atol=0)

    def test_recent128_training_has_only_tail_input_gradients(self):
        from rhpr.sasrec.losses import next_item_loss

        torch.manual_seed(22)
        model = self.model("tail_only", 0)
        items, mask, targets = self.batch((1028,))
        input_activations = []

        def retain_input(module, args, output):
            output.retain_grad()
            input_activations.append(output)

        hook = model.item_embedding.register_forward_hook(retain_input)
        encoded, output_mask, sources = model.encode(items, mask)
        hook.remove()
        loss = next_item_loss(
            name="sampled_softmax", model=model, encoded=encoded, valid_mask=output_mask,
            source_positions=sources, targets=targets, num_negatives=32, loss_tail_size=128,
            original_lengths=mask.sum(dim=1), negative_generator=torch.Generator().manual_seed(23),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        # Input activation gradients distinguish attention inputs from catalogue
        # positives/negatives, which may independently update embedding weights.
        self.assertEqual(int(input_activations[0].grad[0, :900].count_nonzero()), 0)
        self.assertTrue(bool((input_activations[0].grad[0, 900:].abs().sum(dim=-1) > 0).all()))


if __name__ == "__main__":
    unittest.main()
