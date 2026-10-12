from __future__ import annotations

import ast
import hashlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on the local developer image
    torch = None  # type: ignore[assignment]


class _ReferencePacking:
    """Independent slow CPU packing used to execute the unmodified upstream block."""

    @staticmethod
    def dense_to_jagged(values, offsets):
        boundaries = offsets[0].tolist()
        return torch.cat([values[row, :end - start] for row, (start, end)
                          in enumerate(zip(boundaries[:-1], boundaries[1:]))]), offsets[0]

    @staticmethod
    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        boundaries = offsets[0].tolist()
        width = max_lengths[0]
        return torch.stack([F.pad(values[start:min(end, start + width)],
                                 (0, 0, 0, max(width - (end - start), 0)), value=padding_value)
                            for start, end in zip(boundaries[:-1], boundaries[1:])])


def _upstream_namespace():
    source_path = ROOT / "src/rhpr/vendor/hstu/upstream_hstu.py.txt"
    source = source_path.read_bytes()
    # Git's blob identity confirms that the fixture is the exact pinned official
    # source, rather than another local implementation of its attention math.
    blob_header = f"blob {len(source)}\0".encode()
    if hashlib.sha1(blob_header + source).hexdigest() != "78329791d6c87d7826bef07fc89f7021ad197a37":
        raise AssertionError("upstream HSTU source differs from the pinned Git blob")
    parsed = ast.parse(source)
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.ClassDef) and node.name == "HSTU":
            break
        if isinstance(node, ast.ImportFrom) and node.module.startswith("generative_recommenders"):
            continue
        selected.append(node)
    namespace = {"__name__": "official_hstu_reference"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    torch_proxy = types.SimpleNamespace(**torch.__dict__)
    torch_proxy.ops = types.SimpleNamespace(fbgemm=_ReferencePacking)
    namespace["torch"] = torch_proxy
    return namespace


@unittest.skipIf(torch is None, "torch is not installed")
class HSTUModelTest(unittest.TestCase):
    def model(self, kind="identity", *, max_seq_len=16, tail_size=3, n_layers=4):
        from rhpr.sasrec.hstu import HSTUModel

        return HSTUModel(num_items=1200, max_seq_len=max_seq_len, d_model=8,
                         n_heads=2, n_layers=n_layers, ffn_multiplier=4, dropout=0.0,
                         compressor_type=kind, tail_size=tail_size,
                         use_time_features=False)

    def assert_forward_backward_parity(self, native, upstream, width, *, training):
        native.train(training)
        upstream.train(training)
        upstream.load_state_dict(native.state_dict(), strict=True)
        native_input = torch.randn(7, 8, requires_grad=True)
        upstream_input = native_input.detach().clone().requires_grad_()
        offsets = torch.tensor([0, 5, 7])
        kwargs = dict(x_offsets=offsets, all_timestamps=torch.zeros((2, width), dtype=torch.long),
                      invalid_attn_mask=torch.ones(width, width).tril())
        torch.manual_seed(123)
        actual = native(x=native_input, **kwargs)[0]
        torch.manual_seed(123)
        expected = upstream(x=upstream_input, **kwargs)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
        (actual.square().mean() + (actual * weights).mean()).backward()
        (expected.square().mean() + (expected * weights).mean()).backward()
        torch.testing.assert_close(native_input.grad, upstream_input.grad, rtol=1e-6, atol=1e-7)
        for name, parameter in native.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            torch.testing.assert_close(parameter.grad, dict(upstream.named_parameters())[name].grad,
                                       rtol=1e-6, atol=1e-7, msg=name)

    def test_native_block_matches_exact_pinned_upstream_forward_and_backward(self):
        from rhpr.vendor.hstu.native import RelativePositionalBias, SequentialTransductionUnitJagged

        official = _upstream_namespace()
        for training in (False, True):
            with self.subTest(training=training):
                kwargs = dict(embedding_dim=8, linear_hidden_dim=3, attention_dim=2,
                              dropout_ratio=0.25, attn_dropout_ratio=0.0, num_heads=2,
                              linear_activation="silu")
                native = SequentialTransductionUnitJagged(
                    **kwargs, relative_attention_bias_module=RelativePositionalBias(6))
                upstream = official["SequentialTransductionUnitJagged"](
                    **kwargs, relative_attention_bias_module=official["RelativePositionalBias"](6))
                self.assert_forward_backward_parity(native, upstream, 6, training=training)

    def test_fixed_divisor_runtime_attention_matches_official_full_padding(self):
        from rhpr.sasrec.hstu import RuntimeRelativePositionalBias
        from rhpr.vendor.hstu.native import SequentialTransductionUnitJagged

        official = _upstream_namespace()
        kwargs = dict(embedding_dim=8, linear_hidden_dim=3, attention_dim=2,
                      dropout_ratio=0.0, attn_dropout_ratio=0.0, num_heads=2,
                      linear_activation="silu")
        native = SequentialTransductionUnitJagged(
            **kwargs, relative_attention_bias_module=RuntimeRelativePositionalBias(16),
            normalization_length=16)
        upstream = official["SequentialTransductionUnitJagged"](
            **kwargs, relative_attention_bias_module=official["RelativePositionalBias"](16))
        upstream.load_state_dict(native.state_dict(), strict=True)
        native_input = torch.randn(7, 8, requires_grad=True)
        upstream_input = native_input.detach().clone().requires_grad_()
        offsets = torch.tensor([0, 5, 7])
        actual = native(x=native_input, x_offsets=offsets,
                        all_timestamps=torch.zeros(2, 6, dtype=torch.long),
                        invalid_attn_mask=torch.ones(6, 6).tril())[0]
        expected = upstream(x=upstream_input, x_offsets=offsets,
                            all_timestamps=torch.zeros(2, 16, dtype=torch.long),
                            invalid_attn_mask=torch.ones(16, 16).tril())[0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(native_input.grad, upstream_input.grad, rtol=1e-6, atol=1e-7)
        for name, parameter in native.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(upstream.named_parameters())[name].grad,
                                       rtol=1e-6, atol=1e-7, msg=name)

    def test_runtime_relative_bias_is_exact_official_fixed_matrix_crop(self):
        from rhpr.sasrec.hstu import RuntimeRelativePositionalBias
        from rhpr.vendor.hstu.native import RelativePositionalBias

        native, original = RuntimeRelativePositionalBias(16), RelativePositionalBias(16)
        original.load_state_dict(native.state_dict())
        for width in (1, 3, 8, 16):
            with self.subTest(width=width):
                actual = native(torch.zeros((2, width), dtype=torch.long))
                expected = original(torch.zeros((2, 16), dtype=torch.long))[:, :width, :width]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_and_recent_share_parameters_rng_and_fixed_normalization(self):
        torch.manual_seed(314)
        full = self.model(max_seq_len=1028, tail_size=128)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(314)
        recent = self.model("tail_only", max_seq_len=1028, tail_size=128)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(full.state_dict().keys(), recent.state_dict().keys())
        for name, parameter in full.state_dict().items():
            self.assertTrue(torch.equal(parameter, recent.state_dict()[name]), name)
        for model in (full, recent):
            self.assertEqual(model.hstu_normalization_length, 1028)
            for block in model.encoder.layers:
                self.assertEqual(block._normalization_length, 1028)
                self.assertEqual(block._rel_attn_bias._w.numel(), 2055)

    def test_padding_batch_membership_and_future_tokens_do_not_change_valid_past(self):
        for kind in ("identity", "tail_only"):
            with self.subTest(kind=kind):
                model = self.model(kind).eval()
                solo = torch.tensor([[1, 2, 3, 4, 5]])
                batched = torch.tensor([[1, 2, 3, 4, 5, 99, 99, 99], [6, 7, 8, 9, 10, 11, 12, 13]])
                batch_mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]
                with torch.no_grad():
                    solo_encoded, solo_mask, solo_sources = model.encode(solo, torch.ones_like(solo, dtype=torch.bool))
                    batch_encoded, output_mask, sources = model.encode(batched, batch_mask)
                width = int(solo_mask[0].sum())
                torch.testing.assert_close(solo_encoded[0, :width], batch_encoded[0, :width], rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(solo_sources[0, :width], sources[0, :width]))
                self.assertEqual(int(batch_encoded[~output_mask].count_nonzero()), 0)
                items = torch.arange(1, 17)[None, :]
                changed = items.clone()
                changed[:, 14:] += 37
                mask = torch.ones_like(items, dtype=torch.bool)
                with torch.no_grad():
                    original, _, original_sources = model.encode(items, mask)
                    future_changed, _, _ = model.encode(changed, mask)
                query = int(torch.where(original_sources[0] == 13)[0][0])
                torch.testing.assert_close(original[:, query], future_changed[:, query], rtol=0, atol=1e-6)
                self.assertFalse(torch.allclose(original[:, -1], future_changed[:, -1]))

    def test_both_budgets_select_same_original128_targets_and_recency_pe(self):
        from rhpr.sasrec.losses import build_loss_mask, supervised_state_tensors

        raw = torch.arange(1028)[None, :].expand(2, -1)
        mask = raw < torch.tensor([1028, 64])[:, None]
        items, targets = (raw + 1).masked_fill(~mask, 0), (raw + 2).masked_fill(~mask, 0)
        expected_targets = torch.zeros((2, 128), dtype=torch.long)
        expected_targets[0] = targets[0, -128:]
        expected_targets[1, :64] = targets[1, :64]
        for kind in ("identity", "tail_only"):
            with self.subTest(kind=kind):
                model = self.model(kind, max_seq_len=1028, tail_size=128, n_layers=1).eval()
                positions = []
                hook = model.position_embedding.register_forward_pre_hook(lambda module, args: positions.append(args[0].clone()))
                with torch.no_grad():
                    encoded, output_mask, sources = model.encode(items, mask)
                hook.remove()
                selected = build_loss_mask(output_mask, sources, mask.sum(dim=1), 128)
                states, selected_targets, selected_mask = supervised_state_tensors(encoded, selected, sources, targets)
                self.assertEqual(states.shape, (2, 128, 8))
                self.assertTrue(torch.equal(selected_targets, expected_targets))
                self.assertEqual(selected_mask.sum(dim=1).tolist(), [128, 64])
                self.assertTrue(torch.equal(positions[0][0, selected[0]], torch.arange(127, -1, -1)))
                self.assertTrue(torch.equal(positions[0][1, selected[1]], torch.arange(63, -1, -1)))
                if kind == "tail_only":
                    self.assertEqual(encoded.shape[1], 128)

    def test_four_block_tail_loss_has_finite_gradients_and_padding_is_excluded(self):
        from rhpr.sasrec.losses import next_item_loss

        for kind in ("identity", "tail_only"):
            with self.subTest(kind=kind):
                torch.manual_seed(19)
                model = self.model(kind)
                items = torch.tensor([[1, 2, 3, 4, 5, 99, 99, 99], [6, 7, 8, 9, 10, 11, 12, 13]])
                mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]
                targets = (items + 1).masked_fill(~mask, 0)
                activations = []

                def retain_inputs(module, args, output):
                    output.retain_grad()
                    activations.append(output)

                hook = model.item_embedding.register_forward_hook(retain_inputs)
                encoded, output_mask, sources = model.encode(items, mask)
                hook.remove()
                loss = next_item_loss(name="sampled_softmax", model=model, encoded=encoded,
                                     valid_mask=output_mask, source_positions=sources, targets=targets,
                                     num_negatives=32, loss_tail_size=3, original_lengths=mask.sum(dim=1),
                                     negative_generator=torch.Generator().manual_seed(20))
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                self.assertEqual(int(activations[0].grad[~mask].count_nonzero()), 0)
                for block in model.encoder.layers:
                    self.assertTrue(bool(torch.isfinite(block._uvqk.grad).all()))
                    self.assertGreater(float(block._uvqk.grad.abs().sum()), 0)
                    self.assertGreater(float(block._rel_attn_bias._w.grad.abs().sum()), 0)
                if kind == "tail_only":
                    self.assertEqual(int(activations[0].grad[0, :2].count_nonzero()), 0)
                else:
                    self.assertGreater(float(activations[0].grad[0, :2].abs().sum()), 0)

    def test_time_features_unsupported_compressors_and_noncontiguous_masks_fail(self):
        from rhpr.sasrec.hstu import HSTUModel

        kwargs = dict(num_items=50, max_seq_len=16, d_model=8, n_heads=2,
                      n_layers=1, ffn_multiplier=4, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "Time OFF"):
            HSTUModel(**kwargs, use_time_features=True)
        with self.assertRaisesRegex(ValueError, "only identity and tail_only"):
            HSTUModel(**kwargs, compressor_type="recursive_pair_mlp")
        model = HSTUModel(**kwargs)
        items = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "gap_ids"):
            model.encode(items, torch.ones_like(items, dtype=torch.bool), torch.zeros_like(items))
        with self.assertRaisesRegex(ValueError, "right padding"):
            model.encode(items, torch.tensor([[True, False, True]]))


if __name__ == "__main__":
    unittest.main()
