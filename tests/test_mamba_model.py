from __future__ import annotations

import ast
import hashlib
import math
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - local image dependent
    torch = None


def _rearrange(value, pattern, **sizes):
    """Only tensor layouts used by the exact Mamba1 CPU fixture.

    A test-scoped substitute for einops keeps CUDA dependencies out of the CPU
    suite. Unsupported layouts fail loudly; this is never a profiling backend.
    """
    if pattern == "b l d -> d (b l)":
        return value.permute(2, 0, 1).reshape(value.shape[2], -1)
    if pattern == "d (b l) -> b d l":
        return value.reshape(value.shape[0], -1, sizes["l"]).permute(1, 0, 2)
    if pattern == "b d l -> (b l) d":
        return value.permute(0, 2, 1).reshape(-1, value.shape[1])
    if pattern == "(b l) dstate -> b dstate l":
        return value.reshape(-1, sizes["l"], value.shape[1]).permute(0, 2, 1)
    if pattern == "b d l -> b l d":
        return value.transpose(1, 2)
    if pattern == "d -> d 1":
        return value[:, None]
    raise AssertionError(f"unsupported test-scoped rearrange: {pattern}")


def _repeat(value, pattern, **sizes):
    if pattern == "n -> d n":
        return value[None, :].expand(sizes["d"], -1)
    raise AssertionError(f"unsupported test-scoped repeat: {pattern}")


def _exact_source(filename, git_blob):
    source = (ROOT / "src/rhpr/vendor/mamba4rec" / filename).read_bytes()
    if hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest() != git_blob:
        raise AssertionError(f"modified pinned upstream fixture: {filename}")
    return source.decode()


def _cpu_native_mamba():
    """Execute official Mamba1 and its official reference scan only in tests."""
    source = _exact_source("upstream_selective_scan_interface.py.txt",
                           "a41f1359c2463e69a2a8e52035749893f126769a")
    scan_node = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == "selective_scan_ref")
    namespace = dict(torch=torch, nn=nn, F=F, math=math, rearrange=_rearrange,
                     repeat=_repeat, causal_conv1d_fn=None, causal_conv1d_update=None,
                     selective_state_update=None)
    exec(compile(ast.Module(body=[scan_node], type_ignores=[]), "official_scan_reference", "exec"), namespace)
    namespace["selective_scan_fn"] = namespace["selective_scan_ref"]
    source = _exact_source("upstream_mamba_simple.py.txt",
                           "4c8a388217c4cd4d21d5ea4704a5d571be294781")
    mamba_node = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "Mamba")
    exec(compile(ast.Module(body=[mamba_node], type_ignores=[]), "official_native_mamba1", "exec"), namespace)
    return namespace["Mamba"]


def _official_rec_blocks(native_mamba):
    source = _exact_source("upstream_mamba4rec.py.txt",
                           "d75ae50fec7823022ed4a26bd76f0119247f295d")
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
             and node.name in {"MambaLayer", "FeedForward"}]
    namespace = dict(torch=torch, nn=nn, Mamba=native_mamba)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "official_mamba4rec_blocks", "exec"), namespace)
    return namespace


@unittest.skipIf(torch is None, "torch is not installed")
class MambaModelTest(unittest.TestCase):
    def model(self, *, n_layers=2, dropout=0.0, max_seq_len=16, factory=None):
        from rhpr.sasrec.mamba import Mamba4RecModel

        test_metadata = {"execution_path": "test_only_official_cpu_reference",
                         "python_reference_fallback": True}
        with mock.patch("rhpr.sasrec.mamba._require_native_cuda_runtime", return_value=test_metadata), \
                mock.patch("rhpr.vendor.mamba4rec.native.Mamba", factory or _cpu_native_mamba()):
            return Mamba4RecModel(num_items=1200, max_seq_len=max_seq_len,
                                 d_model=8, n_layers=n_layers, dropout=dropout)

    def test_exact_recommendation_blocks_match_upstream_forward_backward(self):
        from rhpr.vendor.mamba4rec.native import MambaLayer

        native_mamba = _cpu_native_mamba()
        official = _official_rec_blocks(native_mamba)
        for layers in (1, 2):
            for training in (False, True):
                with self.subTest(layers=layers, training=training):
                    kwargs = dict(d_model=8, d_state=32, d_conv=4, expand=2,
                                  dropout=0.2, num_layers=layers)
                    with mock.patch("rhpr.vendor.mamba4rec.native.Mamba", native_mamba):
                        actual_block = MambaLayer(**kwargs).train(training)
                    expected_block = official["MambaLayer"](**kwargs).train(training)
                    expected_block.load_state_dict(actual_block.state_dict())
                    actual_input = torch.randn(2, 7, 8, requires_grad=True)
                    expected_input = actual_input.detach().clone().requires_grad_()
                    torch.manual_seed(42)
                    actual = actual_block(actual_input)
                    torch.manual_seed(42)
                    expected = expected_block(expected_input)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    weights = torch.randn_like(actual)
                    (actual * weights).sum().backward()
                    (expected * weights).sum().backward()
                    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
                    for name, parameter in actual_block.named_parameters():
                        torch.testing.assert_close(parameter.grad,
                                                   dict(expected_block.named_parameters())[name].grad,
                                                   rtol=0, atol=0, msg=name)

    def test_wrapper_preserves_every_native_ssm_parameter_and_dt_bias(self):
        native_mamba = _cpu_native_mamba()
        snapshots = []

        def capture_factory(**kwargs):
            block = native_mamba(**kwargs)
            snapshots.append({name: value.detach().clone() for name, value in block.state_dict().items()})
            return block

        model = self.model(factory=capture_factory)
        self.assertEqual(len(snapshots), 2)
        for layer, expected in zip(model.mamba_layers, snapshots):
            for name, value in layer.mamba.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
            delta = F.softplus(layer.mamba.dt_proj.bias)
            self.assertTrue(bool(((delta >= 0.001) & (delta <= 0.1)).all()))
            self.assertTrue(layer.mamba.dt_proj.bias._no_reinit)
            self.assertTrue(layer.mamba.A_log._no_weight_decay)
            self.assertTrue(layer.mamba.D._no_weight_decay)
        self.assertIsNone(model.position_embedding)
        self.assertEqual(int(model.item_embedding.weight[0].count_nonzero()), 0)

    def test_padding_batch_membership_future_causality_and_source_positions(self):
        model = self.model().eval()
        solo = torch.tensor([[1, 2, 3, 4, 5]])
        batch = torch.tensor([[1, 2, 3, 4, 5, 99999, 99999, 99999],
                              [6, 7, 8, 9, 10, 11, 12, 13]])
        mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]
        with torch.no_grad():
            solo_encoded, _, _ = model.encode(solo, torch.ones_like(solo, dtype=torch.bool))
            batch_encoded, output_mask, sources = model.encode(batch, mask)
            last = model.last_state(batch, mask)
        torch.testing.assert_close(solo_encoded[0], batch_encoded[0, :5], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(last[0], solo_encoded[0, -1], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(output_mask, mask))
        expected_sources = torch.arange(8)[None, :].expand(2, -1).masked_fill(~mask, -1)
        self.assertTrue(torch.equal(sources, expected_sources))
        self.assertEqual(int(batch_encoded[~mask].count_nonzero()), 0)
        items = torch.arange(1, 17)[None, :]
        changed = items.clone()
        changed[:, 14:] += 123
        with torch.no_grad():
            original, _, _ = model.encode(items, torch.ones_like(items, dtype=torch.bool))
            future, _, _ = model.encode(changed, torch.ones_like(items, dtype=torch.bool))
        torch.testing.assert_close(original[:, :14], future[:, :14], rtol=0, atol=0)
        self.assertFalse(torch.allclose(original[:, -1], future[:, -1]))

    def test_tail_loss_selects_raw_targets_and_has_finite_prefix_gradients(self):
        from rhpr.sasrec.losses import build_loss_mask, next_item_loss, supervised_state_tensors

        model = self.model()
        items = torch.tensor([[1, 2, 3, 4, 5, 99, 99, 99], [6, 7, 8, 9, 10, 11, 12, 13]])
        mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]
        targets = (items + 1).masked_fill(~mask, 0)
        activations = []

        def save_embeddings(module, args, output):
            output.retain_grad()
            activations.append(output)

        hook = model.item_embedding.register_forward_hook(save_embeddings)
        encoded, output_mask, sources = model.encode(items, mask)
        hook.remove()
        selected = build_loss_mask(output_mask, sources, mask.sum(dim=1), 3)
        _, selected_targets, selected_mask = supervised_state_tensors(encoded, selected, sources, targets)
        self.assertEqual(selected_targets.tolist(), [[4, 5, 6], [12, 13, 14]])
        self.assertTrue(bool(selected_mask.all()))
        loss = next_item_loss(name="sampled_softmax", model=model, encoded=encoded,
                             valid_mask=output_mask, source_positions=sources, targets=targets,
                             num_negatives=32, loss_tail_size=3, original_lengths=mask.sum(dim=1),
                             negative_generator=torch.Generator().manual_seed(24))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(int(activations[0].grad[~mask].count_nonzero()), 0)
        self.assertGreater(float(activations[0].grad[0, :2].abs().sum()), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_invalid_inputs_and_non_native_runtime_fail_loudly(self):
        from rhpr.sasrec.mamba import Mamba4RecModel, _require_native_cuda_runtime

        kwargs = dict(num_items=50, max_seq_len=16, d_model=8, n_layers=1, dropout=0.0)
        for extra, message in (({"use_time_features": True}, "Time OFF"),
                               ({"compressor_type": "tail_only"}, "full-history"),
                               ({"ffn_multiplier": 2}, "ffn_multiplier=4")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                Mamba4RecModel(**kwargs, **extra)
        with mock.patch("rhpr.sasrec.mamba.importlib.import_module", side_effect=ImportError("missing CUDA")), \
                self.assertRaisesRegex(RuntimeError, "no Python/reference timing fallback"):
            _require_native_cuda_runtime()
        model = self.model()
        items = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "right padding"):
            model.encode(items, torch.tensor([[True, False, True]]))
        with self.assertRaisesRegex(ValueError, "gap_ids"):
            model.encode(items, torch.ones_like(items, dtype=torch.bool), torch.zeros_like(items))
        with self.assertRaisesRegex(ValueError, "empty"):
            model.encode(items, torch.zeros_like(items, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            model.encode(torch.ones(1, 17, dtype=torch.long), torch.ones(1, 17, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
