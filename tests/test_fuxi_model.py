from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VENDOR = ROOT / "src/rhpr/vendor/fuxi"
try:
    import torch
    import einops
    from torch.nn import functional as F
except ImportError:
    torch = None


class _ReferencePacking:
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


def _official_block_class():
    """Execute exact official fixtures with only dependency/packing scaffolding."""
    package_name = "_rhpr_official_fuxi_reference"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    torch_proxy = types.SimpleNamespace(**torch.__dict__)
    torch_proxy.ops = types.SimpleNamespace(fbgemm=_ReferencePacking)
    for name in ("linear_attn_fn", "linear_attn", "attn"):
        module = types.ModuleType(f"{package_name}.{name}")
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        fixture = VENDOR / f"upstream_{name}.py.txt"
        exec(compile(fixture.read_bytes(), str(fixture), "exec"), module.__dict__)
        module.torch = torch_proxy
    fixture = VENDOR / "upstream_modules_init.py.txt"
    parsed = ast.parse(fixture.read_bytes())
    ffn_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                     and node.name == "MultistageFeedforwardNeuralNetwork")
    namespace = {"torch": torch_proxy, "F": F}
    exec(compile(ast.Module(body=[ffn_class], type_ignores=[]), str(fixture), "exec"), namespace)
    fixture = VENDOR / "upstream_fuxi_linear.py.txt"
    parsed = ast.parse(fixture.read_bytes())
    block_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                       and node.name == "FuXiLinearBlockJagged")
    namespace.update({"Dict": dict, "Optional": __import__("typing").Optional,
                      "logging": __import__("logging"),
                      "Retention": package.attn.Retention,
                      "LinearPositionalChannel": package.attn.LinearPositionalChannel,
                      "LinearTemporalChannel": package.attn.LinearTemporalChannel})
    exec(compile(ast.Module(body=[block_class], type_ignores=[]), str(fixture), "exec"), namespace)
    return namespace["FuXiLinearBlockJagged"]


class FuXiSourceTest(unittest.TestCase):
    def test_upstream_fixtures_match_known_official_git_blobs(self):
        known = {"upstream_fuxi_linear.py.txt": "5e2455ca24a2d3b40ac68118835eb9c62a3dad8c",
                 "upstream_attn.py.txt": "d5d250a848b841d0008f3276e03b8bb0ed5cedd7",
                 "upstream_linear_attn.py.txt": "4e0635ea93d1324ac6a1ac2249312e2faad06c4d",
                 "upstream_linear_attn_fn.py.txt": "5072526467bb18f1a503538369e1ec96595486f4",
                 "upstream_modules_init.py.txt": "03126af007324cd060c663495e38628d0ec30be3"}
        provenance = json.loads((VENDOR / "provenance.json").read_text())
        self.assertEqual(provenance["upstream_revision"], "5a704061a7ebf0b81afb465f12f93f2747ba4667")
        for name, expected_blob in known.items():
            raw = (VENDOR / name).read_bytes()
            self.assertEqual(hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(), expected_blob)
        for source in provenance["source_files"].values():
            raw = (VENDOR / source["fixture"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])


@unittest.skipIf(torch is None, "torch/einops is not installed")
class FuXiModelTest(unittest.TestCase):
    def model(self, *, max_seq_len=16, n_layers=2):
        from rhpr.sasrec.fuxi import FuXiLinearModel
        return FuXiLinearModel(num_items=100, max_seq_len=max_seq_len, d_model=32,
                               n_heads=4, n_layers=n_layers, ffn_multiplier=2,
                               dropout=0.0, chunk_size=4)

    def test_chunkwise_scan_mixed_precision_gradients_match_autograd_recurrence(self):
        from rhpr.vendor.fuxi.linear_attn_fn import eval_chunkwise_hidden_state_fn

        # Four chunks exercise the recurrence after FP32 decay promotes the
        # reverse accumulator; two chunks would miss the original dtype error.
        for kv_dtype, decay_dtype in ((torch.float16, torch.float32),
                                      (torch.bfloat16, torch.float32),
                                      (torch.float32, torch.float32),
                                      (torch.float64, torch.float64),
                                      (torch.float32, torch.float64)):
            for decay_shape in ((4, 2, 3), (4, 1, 3), (4, 2, 1), (4, 1, 1)):
                with self.subTest(kv_dtype=kv_dtype, decay_dtype=decay_dtype,
                                  decay_shape=decay_shape):
                    torch.manual_seed(93)
                    kv = (torch.randn(4, 2, 3, 2, 3, dtype=kv_dtype) * 0.1).requires_grad_()
                    decay = (torch.rand(decay_shape, dtype=decay_dtype) * 0.3 + 0.5).requires_grad_()
                    reference_kv = kv.detach().clone().requires_grad_()
                    reference_decay = decay.detach().clone().requires_grad_()
                    accumulator_dtype = torch.promote_types(kv_dtype, decay_dtype)
                    state = torch.zeros_like(reference_kv[0], dtype=accumulator_dtype)
                    reference_states = [state]
                    for index in range(3):
                        state = (reference_kv[index].to(accumulator_dtype)
                                 + state * reference_decay[index].to(accumulator_dtype)[..., None, None])
                        reference_states.append(state)
                    expected = torch.stack(reference_states).to(kv_dtype)
                    actual = eval_chunkwise_hidden_state_fn(kv, decay)
                    self.assertEqual(actual.dtype, kv_dtype)
                    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2,
                                 torch.float32: 1e-6, torch.float64: 1e-12}[kv_dtype]
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance * 0.01)
                    # Different weights at each output check every dependency,
                    # including zero gradients for the unused final KV/decay.
                    objective = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
                    (actual.float() * objective).sum().backward()
                    (expected.float() * objective).sum().backward()
                    self.assertEqual(kv.grad.dtype, kv_dtype)
                    self.assertEqual(decay.grad.dtype, decay_dtype)
                    self.assertTrue(bool(torch.isfinite(kv.grad).all()))
                    self.assertTrue(bool(torch.isfinite(decay.grad).all()))
                    torch.testing.assert_close(kv.grad, reference_kv.grad,
                                               rtol=tolerance, atol=tolerance * 0.01)
                    torch.testing.assert_close(decay.grad, reference_decay.grad,
                                               rtol=1e-6, atol=1e-7)
                    self.assertEqual(int(kv.grad[-1].count_nonzero()), 0)
                    self.assertEqual(int(decay.grad[-1].count_nonzero()), 0)

    def test_chunkwise_scan_double_gradcheck_with_broadcast_decay(self):
        from rhpr.vendor.fuxi.linear_attn_fn import eval_chunkwise_hidden_state_fn
        torch.manual_seed(94)
        kv = torch.randn(4, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
        decay = torch.full((4, 2, 1), 0.5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(eval_chunkwise_hidden_state_fn,
                                                (kv, decay), eps=1e-6,
                                                atol=1e-5, rtol=1e-3))

    def test_amp_forward_and_backward_outside_autocast_have_finite_gradients(self):
        items = torch.arange(1, 17)[None, :].expand(2, -1).clone()
        mask = torch.ones_like(items, dtype=torch.bool)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(95)
                model = self.model()
                with torch.autocast(device_type="cpu", dtype=dtype):
                    states, _, _ = model.encode(items, mask)
                    loss = states[:, -4:].float().square().mean()
                # Match normal AMP training: forward is autocast; backward runs
                # outside it. This covers semantic, temporal and position scans.
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(bool(torch.isfinite(states).all()))
                self.assertIsNotNone(model.item_embedding.weight.grad)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_pinned_official_block_forward_backward_parity_without_global_mask(self):
        from rhpr.vendor.fuxi.native import FuXiLinearBlockJagged
        official_class = _official_block_class()
        arguments = dict(embedding_dim=32, linear_hidden_dim=8, attention_dim=8,
                         dropout_ratio=0.25, attn_dropout_ratio=0.0, num_heads=4,
                         linear_activation="silu", id_layer=0, ffn_multiply=2, chunk_size=4,
                         channel_t_config={"num_heads": 8, "base": 2, "start_index": 10,
                                           "base_stride": 3, "aug_current": True,
                                           "learnable_gamma": True},
                         channel_p_config={"max_sequence_length": 16, "dim": 32,
                                           "aug_current": True})
        for training in (False, True):
            native = FuXiLinearBlockJagged(**arguments).train(training)
            official = official_class(**arguments).train(training)
            official.load_state_dict(native.state_dict(), strict=True)
            actual_input = torch.randn(22, 32, requires_grad=True)
            reference_input = actual_input.detach().clone().requires_grad_()
            # Nonzero native-channel timestamps exercise learned temporal decay
            # in this source-parity test. The model wrapper itself rejects them.
            timestamps = (torch.arange(16)[None, :].expand(2, -1) * 100
                          if training else torch.zeros(2, 16, dtype=torch.long))
            common = dict(x_offsets=torch.tensor([0, 13, 22]),
                          all_timestamps=timestamps,
                          past_lengths=torch.tensor([13, 9]), target_timestamps=None,
                          current_length=16)
            torch.manual_seed(92)
            actual = native(x=actual_input, invalid_attn_mask=None, **common)[0]
            torch.manual_seed(92)
            expected = official(x=reference_input, invalid_attn_mask=torch.ones(16, 16).tril(), **common)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            objective = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
            (actual.square().mean() + (actual * objective).mean()).backward()
            (expected.square().mean() + (expected * objective).mean()).backward()
            torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-6, atol=1e-7)
            for name, parameter in native.named_parameters():
                reference_grad = dict(official.named_parameters())[name].grad
                self.assertEqual(parameter.grad is None, reference_grad is None, name)
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                    torch.testing.assert_close(parameter.grad, reference_grad,
                                               rtol=1e-6, atol=1e-7, msg=name)

    def test_future_items_padding_and_batch_membership_do_not_change_valid_past(self):
        torch.manual_seed(71)
        model = self.model().eval()
        solo = torch.tensor([[1, 2, 3, 4, 5]])
        batched = torch.tensor([[1, 2, 3, 4, 5, 99, 99, 99], [6, 7, 8, 9, 10, 11, 12, 13]])
        batch_mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]
        with torch.no_grad():
            solo_states, solo_mask, sources = model.encode(solo, torch.ones_like(solo, dtype=torch.bool))
            batch_states, output_mask, batch_sources = model.encode(batched, batch_mask)
            torch.testing.assert_close(solo_states[0], batch_states[0, :5], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(sources[0], batch_sources[0, :5]))
            self.assertEqual(int(batch_states[~output_mask].count_nonzero()), 0)
            torch.testing.assert_close(model.last_state(batched, batch_mask),
                                       batch_states[torch.arange(2), torch.tensor([4, 7])])
            items = torch.arange(1, 17)[None, :]
            future_changed = items.clone()
            future_changed[:, 13:] += 37
            mask = torch.ones_like(items, dtype=torch.bool)
            before = model.encode(items, mask)[0]
            after = model.encode(future_changed, mask)[0]
            torch.testing.assert_close(before[:, :13], after[:, :13], rtol=0, atol=1e-6)
            self.assertFalse(torch.allclose(before[:, -1], after[:, -1]))

    def test_runtime_positional_crop_matches_official_full_padding(self):
        model = self.model().eval()
        items = torch.arange(1, 6)[None, :]
        padded = F.pad(items, (0, 11))
        mask = torch.ones_like(items, dtype=torch.bool)
        with torch.no_grad():
            short = model.encode(items, mask)[0]
            full = model.encode(padded, padded.ne(0))[0]
        torch.testing.assert_close(short, full[:, :5], rtol=1e-5, atol=1e-6)

    def test_tail_targets_gradient_and_time_off_contract(self):
        from rhpr.sasrec.losses import next_item_loss
        model = self.model()
        items = torch.arange(1, 17)[None, :].expand(2, -1).clone()
        mask = torch.arange(16)[None, :] < torch.tensor([16, 9])[:, None]
        items = items.masked_fill(~mask, 0)
        states, output_mask, sources = model.encode(items, mask)
        loss = next_item_loss(name="sampled_softmax", model=model, encoded=states,
                              valid_mask=output_mask, source_positions=sources,
                              targets=items + 1, num_negatives=8,
                              loss_tail_size=4, original_lengths=mask.sum(dim=1))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertEqual(int(model.item_embedding.weight.grad[0].count_nonzero()), 0)
        with self.assertRaisesRegex(ValueError, "Time OFF"):
            model.encode(items, mask, timestamps=torch.ones_like(items))
        with self.assertRaisesRegex(ValueError, "right padding"):
            model.encode(items, mask.roll(1, dims=1))

    def test_major_matmul_count_matches_actual_native_operators(self):
        model = self.model(n_layers=2)
        for batch_size in (1, 2):
            for length in (13, 16):
                items = torch.ones((batch_size, length), dtype=torch.long)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            with_flops=True) as profile:
                    model.encode(items, torch.ones_like(items, dtype=torch.bool))
                actual = sum(event.flops for event in profile.key_averages()
                             if event.key in {"aten::mm", "aten::bmm", "aten::addmm"})
                self.assertEqual(model.major_matmul_flops(batch_size, length), actual)

    def test_16k_forward_and_backward_never_allocate_global_sequence_square(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from rhpr.sasrec.fuxi import FuXiLinearModel
        width = 16_384
        seen_shapes = []

        class ShapeAudit(TorchDispatchMode):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                result = function(*args, **(kwargs or {}))
                def inspect(value):
                    if isinstance(value, torch.Tensor):
                        shape = tuple(value.shape)
                        if len(shape) >= 2 and shape[-2:] == (width, width):
                            raise AssertionError(f"global L-square allocation in {function}: {shape}")
                        seen_shapes.append(shape)
                    elif isinstance(value, (tuple, list)):
                        for entry in value:
                            inspect(entry)
                inspect(result)
                return result

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ShapeAudit():
                model = FuXiLinearModel(num_items=100, max_seq_len=width, d_model=16,
                                        n_heads=2, n_layers=1, ffn_multiplier=1,
                                        dropout=0.0, chunk_size=128,
                                        fuxi_temporal_heads=2, fuxi_positional_dim=4)
                items = torch.ones((1, width), dtype=torch.long)
                mask = torch.ones_like(items, dtype=torch.bool)
                states, _, _ = model.encode(items, mask)
                states[:, -128:].square().mean().backward()
                self.assertTrue(bool(torch.isfinite(states).all()))
            self.assertIn((128, 128), seen_shapes)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
