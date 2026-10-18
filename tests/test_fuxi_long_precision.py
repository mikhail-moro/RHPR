from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import torch
    import einops
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch/einops is not installed')
class FuXiLongPrecisionTest(unittest.TestCase):
    def _devices(self):
        return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    def _inputs(self, *, dtype, device, requires_grad=False):
        torch.manual_seed(106)
        # Broadcast queries and decays exercise the positional-channel layout.
        chunks, batch, heads, width = 3, 2, 2, 2
        shapes = ((1, 12, heads, width), (1, 12, heads, width),
                  (batch, 12, heads, 3), (chunks, 1, heads),
                  (chunks, 1, heads, 4, 4), (chunks, 1, 4, heads),
                  (chunks, 1, 4, heads))
        values = [torch.randn(shape, device=device, dtype=dtype) * 0.1
                  for shape in shapes]
        for index in (3, 4, 5, 6):
            values[index] = values[index].abs() + 0.4
        values[4] = values[4].tril()
        return [value.detach().requires_grad_(requires_grad) for value in values]

    @staticmethod
    def _reference_parallel(inputs, chunk_size=4):
        q, k, v, cw, local, frwd, bkwd = inputs
        batch, length, heads, value_dim = v.shape
        chunks = length // chunk_size
        def chunk(value):
            return value.reshape(value.shape[0], chunks, chunk_size,
                                 value.shape[2], value.shape[3]).transpose(0, 1)
        q, k, v = [chunk(value) for value in (q, k, v)]
        state = torch.zeros(batch, heads, q.shape[-1], value_dim,
                            dtype=v.dtype, device=v.device)
        outputs = []
        for index in range(chunks):
            cross = torch.einsum('bchk,bhkv->bchv', q[index] * frwd[index, ..., None], state)
            weights = torch.einsum('bnhd,bmhd->bhnm', q[index], k[index]) * local[index]
            within = torch.einsum('bhnm,bmhd->bnhd', weights, v[index])
            outputs.append(cross + within)
            state = (state * cw[index, ..., None, None]
                     + torch.einsum('bchk,bchv->bhkv', k[index] * bkwd[index, ..., None], v[index]))
        return torch.stack(outputs).transpose(0, 1).reshape(batch, length, heads * value_dim)

    def test_recurrent_state_exceeding_fp16_range_is_preserved(self):
        from rhpr.vendor.fuxi.linear_attn_fn import eval_chunkwise_hidden_state_fn
        for device in self._devices():
            with self.subTest(device=device):
                kv = torch.full((220, 1, 1, 1, 1), 400., dtype=torch.float16,
                                device=device, requires_grad=True)
                decay = torch.ones((220, 1, 1), device=device, requires_grad=True)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    states = eval_chunkwise_hidden_state_fn(kv, decay)
                self.assertEqual(states.dtype, torch.float32)
                self.assertEqual(states[-1].item(), 87_600.)
                self.assertTrue(bool(states.isfinite().all()))
                # A finite read and backward exist although the state itself
                # could not be represented in FP16.
                (states[-1].sum() * 0.01).backward()
                self.assertTrue(bool(kv.grad.isfinite().all()))
                self.assertTrue(bool(decay.grad.isfinite().all()))

    def test_parallel_read_stays_finite_after_large_state_and_kv_product(self):
        from rhpr.vendor.fuxi.linear_attn_fn import chunkwise_parallel_forward
        for device in self._devices():
            for length, key_value in ((220, 20.), (2, 300.)):
                with self.subTest(device=device, length=length):
                    q = torch.full((1, length, 1, 1), 0.01, dtype=torch.float16, device=device)
                    k = torch.full_like(q, key_value)
                    v = torch.full_like(q, key_value)
                    ones = torch.ones(length, 1, 1, device=device)
                    inputs = [q, k, v, ones, ones[..., None, None],
                              ones[..., None], ones[..., None]]
                    reference = self._reference_parallel([value.double() for value in inputs], chunk_size=1)
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        actual = chunkwise_parallel_forward(*inputs, chunk_size=1)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(bool(actual.isfinite().all()))
                    torch.testing.assert_close(actual.double(), reference, rtol=2e-6, atol=1e-5)

    def test_autocast_parallel_outputs_and_gradients_match_fp64_equations(self):
        from rhpr.vendor.fuxi.linear_attn_fn import chunkwise_parallel_forward
        for device in self._devices():
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    values = self._inputs(dtype=dtype, device=device, requires_grad=True)
                    reference_values = [value.detach().double().requires_grad_() for value in values]
                    expected = self._reference_parallel(reference_values)
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        actual = chunkwise_parallel_forward(*values, chunk_size=4)
                        self.assertEqual(actual.dtype, torch.float32)
                        objective = torch.linspace(-0.7, 1.1, actual.numel(), device=device).reshape_as(actual)
                        loss = (actual * objective).sum()
                    # Match training: AMP forward, backward outside autocast.
                    loss.backward()
                    (expected * objective.double()).sum().backward()
                    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
                    tolerance = {torch.float16: 2e-3, torch.bfloat16: 2e-2,
                                 torch.float32: 2e-5}[dtype]
                    for value, reference in zip(values, reference_values):
                        self.assertTrue(bool(value.grad.isfinite().all()))
                        torch.testing.assert_close(value.grad.double(), reference.grad,
                                                   rtol=tolerance, atol=tolerance * 0.01)

    def test_custom_state_backward_under_autocast_matches_fp64(self):
        from rhpr.vendor.fuxi.linear_attn_fn import eval_chunkwise_hidden_state_fn
        for device in self._devices():
            torch.manual_seed(109)
            kv = (torch.randn(5, 2, 3, 2, 4, device=device) * .3).requires_grad_()
            decay = (torch.rand(5, 1, 3, device=device) * .2 + .7).requires_grad_()
            rkv = kv.detach().double().requires_grad_()
            rd = decay.detach().double().requires_grad_()
            state = torch.zeros_like(rkv[0])
            references = []
            for index in range(5):
                references.append(state)
                state = state * rd[index, ..., None, None] + rkv[index]
            expected = torch.stack(references)
            with torch.autocast(device_type=device, dtype=torch.float16):
                actual = eval_chunkwise_hidden_state_fn(kv, decay)
                objective = torch.linspace(-.7, 1.3, actual.numel(), device=device).reshape_as(actual)
                (actual * objective).sum().backward()
            (expected * objective.double()).sum().backward()
            torch.testing.assert_close(kv.grad.double(), rkv.grad, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(decay.grad.double(), rd.grad, rtol=1e-6, atol=1e-6)

    def test_training_inference_and_fixed_query_gradients_use_same_recurrence(self):
        from rhpr.vendor.fuxi.linear_attn_fn import chunkwise_parallel_forward
        values = self._inputs(dtype=torch.float16, device='cpu', requires_grad=True)
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            training = chunkwise_parallel_forward(*values, chunk_size=4)
            with torch.no_grad():
                inference = chunkwise_parallel_forward(*values, chunk_size=4)
            fixed_query_values = [value.detach().clone().requires_grad_(index != 0)
                                  for index, value in enumerate(values)]
            fixed_query = chunkwise_parallel_forward(*fixed_query_values, chunk_size=4)
        torch.testing.assert_close(training, inference, rtol=0, atol=0)
        torch.testing.assert_close(training, fixed_query, rtol=0, atol=0)
        fixed_query.square().sum().backward()
        self.assertIsNotNone(fixed_query_values[2].grad)
        self.assertIsNotNone(fixed_query_values[3].grad)
        self.assertTrue(bool(fixed_query_values[2].grad.isfinite().all()))

    def test_temporal_chunk_outputs_gradients_match_dense_causal_fp64(self):
        from rhpr.vendor.fuxi.linear_attn import chunkwise_forward
        for device in self._devices():
            torch.manual_seed(107)
            q, k, v = [(torch.randn(2, 12, 2, width, device=device) * .1).requires_grad_()
                       for width in (2, 2, 3)]
            log_decay = (torch.rand(2, 12, 2, device=device) * .1).requires_grad_()
            values = [q, k, v, log_decay]
            refs = [value.detach().double().requires_grad_() for value in values]
            rq, rk, rv, rd = refs
            cumulative = torch.nn.functional.pad(rd.cumsum(dim=1), (0, 0, 1, 0))
            # Preserve FuXi's interval endpoints: sum log_decay[j:i+1].
            decay = torch.exp(-(cumulative[:, 1:, None] - cumulative[:, None, :-1]).clamp_min(0))
            weights = torch.einsum('bnhd,bmhd->bhnm', rq, rk)
            causal = torch.ones(12, 12, device=device).tril()
            weights = weights * decay.permute(0, 3, 1, 2) * causal
            expected = torch.einsum('bhnm,bmhd->bnhd', weights, rv).reshape(2, 12, 6)
            with torch.autocast(device_type=device, dtype=torch.float16):
                actual = chunkwise_forward(*values, chunk_size=4)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
            objective = torch.linspace(-.5, .9, actual.numel(), device=device).reshape_as(actual)
            (actual * objective).sum().backward()
            (expected * objective.double()).sum().backward()
            for value, reference in zip(values, refs):
                torch.testing.assert_close(value.grad.double(), reference.grad, rtol=3e-5, atol=1e-7)

    def test_native_model_amp_ragged_causality_and_training_inference_parity(self):
        from rhpr.sasrec.fuxi import FuXiLinearModel
        for device in self._devices():
            torch.manual_seed(108)
            model = FuXiLinearModel(num_items=100, max_seq_len=16, d_model=32,
                                    n_heads=4, n_layers=2, ffn_multiplier=2,
                                    dropout=0., chunk_size=4).to(device).eval()
            def check_channel_output(module, arguments, output):
                # All three unnormalized channel reads must reach the existing
                # per-channel LayerNorm without an intervening FP16 cast.
                self.assertEqual(output[0].dtype, torch.float32)
                self.assertTrue(bool(output[0].isfinite().all()))
            hooks = [channel.register_forward_hook(check_channel_output)
                     for block in model.encoder.layers
                     for channel in (block._retention, block._channel_t, block._channel_p)]
            self.addCleanup(lambda handles=hooks: [handle.remove() for handle in handles])
            items = torch.arange(1, 17, device=device)[None].repeat(2, 1)
            mask = torch.arange(16, device=device)[None] < torch.tensor([13, 16], device=device)[:, None]
            items = items.masked_fill(~mask, 0)
            changed = items.clone()
            changed[0, 10:13] += 30
            changed[1, 10:] += 30
            changed[0, 13:] = 99  # masked padding must also stay irrelevant
            with torch.autocast(device_type=device, dtype=torch.float16):
                differentiable = model.encode(items, mask)[0]
                with torch.no_grad():
                    inference = model.encode(items, mask)[0]
                    perturbed = model.encode(changed, mask)[0]
                    solo = model.encode(items[:1, :13], mask[:1, :13])[0]
            self.assertTrue(bool(inference.isfinite().all()))
            torch.testing.assert_close(differentiable, inference, rtol=0, atol=0)
            torch.testing.assert_close(inference[:, :10], perturbed[:, :10], rtol=0, atol=0)
            torch.testing.assert_close(inference[:1, :13], solo, rtol=1e-4, atol=1e-5)
            self.assertEqual(int(inference[~mask].count_nonzero()), 0)
            differentiable[:, :10].square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(parameter.grad.isfinite().all()), name)


if __name__ == '__main__':
    unittest.main()
