# Copyright 2025-2026 USTC-StarTeam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Callable
from einops import rearrange

def attention_accumulator_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Keep attention sums at least FP32, preserving FP64 reference inputs."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


class ChunkwiseHiddenStateFunction(Function):
    """The upstream recurrence with FP32-or-higher states and reads.

    State ``i`` is the state *before* chunk ``i``. The final chunk KV and
    decay therefore have no effect on the returned states.
    """

    @staticmethod
    def forward(ctx, inchunk_kv, chunkwise_decay):
        state_dtype = attention_accumulator_dtype(inchunk_kv, chunkwise_decay)
        with torch.autocast(device_type=inchunk_kv.device.type, enabled=False):
            kv = inchunk_kv.to(state_dtype)
            decay = chunkwise_decay.to(state_dtype)
            hidden_states = torch.zeros_like(kv)
            current_hidden_state = torch.zeros_like(kv[0])
            for i in range(kv.shape[0]):
                hidden_states[i] = current_hidden_state
                if i != kv.shape[0] - 1:
                    current_hidden_state = torch.addcmul(
                        kv[i], current_hidden_state, decay[i, ..., None, None]
                    )
        ctx.kv_dtype = inchunk_kv.dtype
        ctx.decay_dtype = chunkwise_decay.dtype
        ctx.save_for_backward(decay, hidden_states)
        # Do not cast this state back to the autocast KV dtype: finite long
        # prefixes can exceed FP16 range while their subsequent reads are small.
        return hidden_states

    @staticmethod
    def backward(ctx, grad_hidden_states):
        decay, state_list = ctx.saved_tensors
        # Backward may itself run under autocast. Explicit casts alone do not
        # protect the decay-gradient einsum from being downcast again.
        with torch.autocast(device_type=state_list.device.type, enabled=False):
            grad_hidden_states = grad_hidden_states.to(state_list.dtype)
            grad_kv = torch.zeros_like(state_list)
            grad_decay = torch.zeros_like(decay)
            grad_current = torch.zeros_like(state_list[0])
            for i in range(state_list.shape[0] - 1, -1, -1):
                if i != state_list.shape[0] - 1:
                    grad_kv[i] = grad_current
                    contribution = torch.einsum(
                        'bhkv,bhkv->bh', grad_current, state_list[i]
                    )
                    grad_decay[i] = contribution.sum_to_size(decay[i].shape)
                    grad_current = torch.addcmul(
                        grad_hidden_states[i], grad_current,
                        decay[i, ..., None, None],
                    )
                else:
                    grad_current = grad_hidden_states[i]
        return grad_kv.to(ctx.kv_dtype), grad_decay.to(ctx.decay_dtype)


def eval_chunkwise_hidden_state_fn(inchunk_kv, chunkwise_decay):
    return ChunkwiseHiddenStateFunction.apply(inchunk_kv, chunkwise_decay)


def chunkwise_parallel_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunkwise_decay: torch.Tensor,
    inchunk_decay: torch.Tensor,
    inchunk_decay_frwd: torch.Tensor,
    inchunk_decay_bkwd: torch.Tensor,
    chunk_size: int,
):
    # RHPR: an autocast einsum can downcast even explicitly FP32 operands.
    # Protect KV products, the recurrence, both reads, and the returned sum.
    # The existing per-channel normalization in native.py receives this FP32
    # result; no clipping, scaling, or additional normalization is introduced.
    dtype = attention_accumulator_dtype(
        q, k, v, chunkwise_decay, inchunk_decay, inchunk_decay_frwd, inchunk_decay_bkwd
    )
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = (tensor.to(dtype) for tensor in (q, k, v))
        chunkwise_decay = chunkwise_decay.to(dtype)
        inchunk_decay = inchunk_decay.to(dtype)
        inchunk_decay_frwd = inchunk_decay_frwd.to(dtype)
        inchunk_decay_bkwd = inchunk_decay_bkwd.to(dtype)
        n = v.shape[1]
        assert n % chunk_size == 0, "n must be divisible by chunk_size"
        q = rearrange(q, 'b (s c) h d -> s b c h d', c=chunk_size)
        k = rearrange(k, 'b (s c) h d -> s b c h d', c=chunk_size)
        v = rearrange(v, 'b (s c) h d -> s b c h d', c=chunk_size)
        tilde_k = k * inchunk_decay_bkwd.unsqueeze(-1)
        inchunk_kv = torch.einsum('sbchk,sbchv->sbhkv', tilde_k, v)
        # One implementation for training and inference, including the case
        # where q is fixed but v or a decay parameter still requires gradients.
        hidden_states = eval_chunkwise_hidden_state_fn(inchunk_kv, chunkwise_decay)
        inchunk_attnmap = torch.einsum('sbnhd,sbmhd->sbhnm', q, k) * inchunk_decay
        y2 = torch.einsum('sbhnm,sbmhd->sbnhd', inchunk_attnmap, v)
        tilde_q = q * inchunk_decay_frwd.unsqueeze(-1)
        y1 = torch.einsum('sbchk,sbhkv->sbchv', tilde_q, hidden_states)
        return rearrange(y1 + y2, 's b c h d -> b (s c) (h d)')

if __name__ == "__main__":
    # ==================== Consistency Check ====================
    def _test_forward_consistency():
        from .linear_attn import chunkwise_forward
        
        torch.manual_seed(0)
        B, n, h, d_attn, d_v = 2, 64, 3, 4, 5
        chunk_size = 4
        
        q = torch.randn(B, n, h, d_attn)
        k = torch.randn(B, n, h, d_attn)
        v = torch.randn(B, n, h, d_v)
        
        def make_eval(tensor_list):
            return lambda i: tensor_list[i]
        
        # Precompute decay parameters
        num_chunks = n // chunk_size
        cw_list = [torch.randn(B, h) for _ in range(num_chunks)]
        ic_list = [torch.randn(B, h, chunk_size, chunk_size) for _ in range(num_chunks)]
        ic_frwd_list = [torch.randn(B, chunk_size, h) for _ in range(num_chunks)]
        ic_bkwd_list = [torch.randn(B, chunk_size, h) for _ in range(num_chunks)]
        
        # Original
        out1 = chunkwise_forward(
            q, k, v,
            make_eval(cw_list),
            make_eval(ic_list),
            make_eval(ic_frwd_list),
            make_eval(ic_bkwd_list),
            chunk_size
        )
        
        # Custom function
        out2 = chunkwise_parallel_forward(
            q, k, v,
            torch.stack(cw_list, dim=0),
            torch.stack(ic_list, dim=0),
            torch.stack(ic_frwd_list, dim=0),
            torch.stack(ic_bkwd_list, dim=0),
            chunk_size
        )
        
        print("Forward consistency check:")
        print(f"Max difference: {torch.max(torch.abs(out1 - out2)).item():.2e}")
        assert torch.allclose(out1, out2, atol=1e-6), "Forward outputs do not match!"
        print("✅ Forward consistency PASSED!")

    # ==================== Gradcheck ====================
    def test_gradcheck():
        torch.manual_seed(42)
        B, n, h, d_attn, d_v = 5, 12, 2, 3, 4
        chunk_size = 4
        num_chunks = n // chunk_size
        
        # q = torch.randn(B, n, h, d_attn, dtype=torch.double, requires_grad=True)
        # k = torch.randn(B, n, h, d_attn, dtype=torch.double, requires_grad=True)
        # v = torch.randn(B, n, h, d_v, dtype=torch.double, requires_grad=True)
        
        # cw = torch.randn(B, num_chunks, h, dtype=torch.double, requires_grad=True)
        # ic = torch.randn(B, h, num_chunks, chunk_size, chunk_size, dtype=torch.double, requires_grad=True)
        # ic_frwd = torch.randn(B, num_chunks, chunk_size, h, dtype=torch.double, requires_grad=True)
        # ic_bkwd = torch.randn(B, num_chunks, chunk_size, h, dtype=torch.double, requires_grad=True)
        
        # inputs = (q, k, v, cw, ic, ic_frwd, ic_bkwd, chunk_size)
        
        inchunk_kv = torch.randn(num_chunks, B, h, d_attn, d_v, dtype=torch.double, requires_grad=True)
        chunkwise_decay = torch.randn(num_chunks, 1, 1, dtype=torch.double, requires_grad=True)
        
        print("Running gradcheck...")
        try:
            test_passed = torch.autograd.gradcheck(
                ChunkwiseHiddenStateFunction.apply,
                (inchunk_kv, chunkwise_decay),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
                nondet_tol=0.0
            )
            if test_passed:
                print("✅ Gradcheck PASSED!")
            else:
                print("❌ Gradcheck FAILED!")
        except Exception as e:
            print(f"❌ Gradcheck error: {e}")
            raise

    
    _test_forward_consistency()
    print("\n" + "="*50 + "\n")
    test_gradcheck()
