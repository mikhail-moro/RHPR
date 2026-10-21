"""Query-conditioned Mamba2 SSD prefix compressor.

Only the recurrence backend is borrowed from Mamba2: this module keeps the
existing query generator, shared history projections and residual readout.  Each
query owns an independent recurrent state, represented by folding the slot axis
into the scan batch.  Unlike the diagonal-A QSSM, each SSD head uses ``A=a*I``
and a single event step shared by that head's channels and state coordinates.
This is a new architecture and does not accept a diagonal-QSSM checkpoint.

The differentiable Python recurrence is deliberately limited to CPU checks.
CUDA execution requires the original ``mamba_chunk_scan_combined`` operator;
there is no Python-loop GPU fallback and no bundled copy of its kernels.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rhpr.sasrec.query_ssm import _positive_integer, _RecentQueryPrefixCompressor


def ssd_scan_reference(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                       B: torch.Tensor, C: torch.Tensor,
                       dt_bias: torch.Tensor | None = None,
                       dt_softplus: bool = True) -> torch.Tensor:
    """FP32 scalar-identity recurrence matching Mamba2's SSD scan API.

    ``x`` is [batch,length,heads,head_dim], ``dt`` is [batch,length,heads],
    ``A`` is [heads], and B/C are [batch,length,groups,state]. Each group
    supplies B/C to a contiguous block of heads. The zero-state update is
    ``s_t=exp(dt_t*A)*s_(t-1) + dt_t*x_t*B_t`` and ``y_t=s_t @ C_t``.
    The returned output, including its final position, remains differentiable.
    """
    if any(tensor.device.type != "cpu" for tensor in (x, dt, A, B, C)):
        raise ValueError("ssd_scan_reference is restricted to CPU numerical checks")
    if (x.ndim != 4 or x.shape[1] < 1 or dt.shape != x.shape[:3]
            or A.shape != (x.shape[2],) or B.ndim != 4 or B.shape != C.shape
            or B.shape[:2] != x.shape[:2] or B.shape[2] < 1 or B.shape[3] < 1
            or x.shape[2] % B.shape[2]):
        raise ValueError("invalid SSD input shapes or head/group divisibility")
    if dt_bias is not None and (dt_bias.device.type != "cpu"
                               or dt_bias.shape != A.shape):
        raise ValueError("dt_bias must be a CPU vector matching SSD heads")
    steps = dt.float()
    if dt_bias is not None:
        steps = steps + dt_bias.float()[None, None, :]
    if dt_softplus:
        steps = F.softplus(steps)
    values = x.float()
    heads_per_group = x.shape[2] // B.shape[2]
    writes = B.float().repeat_interleave(heads_per_group, dim=2)
    reads = C.float().repeat_interleave(heads_per_group, dim=2)
    state = torch.zeros((x.shape[0], x.shape[2], x.shape[3], B.shape[3]),
                        device=x.device, dtype=torch.float32)
    outputs = []
    for position in range(x.shape[1]):
        step = steps[:, position, :, None, None]
        decay = torch.exp(step * A.float()[None, :, None, None])
        state = (decay * state + step * values[:, position, :, :, None]
                 * writes[:, position, :, None, :])
        outputs.append((state * reads[:, position, :, None, :]).sum(dim=-1))
    return torch.stack(outputs, dim=1).to(dtype=x.dtype)


def native_ssd_available() -> bool:
    """Check the optional original SSD module without initializing CUDA."""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False
    return True


class QuerySSM2Compressor(_RecentQueryPrefixCompressor):
    """One scalar-A SSD recurrence per query, with joint query/history writes.

    B and dt depend on ``SiLU(W_x*x_t + W_q*q_k)``. By default C depends only
    on q, preserving the current QSSM's readout conditioning. ``joint`` also
    makes C depend on the joint query/history features. There is deliberately
    no new convolution, D skip or output gate in this first backend benchmark.
    """

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 128,
                 query_context_size: int = 128, n_heads: int = 4,
                 d_state: int = 16, d_inner: int = 64, gate_rank: int = 32,
                 head_dim: int = 32, chunk_size: int = 64,
                 c_conditioning: str = "query", dropout: float = 0.0,
                 dt_min: float = 1e-4, dt_max: float = 1e-2,
                 query_source: str = "old_suffix") -> None:
        super().__init__(d_model, summary_tokens, tail_size, query_context_size,
                         n_heads, dropout, query_source=query_source)
        for value, name in ((d_state, "d_state"), (d_inner, "d_inner"),
                            (gate_rank, "gate_rank"), (head_dim, "head_dim"),
                            (chunk_size, "chunk_size")):
            _positive_integer(value, name)
        if d_inner % head_dim:
            raise ValueError("d_inner must be divisible by head_dim")
        if chunk_size & (chunk_size - 1):
            raise ValueError("chunk_size must be a power of two")
        if d_state > 256:
            raise ValueError("SSD benchmark supports d_state at most 256")
        if c_conditioning not in {"query", "joint"}:
            raise ValueError("c_conditioning must be query or joint")
        if (isinstance(dt_min, bool) or isinstance(dt_max, bool)
                or not isinstance(dt_min, (int, float)) or not isinstance(dt_max, (int, float))
                or not math.isfinite(dt_min) or not math.isfinite(dt_max)
                or not 1e-4 <= dt_min <= dt_max):
            raise ValueError("finite dt_min and dt_max must satisfy 1e-4 <= dt_min <= dt_max")
        self.d_state, self.d_inner, self.gate_rank = d_state, d_inner, gate_rank
        self.head_dim, self.scan_heads = head_dim, d_inner // head_dim
        self.chunk_size, self.c_conditioning = chunk_size, c_conditioning
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.implementation = f"{query_source}_query_scalar_identity_ssd"
        self.value_projection = nn.Linear(d_model, d_inner)
        self.input_gate = nn.Linear(d_model, gate_rank)
        self.query_gate = nn.Linear(d_model, gate_rank, bias=False)
        self.delta_projection = nn.Linear(gate_rank, self.scan_heads)
        # One B/C group spans all heads, as the current QSSM shares B/C
        # across its inner channels. Head-specific B/C would change capacity.
        self.write_projection = nn.Linear(gate_rank, d_state, bias=False)
        self.read_projection = nn.Linear(d_model, d_state, bias=False)
        self.joint_read_projection = (nn.Linear(gate_rank, d_state, bias=False)
                                      if c_conditioning == "joint" else None)
        self.output_projection = nn.Linear(d_inner, d_model)
        self.A_log = nn.Parameter(torch.arange(1, self.scan_heads + 1,
                                              dtype=torch.float32).log())
        self.A_log._no_weight_decay = True
        nn.init.uniform_(self.delta_projection.weight, -gate_rank ** -.5, gate_rank ** -.5)
        initial_dt = torch.exp(torch.rand(self.scan_heads)
                               * (math.log(self.dt_max) - math.log(self.dt_min))
                               + math.log(self.dt_min)).clamp(min=1e-4)
        with torch.no_grad():
            self.delta_projection.bias.copy_(initial_dt + torch.log(-torch.expm1(-initial_dt)))
        self.delta_projection.bias._no_reinit = True
        self.last_scan_backend: str | None = None

    def build_scan_inputs(self, memory: torch.Tensor,
                          queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build original SSD API tensors, folding [batch,slots] into batch."""
        if (memory.ndim != 3 or memory.shape[-1] != self.d_model or memory.shape[1] < 1
                or queries.shape != (memory.shape[0], self.summary_tokens, self.d_model)):
            raise ValueError("memory and queries must match compressor dimensions")
        batch, length, _ = memory.shape
        projected_values = self.value_projection(memory)
        joint = F.silu(self.input_gate(memory)[:, None, :, :]
                       + self.query_gate(queries)[:, :, None, :])
        values = projected_values[:, None, :, :].expand(-1, self.summary_tokens, -1, -1)
        values = values.reshape(batch * self.summary_tokens, length,
                                self.scan_heads, self.head_dim)
        delta = F.linear(joint, self.delta_projection.weight)
        writes = self.write_projection(joint)
        if self.joint_read_projection is None:
            reads = self.read_projection(queries)[:, :, None, :].expand(-1, -1, length, -1)
        else:
            reads = self.joint_read_projection(joint)
        dtype = values.dtype
        return {
            "x": values.contiguous(),
            "dt": delta.reshape(batch * self.summary_tokens, length, self.scan_heads).to(dtype).contiguous(),
            "A": -self.A_log.float().exp(),
            "B": writes.reshape(batch * self.summary_tokens, length, 1, self.d_state).to(dtype).contiguous(),
            "C": reads.reshape(batch * self.summary_tokens, length, 1, self.d_state).to(dtype).contiguous(),
            "dt_bias": self.delta_projection.bias.float(),
        }

    def scan_output(self, inputs: dict[str, torch.Tensor],
                    backend: str | None = None) -> torch.Tensor:
        device = inputs["x"].device.type
        selected = ("native" if device == "cuda" else "reference") if backend is None else backend
        if selected == "reference":
            if device != "cpu":
                raise RuntimeError("Python SSD reference is disabled on GPU; install native mamba-ssm")
            self.last_scan_backend = "cpu_ssd_reference"
            return ssd_scan_reference(**inputs, dt_softplus=True)
        if selected != "native" or device != "cuda":
            raise ValueError("native SSD scan requires CUDA; CPU accepts only the reference backend")
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError("Query SSM2 requires original mamba-ssm Triton SSD kernels; "
                               "no Python GPU fallback is permitted") from error
        self.last_scan_backend = "mamba2_native_triton_ssd"
        # Consume differentiable output, never return_final_states: native y's
        # backward includes B/dt/A and every preceding state transition.
        return mamba_chunk_scan_combined(**inputs, chunk_size=self.chunk_size,
                                         D=None, z=None, dt_softplus=True,
                                         return_final_states=False)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old, count, queries, query_ends, memory, _ = self._inputs(embeddings, valid_mask)
        outputs = self.scan_output(self.build_scan_inputs(memory, queries))
        outputs = outputs.reshape(embeddings.shape[0], self.summary_tokens,
                                  memory.shape[1], self.d_inner)
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        last_output = outputs[rows, slots, (old - 1).clamp(min=0)[:, None]]
        return self._finish(self.output_projection(last_output), queries, embeddings,
                            lengths, old, count, query_ends)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "scan_operator": "mamba_chunk_scan_combined",
            "state_transition": "head_scalar_A_times_identity",
            "d_inner": self.d_inner, "d_state": self.d_state,
            "head_dim": self.head_dim, "scan_heads": self.scan_heads,
            "chunk_size": self.chunk_size, "bc_groups": 1,
            "queries_folded_into_batch": True, "c_conditioning": self.c_conditioning,
            "b_and_dt_conditioning": "joint_history_and_query",
            "a_initialization": "negative_head_index_1_through_scan_heads",
            "dt_min": self.dt_min, "dt_max": self.dt_max,
            "diagonal_qssm_checkpoint_compatible": False,
            "new_architecture_requires_quality_evaluation": True,
            "convolution": False, "d_skip": False, "output_gate": False,
        }

    def scan_compute_flops(self, old_length: int, batch_size: int = 1) -> dict[str, int | str]:
        """Logical recurrence estimate, not SSD matrix-kernel instruction FLOPs."""
        _positive_integer(old_length, "old_length")
        _positive_integer(batch_size, "batch_size")
        head_events = batch_size * self.summary_tokens * old_length * self.scan_heads
        return {
            "scan_arithmetic_flops": head_events * (1 + self.d_state
                + self.head_dim * (5 * self.d_state - 1)),
            "scan_exp_evaluations": head_events,
            "scan_softplus_evaluations": head_events,
            "unhooked_delta_projection_flops": 2 * head_events * self.gate_rank,
            "scan_cost_policy": "logical_scalar_identity_recurrence_not_actual_ssd_kernel_flops",
        }


def make_query_ssm2_compressor(**kwargs: Any) -> QuerySSM2Compressor:
    """Construct without advancing the shared-backbone CPU RNG stream."""
    with torch.random.fork_rng(devices=[]):
        return QuerySSM2Compressor(**kwargs)


def audit_native_query_ssm2(device: torch.device | str, head_dim: int = 32,
                           chunk_size: int = 64, d_state: int = 16) -> dict[str, Any]:
    """Require native FP16 forward and final-output gradients to match CPU.

    This is an explicitly invoked, tiny CUDA audit, never an import side effect.
    Inputs span one complete chunk plus one event, checking both cross-chunk
    state-passing backward and a partial final chunk.
    ``d_state`` selects the true B/C state width, independently of head_dim.
    All six differentiable API inputs must receive finite, nonzero gradients.
    Any unavailable/unsupported kernel or failed comparison raises immediately.
    """
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError("native SSD audit requires a CUDA device")
    if head_dim not in {32, 64}:
        raise ValueError("native SSD benchmark audit supports head_dim 32 or 64")
    _positive_integer(d_state, "d_state")
    if d_state > 256:
        raise ValueError("SSD benchmark supports d_state at most 256")
    generator = torch.Generator(device="cpu").manual_seed(1701)
    _positive_integer(chunk_size, "chunk_size")
    if chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a power of two")
    heads, state, length, batch = 64 // head_dim, d_state, chunk_size + 1, 2
    data = {
        "x": torch.randn(batch, length, heads, head_dim, generator=generator) * .5,
        "dt": torch.randn(batch, length, heads, generator=generator) * .2,
        "A": -torch.arange(1, heads + 1, dtype=torch.float32),
        "B": torch.randn(batch, length, 1, state, generator=generator) * .3,
        "C": torch.randn(batch, length, 1, state, generator=generator) * .3,
        "dt_bias": torch.full((heads,), -3.),
    }
    reference_inputs = {key: value.to(dtype=torch.float32 if key in {"A", "dt_bias"}
                                      else torch.float16).detach().requires_grad_()
                        for key, value in data.items()}
    native_inputs = {key: value.detach().to(selected_device).requires_grad_()
                     for key, value in reference_inputs.items()}
    model = make_query_ssm2_compressor(d_model=8, summary_tokens=2, n_heads=2,
                                       d_inner=64, d_state=state, head_dim=head_dim,
                                       chunk_size=chunk_size)
    reference = ssd_scan_reference(**reference_inputs)
    native = model.scan_output(native_inputs, backend="native")
    torch.testing.assert_close(native.detach().cpu(), reference.detach(),
                               rtol=2e-2, atol=2e-3)
    weights = torch.linspace(-.9, 1.3, native.shape[-1]).reshape(1, 1, -1)
    (reference[:, -1].float() * weights).square().sum().backward()
    (native[:, -1].float() * weights.to(selected_device)).square().sum().backward()
    gradient_norms, gradient_errors = {}, {}
    for key in reference_inputs:
        expected, actual = reference_inputs[key].grad, native_inputs[key].grad
        if expected is None or actual is None or not bool(torch.isfinite(actual).all()):
            raise FloatingPointError(f"native SSD audit has missing/nonfinite {key} gradient")
        norm = float(actual.float().norm().detach().cpu())
        if norm <= 0:
            raise FloatingPointError(f"native SSD audit has zero {key} gradient")
        actual_cpu = actual.detach().cpu()
        torch.testing.assert_close(actual_cpu, expected, rtol=5e-2, atol=4e-3)
        gradient_norms[key] = norm
        gradient_errors[key] = float((actual_cpu.float() - expected.float()).abs().max())
    early_x_gradient_norm = float(native_inputs["x"].grad[:, 0].float().norm().cpu())
    if early_x_gradient_norm <= 0:
        raise FloatingPointError("native SSD final-output loss does not reach first input")
    return {
        "status": "passed", "backend": model.last_scan_backend,
        "device": str(selected_device), "head_dim": head_dim, "chunk_size": chunk_size,
        "d_state": d_state,
        "batch": batch, "length": length, "partial_chunk_checked": True,
        "multi_chunk_checked": True, "chunk_count": math.ceil(length / chunk_size),
        "input_dtype": "float16", "a_and_dt_bias_dtype": "float32",
        "cpu_accumulation_dtype": "float32", "loss_uses_last_output": True,
        "output_max_abs_error": float((native.detach().cpu().float()
                                        - reference.detach().float()).abs().max()),
        "gradient_norms": gradient_norms, "gradient_max_abs_errors": gradient_errors,
        "early_x_gradient_norm": early_x_gradient_norm,
        "forward_rtol": 2e-2, "forward_atol": 2e-3,
        "gradient_rtol": 5e-2, "gradient_atol": 4e-3,
    }
