"""Query-conditioned fixed-budget prefix compressors.

By default, training queries see the first ``query_context_size`` events of the raw tail,
not its later supervised events.  With a 159-event tail and 128-event query
context this permits 32 causal next-item losses after a single prefix scan.
At evaluation the 128-event tail supplies a fresh query for the next item.

The optional ``old_suffix`` query source instead reads the last available
``query_context_size`` old-prefix events. Its summaries depend only on the old
prefix, so every raw-tail state can receive a causal next-item loss in parallel.
Those summaries can be reused while that prefix and model weights stay fixed.

The selective recurrence uses Mamba's native CUDA selective-scan operator.  A
small, differentiable CPU implementation exists for numerical/unit checks only;
there is deliberately no Python-loop fallback for GPU experiments.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rhpr.sasrec.long_compression import _FixedBudgetPrefixCompressor


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _position_features(recency: torch.Tensor, d_model: int,
                       dtype: torch.dtype, scale: float = 0.02) -> torch.Tensor:
    frequencies = torch.exp(torch.arange(0, d_model, 2, device=recency.device,
                                        dtype=torch.float32) * (-math.log(10000.0) / d_model))
    angles = recency.float()[:, :, None] * frequencies[None, None, :]
    features = torch.empty((*angles.shape[:2], d_model), device=recency.device,
                           dtype=torch.float32)
    features[:, :, 0::2] = angles.sin()
    features[:, :, 1::2] = angles[:, :, :d_model // 2].cos()
    return (features * scale).to(dtype=dtype)


def selective_scan_reference(u: torch.Tensor, delta: torch.Tensor,
                             A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                             delta_bias: torch.Tensor | None = None,
                             delta_softplus: bool = True) -> torch.Tensor:
    """Differentiable FP32 diagonal selective scan, restricted to CPU checks.

    ``u, delta`` are [batch, channels, length], A is [channels, state], and
    input-dependent B/C are [batch, state, length].  This is the same zero-state
    recurrence as native ``selective_scan_fn`` without D or z projections.
    In particular, callers receive differentiable *outputs*, not last_state.
    """
    if any(tensor.device.type != "cpu" for tensor in (u, delta, A, B, C)):
        raise ValueError("selective_scan_reference is restricted to CPU numerical checks")
    if (u.ndim != 3 or delta.shape != u.shape or A.ndim != 2
            or A.shape[0] != u.shape[1] or B.shape != (u.shape[0], A.shape[1], u.shape[2])
            or C.shape != B.shape or u.shape[2] < 1):
        raise ValueError("invalid selective-scan input shapes")
    if delta_bias is not None and (delta_bias.device.type != "cpu"
                                   or delta_bias.shape != (u.shape[1],)):
        raise ValueError("delta_bias must be a CPU vector matching scan channels")
    original_dtype = u.dtype
    values, steps = u.float(), delta.float()
    if delta_bias is not None:
        steps = steps + delta_bias.float()[None, :, None]
    if delta_softplus:
        steps = F.softplus(steps)
    state = torch.zeros((u.shape[0], u.shape[1], A.shape[1]), device=u.device,
                        dtype=torch.float32)
    outputs = []
    for index in range(u.shape[2]):
        step = steps[:, :, index, None]
        decay = torch.exp(step * A.float()[None, :, :])
        write = step * B.float()[:, None, :, index] * values[:, :, index, None]
        state = decay * state + write
        outputs.append((state * C.float()[:, None, :, index]).sum(dim=-1))
    return torch.stack(outputs, dim=-1).to(dtype=original_dtype)


def native_selective_scan_available() -> bool:
    """Check the optional native extension without requiring CUDA at import."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
        import selective_scan_cuda  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False
    return True


class RecentQueryGenerator(nn.Module):
    """Learned slots cross-attend to an occupied, chronological query context."""

    def __init__(self, d_model: int, summary_tokens: int, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        _positive_integer(d_model, "d_model")
        _positive_integer(summary_tokens, "summary_tokens")
        _positive_integer(n_heads, "n_heads")
        if d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError("d_model must be divisible by n_heads and dropout in [0,1)")
        self.d_model, self.summary_tokens = d_model, summary_tokens
        self.queries = nn.Parameter(torch.empty(summary_tokens, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def slot_queries(self, batch_size: int) -> torch.Tensor:
        return self.queries[None, :, :].expand(batch_size, -1, -1)

    def forward(self, recent_embeddings: torch.Tensor, recent_mask: torch.Tensor) -> torch.Tensor:
        if (recent_embeddings.ndim != 3 or recent_embeddings.shape[-1] != self.d_model
                or recent_mask.shape != recent_embeddings.shape[:2] or recent_mask.dtype != torch.bool
                or recent_embeddings.shape[1] < 1):
            raise ValueError("recent context and boolean mask must have matching nonempty shapes")
        lengths = recent_mask.sum(dim=1)
        offsets = torch.arange(recent_embeddings.shape[1], device=recent_embeddings.device)[None, :]
        recency = (lengths[:, None] - 1 - offsets).clamp(min=0)
        memory = self.context_norm(recent_embeddings + _position_features(
            recency, self.d_model, recent_embeddings.dtype))
        memory = memory.masked_fill(~recent_mask[:, :, None], 0)
        safe_mask = recent_mask.clone()
        safe_mask[:, 0] |= lengths == 0
        queries = self.slot_queries(recent_embeddings.shape[0])
        readout = self.attention(self.query_norm(queries), memory, memory,
                                 key_padding_mask=~safe_mask, need_weights=False)[0]
        return queries + self.dropout(readout)


class _RecentQueryPrefixCompressor(_FixedBudgetPrefixCompressor):
    source_position_policy = "latest_input_of_query_and_global_old_prefix"

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 159,
                 query_context_size: int = 128, n_heads: int = 4,
                 dropout: float = 0.0, query_source: str = "recent_start") -> None:
        super().__init__(summary_tokens, tail_size)
        _positive_integer(d_model, "d_model")
        _positive_integer(query_context_size, "query_context_size")
        if query_source not in {"recent_start", "old_suffix"}:
            raise ValueError("query_source must be recent_start or old_suffix")
        if query_source == "recent_start" and query_context_size > tail_size:
            raise ValueError("query_context_size must not exceed tail_size")
        self.d_model, self.query_context_size = d_model, query_context_size
        self.query_source = query_source
        if query_source == "old_suffix":
            self.source_position_policy = "latest_input_of_global_old_prefix"
        self.query_generator = RecentQueryGenerator(d_model, summary_tokens, n_heads, dropout)
        self.memory_norm = nn.LayerNorm(d_model)
        self.readout_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def _inputs(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, ...]:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.d_model:
            raise ValueError("embedding width must match d_model")
        if self.query_source == "recent_start" and self.tail_size < self.query_context_size:
            raise ValueError("tail_size must not be smaller than query_context_size")
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        context_width = min(self.query_context_size, embeddings.shape[1])
        context_offsets = torch.arange(context_width, device=embeddings.device)[None, :]
        if self.query_source == "old_suffix":
            context_lengths = old_lengths.clamp(max=self.query_context_size)
            context_starts = old_lengths - context_lengths
        else:
            context_lengths = (lengths - old_lengths).clamp(max=self.query_context_size)
            context_starts = old_lengths
        context_mask = context_offsets < context_lengths[:, None]
        context_sources = (context_starts[:, None] + context_offsets).clamp(max=embeddings.shape[1] - 1)
        context = embeddings[rows, context_sources].masked_fill(~context_mask[:, :, None], 0)
        queries = self.query_generator(context, context_mask)
        query_endpoints = (old_lengths - 1 if self.query_source == "old_suffix"
                           else old_lengths + context_lengths - 1)
        width = max(1, int(old_lengths.max().item()))
        offsets = torch.arange(width, device=embeddings.device)[None, :]
        memory_mask = offsets < old_lengths[:, None]
        memory = embeddings[:, :width].masked_fill(~memory_mask[:, :, None], 0)
        memory = self.memory_norm(memory + _position_features(
            query_endpoints[:, None] - offsets, self.d_model, embeddings.dtype))
        memory = memory.masked_fill(~memory_mask[:, :, None], 0)
        return lengths, old_lengths, summary_lengths, queries, query_endpoints, memory, memory_mask

    def _finish(self, readout: torch.Tensor, queries: torch.Tensor,
                embeddings: torch.Tensor, lengths: torch.Tensor,
                old_lengths: torch.Tensor, summary_lengths: torch.Tensor,
                query_endpoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summaries = queries + self.dropout(self.readout_norm(readout))
        summaries = summaries + self.dropout(self.ffn(self.ffn_norm(summaries)))
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        active = slots < summary_lengths[:, None]
        summaries = summaries.masked_fill(~active[:, :, None], 0)
        # The endpoint covers both query support and global old-memory support.
        # These are feature/position metadata, never additional loss targets.
        sources = query_endpoints[:, None].expand(-1, self.summary_tokens).masked_fill(~active, -1)
        return self._pack(summaries, sources, embeddings, lengths, old_lengths, summary_lengths)


class QueryCrossAttentionCompressor(_RecentQueryPrefixCompressor):
    implementation = "recent_query_global_cross_attention"

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 159,
                 query_context_size: int = 128, n_heads: int = 4,
                 dropout: float = 0.0, query_source: str = "recent_start") -> None:
        super().__init__(d_model, summary_tokens, tail_size, query_context_size, n_heads, dropout,
                         query_source=query_source)
        if query_source == "old_suffix":
            self.implementation = "old_suffix_query_global_cross_attention"
        self.cross_query_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old, count, queries, query_ends, memory, memory_mask = self._inputs(embeddings, valid_mask)
        safe_mask = memory_mask.clone()
        safe_mask[:, 0] |= old == 0
        readout = self.cross_attention(self.cross_query_norm(queries), memory, memory,
                                       key_padding_mask=~safe_mask, need_weights=False)[0]
        return self._finish(readout, queries, embeddings, lengths, old, count, query_ends)


class QuerySSMCompressor(_RecentQueryPrefixCompressor):
    """Zero-state diagonal SSM, with context-conditioned writes or readout only.

    For each slot j, r_tj=SiLU(W_x memory_t + W_q write_query_j).
    Delta_tj=softplus(W_delta r_tj + dt_bias), B_tj=W_B r_tj,
    A=-exp(A_log), and C_j=W_C context_query_j.  The recurrence is
    h_tj=exp(Delta_tj A) h_(t-1)j + Delta_tj B_tj u_t.
    The final *output* y_Nj=C_j h_Nj carries gradients through the native scan.

    ``slot_only`` keeps exactly the same parameterization and uses static slot
    queries for Delta/B, while the context query still supplies C and the common
    residual/readout block.  History value/input projections are shared over K.
    """

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 159,
                 query_context_size: int = 128, n_heads: int = 4,
                 d_state: int = 16, d_inner: int = 128, gate_rank: int = 32,
                 write_conditioning: str = "recent", dropout: float = 0.0,
                 dt_min: float = 1e-4, dt_max: float = 0.1,
                 query_source: str = "recent_start") -> None:
        super().__init__(d_model, summary_tokens, tail_size, query_context_size, n_heads, dropout,
                         query_source=query_source)
        for value, name in ((d_state, "d_state"), (d_inner, "d_inner"), (gate_rank, "gate_rank")):
            _positive_integer(value, name)
        if d_state > 256:
            raise ValueError("native selective scan supports d_state at most 256")
        if write_conditioning not in {"recent", "slot_only"}:
            raise ValueError("write_conditioning must be recent or slot_only")
        if (isinstance(dt_min, bool) or isinstance(dt_max, bool)
                or not isinstance(dt_min, (int, float)) or not isinstance(dt_max, (int, float))
                or not math.isfinite(dt_min) or not math.isfinite(dt_max)
                or not 1e-4 <= dt_min <= dt_max):
            raise ValueError("finite dt_min and dt_max must satisfy 1e-4 <= dt_min <= dt_max")
        self.d_state, self.d_inner, self.gate_rank = d_state, d_inner, gate_rank
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.write_conditioning = write_conditioning
        self.implementation = ("recent_query_selective_ssm" if write_conditioning == "recent"
                               else "recent_query_readout_static_write_ssm")
        if query_source == "old_suffix":
            self.implementation = ("old_suffix_query_selective_ssm" if write_conditioning == "recent"
                                   else "old_suffix_query_readout_static_write_ssm")
        self.value_projection = nn.Linear(d_model, d_inner)
        self.input_gate = nn.Linear(d_model, gate_rank)
        self.query_gate = nn.Linear(d_model, gate_rank, bias=False)
        self.delta_projection = nn.Linear(gate_rank, d_inner)
        self.write_projection = nn.Linear(gate_rank, d_state, bias=False)
        self.read_projection = nn.Linear(d_model, d_state, bias=False)
        self.output_projection = nn.Linear(d_inner, d_model)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(a.log()[None, :].repeat(d_inner, 1))
        self.A_log._no_weight_decay = True
        # Mamba-style positive small initial steps.  The 1e-4 lower endpoint
        # includes ~10K-event initial decay timescales for A=-1, appropriate to
        # the 8K prefix. Delta is a learned event step, not wall-clock time.
        # Preserve these values:
        # generic blanket Linear initialization must not reset dt/A afterward.
        nn.init.uniform_(self.delta_projection.weight, -gate_rank ** -0.5, gate_rank ** -0.5)
        initial_dt = torch.exp(torch.rand(d_inner) * (math.log(self.dt_max) - math.log(self.dt_min))
                               + math.log(self.dt_min)).clamp(min=1e-4)
        inverse_softplus = initial_dt + torch.log(-torch.expm1(-initial_dt))
        with torch.no_grad():
            self.delta_projection.bias.copy_(inverse_softplus)
        self.delta_projection.bias._no_reinit = True
        self.last_scan_backend: str | None = None

    def build_scan_inputs(self, memory: torch.Tensor, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose native scan tensors for numerical/conditioning audits.

        ``memory`` is [B,L,d_model] after shared memory normalization and raw
        recency features. ``queries`` is [B,K,d_model].  Delta excludes its bias
        because the native operator applies that FP32 bias before softplus.
        """
        if (memory.ndim != 3 or memory.shape[-1] != self.d_model or queries.ndim != 3
                or queries.shape != (memory.shape[0], self.summary_tokens, self.d_model)):
            raise ValueError("memory and queries must match compressor dimensions")
        batch, length, _ = memory.shape
        write_queries = (queries if self.write_conditioning == "recent"
                         else self.query_generator.slot_queries(batch))
        # Expensive d_model history projections happen once, before K expansion.
        projected_values = self.value_projection(memory)
        projected_input = self.input_gate(memory)
        joint = F.silu(projected_input[:, None, :, :]
                       + self.query_gate(write_queries)[:, :, None, :])
        delta = F.linear(joint, self.delta_projection.weight).permute(0, 1, 3, 2)
        writes = self.write_projection(joint).permute(0, 1, 3, 2)
        reads = self.read_projection(queries)[:, :, :, None].expand(-1, -1, -1, length)
        values = projected_values[:, None, :, :].expand(-1, self.summary_tokens, -1, -1)
        values = values.permute(0, 1, 3, 2).reshape(batch * self.summary_tokens, self.d_inner, length)
        # Native B/C must have the same scalar type as u; A and dt bias are FP32.
        dtype = values.dtype
        return {"u": values.contiguous(),
                "delta": delta.reshape(batch * self.summary_tokens, self.d_inner, length).to(dtype).contiguous(),
                "A": -self.A_log.float().exp(),
                "B": writes.reshape(batch * self.summary_tokens, self.d_state, length).to(dtype).contiguous(),
                "C": reads.reshape(batch * self.summary_tokens, self.d_state, length).to(dtype).contiguous(),
                "delta_bias": self.delta_projection.bias.float()}

    def scan_output(self, inputs: dict[str, torch.Tensor], backend: str | None = None) -> torch.Tensor:
        device = inputs["u"].device.type
        selected = ("native" if device == "cuda" else "reference") if backend is None else backend
        if selected == "reference":
            if device != "cpu":
                raise RuntimeError("Python reference scan is disabled on GPU; install native mamba-ssm")
            self.last_scan_backend = "cpu_reference"
            return selective_scan_reference(**inputs, delta_softplus=True)
        if selected != "native" or device != "cuda":
            raise ValueError("native selective scan requires CUDA; CPU accepts only the reference backend")
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError("Query SSM requires the native mamba-ssm CUDA selective-scan extension; "
                               "no Python GPU fallback is permitted") from error
        self.last_scan_backend = "mamba_native_cuda"
        # return_last_state=False is intentional: returned y has a native
        # backward through the whole recurrence, whereas last_state does not.
        return selective_scan_fn(**inputs, D=None, z=None, delta_softplus=True, return_last_state=False)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old, count, queries, query_ends, memory, _ = self._inputs(embeddings, valid_mask)
        inputs = self.build_scan_inputs(memory, queries)
        outputs = self.scan_output(inputs)
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        outputs = outputs.reshape(embeddings.shape[0], self.summary_tokens,
                                  self.d_inner, memory.shape[1]).permute(0, 1, 3, 2)
        last_output = outputs[rows, slots, (old - 1).clamp(min=0)[:, None]]
        readout = self.output_projection(last_output)
        return self._finish(readout, queries, embeddings, lengths, old, count, query_ends)

    def scan_compute_flops(self, old_length: int, batch_size: int = 1) -> dict[str, int | str]:
        """Supplement Linear/MHA hooks with explicit scan arithmetic estimates.

        Per state/event: Delta*A, exp, two write multiplies, decay multiply,
        state add, readout multiply and reduction.  Transcendentals are counted
        separately, since equating exp/softplus with a hardware FLOP is misleading.
        Projection GEMMs and gate SiLU are excluded here (reported by hooks).
        """
        _positive_integer(old_length, "old_length")
        _positive_integer(batch_size, "batch_size")
        streams = batch_size * self.summary_tokens * old_length * self.d_inner
        return {"scan_arithmetic_flops": streams * (7 * self.d_state - 1),
                "scan_exp_evaluations": streams * self.d_state,
                "scan_softplus_evaluations": streams,
                # F.linear deliberately omits dt bias until native softplus;
                # nn.Linear hooks therefore do not observe this projection.
                "unhooked_delta_projection_flops": (2 * batch_size * self.summary_tokens
                                                      * old_length * self.gate_rank * self.d_inner),
                "scan_cost_policy": "explicit_recurrence_and_readout_excluding_projection_gemms"}


def make_query_compressor(*, kind: str, d_model: int, summary_tokens: int,
                          tail_size: int = 159, query_context_size: int = 128,
                          n_heads: int = 4, d_state: int = 16,
                          d_inner: int = 128, gate_rank: int = 32,
                          dropout: float = 0.0, dt_min: float = 1e-4,
                          dt_max: float = 0.1, query_source: str = "recent_start") -> nn.Module:
    """Construct independently without advancing shared-backbone CPU RNG."""
    shared: dict[str, Any] = dict(d_model=d_model, summary_tokens=summary_tokens,
                                  tail_size=tail_size, query_context_size=query_context_size,
                                  n_heads=n_heads, dropout=dropout, query_source=query_source)
    with torch.random.fork_rng(devices=[]):
        if kind == "query_cross_attention":
            return QueryCrossAttentionCompressor(**shared)
        if kind in {"query_ssm", "query_ssm_readout"}:
            return QuerySSMCompressor(**shared, d_state=d_state, d_inner=d_inner,
                                      gate_rank=gate_rank, dt_min=dt_min, dt_max=dt_max,
                                      write_conditioning="recent" if kind == "query_ssm" else "slot_only")
    raise ValueError(f"unknown query compressor: {kind}")
