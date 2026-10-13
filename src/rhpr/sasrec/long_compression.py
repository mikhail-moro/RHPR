"""Fixed-budget compression of a long prefix, followed by an untouched raw tail.

These modules follow the existing next-item adapter contract.  Returned source
positions identify the latest raw input used by a summary, not a supervised
summary target.  Global learned-query slots therefore all carry the old-prefix
endpoint: every slot may use every old event, but no slot reads the recent tail.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _FixedBudgetPrefixCompressor(nn.Module):
    def __init__(self, summary_tokens: int, tail_size: int = 128) -> None:
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (summary_tokens, tail_size)):
            raise ValueError("summary_tokens and tail_size must be positive integers")
        self.summary_tokens = summary_tokens
        self.tail_size = tail_size

    def _lengths(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]
                or valid_mask.dtype != torch.bool or embeddings.shape[1] < 1):
            raise ValueError("embeddings and boolean valid_mask must have matching nonempty batch/length shapes")
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("prefix compression requires nonempty histories")
        offsets = torch.arange(valid_mask.shape[1], device=valid_mask.device)
        if not torch.equal(valid_mask, offsets[None, :] < lengths[:, None]):
            raise ValueError("prefix compression requires contiguous histories with right padding")
        old_lengths = (lengths - self.tail_size).clamp(min=0)
        summary_lengths = old_lengths.clamp(max=self.summary_tokens)
        return lengths, old_lengths, summary_lengths

    def _regions(self, embeddings: torch.Tensor, old_lengths: torch.Tensor,
                 summary_lengths: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Near-equal contiguous regions, independent of other batch members."""
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        divisor = summary_lengths.clamp(min=1)[:, None]
        starts = torch.div(slots * old_lengths[:, None], divisor, rounding_mode="floor")
        ends = torch.div((slots + 1) * old_lengths[:, None], divisor, rounding_mode="floor")
        active = slots < summary_lengths[:, None]
        widths = (ends - starts).masked_fill(~active, 0)
        width = max(1, int(widths.max().item()))
        offsets = torch.arange(width, device=embeddings.device)[None, None, :]
        positions = starts[:, :, None] + offsets
        region_mask = active[:, :, None] & (offsets < widths[:, :, None])
        batch = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None, None]
        # The mask excludes raw tail and padding; invalid indices use a harmless
        # in-bounds gather and are zeroed before any projection or reduction.
        regions = embeddings[batch, positions.clamp(min=0, max=embeddings.shape[1] - 1)]
        regions = regions.masked_fill(~region_mask[:, :, :, None], 0)
        endpoints = (ends - 1).masked_fill(~active, -1)
        return regions, region_mask, endpoints

    def _pack(self, summaries: torch.Tensor, summary_sources: torch.Tensor,
              embeddings: torch.Tensor, lengths: torch.Tensor,
              old_lengths: torch.Tensor, summary_lengths: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tail_lengths = lengths - old_lengths
        output_lengths = summary_lengths + tail_lengths
        offsets = torch.arange(int(output_lengths.max().item()), device=embeddings.device)[None, :]
        output_mask = offsets < output_lengths[:, None]
        from_summary = offsets < summary_lengths[:, None]
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        summary_columns = offsets.expand(embeddings.shape[0], -1).clamp(max=self.summary_tokens - 1)
        tail_sources = old_lengths[:, None] + offsets - summary_lengths[:, None]
        tail_values = embeddings[rows, tail_sources.clamp(min=0, max=embeddings.shape[1] - 1)]
        values = torch.where(from_summary[:, :, None], summaries[rows, summary_columns], tail_values)
        sources = torch.where(from_summary, summary_sources[rows, summary_columns], tail_sources)
        return (values.masked_fill(~output_mask[:, :, None], 0), output_mask,
                sources.masked_fill(~output_mask, -1))


class FixedBudgetMeanCompressor(_FixedBudgetPrefixCompressor):
    """One arithmetic mean per near-equal chronological old-prefix region."""
    implementation = "contiguous_region_mean"

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        regions, region_mask, endpoints = self._regions(embeddings, old_lengths, summary_lengths)
        counts = region_mask.sum(dim=2).clamp(min=1)
        summaries = regions.sum(dim=2) / counts[:, :, None]
        return self._pack(summaries, endpoints, embeddings, lengths, old_lengths, summary_lengths)


class UniformSampleCompressor(_FixedBudgetPrefixCompressor):
    """Keep the most recent real event of every equal chronological region."""
    implementation = "uniform_region_endpoint_sample"

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        ends = torch.div((slots + 1) * old_lengths[:, None], summary_lengths.clamp(min=1)[:, None],
                         rounding_mode="floor")
        endpoints = (ends - 1).masked_fill(slots >= summary_lengths[:, None], -1)
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        summaries = embeddings[rows, endpoints.clamp(min=0, max=embeddings.shape[1] - 1)]
        summaries = summaries.masked_fill((endpoints < 0)[:, :, None], 0)
        return self._pack(summaries, endpoints, embeddings, lengths, old_lengths, summary_lengths)


class HierarchicalPairMLPCompressor(_FixedBudgetPrefixCompressor):
    """Shared ordered-group MLP at every level, with a count-weighted mean skip.

    With group_arity=2 this is a binary PairMLP.  group_arity=4 is its explicit
    four-way extension, reducing both hierarchy depth and total projection work.
    There is no initial heuristic pooling: all old events enter the first learned
    merge.  Regions fix the exact budget, including a non-power-of-two prefix.
    Zero residual initialization gives the same regional mean as the heuristic;
    counts prevent odd or padded groups from changing event weights.
    """
    implementation = "shared_residual_ordered_group_mlp"

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 128,
                 group_arity: int = 4) -> None:
        super().__init__(summary_tokens, tail_size)
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 1:
            raise ValueError("d_model must be a positive integer")
        if isinstance(group_arity, bool) or group_arity not in (2, 4):
            raise ValueError("group_arity must be 2 or 4")
        self.d_model = d_model
        self.group_arity = group_arity
        self.projection = nn.Linear(group_arity * d_model, d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.shape[-1] != self.d_model:
            raise ValueError("embedding width must match d_model")
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        states, region_mask, endpoints = self._regions(embeddings, old_lengths, summary_lengths)
        counts = region_mask.to(dtype=torch.long)
        while states.shape[2] > 1:
            padding = (-states.shape[2]) % self.group_arity
            if padding:
                states = F.pad(states, (0, 0, 0, padding))
                counts = F.pad(counts, (0, padding))
            batch, slots, width, dimensions = states.shape
            states = states.reshape(batch, slots, width // self.group_arity,
                                    self.group_arity, dimensions)
            child_counts = counts.reshape(batch, slots, width // self.group_arity, self.group_arity)
            counts = child_counts.sum(dim=-1)
            weighted = (states * child_counts[:, :, :, :, None]).sum(dim=-2)
            mean = weighted / counts.clamp(min=1)[:, :, :, None]
            delta = F.gelu(self.projection(states.flatten(start_dim=-2)))
            merged = mean + delta
            # Singletons carry through unchanged, including at partial last bins.
            states = torch.where((child_counts.gt(0).sum(dim=-1) > 1)[:, :, :, None], merged, mean)
            states = states.masked_fill((counts == 0)[:, :, :, None], 0)
        return self._pack(states[:, :, 0], endpoints, embeddings, lengths, old_lengths, summary_lengths)


class LearnedQueryCompressor(_FixedBudgetPrefixCompressor):
    """One randomly initialized query block; no pretrained vision Q-Former.

    Learned queries self-attend, cross-attend to all old embeddings plus fixed
    sinusoidal raw-recency features, then use a four-times-width GELU FFN.
    Prefix K/V never contain raw recent-tail events.  Every query has global old
    support, so all summary source positions equal the old-prefix endpoint.
    Query identity and downstream HSTU slot-relative bias distinguish summaries.
    """
    implementation = "one_block_learned_query_self_cross_attention"
    source_position_policy = "latest_input_of_global_old_prefix"

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 128,
                 n_heads: int = 4, ffn_multiplier: int = 4, dropout: float = 0.0,
                 positional_scale: float = 0.02) -> None:
        super().__init__(summary_tokens, tail_size)
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in (d_model, n_heads, ffn_multiplier)) or d_model % n_heads):
            raise ValueError("positive d_model must be divisible by positive n_heads")
        if not 0 <= dropout < 1 or not math.isfinite(positional_scale) or positional_scale <= 0:
            raise ValueError("dropout must be in [0,1) and positional_scale must be finite and positive")
        self.d_model, self.n_heads = d_model, n_heads
        self.ffn_multiplier, self.positional_scale = ffn_multiplier, float(positional_scale)
        self.queries = nn.Parameter(torch.empty(summary_tokens, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_multiplier * d_model), nn.GELU(),
                                 nn.Linear(ffn_multiplier * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def _position_features(self, lengths: torch.Tensor, width: int,
                           dtype: torch.dtype) -> torch.Tensor:
        offsets = torch.arange(width, device=lengths.device, dtype=torch.float32)
        recency = lengths[:, None].to(dtype=torch.float32) - 1 - offsets[None, :]
        frequencies = torch.exp(torch.arange(0, self.d_model, 2, device=lengths.device,
                                            dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        angles = recency[:, :, None] * frequencies[None, None, :]
        features = torch.empty((*angles.shape[:2], self.d_model), device=lengths.device, dtype=torch.float32)
        features[:, :, 0::2] = angles.sin()
        features[:, :, 1::2] = angles[:, :, :self.d_model // 2].cos()
        return (features * self.positional_scale).to(dtype=dtype)

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.shape[-1] != self.d_model:
            raise ValueError("embedding width must match d_model")
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        width = max(1, int(old_lengths.max().item()))
        memory_mask = torch.arange(width, device=embeddings.device)[None, :] < old_lengths[:, None]
        memory = embeddings[:, :width].masked_fill(~memory_mask[:, :, None], 0)
        memory = self.memory_norm(memory + self._position_features(lengths, width, embeddings.dtype))
        # A dummy zero K/V avoids all-masked softmax on rows with no old events.
        memory = memory.masked_fill(~memory_mask[:, :, None], 0)
        safe_memory_mask = memory_mask.clone()
        safe_memory_mask[:, 0] |= old_lengths == 0
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        query_mask = slots < summary_lengths[:, None]
        safe_query_mask = query_mask.clone()
        safe_query_mask[:, 0] |= summary_lengths == 0
        queries = self.queries[None, :, :].expand(embeddings.shape[0], -1, -1)
        normalized = self.self_norm(queries)
        queries = queries + self.dropout(self.self_attention(
            normalized, normalized, normalized, key_padding_mask=~safe_query_mask, need_weights=False)[0])
        queries = queries + self.dropout(self.cross_attention(
            self.cross_norm(queries), memory, memory,
            key_padding_mask=~safe_memory_mask, need_weights=False)[0])
        queries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        queries = queries.masked_fill(~query_mask[:, :, None], 0)
        sources = (old_lengths[:, None] - 1).expand(-1, self.summary_tokens).masked_fill(~query_mask, -1)
        return self._pack(queries, sources, embeddings, lengths, old_lengths, summary_lengths)


def make_prefix_compressor(*, kind: str, d_model: int, summary_tokens: int,
                           tail_size: int = 128, group_arity: int = 4,
                           n_heads: int = 4, dropout: float = 0.0) -> nn.Module:
    """Build a compressor without affecting the caller's shared-backbone RNG."""
    with torch.random.fork_rng(devices=[]):
        if kind == "mean":
            return FixedBudgetMeanCompressor(summary_tokens, tail_size)
        if kind == "uniform":
            return UniformSampleCompressor(summary_tokens, tail_size)
        if kind == "pair_mlp":
            return HierarchicalPairMLPCompressor(d_model, summary_tokens, tail_size, group_arity)
        if kind == "qformer":
            return LearnedQueryCompressor(d_model, summary_tokens, tail_size, n_heads=n_heads, dropout=dropout)
    raise ValueError(f"unknown long-prefix compressor: {kind}")
