"""SASRec training/evaluation adapter around Meta's pinned native HSTU blocks.

The transduction block is vendored rather than reimplemented. This wrapper owns
the shared experiment input policy, jagged packing, runtime bias crop and fixed
normalization length. It does not reproduce the full upstream training recipe.
"""
from __future__ import annotations

import torch
from torch import nn

from rhpr.sasrec.model import (
    IdentitySequenceCompressor,
    RecursivePairMLPSequenceCompressor,
    TailOnlySequenceCompressor,
)
from rhpr.vendor.hstu import packing
from rhpr.vendor.hstu.native import RelativePositionalBias, SequentialTransductionUnitJagged


HSTU_UPSTREAM_REVISION = "876dcefb40cea14e87660adfd7b965869e6cd403"
HSTU_IMPLEMENTATION = "meta_native_pytorch_jagged_with_pytorch_packing"


class RuntimeRelativePositionalBias(RelativePositionalBias):
    """Select the exact top-left crop of the upstream fixed-size bias matrix.

    Keep the same parameter table and distance indexing for both context budgets;
    materialize only the current padded attention width, not the full raw window.
    all_timestamps is only a shape carrier here and contains no temporal features.
    """

    def forward(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        width = all_timestamps.shape[-1]
        if width < 1 or width > self._max_seq_len:
            raise ValueError("runtime relative-bias width must fit the fixed parameter table")
        positions = torch.arange(width, device=self._w.device)
        indexes = positions[None, :] - positions[:, None] + self._max_seq_len - 1
        return self._w[indexes].unsqueeze(0)


class HSTUEncoder(nn.Module):
    """Run the official block stack on jagged valid states under caller AMP."""

    def __init__(self, *, d_model: int, n_layers: int, n_heads: int,
                 linear_dim: int, attention_dim: int, dropout: float,
                 normalization_length: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SequentialTransductionUnitJagged(
                embedding_dim=d_model,
                linear_hidden_dim=linear_dim,
                attention_dim=attention_dim,
                dropout_ratio=dropout,
                attn_dropout_ratio=0.0,
                num_heads=n_heads,
                linear_activation="silu",
                relative_attention_bias_module=RuntimeRelativePositionalBias(normalization_length),
                normalization="rel_bias",
                linear_config="uvqk",
                concat_ua=False,
                normalization_length=normalization_length,
            )
            for _ in range(n_layers)
        ])

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        width = embeddings.shape[1]
        lengths = valid_mask.sum(dim=1)
        offsets = torch.nn.functional.pad(lengths.cumsum(dim=0), (1, 0))
        jagged = packing.dense_to_jagged(embeddings, [offsets])[0]
        causal_visibility = torch.ones((width, width), device=embeddings.device,
                                       dtype=embeddings.dtype).tril()
        # Upstream applies relative positional bias only for a non-None payload.
        # This zero shape carrier never reads timestamps or next-event metadata.
        position_payload = torch.zeros(valid_mask.shape, device=embeddings.device, dtype=torch.long)
        for layer in self.layers:
            jagged, _ = layer(
                x=jagged, x_offsets=offsets, all_timestamps=position_payload,
                invalid_attn_mask=causal_visibility,
            )
        return packing.jagged_to_padded_dense(jagged, [offsets], [width])


class HSTUModel(nn.Module):
    """Expose our next-item model API using official HSTU transduction blocks."""

    def __init__(self, *, num_items: int, max_seq_len: int, d_model: int,
                 n_heads: int, n_layers: int, ffn_multiplier: int, dropout: float,
                 compressor_type: str = "identity", compression_depth: int = 1,
                 tail_size: int = 128, use_time_features: bool = False,
                 num_gap_buckets: int = 33, layer_norm_eps: float = 1e-9,
                 initializer_range: float = 0.02, hstu_linear_dim: int | None = None,
                 hstu_attention_dim: int | None = None) -> None:
        super().__init__()
        if min(num_items, max_seq_len, d_model, n_heads, n_layers, ffn_multiplier, tail_size) < 1:
            raise ValueError("model dimensions and tail_size must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if compressor_type not in {"identity", "tail_only"}:
            raise ValueError("HSTU supports only identity and tail_only inputs in this experiment")
        if use_time_features:
            raise ValueError("HSTU comparison requires Time OFF")
        if num_gap_buckets < 2 or not 0 <= dropout < 1:
            raise ValueError("num_gap_buckets must be at least two and dropout must be in [0,1)")
        linear_dim = d_model // n_heads if hstu_linear_dim is None else hstu_linear_dim
        attention_dim = d_model // n_heads if hstu_attention_dim is None else hstu_attention_dim
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (linear_dim, attention_dim)):
            raise ValueError("HSTU per-head linear and attention dimensions must be positive integers")
        self.architecture = "hstu"
        self.attention_implementation = HSTU_IMPLEMENTATION
        self.upstream_revision = HSTU_UPSTREAM_REVISION
        self.num_items, self.max_seq_len, self.d_model = int(num_items), int(max_seq_len), int(d_model)
        self.n_heads, self.n_layers, self.ffn_multiplier = int(n_heads), int(n_layers), int(ffn_multiplier)
        self.compressor_type = compressor_type
        self.compression_depth = RecursivePairMLPSequenceCompressor.validate_depth(compression_depth)
        self.tail_size = int(tail_size)
        self.use_time_features = False
        self.num_gap_buckets = int(num_gap_buckets)
        self.gap_embedding = None
        self.hstu_linear_dim = self.hstu_dv = linear_dim
        self.hstu_attention_dim = self.hstu_dqk = attention_dim
        self.hstu_normalization_length = self.max_seq_len
        self.hstu_context_len = min(self.max_seq_len, tail_size) if compressor_type == "tail_only" else self.max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout)
        self.compressor = (TailOnlySequenceCompressor(tail_size)
                           if compressor_type == "tail_only" else IdentitySequenceCompressor())
        # Both input budgets instantiate the same tables and official block shapes,
        # so a matching seed produces identical parameter state and sampling RNG.
        with torch.no_grad():
            for embedding in (self.item_embedding, self.position_embedding):
                nn.init.trunc_normal_(embedding.weight, std=initializer_range,
                                      a=-2 * initializer_range, b=2 * initializer_range)
            self.item_embedding.weight[0].zero_()
        self.encoder = HSTUEncoder(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            linear_dim=linear_dim, attention_dim=attention_dim, dropout=dropout,
            normalization_length=max_seq_len,
        )

    def encode(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
               gap_ids: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gap_ids is not None:
            raise ValueError("HSTU Time OFF does not accept gap_ids")
        if item_ids.ndim != 2 or item_ids.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
            raise ValueError("item_ids and boolean valid_mask must have matching [batch,length] shapes")
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("empty sequences are not supported")
        if item_ids.shape[1] > self.max_seq_len or torch.any(lengths > self.max_seq_len):
            raise ValueError("raw sequence exceeds max_seq_len")
        if not torch.equal(valid_mask, torch.arange(item_ids.shape[1], device=item_ids.device)[None, :] < lengths[:, None]):
            raise ValueError("HSTU requires contiguous histories with right padding")
        embeddings, output_mask, sources = self.compressor(self.item_embedding(item_ids), valid_mask)
        positions = (lengths[:, None] - 1 - sources).clamp(min=0, max=self.max_seq_len - 1)
        positions = positions.masked_fill(~output_mask, 0)
        embeddings = self.input_dropout(self.input_norm(embeddings + self.position_embedding(positions)))
        embeddings = embeddings.masked_fill(~output_mask.unsqueeze(-1), 0)
        encoded = self.encoder(embeddings, output_mask)
        return encoded.masked_fill(~output_mask.unsqueeze(-1), 0), output_mask, sources

    def last_state(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
                   gap_ids: torch.Tensor | None = None) -> torch.Tensor:
        encoded, output_mask, _ = self.encode(item_ids, valid_mask, gap_ids)
        indexes = output_mask.sum(dim=1) - 1
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), indexes]
