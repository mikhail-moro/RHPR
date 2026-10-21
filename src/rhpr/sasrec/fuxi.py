"""Next-item adapter for the pinned official FuXi-Linear chunkwise blocks.

This profile adapter retains semantic, positional and temporal channel compute.
Time OFF supplies an all-zero timestamp payload to the temporal channel. No next
interaction timestamp is read; this is not the published temporal training recipe.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from rhpr.vendor.hstu import packing
from rhpr.vendor.fuxi.native import FuXiLinearBlockJagged


FUXI_UPSTREAM_REVISION = "5a704061a7ebf0b81afb465f12f93f2747ba4667"
FUXI_IMPLEMENTATION = "official_native_pytorch_chunkwise_with_scoped_packing"


class FuXiLinearEncoder(nn.Module):
    def __init__(self, *, max_seq_len: int, d_model: int, n_layers: int,
                 n_heads: int, ffn_multiplier: int, dropout: float,
                 chunk_size: int, temporal_heads: int, positional_dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.layers = nn.ModuleList([
            FuXiLinearBlockJagged(
                embedding_dim=d_model, linear_hidden_dim=d_model // n_heads,
                attention_dim=d_model // n_heads, num_heads=n_heads,
                dropout_ratio=dropout, attn_dropout_ratio=0.0,
                linear_activation="silu", id_layer=index, use_rope=False,
                ffn_multiply=ffn_multiplier, chunk_size=chunk_size,
                channel_t_config={"num_heads": temporal_heads, "base": 2,
                                  "start_index": 10, "base_stride": 3,
                                  "aug_current": True, "learnable_gamma": True},
                channel_p_config={"max_sequence_length": max_seq_len,
                                  "dim": positional_dim, "aug_current": True},
            ) for index in range(n_layers)
        ])

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        width = embeddings.shape[1]
        padded_width = math.ceil(width / self.chunk_size) * self.chunk_size
        lengths = valid_mask.sum(dim=1)
        offsets = F.pad(lengths.cumsum(dim=0), (1, 0))
        jagged = packing.dense_to_jagged(embeddings, [offsets])[0]
        timestamps = torch.zeros((embeddings.shape[0], padded_width),
                                 device=embeddings.device, dtype=torch.long)
        temporal_buffer = None
        for layer in self.layers:
            # Every official channel ignores the global attention mask in its
            # chunkwise branch. Local chunk_size-square masks are still native.
            jagged, _, temporal_buffer = layer(
                x=jagged, x_offsets=offsets, all_timestamps=timestamps,
                invalid_attn_mask=None, past_lengths=lengths,
                target_timestamps=None, current_length=padded_width,
                channel_t_buffer=temporal_buffer,
            )
        return packing.jagged_to_padded_dense(jagged, [offsets], [width])


class FuXiLinearModel(nn.Module):
    """Expose the shared recommendation API without a sequence-square mask."""

    def __init__(self, *, num_items: int, max_seq_len: int, d_model: int,
                 n_heads: int, n_layers: int, ffn_multiplier: int, dropout: float,
                 chunk_size: int = 128, use_time_features: bool = False,
                 layer_norm_eps: float = 1e-9, initializer_range: float = 0.02,
                 fuxi_temporal_heads: int = 8, fuxi_positional_dim: int = 32) -> None:
        super().__init__()
        dimensions = (num_items, max_seq_len, d_model, n_heads, n_layers,
                      ffn_multiplier, chunk_size, fuxi_temporal_heads, fuxi_positional_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in dimensions):
            raise ValueError("FuXi model dimensions must be positive integers")
        if d_model % n_heads or d_model % (2 * fuxi_temporal_heads):
            raise ValueError("d_model must divide evenly across semantic and temporal heads")
        if fuxi_positional_dim % 2:
            raise ValueError("FuXi positional dimension must be even")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if use_time_features:
            raise ValueError("the initial FuXi profile uses Time OFF")
        self.architecture = "fuxi_linear"
        self.attention_implementation = FUXI_IMPLEMENTATION
        self.upstream_revision = FUXI_UPSTREAM_REVISION
        self.num_items, self.max_seq_len, self.d_model = num_items, max_seq_len, d_model
        self.n_heads, self.n_layers, self.ffn_multiplier = n_heads, n_layers, ffn_multiplier
        self.compressor_type, self.compression_depth, self.tail_size = "identity", 0, 128
        self.use_time_features = False
        self.fuxi_chunk_size = chunk_size
        self.fuxi_temporal_heads = fuxi_temporal_heads
        self.fuxi_positional_dim = fuxi_positional_dim
        self.fuxi_dv = self.fuxi_dqk = d_model // n_heads
        self.fuxi_time_mode = "zero_payload_all_channels_no_future_timestamps"
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout)
        with torch.no_grad():
            nn.init.trunc_normal_(self.item_embedding.weight, std=initializer_range,
                                  a=-2 * initializer_range, b=2 * initializer_range)
            self.item_embedding.weight[0].zero_()
        self.encoder = FuXiLinearEncoder(
            max_seq_len=max_seq_len, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, ffn_multiplier=ffn_multiplier, dropout=dropout,
            chunk_size=chunk_size, temporal_heads=fuxi_temporal_heads,
            positional_dim=fuxi_positional_dim,
        )

    def major_matmul_flops(self, batch_size: int, sequence_length: int) -> int:
        """Count native forward GEMM/einsum multiply-adds for full valid rows.

        A multiply-add counts as two FLOPs. Includes all three channels' local
        attention, chunk KV summaries/cross-chunk reads, projections and gated
        FFN. The positional QK map is shared across batch rows, as upstream.
        Runtime channel work uses a rounded chunk width; projections/FFN use
        valid jagged rows. Excludes input/catalog lookup, loss, normalization,
        nonlinearities, decay/gates, packing and elementwise recurrent scan.
        This is a major-matmul count, not total forward/backward operations.
        """
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (batch_size, sequence_length)):
            raise ValueError("batch_size and sequence_length must be positive integers")
        if sequence_length > self.max_seq_len:
            raise ValueError("sequence_length exceeds max_seq_len")
        b, length, d = batch_size, sequence_length, self.d_model
        c = self.fuxi_chunk_size
        width = math.ceil(length / c) * c
        h, dqk, dv = self.n_heads, self.fuxi_dqk, self.fuxi_dv
        ht, dp = self.fuxi_temporal_heads, self.fuxi_positional_dim
        # UVQK: D -> 6D; temporal/positional value projections: two D -> D;
        # output projection: 3D -> D; gated FFN: three D <-> multiplier*D.
        projections = 2 * b * length * d * d * (11 + 3 * self.ffn_multiplier)
        semantic = 4 * b * width * h * dqk * dv + 2 * b * width * c * h * (dqk + dv)
        # Temporal QK head width is two; temporal value width D/(2*Ht).
        temporal = 8 * b * width * d + 8 * b * width * c * ht + 2 * b * width * c * d
        positional = 4 * b * width * dp * d + 2 * width * c * dp + 2 * b * width * c * d
        return int(self.n_layers * (projections + semantic + temporal + positional))

    def encode(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
               gap_ids: torch.Tensor | None = None,
               timestamps: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gap_ids is not None or timestamps is not None:
            raise ValueError("FuXi Time OFF rejects gap_ids and timestamp features")
        if item_ids.ndim != 2 or item_ids.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
            raise ValueError("item_ids and boolean valid_mask need matching [batch,length] shapes")
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("empty histories are not supported")
        if item_ids.shape[1] > self.max_seq_len:
            raise ValueError("history exceeds max_seq_len")
        columns = torch.arange(item_ids.shape[1], device=item_ids.device)
        if not torch.equal(valid_mask, columns[None, :] < lengths[:, None]):
            raise ValueError("FuXi requires contiguous histories with right padding")
        embeddings = self.input_dropout(self.input_norm(self.item_embedding(item_ids)))
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0)
        encoded = self.encoder(embeddings, valid_mask)
        sources = columns.expand(item_ids.shape[0], -1).masked_fill(~valid_mask, -1)
        return encoded.masked_fill(~valid_mask.unsqueeze(-1), 0), valid_mask, sources

    def last_state(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
                   gap_ids: torch.Tensor | None = None,
                   timestamps: torch.Tensor | None = None) -> torch.Tensor:
        states, output_mask, _ = self.encode(item_ids, valid_mask, gap_ids, timestamps)
        indexes = output_mask.sum(dim=1) - 1
        return states[torch.arange(states.shape[0], device=states.device), indexes]
