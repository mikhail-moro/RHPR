from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class SequenceCompressor(Protocol):
    """Contract for causal modules that may shorten the temporal axis."""

    def __call__(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class IdentitySequenceCompressor(nn.Module):
    """No-op compressor used by the uncompressed SASRec baseline."""

    def forward(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = embeddings.shape
        source_positions = torch.arange(sequence_length, device=embeddings.device).expand(batch_size, -1)
        source_positions = source_positions.masked_fill(~valid_mask, -1)
        return embeddings, valid_mask, source_positions


class SASRecModel(nn.Module):
    """Small PyTorch SASRec based on the official Yambda benchmark implementation."""

    def __init__(
        self,
        *,
        num_items: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_multiplier: int,
        dropout: float,
        compressor: nn.Module | None = None,
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if max_seq_len < 1 or n_layers < 1 or ffn_multiplier < 1:
            raise ValueError("max_seq_len, n_layers, and ffn_multiplier must be positive")

        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.compressor = compressor if compressor is not None else IdentitySequenceCompressor()
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_multiplier * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self._initialize(initializer_range)

    @torch.no_grad()
    def _initialize(self, initializer_range: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.item_embedding.weight[0].zero_()

    def encode(
        self, item_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.item_embedding(item_ids)
        embeddings, compressed_mask, source_positions = self.compressor(embeddings, valid_mask)
        if embeddings.shape[:2] != compressed_mask.shape or compressed_mask.shape != source_positions.shape:
            raise ValueError("compressor returned incompatible tensor shapes")
        sequence_length = embeddings.shape[1]
        if sequence_length > self.max_seq_len:
            raise ValueError("compressed sequence exceeds max_seq_len")

        lengths = compressed_mask.sum(dim=1)
        offsets = torch.arange(sequence_length, device=embeddings.device).expand(embeddings.shape[0], -1)
        positions = (lengths[:, None] - 1 - offsets).clamp(min=0, max=self.max_seq_len - 1)
        embeddings = embeddings + self.position_embedding(positions)
        embeddings = self.input_dropout(self.input_norm(embeddings))
        embeddings = embeddings.masked_fill(~compressed_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=embeddings.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.encoder(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=~compressed_mask,
        )
        encoded = encoded.masked_fill(~compressed_mask.unsqueeze(-1), 0.0)
        return encoded, compressed_mask, source_positions

    def last_state(self, item_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        encoded, compressed_mask, _ = self.encode(item_ids, valid_mask)
        last_indexes = compressed_mask.sum(dim=1) - 1
        if torch.any(last_indexes < 0):
            raise ValueError("empty sequences are not supported")
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), last_indexes]
