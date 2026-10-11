from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


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


def paired_lengths(lengths: torch.Tensor, tail_size: int = 128) -> torch.Tensor:
    """Length budget shared by mean, learned pairing, and recent truncation."""
    if tail_size < 1:
        raise ValueError("tail_size must be positive")
    tail = lengths.clamp(max=tail_size)
    old = lengths - tail
    return tail + torch.div(old + 1, 2, rounding_mode="floor")


def _gather_embeddings(embeddings: torch.Tensor, source_positions: torch.Tensor) -> torch.Tensor:
    safe_positions = source_positions.clamp(min=0, max=embeddings.shape[1] - 1)
    return torch.gather(
        embeddings, 1, safe_positions.unsqueeze(-1).expand(-1, -1, embeddings.shape[2])
    )


class RecentSequenceCompressor(nn.Module):
    """Keep the most recent events at the same length budget as pair merging."""

    def __init__(self, tail_size: int = 128) -> None:
        super().__init__()
        if tail_size < 1:
            raise ValueError("tail_size must be positive")
        self.tail_size = int(tail_size)

    def forward(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = valid_mask.sum(dim=1)
        output_lengths = paired_lengths(lengths, self.tail_size)
        width = int(output_lengths.max().item())
        offsets = torch.arange(width, device=embeddings.device).expand(embeddings.shape[0], -1)
        output_mask = offsets < output_lengths[:, None]
        source_positions = offsets + (lengths - output_lengths)[:, None]
        source_positions = source_positions.masked_fill(~output_mask, -1)
        output = _gather_embeddings(embeddings, source_positions)
        output = output.masked_fill(~output_mask.unsqueeze(-1), 0.0)
        return output, output_mask, source_positions


class MeanSequenceCompressor(nn.Module):
    """Merge adjacent old pairs and retain an odd singleton and recent tail."""

    def __init__(self, tail_size: int = 128) -> None:
        super().__init__()
        if tail_size < 1:
            raise ValueError("tail_size must be positive")
        self.tail_size = int(tail_size)

    def merge_pairs(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left + right) / 2

    def pair_count(self, lengths: torch.Tensor) -> int:
        old = (lengths - self.tail_size).clamp(min=0)
        return int(torch.div(old, 2, rounding_mode="floor").sum().item())

    def forward(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = valid_mask.sum(dim=1)
        old = (lengths - self.tail_size).clamp(min=0)
        old_outputs = torch.div(old + 1, 2, rounding_mode="floor")
        output_lengths = paired_lengths(lengths, self.tail_size)
        width = int(output_lengths.max().item())
        offsets = torch.arange(width, device=embeddings.device).expand(embeddings.shape[0], -1)
        output_mask = offsets < output_lengths[:, None]
        old_endpoints = torch.minimum(2 * offsets + 1, old[:, None] - 1)
        tail_positions = old[:, None] + offsets - old_outputs[:, None]
        source_positions = torch.where(offsets < old_outputs[:, None], old_endpoints, tail_positions)
        source_positions = source_positions.masked_fill(~output_mask, -1)
        output = _gather_embeddings(embeddings, source_positions)

        # Only real pairs reach the projection: singleton, tail, and padding do not.
        pair_mask = offsets < torch.div(old, 2, rounding_mode="floor")[:, None]
        rows, columns = torch.where(pair_mask)
        if rows.numel():
            left = embeddings[rows, 2 * columns]
            right = embeddings[rows, 2 * columns + 1]
            output[rows, columns] = self.merge_pairs(left, right)
        output = output.masked_fill(~output_mask.unsqueeze(-1), 0.0)
        return output, output_mask, source_positions


class PairMLPSequenceCompressor(MeanSequenceCompressor):
    """Mean residual plus one shared [2E, E] projection and GELU."""

    def __init__(self, d_model: int, tail_size: int = 128) -> None:
        super().__init__(tail_size)
        self.projection = nn.Linear(2 * d_model, d_model)
        self.reset_residual()

    @torch.no_grad()
    def reset_residual(self) -> None:
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def merge_pairs(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return super().merge_pairs(left, right) + F.gelu(self.projection(torch.cat((left, right), dim=-1)))


class RecursivePairMLPSequenceCompressor(PairMLPSequenceCompressor):
    """Reuse one pair projection while retaining the same raw recent tail.

    Each pass pairs the current old prefix from its oldest end. Unpaired old
    singletons pass through unchanged. Raw endpoints compose across passes,
    so metadata and downstream positions always refer to original events.
    """

    def __init__(self, d_model: int, tail_size: int = 128, depth: int = 1) -> None:
        self.depth = depth
        super().__init__(d_model, tail_size)

    @staticmethod
    def validate_depth(depth: int) -> int:
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 3:
            raise ValueError("compression depth must be an integer between 0 and 3")
        return depth

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, depth: int) -> None:
        self._depth = self.validate_depth(depth)

    def pair_count(self, lengths: torch.Tensor) -> int:
        old = (lengths - self.tail_size).clamp(min=0)
        total = torch.zeros((), dtype=old.dtype, device=old.device)
        for _ in range(self.depth):
            total = total + torch.div(old, 2, rounding_mode="floor").sum()
            old = torch.div(old + 1, 2, rounding_mode="floor")
        return int(total.item())

    def _compress(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth == 0:
            return IdentitySequenceCompressor()(embeddings, valid_mask)
        output, output_mask = embeddings, valid_mask
        for level in range(depth):
            output, output_mask, local_endpoints = super().forward(output, output_mask)
            if level == 0:
                raw_endpoints = local_endpoints
            else:
                raw_endpoints = torch.gather(raw_endpoints, 1, local_endpoints.clamp(min=0))
                raw_endpoints = raw_endpoints.masked_fill(~output_mask, -1)
        return output, output_mask, raw_endpoints

    def forward_with_metadata(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor, depth: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_depth = self.depth if depth is None else self.validate_depth(depth)
        output, output_mask, raw_endpoints = self._compress(embeddings, valid_mask, selected_depth)
        # Outputs form a contiguous partition of the raw history. Endpoint
        # differences therefore count events exactly, including odd carries.
        previous_endpoints = F.pad(raw_endpoints[:, :-1], (1, 0), value=-1)
        source_counts = (raw_endpoints - previous_endpoints).masked_fill(~output_mask, 0)
        return output, output_mask, raw_endpoints, source_counts

    def forward(
        self, embeddings: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compress(embeddings, valid_mask, self.depth)


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
        compressor_type: str = "identity",
        compression_depth: int = 1,
        tail_size: int = 128,
        use_time_features: bool = False,
        num_gap_buckets: int = 33,
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if max_seq_len < 1 or n_layers < 1 or ffn_multiplier < 1:
            raise ValueError("max_seq_len, n_layers, and ffn_multiplier must be positive")
        if compressor_type not in {"identity", "recent", "mean", "pair_mlp", "recursive_pair_mlp"}:
            raise ValueError(f"unknown compressor_type: {compressor_type}")
        if compressor is not None and compressor_type != "identity":
            raise ValueError("choose a custom compressor or compressor_type, not both")
        if tail_size < 1 or num_gap_buckets < 2:
            raise ValueError("tail_size must be positive and num_gap_buckets at least two")

        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ffn_multiplier = int(ffn_multiplier)
        self.compressor_type = "custom" if compressor is not None else compressor_type
        self.compression_depth = RecursivePairMLPSequenceCompressor.validate_depth(compression_depth)
        self.tail_size = int(tail_size)
        self.use_time_features = bool(use_time_features)
        self.num_gap_buckets = int(num_gap_buckets)
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
        # Initialize the original backbone first, preserving baseline parameter RNG.
        # Extra modules must neither shift shared initialization nor future sampling RNG.
        with torch.random.fork_rng(devices=[]):
            if compressor is None:
                if compressor_type == "recent":
                    self.compressor = RecentSequenceCompressor(tail_size)
                elif compressor_type == "mean":
                    self.compressor = MeanSequenceCompressor(tail_size)
                elif compressor_type == "pair_mlp":
                    self.compressor = PairMLPSequenceCompressor(d_model, tail_size)
                elif compressor_type == "recursive_pair_mlp":
                    self.compressor = RecursivePairMLPSequenceCompressor(d_model, tail_size, self.compression_depth)
            self.gap_embedding = nn.Embedding(num_gap_buckets, d_model) if use_time_features else None
            if self.gap_embedding is not None:
                nn.init.zeros_(self.gap_embedding.weight)
        if isinstance(self.compressor, PairMLPSequenceCompressor):
            self.compressor.reset_residual()

    def set_compression_depth(self, depth: int) -> None:
        """Select a deterministic exit without replacing or reinitializing weights."""
        if not isinstance(self.compressor, RecursivePairMLPSequenceCompressor):
            raise ValueError("set_compression_depth requires a recursive pair compressor")
        self.compressor.depth = depth
        self.compression_depth = self.compressor.depth

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
        self, item_ids: torch.Tensor, valid_mask: torch.Tensor, gap_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if item_ids.shape != valid_mask.shape or item_ids.ndim != 2:
            raise ValueError("item_ids and valid_mask must have matching [batch, length] shapes")
        original_lengths = valid_mask.sum(dim=1)
        if torch.any(original_lengths < 1):
            raise ValueError("empty sequences are not supported")
        if torch.any(original_lengths > self.max_seq_len):
            raise ValueError("raw sequence exceeds max_seq_len")
        embeddings = self.item_embedding(item_ids)
        if self.gap_embedding is not None:
            if gap_ids is None or gap_ids.shape != item_ids.shape:
                raise ValueError("time features require gap_ids with the same shape as item_ids")
            embeddings = embeddings + self.gap_embedding(gap_ids)
        embeddings, compressed_mask, source_positions = self.compressor(embeddings, valid_mask)
        if embeddings.shape[:2] != compressed_mask.shape or compressed_mask.shape != source_positions.shape:
            raise ValueError("compressor returned incompatible tensor shapes")
        sequence_length = embeddings.shape[1]
        if sequence_length > self.max_seq_len:
            raise ValueError("compressed sequence exceeds max_seq_len")

        positions = (original_lengths[:, None] - 1 - source_positions).clamp(min=0, max=self.max_seq_len - 1)
        positions = positions.masked_fill(~compressed_mask, 0)
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

    def last_state(
        self, item_ids: torch.Tensor, valid_mask: torch.Tensor, gap_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded, compressed_mask, _ = self.encode(item_ids, valid_mask, gap_ids)
        last_indexes = compressed_mask.sum(dim=1) - 1
        if torch.any(last_indexes < 0):
            raise ValueError("empty sequences are not supported")
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), last_indexes]
