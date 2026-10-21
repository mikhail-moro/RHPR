"""One-ragged-axis PyTorch compatibility for the upstream FBGEMM packing API.

These operations only rearrange values. They contain no attention, normalization,
activation or HSTU math. No global torch.ops registration or binary dependency is
needed. The adapter supports the forms exercised by the pinned native HSTU block.
"""
from __future__ import annotations

import torch


def dense_to_jagged(values: torch.Tensor, offsets: list[torch.Tensor]):
    if len(offsets) != 1 or values.ndim < 2:
        raise ValueError("only one ragged axis with dense [B,N,...] values is supported")
    row_offsets = offsets[0]
    if row_offsets.ndim != 1 or row_offsets.numel() != values.shape[0] + 1:
        raise ValueError("offsets must contain one boundary per batch row plus the end")
    lengths = row_offsets[1:] - row_offsets[:-1]
    valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
    return values[valid], row_offsets


def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: list[torch.Tensor],
    max_lengths: list[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    if len(offsets) != 1 or len(max_lengths) != 1:
        raise ValueError("only one ragged axis is supported")
    row_offsets = offsets[0]
    lengths = row_offsets[1:] - row_offsets[:-1]
    width = max_lengths[0]
    columns = torch.arange(width, device=values.device)
    output_shape = (lengths.numel(), width, *values.shape[1:])
    if values.shape[0] == 0:
        return values.new_full(output_shape, padding_value) + values.sum() * 0
    sources = row_offsets[:-1, None] + columns[None, :]
    valid = columns[None, :] < lengths[:, None]
    output = values[sources.clamp(max=values.shape[0] - 1)]
    mask_shape = (*valid.shape, *([1] * (values.ndim - 1)))
    return output.masked_fill(~valid.reshape(mask_shape), padding_value)
