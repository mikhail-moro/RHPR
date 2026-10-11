from __future__ import annotations

import torch
from torch.nn import functional as F

from rhpr.sasrec.model import SASRecModel


def _aligned_targets(targets: torch.Tensor, source_positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    safe_positions = source_positions.clamp(min=0)
    aligned = torch.gather(targets, dim=1, index=safe_positions)
    return aligned.masked_fill(~valid_mask, 0)


def build_loss_mask(
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    original_lengths: torch.Tensor | None = None,
    loss_tail_size: int | None = None,
) -> torch.Tensor:
    """Choose supervised source events without changing attention visibility."""
    if valid_mask.shape != source_positions.shape:
        raise ValueError("valid_mask and source_positions must have matching shapes")
    if loss_tail_size is None:
        return valid_mask
    if loss_tail_size < 1:
        raise ValueError("loss_tail_size must be positive")
    if original_lengths is None or original_lengths.shape != (valid_mask.shape[0],):
        raise ValueError("tail loss requires original_lengths with shape [batch]")
    starts = (original_lengths - loss_tail_size).clamp(min=0)
    return valid_mask & (source_positions >= starts[:, None])


def supervised_state_tensors(
    encoded: torch.Tensor,
    loss_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physically pack selected states before computing sampled-softmax logits."""
    counts = loss_mask.sum(dim=1)
    width = int(counts.max().item())
    if width < 1:
        raise ValueError("loss must select at least one valid target")
    rows, columns = torch.where(loss_mask)
    packed_columns = (loss_mask.cumsum(dim=1) - 1)[rows, columns]
    states = encoded.new_zeros((encoded.shape[0], width, encoded.shape[2]))
    states[rows, packed_columns] = encoded[rows, columns]
    aligned = targets.new_zeros((encoded.shape[0], width))
    aligned[rows, packed_columns] = targets[rows, source_positions[rows, columns]]
    packed_mask = torch.arange(width, device=encoded.device)[None, :] < counts[:, None]
    return states, aligned, packed_mask


def _sample_not_equal(
    shape: torch.Size,
    positives: torch.Tensor,
    num_items: int,
    device: torch.device,
    negative_generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_items < 2:
        raise ValueError("negative sampling requires at least two catalogue items")
    negatives = torch.randint(1, num_items + 1, shape, device=device, generator=negative_generator)
    collision = negatives == positives
    while torch.any(collision):
        negatives[collision] = torch.randint(
            1, num_items + 1, (int(collision.sum()),), device=device, generator=negative_generator
        )
        collision = negatives == positives
    return negatives


def bce_next_item_loss(
    model: SASRecModel,
    encoded: torch.Tensor,
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
    negative_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Original SASRec BCE with one catalogue-uniform negative per target."""

    states, aligned, loss_mask = supervised_state_tensors(encoded, valid_mask, source_positions, targets)
    negatives = _sample_not_equal(aligned.shape, aligned, model.num_items, aligned.device, negative_generator)
    positive_scores = (states * model.item_embedding(aligned)).sum(dim=-1)
    negative_scores = (states * model.item_embedding(negatives)).sum(dim=-1)
    scores = torch.cat((positive_scores[loss_mask], negative_scores[loss_mask]))
    labels = torch.cat((torch.ones_like(positive_scores[loss_mask]), torch.zeros_like(negative_scores[loss_mask])))
    return F.binary_cross_entropy_with_logits(scores, labels)


def sampled_softmax_next_item_loss(
    model: SASRecModel,
    encoded: torch.Tensor,
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int,
    negative_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sampled CE with one catalogue-uniform negative set shared within each sequence."""

    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    states, aligned, loss_mask = supervised_state_tensors(encoded, valid_mask, source_positions, targets)
    batch_size = encoded.shape[0]
    negative_ids = torch.randint(
        1, model.num_items + 1, (batch_size, num_negatives), device=encoded.device, generator=negative_generator
    )
    positive_scores = (states * model.item_embedding(aligned)).sum(dim=-1, keepdim=True)
    negative_embeddings = model.item_embedding(negative_ids)
    negative_scores = torch.einsum("bld,bkd->blk", states, negative_embeddings)
    collisions = negative_ids[:, None, :] == aligned[:, :, None]
    negative_scores = negative_scores.masked_fill(collisions, torch.finfo(negative_scores.dtype).min)
    logits = torch.cat((positive_scores, negative_scores), dim=-1)
    flat_logits = logits[loss_mask]
    labels = torch.zeros(flat_logits.shape[0], device=encoded.device, dtype=torch.long)
    return F.cross_entropy(flat_logits, labels)


def next_item_loss(
    *,
    name: str,
    model: SASRecModel,
    encoded: torch.Tensor,
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int,
    loss_tail_size: int | None = None,
    original_lengths: torch.Tensor | None = None,
    negative_generator: torch.Generator | None = None,
) -> torch.Tensor:
    loss_mask = build_loss_mask(valid_mask, source_positions, original_lengths, loss_tail_size)
    if name == "bce":
        return bce_next_item_loss(model, encoded, loss_mask, source_positions, targets, negative_generator)
    if name == "sampled_softmax":
        return sampled_softmax_next_item_loss(
            model, encoded, loss_mask, source_positions, targets, num_negatives, negative_generator
        )
    raise ValueError(f"unsupported loss: {name}")
