from __future__ import annotations

import torch
from torch.nn import functional as F

from rhpr.sasrec.model import SASRecModel


def _aligned_targets(targets: torch.Tensor, source_positions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    safe_positions = source_positions.clamp(min=0)
    aligned = torch.gather(targets, dim=1, index=safe_positions)
    return aligned.masked_fill(~valid_mask, 0)


def _sample_not_equal(shape: torch.Size, positives: torch.Tensor, num_items: int, device: torch.device) -> torch.Tensor:
    if num_items < 2:
        raise ValueError("negative sampling requires at least two catalogue items")
    negatives = torch.randint(1, num_items + 1, shape, device=device)
    collision = negatives == positives
    while torch.any(collision):
        negatives[collision] = torch.randint(1, num_items + 1, (int(collision.sum()),), device=device)
        collision = negatives == positives
    return negatives


def bce_next_item_loss(
    model: SASRecModel,
    encoded: torch.Tensor,
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Original SASRec BCE with one catalogue-uniform negative per target."""

    aligned = _aligned_targets(targets, source_positions, valid_mask)
    negatives = _sample_not_equal(aligned.shape, aligned, model.num_items, aligned.device)
    positive_scores = (encoded * model.item_embedding(aligned)).sum(dim=-1)
    negative_scores = (encoded * model.item_embedding(negatives)).sum(dim=-1)
    scores = torch.cat((positive_scores[valid_mask], negative_scores[valid_mask]))
    labels = torch.cat((torch.ones_like(positive_scores[valid_mask]), torch.zeros_like(negative_scores[valid_mask])))
    return F.binary_cross_entropy_with_logits(scores, labels)


def sampled_softmax_next_item_loss(
    model: SASRecModel,
    encoded: torch.Tensor,
    valid_mask: torch.Tensor,
    source_positions: torch.Tensor,
    targets: torch.Tensor,
    num_negatives: int,
) -> torch.Tensor:
    """Sampled CE with one catalogue-uniform negative set shared within each sequence."""

    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    aligned = _aligned_targets(targets, source_positions, valid_mask)
    batch_size = encoded.shape[0]
    negative_ids = torch.randint(
        1, model.num_items + 1, (batch_size, num_negatives), device=encoded.device
    )
    positive_scores = (encoded * model.item_embedding(aligned)).sum(dim=-1, keepdim=True)
    negative_embeddings = model.item_embedding(negative_ids)
    negative_scores = torch.einsum("bld,bkd->blk", encoded, negative_embeddings)
    collisions = negative_ids[:, None, :] == aligned[:, :, None]
    negative_scores = negative_scores.masked_fill(collisions, torch.finfo(negative_scores.dtype).min)
    logits = torch.cat((positive_scores, negative_scores), dim=-1)
    flat_logits = logits[valid_mask]
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
) -> torch.Tensor:
    if name == "bce":
        return bce_next_item_loss(model, encoded, valid_mask, source_positions, targets)
    if name == "sampled_softmax":
        return sampled_softmax_next_item_loss(
            model, encoded, valid_mask, source_positions, targets, num_negatives
        )
    raise ValueError(f"unsupported loss: {name}")
