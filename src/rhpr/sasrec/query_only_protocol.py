"""Short-context control for the old-suffix-conditioned query compressors.

The learned query generator and summary FFN match the full-prefix arms, but
there is no history readout.  Only the final 128 old events and 128 raw events
are transferred to the model.  The original final-window descriptors, targets,
negative samples and SASRec initialization remain those of the parent pilot.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from rhpr.sasrec.long_history_data import LongHistoryCohort
from rhpr.sasrec.long_quality_schedule import QualityWindow, quality_negative_ids
from rhpr.sasrec.model import SASRecModel
from rhpr.sasrec.query_prefix_protocol import build_prefix_model
from rhpr.sasrec.query_ssm import _RecentQueryPrefixCompressor


QUERY_ONLY_KIND = "yambda_query_only_control"


def query_only_variant() -> dict[str, Any]:
    return {"name": "query_only_m4", "kind": "query_only", "summary_tokens": 4}


def query_only_parent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the pinned parent settings without changing their context cap."""
    if config.get("kind") != QUERY_ONLY_KIND:
        raise ValueError(f"kind must be {QUERY_ONLY_KIND}")
    parent = config.get("parent", {}).get("experiment_config")
    if not isinstance(parent, dict):
        raise ValueError("parent.experiment_config must contain the original pilot configuration")
    return parent


def _check_variant(variant: dict[str, Any]) -> None:
    if variant != query_only_variant():
        raise ValueError("the QueryOnly control requires exactly query_only_m4")


class QueryOnlyPrefixCompressor(_RecentQueryPrefixCompressor):
    """Context-conditioned learned slots and FFN, with no long-history readout.

    Calling the shared base constructor preserves the exact generator/FFN
    initialization order of Cross Attention and QSSM.  The unused readout and
    memory norms have no random initialization and are removed immediately.
    Old-event gathering is direct: full-history memory is never normalized or
    projected, even if a caller supplies more than the usual 256 input events.
    """

    implementation = "old_suffix_query_only_residual_ffn"

    def __init__(self, d_model: int, summary_tokens: int = 4, tail_size: int = 128,
                 query_context_size: int = 128, n_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__(d_model, summary_tokens, tail_size, query_context_size,
                         n_heads, dropout, query_source="old_suffix")
        del self.memory_norm
        del self.readout_norm
        self.source_position_policy = "latest_input_of_old_suffix_query"

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.d_model:
            raise ValueError("embedding width must match d_model")
        lengths, old_lengths, summary_lengths = self._lengths(embeddings, valid_mask)
        context_lengths = old_lengths.clamp(max=self.query_context_size)
        context_starts = old_lengths - context_lengths
        width = min(self.query_context_size, embeddings.shape[1])
        offsets = torch.arange(width, device=embeddings.device)[None, :]
        context_mask = offsets < context_lengths[:, None]
        rows = torch.arange(embeddings.shape[0], device=embeddings.device)[:, None]
        sources = (context_starts[:, None] + offsets).clamp(max=embeddings.shape[1] - 1)
        context = embeddings[rows, sources].masked_fill(~context_mask[:, :, None], 0)
        queries = self.query_generator(context, context_mask)
        summaries = queries + self.dropout(self.ffn(self.ffn_norm(queries)))
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        active = slots < summary_lengths[:, None]
        summaries = summaries.masked_fill(~active[:, :, None], 0)
        summary_sources = (old_lengths - 1)[:, None].expand(-1, self.summary_tokens)
        summary_sources = summary_sources.masked_fill(~active, -1)
        return self._pack(summaries, summary_sources, embeddings,
                          lengths, old_lengths, summary_lengths)


def build_query_only_model(config: dict[str, Any], variant: dict[str, Any],
                           num_items: int) -> SASRecModel:
    """Attach the short control after the unchanged parent SASRec initializer."""
    _check_variant(variant)
    parent = query_only_parent_config(config)
    dims, train, compressor = [parent[key] for key in ("model", "training", "compressor")]
    if (train["raw_tail_size"] != 128 or train["query_context_size"] != 128
            or compressor["query_context_size"] != 128
            or compressor.get("query_source") != "old_suffix"):
        raise ValueError("the QueryOnly control must preserve old-suffix128 and raw128")
    model = build_prefix_model(parent,
        {"name": "recent128", "kind": "recent", "summary_tokens": 0}, num_items)
    with torch.random.fork_rng(devices=[]):
        model.compressor = QueryOnlyPrefixCompressor(d_model=dims["d_model"],
            summary_tokens=variant["summary_tokens"], tail_size=train["raw_tail_size"],
            query_context_size=compressor["query_context_size"], n_heads=dims["n_heads"],
            dropout=compressor["dropout"])
    model.compressor_type = "custom"
    return model


def query_only_batch_arrays(cohort: LongHistoryCohort, windows: Sequence[QualityWindow],
                            config: dict[str, Any], variant: dict[str, Any]
                            ) -> dict[str, np.ndarray]:
    """Crop old-query128 + raw128 before lookup, retaining original identities.

    The local target axis shifts with the crop, while its last128 labels and
    identity-bound catalogue negatives equal the parent full-prefix batches.
    The full cohort and final input boundary remain untouched; over-cap users
    and nonfinal windows fail rather than silently changing the experiment.
    """
    _check_variant(variant)
    if not windows:
        raise ValueError("a nonempty physical batch is required")
    train = query_only_parent_config(config)["training"]
    if train["raw_tail_size"] != 128 or train["query_context_size"] != 128:
        raise ValueError("the QueryOnly control requires query128 and raw128")
    width = train["raw_tail_size"] + train["query_context_size"]
    shape = (len(windows), width)
    item_ids = np.zeros(shape, dtype=np.int64)
    targets = np.zeros(shape, dtype=np.int64)
    valid_mask = np.ones(shape, dtype=np.bool_)
    negatives = []
    for index, window in enumerate(windows):
        if not 0 <= window.user_index < len(cohort.users):
            raise ValueError("window user_index is outside the cohort")
        user = cohort.users[window.user_index]
        if user.user_id != window.user_id or window.end != len(user.train) - 1:
            raise ValueError("QueryOnly window must end at the original final train-only input")
        if window.end > train["raw_context_length"]:
            raise ValueError("complete train prefix exceeds safety cap; truncation is forbidden")
        if window.end < width:
            raise ValueError("train prefix must contain the complete query128 and raw128")
        start = window.end - width
        item_ids[index] = user.train[start:window.end]
        targets[index] = user.train[start + 1:window.end + 1]
        negatives.append(quality_negative_ids(window, num_items=cohort.num_items,
                                              num_negatives=train["num_negatives"]))
    return {"item_ids": item_ids, "targets": targets, "valid_mask": valid_mask,
            "negative_ids": np.asarray(negatives, dtype=np.int64)}
