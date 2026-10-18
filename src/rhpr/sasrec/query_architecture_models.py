"""Matched compressor architectures at the fixed parent learning rate.

The original initializer remains the control: SASRec, query generation and the
single shared summary FFN keep their exact initial weights. Additional CA reads
refine the summaries before that FFN; they never read the supervised raw tail.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rhpr.sasrec.query_proxy_protocol import build_hpo_model
from rhpr.sasrec.query_ssm import QueryCrossAttentionCompressor, make_query_compressor


class IterativeQueryCrossAttentionCompressor(QueryCrossAttentionCompressor):
    """One or two global reads, keeping the original query generator and FFN.

    With one read this has exactly the original parameters and computation.
    Global attention heads can change independently of the query generator's
    heads; the existing projection weights and their initialization do not.
    """

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 159,
                 query_context_size: int = 128, n_heads: int = 4,
                 dropout: float = 0.0, query_source: str = "recent_start",
                 readout_depth: int = 1, cross_attention_heads: int = 4) -> None:
        if type(readout_depth) is not int or readout_depth not in (1, 2):
            raise ValueError("readout_depth must be one or two")
        if (type(cross_attention_heads) is not int or cross_attention_heads < 1
                or d_model % cross_attention_heads):
            raise ValueError("cross_attention_heads must divide d_model")
        super().__init__(d_model, summary_tokens, tail_size, query_context_size,
                         n_heads, dropout, query_source)
        self.readout_depth = readout_depth
        self.cross_attention_heads = cross_attention_heads
        # MultiheadAttention initializes the same projection matrices regardless
        # of the number of heads. Change only their split, preserving every bit
        # of the control's state and leaving query_generator.attention alone.
        self.cross_attention.num_heads = cross_attention_heads
        self.cross_attention.head_dim = d_model // cross_attention_heads
        # Construct after every original module so shared initialization is exact.
        self.extra_cross_query_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(readout_depth - 1)])
        self.extra_cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, cross_attention_heads, dropout=dropout,
                                  batch_first=True) for _ in range(readout_depth - 1)])
        self.extra_readout_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(readout_depth - 1)])
        if readout_depth > 1:
            self.implementation += "_iterative_read2"

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.readout_depth == 1:
            return super().forward(embeddings, valid_mask)
        lengths, old, count, queries, query_ends, memory, memory_mask = self._inputs(
            embeddings, valid_mask)
        safe_mask = memory_mask.clone()
        safe_mask[:, 0] |= old == 0
        readout = self.cross_attention(self.cross_query_norm(queries), memory, memory,
            key_padding_mask=~safe_mask, need_weights=False)[0]
        summaries = queries + self.dropout(self.readout_norm(readout))
        for query_norm, attention, readout_norm in zip(
                self.extra_cross_query_norms, self.extra_cross_attentions,
                self.extra_readout_norms):
            readout = attention(query_norm(summaries), memory, memory,
                key_padding_mask=~safe_mask, need_weights=False)[0]
            summaries = summaries + self.dropout(readout_norm(readout))
        # Exactly one original FFN, after the final global read. Packing follows
        # the parent: occupied summaries precede the unchanged causal raw tail.
        summaries = summaries + self.dropout(self.ffn(self.ffn_norm(summaries)))
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        active = slots < count[:, None]
        summaries = summaries.masked_fill(~active[:, :, None], 0)
        sources = query_ends[:, None].expand(-1, self.summary_tokens).masked_fill(~active, -1)
        return self._pack(summaries, sources, embeddings, lengths, old, count)


def build_architecture_model(config: dict[str, Any], variant: dict[str, Any],
                             trial: dict[str, Any], num_items: int) -> nn.Module:
    """Build a registered architecture without changing common initialization.

    Both baseline arms return ``build_hpo_model`` directly. Other architectures
    start from that same initializer and replace only their compressor under a
    CPU RNG fork, preserving the following dropout/training RNG as well.
    Full experiment registration and scheduling belong to the protocol module.
    """
    registered = config.get("architecture_trials", {}).get(variant.get("name"), [])
    if trial not in registered:
        raise ValueError("a registered trial for this compressor variant is required")
    if any(type(trial.get(key)) not in (int, float) or trial[key] != .001
           for key in ("backbone_lr", "compressor_lr")):
        raise ValueError("architecture trials fix both learning rates at 0.001")
    model = build_hpo_model(config, variant, num_items)
    original = config["parent"]["experiment_config"]
    dims, train, compressor = (original[key] for key in ("model", "training", "compressor"))
    if variant["kind"] == "query_cross_attention":
        depth, heads = trial["readout_depth"], trial["cross_attention_heads"]
        if type(depth) is not int or type(heads) is not int:
            raise ValueError("CA readout depth and heads must be integers")
        if depth == 1 and heads == dims["n_heads"]:
            return model
        shared = {key: compressor[key] for key in
                  ("query_context_size", "dropout", "query_source")}
        with torch.random.fork_rng(devices=[]):
            model.compressor = IterativeQueryCrossAttentionCompressor(
                d_model=dims["d_model"], summary_tokens=variant["summary_tokens"],
                tail_size=train["raw_tail_size"], n_heads=dims["n_heads"],
                readout_depth=depth, cross_attention_heads=heads, **shared)
        return model
    if variant["kind"] == "query_ssm":
        changed = dict(compressor)
        for key in ("d_inner", "d_state", "gate_rank"):
            value = trial[key]
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            changed[key] = value
        if changed == compressor:
            return model
        model.compressor = make_query_compressor(kind=variant["kind"],
            d_model=dims["d_model"], summary_tokens=variant["summary_tokens"],
            tail_size=train["raw_tail_size"], n_heads=dims["n_heads"], **changed)
        return model
    raise ValueError("architecture trials require Cross Attention or QSSM")
