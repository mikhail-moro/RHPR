"""Auditable log-space catalogue scoring and beam search.

Raw AR and locally renormalized scores are deliberately different objectives.
Exact search is an oracle for that model's ranking, not for recommendation truth.
Encoder outputs are reused; no KV cache is needed for four-token target paths.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from rhpr.gr.study_store import prefix_key

SCORINGS = ("raw", "local")


def retrieval_metrics(rank):
    return {"recall5": float(rank is not None and rank <= 5),
            "recall10": float(rank is not None and rank <= 10),
            "ndcg5": 1 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.,
            "ndcg10": 1 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.}


class ModelScorer:
    def __init__(self, model, batch, index):
        import torch
        if model.training:
            raise ValueError("Diagnostics require model.eval()")
        self.model, self.index = model, index
        self.device = next(model.parameters()).device
        self.batch_size = len(batch["input_ids"])
        with torch.inference_mode():
            self.encoded, self.mask = model.encoder_forward_pass(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])

    def logits(self, prefixes, owners):
        import torch
        depth = len(prefixes[0])
        owner = torch.as_tensor(owners, device=self.device, dtype=torch.long)
        future = torch.as_tensor(prefixes, device=self.device, dtype=torch.long) if depth else None
        with torch.inference_mode():
            hidden = self.model.decoder_forward_pass(future_ids=future,
                encoder_output=self.encoded[owner], attention_mask_for_encoder=self.mask[owner], use_cache=False)
            result = self.model.decoder.decoder_mlp[depth](hidden[:, -1]).float()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite diagnostic logits")
        return result

    def target_logits(self, targets):
        """All correct-path decisions in one causal teacher-forced decoder pass."""
        import torch
        future = torch.as_tensor(targets[:, :-1].copy(), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            hidden = self.model.decoder_forward_pass(future_ids=future,
                encoder_output=self.encoded, attention_mask_for_encoder=self.mask, use_cache=False)
            logits = [head(hidden[:, level]).float() for level, head in enumerate(self.model.decoder.decoder_mlp)]
        if not all(torch.isfinite(value).all() for value in logits):
            raise FloatingPointError("Nonfinite teacher-forced diagnostic logits")
        return logits


@lru_cache(maxsize=8)
def _catalogue_masks(index, device):
    import torch
    children = {}
    for sid in index.sids:
        for level, token in enumerate(sid):
            children.setdefault(tuple(map(int, sid[:level])), set()).add(int(token))
    lookup = {prefix: i for i, prefix in enumerate(sorted(children))}
    mask = np.zeros((len(lookup), index.width), dtype=bool)
    for prefix, tokens in children.items():
        mask[lookup[prefix], list(tokens)] = True
    return lookup, torch.as_tensor(mask, device=device)


def probabilities(logits, prefixes, index):
    import torch
    raw = logits.log_softmax(-1)
    lookup, masks = _catalogue_masks(index, logits.device)
    try:
        selected = [lookup[tuple(prefix)] for prefix in prefixes]
    except KeyError as error:
        raise ValueError("Scored a dead prefix") from error
    mask = masks[torch.as_tensor(selected, dtype=torch.long, device=logits.device)]
    valid_logmass = raw.masked_fill(~mask, -torch.inf).logsumexp(-1)
    local = (raw - valid_logmass[:, None]).masked_fill(~mask, -torch.inf)
    # Raw scores keep their full-vocabulary normalizer, but search is restricted
    # to valid catalogue paths under BOTH scoring rules.
    return raw.masked_fill(~mask, -torch.inf), local, valid_logmass


def target_diagnostics(scorer, targets, case_ids, run_id, epoch):
    rows, totals = [], np.zeros(len(targets), dtype=np.float64)
    all_logits = scorer.target_logits(targets)
    for level in range(scorer.index.sid_length):
        prefixes = [list(map(int, sid[:level])) for sid in targets]
        logits = all_logits[level]
        raw, local, mass = probabilities(logits, prefixes, scorer.index)
        values = logits.cpu().numpy()
        raw, local, mass = raw.cpu().numpy(), local.cpu().numpy(), mass.cpu().numpy()
        for i, sid in enumerate(targets):
            token = int(sid[level])
            alternatives = values[i].copy()
            alternatives[token] = -np.inf
            rows.append({"run_id": run_id, "epoch": epoch, "case_id": case_ids[i],
                "level": level, "prefix": prefix_key(sid[:level]), "token": token,
                "logits": values[i].tolist(), "raw_logprob": float(raw[i, token]),
                "local_logprob": float(local[i, token]),
                "token_rank": int(1 + (values[i] > values[i, token]).sum()),
                "best_other_token": int(alternatives.argmax()), "valid_mass": float(np.exp(mass[i]))})
            totals[i] += raw[i, token]
    return rows, totals


def beam_search(scorer, width, scoring, *, targets=None):
    """Variable valid beam counts, stable lexicographic prefix tie-breaking.

    Unlike the source implementation, narrow catalogue branches cannot create
    invalid zero-probability beams. No probability products or FP16 underflow.
    """
    if type(width) is not int or width < 1 or scoring not in SCORINGS:
        raise ValueError("Invalid beam request")
    beams = [[((), 0.)] for _ in range(scorer.batch_size)]
    trace = []
    for level in range(scorer.index.sid_length):
        prefixes, owners, scores = [], [], []
        for owner, group in enumerate(beams):
            for prefix, score in group:
                prefixes.append(list(prefix))
                owners.append(owner)
                scores.append(score)
        logits = scorer.logits(prefixes, owners)
        raw, local, _ = probabilities(logits, prefixes, scorer.index)
        logp = (raw if scoring == "raw" else local).cpu().numpy()
        candidates = [[] for _ in beams]
        for i, (prefix, owner, score) in enumerate(zip(prefixes, owners, scores, strict=True)):
            for token in scorer.index.trie.allowed_next(prefix):
                candidates[owner].append(((*prefix, token), score + float(logp[i, token])))
        beams = [sorted(group, key=lambda pair: (-pair[1], pair[0]))[:width] for group in candidates]
        for owner, group in enumerate(beams):
            correct = tuple(map(int, targets[owner][:level + 1])) if targets is not None else None
            for rank, (prefix, score) in enumerate(group, 1):
                trace.append({"owner": owner, "level": level, "prefix": prefix_key(prefix), "rank": rank,
                              "log_score": score, "target_prefix": prefix == correct})
            if correct is not None and not any(prefix == correct for prefix, _ in group):
                trace.append({"owner": owner, "level": level, "prefix": prefix_key(correct), "rank": None,
                              "log_score": None, "target_prefix": True})
    return beams, trace


def exact_catalogue(scorer, owner, *, chunk_size=128):
    """Visit each VALID prefix once; score all leaves under both rules.

    Returns vectors in original catalogue-item order. Keeping the raw vector
    also permits later catalogue-normalized likelihood/calibration analysis.
    """
    index = scorer.index
    parents = {(): (0., 0.)}
    for depth in range(index.sid_length):
        prefixes = sorted(parents)
        children = {}
        for begin in range(0, len(prefixes), chunk_size):
            chosen = prefixes[begin:begin + chunk_size]
            logits = scorer.logits(chosen, [owner] * len(chosen))
            raw, local, _ = probabilities(logits, chosen, index)
            raw, local = raw.cpu().numpy(), local.cpu().numpy()
            for i, prefix in enumerate(chosen):
                base_raw, base_local = parents[prefix]
                for token in index.trie.allowed_next(prefix):
                    children[(*prefix, token)] = (base_raw + float(raw[i, token]), base_local + float(local[i, token]))
        parents = children
    values = np.asarray([parents[tuple(sid)] for sid in index.sids], dtype=np.float64)
    result = np.empty_like(values)
    result[index.item_ids] = values
    return {name: result[:, i] for i, name in enumerate(SCORINGS)}


def mask_history_events(batch, positions, hierarchy):
    """Remove events from attention without shifting any surviving positions.

    Positions refer to the actual right-padded model input, not the full user
    history. This intervention measures sensitivity of a fixed trained model.
    """
    result = {name: value.clone() for name, value in batch.items()}
    length = result["input_ids"].shape[1]
    for position in sorted(set(positions)):
        if type(position) is not int or position < 0 or (position + 1) * hierarchy > length:
            raise ValueError("Invalid history event position")
        sl = slice(position * hierarchy, (position + 1) * hierarchy)
        if not result["attention_mask_encoder"][:, sl].all():
            raise ValueError("Cannot intervene on padding")
        result["attention_mask_encoder"][:, sl] = 0
        result["input_ids"][:, sl] = -1
    if not result["attention_mask_encoder"].any(-1).all():
        raise ValueError("Intervention would mask the entire history")
    return result
