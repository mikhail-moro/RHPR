"""Static-shape XLA adapters preserving TIGER windows and constrained beams."""
from __future__ import annotations

import inspect
import math
import textwrap
from types import MethodType


def fast_windows(rows, *, generator, collate, hierarchy=4, maximum=512, sequence_length=100, labels=None):
    """Select precisely the source's sorted unique sampled window indices.

    The source enumerates every possible window and tests membership in sampled
    indices. Invert that enumeration instead; duplicate draws are still emitted
    once, and emitted windows retain source order. Padding/masking stay upstream.
    """
    import torch
    sizes = [len(row["sequence_data"]) // hierarchy for row in rows]
    totals = [n * (n - 1) // 2 for n in sizes]
    total = sum(totals)
    if total < 1:
        raise ValueError("No context/target windows in source batch")
    selected = (torch.randint(total, (maximum,), generator=generator).unique(sorted=True).tolist()
                if total > maximum else list(range(total)))
    result, user, base = [], 0, 0
    for index in selected:
        while index >= base + totals[user]:
            base += totals[user]
            user += 1
        local = index - base
        end = (1 + math.isqrt(1 + 8 * local)) // 2 + 1
        start = local - (end - 1) * (end - 2) // 2
        row = rows[user]
        result.append({"sequence_data": row["sequence_data"][start * hierarchy:end * hierarchy],
                       "user_id": row["user_id"]})
    return collate(result, labels=labels, sequence_length=sequence_length,
                   masking_token=-1, padding_token=-1, oov_token=None)


def kwargs_from_collation(pair, hierarchy=4):
    inputs, labels = pair
    return {"input_ids": inputs.transformed_sequences["sequence_data"],
        "attention_mask_encoder": inputs.mask,
        "user_id": inputs.transformed_sequences["user_id"],
        "future_ids": labels.labels["sequence_data"].reshape(len(inputs.mask), hierarchy)}


def epoch_batches(rows, *, seed, source_size, physical_size, collate, labels, sequence_length=100):
    """Buffer source outputs into fixed-size model batches; retain final remainder."""
    import torch
    from rhpr.gr.genpas_epochs import epoch_order
    rng = torch.Generator().manual_seed(seed)
    order, pending = epoch_order(len(rows), seed, 0), None
    for begin in range(0, len(rows), source_size):
        pair = fast_windows([rows[i] for i in order[begin:begin + source_size]], generator=rng,
            collate=collate, labels=labels, sequence_length=sequence_length)
        part = kwargs_from_collation(pair)
        pending = part if pending is None else {name: torch.cat((pending[name], part[name])) for name in part}
        while len(pending["future_ids"]) >= physical_size:
            yield {name: value[:physical_size] for name, value in pending.items()}
            pending = {name: value[physical_size:] for name, value in pending.items()}
    if pending is not None and len(pending["future_ids"]):
        yield pending


def padded_batch(batch, size):
    import torch
    count = len(batch["future_ids"])
    if not 0 < count <= size:
        raise ValueError("Invalid physical batch size")
    padded = {name: torch.cat((value, value[-1:].expand(size - count, *value.shape[1:])))
              if count < size else value for name, value in batch.items()}
    return padded, (torch.arange(size) < count).float()


def per_example_loss(model, batch):
    import torch.nn.functional as F
    hidden = model(**batch)[:, :-1]
    return sum(F.cross_entropy(head(hidden[:, level]).float(), batch["future_ids"][:, level], reduction="none")
               for level, head in enumerate(model.decoder.decoder_mlp))


def _prefix_membership(self, prefix, batch_size=100000):
    """Fixed-iteration lower_bound; no dynamic boolean indexing or CPU round-trip."""
    import torch
    keys = getattr(self, f"rhpr_prefix_{prefix.shape[1]}")
    target = torch.zeros_like(prefix[:, 0])
    for column in range(prefix.shape[1]):
        target = target * self.num_embeddings_per_hierarchy + prefix[:, column]
    low, high = torch.zeros_like(target), torch.full_like(target, len(keys))
    for _ in range((len(keys) + 1).bit_length()):
        middle = (low + high) // 2
        less = keys[middle.clamp(max=len(keys) - 1)] < target
        active = low < high
        low = torch.where(active & less, middle + 1, low)
        high = torch.where(active & ~less, middle, high)
    return (low < len(keys)) & (keys[low.clamp(max=len(keys) - 1)] == target)


def adapt_beams(model):
    """Replace only prefix membership implementation and two mask assignments.

    The released generate function is otherwise executed unchanged. Inputs are
    from a SHA-verified source tree; replacements assert the expected source.
    """
    import numpy as np
    import torch
    values = model.codebooks.detach().cpu().numpy().astype(np.int64)
    width = model.num_embeddings_per_hierarchy
    if width ** values.shape[1] > np.iinfo(np.int64).max:
        raise ValueError("Prefix key does not fit int64")
    for depth in range(1, values.shape[1] + 1):
        keys = np.zeros(len(values), dtype=np.int64)
        for column in range(depth):
            keys = keys * width + values[:, column]
        model.register_buffer(f"rhpr_prefix_{depth}", torch.from_numpy(np.unique(keys)), persistent=False)
    native = type(model).generate
    source = textwrap.dedent(inspect.getsource(native))
    replacements = {
        'candidate_logits[:, ~valid_prefix_mask] = float("-inf")':
            'candidate_logits = candidate_logits.masked_fill(~valid_prefix_mask.unsqueeze(0), float("-inf"))',
        'candidate_logits[~valid_prefix_mask] = float("-inf")':
            'candidate_logits = candidate_logits.masked_fill(~valid_prefix_mask, float("-inf"))',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Pinned GenPAS beam mask source changed")
        source = source.replace(before, after)
    namespace = dict(native.__globals__)
    exec(compile(source, "<verified-genpas-static-beams>", "exec"), namespace)
    model.generate = MethodType(namespace["generate"], model)
    model._check_valid_prefix = MethodType(_prefix_membership, model)
    return model
