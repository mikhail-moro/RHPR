"""Paper-batch adapter around GRID's unchanged contiguous-window collator.

The supplied author rows remain full length. GRID samples contiguous windows,
truncates their suffix to the SID budget, masks the final item and supplies its
four codes as labels. Only batching is adapted: augmented rows are buffered in
order to produce the article's fixed number of examples per optimizer update.

The normalized-row input uses a seeded shuffle cycle instead of reproducing
TensorFlow's file/worker/buffer shuffle. That I/O distinction is recorded in the
stream report; the augmentation and masking operations themselves are upstream.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import time
from types import SimpleNamespace


def _positive_integer(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@lru_cache(maxsize=1)
def _upstream():
    # load_grid verifies the pinned source before adding its src namespace.
    from rhpr.gr.grid import load_grid
    load_grid()
    from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate
    from src.data.loading.components.label_function import NextKTokenMasking
    return collate_with_sid_causal_duplicate, NextKTokenMasking


def _checked_sids(sids):
    import numpy as np
    values = np.asarray(sids)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or not np.issubdtype(values.dtype, np.integer) or (values < 0).any()):
        raise ValueError("sids must be a nonempty nonnegative integer item-by-code matrix")
    return values


def _rows(users, sids, source_indices):
    import numpy as np
    import torch
    rows = []
    for index in source_indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(users):
            raise ValueError("source indices must identify existing training users")
        user = users[int(index)]
        history = np.asarray(user.train)
        if (history.ndim != 1 or len(history) < 2
                or not np.issubdtype(history.dtype, np.integer)
                or (history < 0).any() or (history >= len(sids)).any()):
            raise ValueError("each source training row needs at least two known items")
        rows.append({
            "sequence_data": torch.as_tensor(sids[history].reshape(-1).copy(), dtype=torch.long),
            "user_id": torch.tensor([user.user_id], dtype=torch.long),
        })
    if not rows:
        raise ValueError("at least one source training user is required")
    return rows


def _collate_rows(rows, hierarchy: int, sequence_length: int, max_augmented_batch: int):
    collator, label_type = _upstream()
    model_input, label_data = collator(
        batch=rows, sequence_field_name="sequence_data", sid_hierarchy=hierarchy,
        labels={"sequence_data": SimpleNamespace(transform=label_type(next_k=hierarchy))},
        sequence_length=sequence_length, masking_token=-1, padding_token=-1,
        oov_token=None, max_batch_size=max_augmented_batch,
    )
    count = model_input.mask.shape[0]
    batch = {
        "input_ids": model_input.transformed_sequences["sequence_data"],
        "attention_mask_encoder": model_input.mask,
        "user_id": model_input.transformed_sequences["user_id"],
        "future_ids": label_data.labels["sequence_data"].reshape(count, hierarchy),
    }
    if batch["input_ids"].shape != (count, sequence_length):
        raise ValueError("upstream collator returned a different SID sequence budget")
    # Keep the target's masked/padded positions in the input, exactly as GRID's
    # model_step does. Repacking to 116 positions changes the input contract.
    return batch


def collate_paper_rows(users, sids, source_indices, *, sequence_length: int = 120,
                       max_augmented_batch: int = 512):
    """Return CPU model kwargs from the actual GRID collator and caller RNG.

    Tests and callers can establish a torch RNG state to reproduce the author
    collator exactly. For an isolated deterministic stream use PaperBatchStream.
    Validation and test fields on each user are deliberately never accessed.
    """
    values = _checked_sids(sids)
    length = _positive_integer(sequence_length, "sequence_length")
    maximum = _positive_integer(max_augmented_batch, "max_augmented_batch")
    hierarchy = values.shape[1]
    if length % hierarchy or length < 2 * hierarchy:
        raise ValueError("sequence_length must hold at least two complete SIDs")
    return _collate_rows(_rows(users, values, source_indices), hierarchy, length, maximum)


def move_batch(batch: dict, device: str) -> dict:
    """Move model tensors without changing labels, padding or precision."""
    return {name: value.to(device) for name, value in batch.items()}


class PaperBatchStream:
    """Infinite, deterministic upstream augmentation with example-counted batches.

    Consumed source rows are reshuffled after a full cycle. Windows already
    produced by the upstream collator are buffered, never discarded to force a
    fixed optimizer batch. Collator CPU RNG is isolated from model/dropout RNG.
    """

    def __init__(self, users, sids, *, seed: int = 42, source_batch_size: int = 32,
                 effective_batch_size: int = 256, sequence_length: int = 120,
                 max_augmented_batch: int = 512):
        import numpy as np
        import torch
        self.users = tuple(users)
        if not self.users:
            raise ValueError("at least one training user is required")
        self.sids = _checked_sids(sids)
        self.source_batch_size = _positive_integer(source_batch_size, "source_batch_size")
        self.effective_batch_size = _positive_integer(effective_batch_size, "effective_batch_size")
        self.sequence_length = _positive_integer(sequence_length, "sequence_length")
        self.max_augmented_batch = _positive_integer(max_augmented_batch, "max_augmented_batch")
        if (self.sequence_length % self.sids.shape[1]
                or self.sequence_length < 2 * self.sids.shape[1]):
            raise ValueError("sequence_length must hold at least two complete SIDs")
        # Pre-map complete rows once. Repeating this inside each optimizer step
        # would profile an avoidable normalized-data adapter overhead.
        self._rows = _rows(self.users, self.sids, range(len(self.users)))
        self._shuffle = np.random.default_rng(seed)
        self._order = self._shuffle.permutation(len(self.users))
        self._position = 0
        self._rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self._pending = deque()
        self._pending_offset = 0
        self._buffered = 0
        self._stats = {
            "source_batches": 0, "source_rows_consumed": 0, "source_cycles_started": 1,
            "augmented_examples_generated": 0, "model_examples_yielded": 0,
            "model_batches_yielded": 0, "source_collate_seconds": 0.0,
            "last_source_collate_seconds": 0.0, "cpu_next_batch_seconds": 0.0,
        }

    def _source_indices(self):
        import numpy as np
        chunks, remaining = [], self.source_batch_size
        while remaining:
            available = len(self._order) - self._position
            if not available:
                self._order = self._shuffle.permutation(len(self.users))
                self._position = 0
                self._stats["source_cycles_started"] += 1
                available = len(self._order)
            count = min(available, remaining)
            chunks.append(self._order[self._position:self._position + count])
            self._position += count
            remaining -= count
        return np.concatenate(chunks)

    def _fill(self):
        import torch
        rows = [self._rows[int(index)] for index in self._source_indices()]
        started = time.perf_counter()
        # fork_rng(devices=[]) preserves the global CPU RNG and never touches
        # CUDA/dropout state, including when a GPU is already initialized.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._rng)
            batch = _collate_rows(rows, self.sids.shape[1], self.sequence_length,
                                  self.max_augmented_batch)
            self._rng = torch.get_rng_state()
        elapsed = time.perf_counter() - started
        size = len(batch["future_ids"])
        if not size:
            raise ValueError("upstream augmentation produced no supervised examples")
        self._pending.append(batch)
        self._buffered += size
        self._stats["source_batches"] += 1
        self._stats["source_rows_consumed"] += len(rows)
        self._stats["augmented_examples_generated"] += size
        self._stats["source_collate_seconds"] += elapsed
        self._stats["last_source_collate_seconds"] = elapsed

    def next_batch(self, device: str = "cpu") -> dict:
        import torch
        started = time.perf_counter()
        while self._buffered < self.effective_batch_size:
            self._fill()
        pieces = []
        remaining = self.effective_batch_size
        while remaining:
            first = self._pending[0]
            available = len(first["future_ids"]) - self._pending_offset
            take = min(remaining, available)
            pieces.append({key: value[self._pending_offset:self._pending_offset + take]
                           for key, value in first.items()})
            self._pending_offset += take
            self._buffered -= take
            remaining -= take
            if self._pending_offset == len(first["future_ids"]):
                self._pending.popleft()
                self._pending_offset = 0
        batch = {key: torch.cat([piece[key] for piece in pieces], dim=0) for key in pieces[0]}
        self._stats["model_batches_yielded"] += 1
        self._stats["model_examples_yielded"] += self.effective_batch_size
        self._stats["cpu_next_batch_seconds"] += time.perf_counter() - started
        return move_batch(batch, device)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def report(self) -> dict:
        return {
            "augmentation": "pinned GRID collate_with_sid_causal_duplicate + NextKTokenMasking",
            "source_order": "seeded normalized-row shuffle cycles; not TensorFlow file/worker/buffer order",
            "effective_batch_policy": "ordered buffering of all augmented rows; fixed model examples",
            "source_users": len(self.users), "source_batch_size": self.source_batch_size,
            "max_augmented_batch": self.max_augmented_batch,
            "effective_batch_size": self.effective_batch_size,
            "sid_tokens_including_target": self.sequence_length,
            "max_context_items": self.sequence_length // self.sids.shape[1] - 1,
            "buffered_examples": self._buffered, "test_targets_read": False,
            **self._stats,
        }
