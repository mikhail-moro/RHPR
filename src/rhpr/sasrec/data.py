from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SequenceSplit:
    """Per-user temporal leave-two-out split."""

    user_id: int
    train: np.ndarray
    validation: int
    test: int


@dataclass(frozen=True)
class YambdaSequences:
    """Mapped Yambda sequences and the corresponding external item catalogue."""

    splits: tuple[SequenceSplit, ...]
    catalog_item_ids: np.ndarray

    @property
    def num_items(self) -> int:
        return int(self.catalog_item_ids.size)


def split_sequence(user_id: int, item_ids: Sequence[int] | np.ndarray) -> SequenceSplit:
    """Split a chronological sequence as n-2 / 1 / 1 without dropping events."""

    items = np.asarray(item_ids, dtype=np.int64)
    if items.ndim != 1:
        raise ValueError("item_ids must be one-dimensional")
    if items.size < 3:
        raise ValueError(f"user {user_id} has fewer than three interactions")
    return SequenceSplit(
        user_id=int(user_id),
        train=items[:-2].copy(),
        validation=int(items[-2]),
        test=int(items[-1]),
    )


def make_training_pair(train: Sequence[int] | np.ndarray, max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a shifted SASRec sample from the last max_seq_len + 1 events."""

    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    items = np.asarray(train, dtype=np.int64)
    if items.ndim != 1:
        raise ValueError("train must be one-dimensional")
    if items.size < 2:
        raise ValueError("training sequence must contain at least two interactions")
    window = items[-(max_seq_len + 1) :]
    return window[:-1].copy(), window[1:].copy()


def validation_example(split: SequenceSplit, max_seq_len: int) -> tuple[np.ndarray, int]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    return split.train[-max_seq_len:].copy(), split.validation


def test_example(split: SequenceSplit, max_seq_len: int) -> tuple[np.ndarray, int]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    context = np.concatenate((split.train, np.asarray([split.validation], dtype=np.int64)))
    return context[-max_seq_len:].copy(), split.test


def _chronological_items(items: Sequence[int], timestamps: Sequence[int] | None) -> np.ndarray:
    # Kaggle's Polars build can materialize a Parquet list cell as either (N,)
    # or (1, N). Both represent one user sequence, so normalize the boundary.
    item_array = np.asarray(items, dtype=np.int64).reshape(-1)
    if timestamps is None:
        return item_array
    timestamp_array = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if timestamp_array.shape != item_array.shape:
        raise ValueError("item and timestamp lists have different lengths")
    if timestamp_array.size > 1 and np.any(timestamp_array[1:] < timestamp_array[:-1]):
        order = np.argsort(timestamp_array, kind="stable")
        item_array = item_array[order]
    return item_array


def load_yambda_sequences(path: str | Path) -> YambdaSequences:
    """Load sequential Yambda parquet and map catalogue ids to 1..N (0 is padding)."""

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised in the Kaggle image
        raise RuntimeError("the yambda_sasrec experiment requires polars") from exc

    parquet_path = Path(path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Yambda parquet not found: {parquet_path}")

    schema = pl.read_parquet_schema(parquet_path)
    if {"uid", "item_ids"}.issubset(schema):
        columns = ["uid", "item_ids"]
        if "timestamps" in schema:
            columns.append("timestamps")
        frame = pl.read_parquet(parquet_path, columns=columns)
    elif {"uid", "item_id", "timestamp"}.issubset(schema):
        frame = (
            pl.read_parquet(parquet_path, columns=["uid", "item_id", "timestamp"])
            .sort(["uid", "timestamp"], maintain_order=True)
            .group_by("uid", maintain_order=True)
            .agg(
                pl.col("item_id").alias("item_ids"),
                pl.col("timestamp").alias("timestamps"),
            )
        )
    else:
        raise ValueError(
            "expected sequential uid/item_ids columns or flat uid/item_id/timestamp columns, "
            f"got {list(schema)}"
        )

    has_timestamps = "timestamps" in frame.columns
    user_ids: list[int] = []
    chronological: list[np.ndarray] = []
    for row in frame.iter_rows():
        user_id, items = row[:2]
        timestamps = row[2] if has_timestamps else None
        user_ids.append(int(user_id))
        chronological.append(_chronological_items(items, timestamps))
    del frame

    if not chronological or any(items.size < 3 for items in chronological):
        raise ValueError("all users must have at least three interactions for n-2/1/1 splitting")

    catalog = np.unique(np.concatenate(chronological)).astype(np.int64, copy=False)
    splits = []
    for user_id, items in zip(user_ids, chronological, strict=True):
        mapped = np.searchsorted(catalog, items).astype(np.int64, copy=False) + 1
        splits.append(split_sequence(int(user_id), mapped))

    return YambdaSequences(splits=tuple(splits), catalog_item_ids=catalog)


def cold_target_fraction(splits: Sequence[SequenceSplit], *, target: str) -> float:
    """Return the fraction of validation/test targets absent from positive train events."""

    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")
    if not splits:
        raise ValueError("splits must not be empty")
    train_items = {int(item) for split in splits for item in split.train}
    cold = sum(int(getattr(split, target) not in train_items) for split in splits)
    return cold / len(splits)
