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
    # Full chronological timeline, including both held-out events, in seconds.
    # Keeping its beginning lets a cropped context retain its true incoming gap.
    timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.timestamps is not None:
            times = _timestamps_in_seconds(self.timestamps, "seconds")
            if times.ndim != 1 or times.size != self.train.size + 2:
                raise ValueError("timestamps must cover train, validation, and test events")
            if np.any(np.diff(times) < 0):
                raise ValueError("timestamps must be chronological")
            object.__setattr__(self, "timestamps", times)

    @property
    def train_timestamps(self) -> np.ndarray | None:
        return None if self.timestamps is None else self.timestamps[:-2]

    @property
    def validation_timestamp(self) -> float | None:
        return None if self.timestamps is None else float(self.timestamps[-2])

    @property
    def test_timestamp(self) -> float | None:
        return None if self.timestamps is None else float(self.timestamps[-1])


@dataclass(frozen=True)
class YambdaSequences:
    """Mapped Yambda sequences and the corresponding external item catalogue."""

    splits: tuple[SequenceSplit, ...]
    catalog_item_ids: np.ndarray

    @property
    def num_items(self) -> int:
        return int(self.catalog_item_ids.size)


def split_sequence(
    user_id: int,
    item_ids: Sequence[int] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
) -> SequenceSplit:
    """Split a chronological sequence as n-2 / 1 / 1 without dropping events."""

    items = np.asarray(item_ids, dtype=np.int64)
    if items.ndim != 1:
        raise ValueError("item_ids must be one-dimensional")
    if items.size < 3:
        raise ValueError(f"user {user_id} has fewer than three interactions")
    times = None if timestamps is None else _timestamps_in_seconds(timestamps, "seconds")
    if times is not None:
        if times.ndim != 1 or times.shape != items.shape:
            raise ValueError("item and timestamp lists have different lengths")
        if np.any(np.diff(times) < 0):
            raise ValueError("split_sequence timestamps must be chronological")
    return SequenceSplit(
        user_id=int(user_id),
        train=items[:-2].copy(),
        validation=int(items[-2]),
        test=int(items[-1]),
        timestamps=None if times is None else times.copy(),
    )


def _timestamps_in_seconds(timestamps: Sequence[float] | np.ndarray, timestamp_unit: str) -> np.ndarray:
    divisors = {"seconds": 1.0, "milliseconds": 1_000.0, "microseconds": 1_000_000.0, "nanoseconds": 1_000_000_000.0}
    if timestamp_unit not in divisors:
        raise ValueError(f"unsupported timestamp_unit: {timestamp_unit}")
    values = np.asarray(timestamps, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps must be finite")
    if timestamp_unit == "seconds" and values.size and np.max(np.abs(values)) >= 100_000_000_000:
        raise ValueError("timestamps look like Unix milliseconds or finer; declare timestamp_unit explicitly")
    return values / divisors[timestamp_unit]


def _gap_ids(
    timestamps: np.ndarray | None, *, num_buckets: int, time_unit: float
) -> np.ndarray:
    if num_buckets < 1:
        raise ValueError("num_buckets must be positive")
    if not np.isfinite(time_unit) or time_unit <= 0:
        raise ValueError("time_unit must be finite and positive seconds")
    if timestamps is None:
        raise ValueError("time-enabled inputs require observed timestamps")
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be a finite one-dimensional sequence")
    deltas = np.diff(times)
    if np.any(deltas < 0):
        raise ValueError("timestamps must be chronological")
    result = np.empty(times.size, dtype=np.int64)
    if times.size:
        result[0] = num_buckets  # START, distinct from a simultaneous-event gap.
        result[1:] = np.clip(np.floor(np.log2(1.0 + deltas / time_unit)), 0, num_buckets - 1).astype(np.int64)
    return result


def training_gap_ids(
    split: SequenceSplit, max_seq_len: int, *, num_buckets: int = 32, time_unit: float = 1.0
) -> np.ndarray:
    """Incoming observed gaps aligned to shifted training inputs, before cropping."""
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if split.train.size < 2:
        raise ValueError("training sequence must contain at least two interactions")
    gaps = _gap_ids(split.train_timestamps, num_buckets=num_buckets, time_unit=time_unit)
    return gaps[-(max_seq_len + 1) : -1].copy()


def evaluation_gap_ids(
    split: SequenceSplit,
    max_seq_len: int,
    split_name: str,
    *,
    num_buckets: int = 32,
    time_unit: float = 1.0,
) -> np.ndarray:
    """Incoming gaps from context only; neither held-out target gap is consumed."""
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if split_name == "validation":
        times = split.train_timestamps
    elif split_name == "test":
        times = None if split.timestamps is None else split.timestamps[:-1]
    else:
        raise ValueError("split_name must be validation or test")
    return _gap_ids(times, num_buckets=num_buckets, time_unit=time_unit)[-max_seq_len:].copy()


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


def _chronological_events(
    items: Sequence[int], timestamps: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray | None]:
    # Kaggle's Polars build can materialize a Parquet list cell as either (N,)
    # or (1, N). Both represent one user sequence, so normalize the boundary.
    item_array = np.asarray(items, dtype=np.int64).reshape(-1)
    if timestamps is None:
        return item_array, None
    timestamp_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamp_array.shape != item_array.shape:
        raise ValueError("item and timestamp lists have different lengths")
    if timestamp_array.size > 1 and np.any(timestamp_array[1:] < timestamp_array[:-1]):
        order = np.argsort(timestamp_array, kind="stable")
        item_array = item_array[order]
        timestamp_array = timestamp_array[order]
    return item_array, timestamp_array


def _chronological_items(items: Sequence[int], timestamps: Sequence[int] | None) -> np.ndarray:
    return _chronological_events(items, timestamps)[0]


def resolve_yambda_parquet_path(path: str | Path) -> Path:
    """Accept both Kaggle mount layouts observed for the same Yambda source."""

    requested = Path(path)
    modern_root = Path("/kaggle/input/datasets/ploshkin/yambda")
    legacy_root = Path("/kaggle/input/yambda")
    candidates = [requested]
    for source_root, alternate_root in ((modern_root, legacy_root), (legacy_root, modern_root)):
        try:
            candidates.append(alternate_root / requested.relative_to(source_root))
        except ValueError:
            pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Yambda parquet not found; checked: {', '.join(map(str, candidates))}")


def load_yambda_sequences(path: str | Path, *, timestamp_unit: str = "seconds") -> YambdaSequences:
    """Load sequential Yambda parquet and map catalogue ids to 1..N (0 is padding)."""

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised in the Kaggle image
        raise RuntimeError("the yambda_sasrec experiment requires polars") from exc

    parquet_path = resolve_yambda_parquet_path(path)

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
    chronological_times: list[np.ndarray | None] = []
    for row in frame.iter_rows():
        user_id, items = row[:2]
        timestamps = row[2] if has_timestamps else None
        user_ids.append(int(user_id))
        item_array, timestamp_array = _chronological_events(items, timestamps)
        chronological.append(item_array)
        chronological_times.append(
            None if timestamp_array is None else _timestamps_in_seconds(timestamp_array, timestamp_unit)
        )
    del frame

    if not chronological or any(items.size < 3 for items in chronological):
        raise ValueError("all users must have at least three interactions for n-2/1/1 splitting")

    catalog = np.unique(np.concatenate(chronological)).astype(np.int64, copy=False)
    splits = []
    for user_id, items, timestamps in zip(user_ids, chronological, chronological_times, strict=True):
        mapped = np.searchsorted(catalog, items).astype(np.int64, copy=False) + 1
        splits.append(split_sequence(int(user_id), mapped, timestamps))

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
