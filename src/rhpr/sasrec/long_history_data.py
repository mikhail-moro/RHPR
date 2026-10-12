"""Train-visible, streaming preparation of long sequential Yambda histories.

The profiling cohort is selected by an explicit global time boundary. Catalogue
IDs and eligibility depend only on events before that boundary; this module
does not construct validation/test targets or publish individual event arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rhpr.sasrec.data import resolve_yambda_parquet_path


@dataclass(frozen=True)
class LongHistoryUser:
    """One chronological training prefix with mapped IDs; zero is padding."""

    user_id: int
    train: np.ndarray
    timestamps: np.ndarray

    @property
    def train_timestamps(self) -> np.ndarray:
        return self.timestamps


@dataclass(frozen=True)
class LongHistoryCohort:
    users: tuple[LongHistoryUser, ...]
    catalog_item_ids: np.ndarray
    summary: dict[str, Any]

    @property
    def num_items(self) -> int:
        return int(self.catalog_item_ids.size)

    @property
    def train_events(self) -> int:
        return sum(int(user.train.size) for user in self.users)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer_array(values: np.ndarray, name: str, *, unsigned_32: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.size and (array.min() < 0 or (unsigned_32 and array.max() > np.iinfo(np.uint32).max)):
        raise ValueError(f"{name} must contain nonnegative uint32-compatible values")
    return array


def _distribution(values: np.ndarray) -> dict[str, int | float]:
    return {
        "count": int(values.size), "min": int(values.min()), "max": int(values.max()),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
    }


def _hash_array(hasher: Any, array: np.ndarray, dtype: str) -> None:
    # Explicit little-endian types make signatures independent of source dtype
    # and native endianness. Callers frame each user with UID and event count.
    hasher.update(np.ascontiguousarray(array, dtype=dtype).tobytes())


def _cohort_content_summary(users: tuple[LongHistoryUser, ...], catalog: np.ndarray) -> dict[str, Any]:
    uid_length_hash, timestamp_hash, train_item_hash = (hashlib.sha256() for _ in range(3))
    for user in users:
        frame = np.asarray([user.user_id, user.train.size], dtype="<i8")
        for hasher in (uid_length_hash, timestamp_hash, train_item_hash):
            _hash_array(hasher, frame, "<i8")
        _hash_array(timestamp_hash, user.timestamps, "<u4")
        _hash_array(train_item_hash, user.train, "<i4")
    train_lengths = np.asarray([user.train.size for user in users], dtype=np.int64)
    return {
        "user_count": len(users), "train_events": int(train_lengths.sum()), "num_items": int(catalog.size),
        "train_length_distribution": _distribution(train_lengths),
        "cohort_uid_length_sha256": uid_length_hash.hexdigest(),
        "cohort_train_timestamps_sha256": timestamp_hash.hexdigest(),
        "cohort_mapped_train_items_sha256": train_item_hash.hexdigest(),
        "catalog_sha256": hashlib.sha256(np.asarray(catalog, dtype="<i8").tobytes()).hexdigest(),
        "mapped_train_array_bytes": int(sum(user.train.nbytes for user in users)),
        "train_timestamp_array_bytes": int(sum(user.timestamps.nbytes for user in users)),
        "catalog_array_bytes": int(catalog.nbytes),
    }


def _validate_cohort_content(cohort: LongHistoryCohort) -> dict[str, Any]:
    summary = cohort.summary
    train_end = _positive_integer(summary.get("train_end_timestamp"), "train_end_timestamp")
    minimum = _positive_integer(summary.get("min_train_events"), "min_train_events")
    window = _positive_integer(summary.get("max_seq_len"), "max_seq_len")
    if minimum < window + 1 or summary.get("timestamp_unit") != "seconds":
        raise ValueError("prepared cohort must have seconds timestamps and full shifted training windows")
    catalog = np.asarray(cohort.catalog_item_ids)
    if catalog.ndim != 1 or not catalog.size or catalog.dtype != np.dtype("int64"):
        raise ValueError("prepared external catalogue must be a nonempty int64 array")
    if catalog[0] < 0 or np.any(catalog[1:] <= catalog[:-1]):
        raise ValueError("prepared external catalogue must be strictly sorted and nonnegative")
    if not cohort.users:
        raise ValueError("prepared cohort must contain users")
    previous_uid = -1
    for user in cohort.users:
        if user.user_id <= previous_uid or user.user_id > np.iinfo(np.int64).max:
            raise ValueError("prepared user IDs must be distinct sorted nonnegative int64 values")
        previous_uid = user.user_id
        items, timestamps = user.train, user.timestamps
        if items.ndim != 1 or items.dtype != np.dtype("int32") or items.size < minimum:
            raise ValueError("prepared mapped train arrays must be int32 and satisfy eligibility")
        if items.min() < 1 or items.max() > catalog.size:
            raise ValueError("prepared mapped IDs must lie in 1..num_items")
        if timestamps.ndim != 1 or timestamps.dtype != np.dtype("uint32") or timestamps.size != items.size:
            raise ValueError("prepared timestamps must be aligned uint32 observed seconds")
        if np.any(timestamps[1:] < timestamps[:-1]) or int(timestamps[-1]) >= train_end:
            raise ValueError("prepared timestamps must be chronological and strictly before train_end_timestamp")
    actual = _cohort_content_summary(cohort.users, catalog)
    for key, value in actual.items():
        if summary.get(key) != value:
            raise ValueError(f"prepared cohort content disagrees with summary: {key}")
    if summary.get("full_shifted_window_endings") != sum(user.train.size - window for user in cohort.users):
        raise ValueError("prepared cohort content disagrees with summary: full_shifted_window_endings")
    if summary.get("catalog_source") != "eligible_users_events_strictly_before_train_end":
        raise ValueError("prepared catalogue must be train-only")
    return actual


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_long_history_cohort(
    path: str | Path, *, train_end_timestamp: int, max_seq_len: int = 16_384,
    min_train_events: int | None = None, batch_size: int = 16,
    output_dir: str | Path | None = None,
) -> LongHistoryCohort:
    """Stream list-valued Yambda Parquet and retain eligible train prefixes.

    Timestamps are the source's observed integer seconds. Membership is strict
    ``timestamp < train_end_timestamp``. A shifted window of ``max_seq_len``
    input events needs at least ``max_seq_len + 1`` training events. All listens,
    including repeats and skips, are retained; only UID/items/timestamps are
    decoded. The train-only sorted catalogue maps external IDs to 1..N.

    ``output_dir`` receives aggregate JSON and one diagnostic row per source
    user, never mapped event arrays or held-out targets. Streamed Arrow batches
    avoid materializing Python lists for the hundreds of millions of events.
    """
    train_end = _positive_integer(train_end_timestamp, "train_end_timestamp")
    if train_end > np.iinfo(np.uint32).max + 1:
        raise ValueError("train_end_timestamp exceeds the supported uint32 seconds range")
    window = _positive_integer(max_seq_len, "max_seq_len")
    minimum = window + 1 if min_train_events is None else _positive_integer(min_train_events, "min_train_events")
    if minimum < window + 1:
        raise ValueError("min_train_events must allow one full shifted training window")
    batch_rows = _positive_integer(batch_size, "batch_size")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - available in Kaggle
        raise RuntimeError("long-history preparation requires pyarrow") from exc

    source_path = resolve_yambda_parquet_path(path)
    parquet = pq.ParquetFile(source_path)
    schema = parquet.schema_arrow
    if {"uid", "item_id", "timestamp"}.issubset(schema.names):
        item_column, time_column = "item_id", "timestamp"
    elif {"uid", "item_ids", "timestamps"}.issubset(schema.names):
        item_column, time_column = "item_ids", "timestamps"
    else:
        raise ValueError("expected sequential uid/item_id/timestamp list columns")
    for name in (item_column, time_column):
        column_type = schema.field(name).type
        if not (pa.types.is_list(column_type) or pa.types.is_large_list(column_type)
                or pa.types.is_fixed_size_list(column_type)):
            raise ValueError(f"{name} must be a list column; flat events are not supported")
    if not pa.types.is_integer(schema.field("uid").type):
        raise ValueError("uid must be an integer column")

    selected: list[tuple[int, np.ndarray, np.ndarray]] = []
    profiles: list[dict[str, int | bool]] = []
    seen_users: set[int] = set()
    source_event_count = source_train_events = sorted_users = 0
    source_min_timestamp: int | None = None
    source_max_timestamp: int | None = None
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["uid", item_column, time_column], use_threads=False):
        uid_values, item_values, time_values = (batch.column(index) for index in range(3))
        for row in range(batch.num_rows):
            uid_scalar = uid_values[row]
            if not uid_scalar.is_valid or not item_values[row].is_valid or not time_values[row].is_valid:
                raise ValueError("uid/item/timestamp cells must not be null")
            user_id = int(uid_scalar.as_py())
            if user_id < 0 or user_id > np.iinfo(np.int64).max or user_id in seen_users:
                raise ValueError("source user IDs must be distinct nonnegative int64 values")
            seen_users.add(user_id)
            arrow_items, arrow_times = item_values[row].values, time_values[row].values
            if arrow_items.null_count or arrow_times.null_count:
                raise ValueError("item/timestamp list elements must not be null")
            items = _integer_array(arrow_items.to_numpy(zero_copy_only=False), "item IDs", unsigned_32=True)
            timestamps = _integer_array(arrow_times.to_numpy(zero_copy_only=False), "timestamps", unsigned_32=True)
            if items.size != timestamps.size:
                raise ValueError("item and timestamp lists have different lengths")
            source_event_count += int(items.size)
            if timestamps.size:
                row_min, row_max = int(timestamps.min()), int(timestamps.max())
                source_min_timestamp = row_min if source_min_timestamp is None else min(source_min_timestamp, row_min)
                source_max_timestamp = row_max if source_max_timestamp is None else max(source_max_timestamp, row_max)
            reordered = bool(timestamps.size > 1 and np.any(timestamps[1:] < timestamps[:-1]))
            if reordered:
                order = np.argsort(timestamps, kind="stable")
                timestamps, items = timestamps[order], items[order]
                sorted_users += 1
            train_length = int(np.searchsorted(timestamps, train_end, side="left"))
            source_train_events += train_length
            eligible = train_length >= minimum
            profiles.append({
                "user_id": user_id, "source_events": int(items.size), "train_events": train_length,
                "eligible": eligible, "stable_time_sort_applied": reordered,
                "full_shifted_window_endings": max(0, train_length - window),
            })
            if eligible:
                # Compact copied arrays release each Arrow batch. External IDs
                # are remapped after constructing the selected train catalogue.
                selected.append((user_id, items[:train_length].astype(np.uint32, copy=True),
                                 timestamps[:train_length].astype(np.uint32, copy=True)))
    if not selected:
        raise ValueError(f"no users have {minimum} training events before timestamp {train_end}")
    selected.sort(key=lambda row: row[0])
    profiles.sort(key=lambda row: int(row["user_id"]))
    # This temporary concat contains uint32 IDs, not Python event objects.
    # Its lifetime ends before the remapped user arrays are constructed.
    catalog = np.unique(np.concatenate([row[1] for row in selected])).astype(np.int64)
    if catalog.size >= np.iinfo(np.int32).max:
        raise ValueError("catalogue cannot be represented with int32 mapped IDs")
    users: list[LongHistoryUser] = []
    for index, (user_id, items, timestamps) in enumerate(selected):
        mapped = (np.searchsorted(catalog, items) + 1).astype(np.int32)
        user = LongHistoryUser(user_id=user_id, train=mapped, timestamps=timestamps)
        users.append(user)
        # Release external per-user IDs as mapping progresses, leaving only
        # mapped training arrays and observed training timestamps in memory.
        selected[index] = (user_id, np.empty(0, dtype=np.uint32), timestamps)
    train_lengths = np.asarray([user.train.size for user in users], dtype=np.int64)
    summary: dict[str, Any] = {
        **_cohort_content_summary(tuple(users), catalog),
        "source_path": str(source_path), "source_size_bytes": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns, "source_user_count": len(seen_users),
        "source_event_count": source_event_count, "source_train_event_count": source_train_events,
        "source_min_timestamp": source_min_timestamp, "source_max_timestamp": source_max_timestamp,
        "timestamp_unit": "seconds", "train_end_timestamp": train_end,
        "train_boundary": "timestamp < train_end_timestamp", "max_seq_len": window,
        "min_train_events": minimum,
        "full_shifted_window_endings": int(np.sum(train_lengths - window)),
        "source_users_time_sorted": sorted_users, "item_mapping": "sorted_train_external_ids_to_1_based_int32",
        "catalog_source": "eligible_users_events_strictly_before_train_end",
        "all_listens_retained": True, "held_out_targets_inspected": False,
        "stream_batch_rows": batch_rows,
    }
    cohort = LongHistoryCohort(users=tuple(users), catalog_item_ids=catalog, summary=summary)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "cohort-summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pq.write_table(pa.Table.from_pylist(profiles), output / "per-user-profile.parquet", compression="zstd")
    return cohort


def write_prepared_cohort(
    cohort: LongHistoryCohort, directory: str | Path, *, batch_size: int = 16,
) -> dict[str, Any]:
    """Write compact train-only arrays for a private Kaggle profiling dataset.

    Parquet is streamed in small groups. No validation/test rows or source event
    IDs outside the training prefix are persisted. A content-checked summary
    records byte hashes for both files as well as canonical cohort signatures.
    """
    _validate_cohort_content(cohort)
    batch_rows = _positive_integer(batch_size, "batch_size")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long-history preparation requires pyarrow") from exc
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("uid", pa.int64()), pa.field("item_id", pa.list_(pa.int32())),
        pa.field("timestamp", pa.list_(pa.uint32())),
    ], metadata={b"rhpr_long_history_format": b"1", b"item_id_domain": b"mapped_1_based_pad_0"})
    train_path, catalog_path = output / "train.parquet", output / "catalog.parquet"
    with pq.ParquetWriter(train_path, schema, compression="zstd", use_dictionary=False) as writer:
        for start in range(0, len(cohort.users), batch_rows):
            group = cohort.users[start:start + batch_rows]
            lengths = np.asarray([user.train.size for user in group], dtype=np.int32)
            offsets = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
            ids = pa.ListArray.from_arrays(pa.array(offsets), pa.array(np.concatenate([user.train for user in group])))
            times = pa.ListArray.from_arrays(pa.array(offsets), pa.array(np.concatenate([user.timestamps for user in group])))
            writer.write_table(pa.Table.from_arrays([pa.array([user.user_id for user in group], type=pa.int64()),
                                                    ids, times], schema=schema))
    pq.write_table(pa.table({"external_item_id": cohort.catalog_item_ids}), catalog_path,
                   compression="zstd", use_dictionary=False)
    summary = {
        **cohort.summary, "prepared_format_version": 1,
        "prepared_files": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in (train_path, catalog_path)
        },
    }
    (output / "cohort-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_prepared_cohort(directory: str | Path, *, batch_size: int = 16) -> LongHistoryCohort:
    """Verify and load a train-only prepared private dataset from its mount."""
    batch_rows = _positive_integer(batch_size, "batch_size")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long-history preparation requires pyarrow") from exc
    root = Path(directory)
    summary = json.loads((root / "cohort-summary.json").read_text(encoding="utf-8"))
    if summary.get("prepared_format_version") != 1:
        raise ValueError("unsupported prepared long-history format version")
    files = summary.get("prepared_files")
    if not isinstance(files, dict) or set(files) != {"train.parquet", "catalog.parquet"}:
        raise ValueError("prepared file manifest must contain train.parquet and catalog.parquet")
    for name in ("train.parquet", "catalog.parquet"):
        path = root / name
        expected = files[name]
        if not isinstance(expected, dict) or path.stat().st_size != expected.get("bytes"):
            raise ValueError(f"prepared file size mismatch: {name}")
        if _file_sha256(path) != expected.get("sha256"):
            raise ValueError(f"prepared file SHA256 mismatch: {name}")
    catalog_table = pq.read_table(root / "catalog.parquet")
    if catalog_table.column_names != ["external_item_id"] or catalog_table.schema.field(0).type != pa.int64():
        raise ValueError("prepared catalogue schema must be external_item_id:int64")
    catalog = catalog_table.column(0).combine_chunks().to_numpy(zero_copy_only=False).copy()
    parquet = pq.ParquetFile(root / "train.parquet")
    expected_schema = {"uid": pa.int64(), "item_id": pa.list_(pa.int32()), "timestamp": pa.list_(pa.uint32())}
    if parquet.schema_arrow.names != list(expected_schema):
        raise ValueError("prepared training schema must contain uid/item_id/timestamp")
    for name, dtype in expected_schema.items():
        if parquet.schema_arrow.field(name).type != dtype:
            raise ValueError(f"prepared training column type mismatch: {name}")
    users: list[LongHistoryUser] = []
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(expected_schema), use_threads=False):
        for row in range(batch.num_rows):
            uid, items, times = (batch.column(index)[row] for index in range(3))
            if not uid.is_valid or not items.is_valid or not times.is_valid:
                raise ValueError("prepared user/item/timestamp cells must not be null")
            if items.values.null_count or times.values.null_count:
                raise ValueError("prepared event list elements must not be null")
            users.append(LongHistoryUser(int(uid.as_py()), items.values.to_numpy(zero_copy_only=False).copy(),
                                         times.values.to_numpy(zero_copy_only=False).copy()))
    cohort = LongHistoryCohort(tuple(users), catalog, summary)
    _validate_cohort_content(cohort)
    return cohort


def make_long_training_pair(
    user: LongHistoryUser, *, max_seq_len: int, end_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full shifted inputs/targets and input timestamps for one ending.

    ``end_index`` is an exclusive offset into the training prefix, including the
    last next-item target. Valid endings are ``max_seq_len + 1 .. len(train)``.
    Target timestamps are never supplied to the model. Returned arrays are
    copies so a collator cannot accidentally mutate the stored training prefix.
    """
    window = _positive_integer(max_seq_len, "max_seq_len")
    end = _positive_integer(end_index, "end_index")
    if end < window + 1 or end > user.train.size:
        raise ValueError("end_index must provide a full shifted window inside training")
    if user.timestamps.ndim != 1 or user.timestamps.size != user.train.size:
        raise ValueError("training timestamps must align with training events")
    start = end - window - 1
    return (user.train[start:end - 1].copy(), user.train[start + 1:end].copy(),
            user.timestamps[start:end - 1].copy())
