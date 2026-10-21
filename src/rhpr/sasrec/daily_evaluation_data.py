"""Daily next-item anchors with past-only rolling contexts and fixed weights.

The train cohort and its catalogue are unchanged. Every day's first event is
selected before catalogue mapping. Unknown identities retain chronological
slots with item ID zero; observed-slot masks must therefore use sequence length.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rhpr.sasrec.data import resolve_yambda_parquet_path
from rhpr.sasrec.history_features import (
    CATALOGUE_FREQUENCY_FEATURE_FIELDS, HISTORY_FEATURE_FIELDS, history_features,
)
from rhpr.sasrec.long_evaluation_data import COHORT_SIGNATURES, _mapped_ids, _positive_int, _sha256_file
from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser


@dataclass(frozen=True)
class DailyEvaluationUser:
    user_id: int
    future_items: np.ndarray
    future_timestamps: np.ndarray


@dataclass(frozen=True)
class DailyEvaluationAnchor:
    user_id: int
    anchor_id: str
    day: int
    future_index: int
    target: int
    target_timestamp: int
    reference_context_hash: str


@dataclass(frozen=True)
class DailyEvaluationData:
    users: tuple[DailyEvaluationUser, ...]
    anchors: tuple[DailyEvaluationAnchor, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class DailyEvaluationExample:
    user_id: int
    anchor_id: str
    day: int
    context: np.ndarray
    timestamps: np.ndarray
    target: int
    target_timestamp: int  # Diagnostic only, never a model input or routing feature.
    reference_context_hash: str


def daily_context_sha256(user_id: int, context: np.ndarray, timestamps: np.ndarray) -> str:
    """Canonical identity of a full reference context, including occupied OOVs."""
    digest = hashlib.sha256()
    digest.update(np.asarray([user_id, context.size], dtype="<i8").tobytes())
    digest.update(np.asarray(context, dtype="<i4").tobytes())
    digest.update(np.asarray(timestamps, dtype="<u4").tobytes())
    return digest.hexdigest()


def _past_context(
    train: LongHistoryUser, user: DailyEvaluationUser, future_index: int, length: int,
) -> tuple[np.ndarray, np.ndarray]:
    future_start = max(0, future_index - length)
    future_count = future_index - future_start
    train_count = min(int(train.train.size), length - future_count)
    return (
        np.concatenate((train.train[-train_count:] if train_count else train.train[:0],
                        user.future_items[future_start:future_index])),
        np.concatenate((train.timestamps[-train_count:] if train_count else train.timestamps[:0],
                        user.future_timestamps[future_start:future_index])),
    )


def _content_summary(data: DailyEvaluationData) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_events = total_unknown = 0
    for user in data.users:
        digest.update(np.asarray([user.user_id, user.future_items.size], dtype="<i8").tobytes())
        digest.update(np.asarray(user.future_items, dtype="<i4").tobytes())
        digest.update(np.asarray(user.future_timestamps, dtype="<u4").tobytes())
        total_events += int(user.future_items.size)
        total_unknown += int(np.count_nonzero(user.future_items == 0))
    for anchor in data.anchors:
        digest.update(np.asarray([anchor.user_id, anchor.day, anchor.future_index,
                                  anchor.target, anchor.target_timestamp], dtype="<i8").tobytes())
        digest.update(bytes.fromhex(anchor.reference_context_hash))
    result: dict[str, Any] = {
        "cohort_users": len(data.users), "future_events": total_events,
        "future_oov_events": total_unknown, "evaluation_content_sha256": digest.hexdigest(),
    }
    seconds = data.summary["day_seconds"]
    for split, low, high in (
        ("validation", data.summary["train_end_timestamp"], data.summary["validation_end_timestamp"]),
        ("test", data.summary["validation_end_timestamp"], data.summary["test_end_timestamp"]),
    ):
        selected = [anchor for anchor in data.anchors if low <= anchor.target_timestamp < high]
        uid_counts = {user.user_id: 0 for user in data.users}
        for anchor in selected:
            uid_counts[anchor.user_id] += 1
        counts = np.asarray(list(uid_counts.values()), dtype=np.int32)
        result.update({
            split + "_anchors": len(selected),
            split + "_users": int(np.count_nonzero(counts)),
            split + "_oov_targets": sum(anchor.target == 0 for anchor in selected),
            split + "_warm_targets": sum(anchor.target > 0 for anchor in selected),
            split + "_anchors_per_user_buckets": {str(i): int(np.count_nonzero(counts == i))
                                                   for i in range((high - low) // seconds + 1)},
            split + "_daily_support": [{
                "day": day, "anchors": sum(anchor.day == day for anchor in selected),
                "oov_targets": sum(anchor.day == day and anchor.target == 0 for anchor in selected),
            } for day in range(low // seconds, high // seconds)],
        })
    return result


def _validate_content(data: DailyEvaluationData, cohort: LongHistoryCohort) -> None:
    summary = data.summary
    if summary.get("daily_evaluation_format_version") != 1:
        raise ValueError("unsupported daily evaluation format version")
    for key in COHORT_SIGNATURES:
        if summary.get(key) != cohort.summary.get(key):
            raise ValueError(f"daily evaluation cohort mismatch: {key}")
    train_end = _positive_int(summary.get("train_end_timestamp"), "train_end_timestamp")
    validation_end = _positive_int(summary.get("validation_end_timestamp"), "validation_end_timestamp")
    test_end = _positive_int(summary.get("test_end_timestamp"), "test_end_timestamp")
    seconds = _positive_int(summary.get("day_seconds"), "day_seconds")
    reference_length = _positive_int(summary.get("reference_context_length"), "reference_context_length")
    if (train_end != cohort.summary.get("train_end_timestamp") or not train_end < validation_end < test_end
            or any(boundary % seconds for boundary in (train_end, validation_end, test_end))):
        raise ValueError("daily evaluation boundaries must be aligned days after the training cutoff")
    if summary.get("anchor_selection") != "first_chronological_event_per_user_per_day_before_catalog_mapping":
        raise ValueError("unsupported daily anchor selection")
    if summary.get("unknown_context_policy") != "mapped_zero_embedding_keep_observed_slots":
        raise ValueError("unsupported daily unknown context policy")
    if [user.user_id for user in data.users] != [user.user_id for user in cohort.users]:
        raise ValueError("daily evaluation must retain every training user in order")
    expected_keys: list[tuple[int, int, int]] = []
    for user in data.users:
        items, times = user.future_items, user.future_timestamps
        if items.ndim != 1 or items.dtype != np.dtype("int32"):
            raise ValueError("future item arrays must be one-dimensional int32")
        if times.dtype != np.dtype("uint32") or times.shape != items.shape:
            raise ValueError("future timestamps must be aligned uint32 arrays")
        if items.size:
            if (items.min() < 0 or items.max() > cohort.num_items or times[0] < train_end
                    or times[-1] >= test_end or np.any(times[1:] < times[:-1])):
                raise ValueError("future events must be chronological and inside evaluation periods")
            days, indices = np.unique(times.astype(np.int64) // seconds, return_index=True)
            expected_keys.extend((user.user_id, int(day), int(index)) for day, index in zip(days, indices))
    if [(a.user_id, a.day, a.future_index) for a in data.anchors] != expected_keys:
        raise ValueError("daily anchors must be each user's first event of every available day")
    trains = {user.user_id: user for user in cohort.users}
    users = {user.user_id: user for user in data.users}
    for anchor in data.anchors:
        user, train = users[anchor.user_id], trains[anchor.user_id]
        if (anchor.anchor_id != f"{anchor.user_id}:{anchor.day}"
                or type(anchor.target) is not int or not 0 <= anchor.target <= cohort.num_items
                or anchor.target != int(user.future_items[anchor.future_index])
                or anchor.target_timestamp != int(user.future_timestamps[anchor.future_index])):
            raise ValueError("daily anchor identity, target or timestamp mismatch")
        items, times = _past_context(train, user, anchor.future_index, reference_length)
        if times.size and int(times[-1]) >= anchor.target_timestamp:
            raise ValueError("daily reference context must be strictly before its anchor timestamp")
        if daily_context_sha256(anchor.user_id, items, times) != anchor.reference_context_hash:
            raise ValueError("daily reference context hash mismatch")
    for key, value in _content_summary(data).items():
        if summary.get(key) != value:
            raise ValueError(f"daily evaluation content disagrees with summary: {key}")


def prepare_daily_evaluation_data(
    path: str | Path, cohort: LongHistoryCohort, *, validation_end_timestamp: int,
    test_end_timestamp: int, day_seconds: int = 86_400,
    reference_context_length: int = 16_384, batch_size: int = 32,
) -> DailyEvaluationData:
    """Select first daily events before mapping, strictly checking train identity."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_end = _positive_int(cohort.summary.get("train_end_timestamp"), "train_end_timestamp")
    validation_end = _positive_int(validation_end_timestamp, "validation_end_timestamp")
    test_end = _positive_int(test_end_timestamp, "test_end_timestamp")
    seconds = _positive_int(day_seconds, "day_seconds")
    length = _positive_int(reference_context_length, "reference_context_length")
    batch_rows = _positive_int(batch_size, "batch_size")
    if (not train_end < validation_end < test_end <= np.iinfo(np.uint32).max + 1
            or any(boundary % seconds for boundary in (train_end, validation_end, test_end))):
        raise ValueError("evaluation boundaries must be ordered and aligned to days")
    source = resolve_yambda_parquet_path(path)
    parquet = pq.ParquetFile(source)
    if {"uid", "item_id", "timestamp"}.issubset(parquet.schema_arrow.names):
        columns = ["uid", "item_id", "timestamp"]
    elif {"uid", "item_ids", "timestamps"}.issubset(parquet.schema_arrow.names):
        columns = ["uid", "item_ids", "timestamps"]
    else:
        raise ValueError("expected sequential UID/item/timestamp list columns")
    if not pa.types.is_integer(parquet.schema_arrow.field("uid").type):
        raise ValueError("uid must be an integer column")
    for name in columns[1:]:
        typ = parquet.schema_arrow.field(name).type
        if not (pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ)):
            raise ValueError("item/timestamp must be list columns")
    trains = {user.user_id: user for user in cohort.users}
    selected: dict[int, DailyEvaluationUser] = {}
    selected_daily_offsets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    scanned = reordered = validation_events = test_events = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns, use_threads=False):
        for row in range(batch.num_rows):
            uid_cell = batch.column(0)[row]
            if not uid_cell.is_valid:
                raise ValueError("source uid must not be null")
            uid = int(uid_cell.as_py())
            scanned += 1
            if uid not in trains:
                continue
            if uid in selected:
                raise ValueError("selected source user IDs must be distinct")
            cells = [batch.column(index)[row] for index in (1, 2)]
            if any(not cell.is_valid or cell.values.null_count for cell in cells):
                raise ValueError("selected source cells and event elements must not be null")
            items, times = (cell.values.to_numpy(zero_copy_only=False) for cell in cells)
            if items.shape != times.shape:
                raise ValueError("selected item and timestamp lists must align")
            for values in (items, times):
                if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
                        or (values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint32).max))):
                    raise ValueError("selected source events must be nonnegative uint32-compatible integers")
            if np.any(times[1:] < times[:-1]):
                order = np.argsort(times, kind="stable")
                items, times = items[order], times[order]
                reordered += 1
            ends = np.searchsorted(times, [train_end, validation_end, test_end], side="left")
            train = trains[uid]
            if (int(ends[0]) != train.train.size or not np.array_equal(times[:ends[0]], train.timestamps)
                    or not np.array_equal(items[:ends[0]], cohort.catalog_item_ids[train.train - 1])):
                raise ValueError(f"source training prefix differs from fixed prepared cohort for UID {uid}")
            validation_events += int(ends[1] - ends[0])
            test_events += int(ends[2] - ends[1])
            # Determine anchors from raw chronology before any catalogue lookup.
            selected_daily_offsets[uid] = np.unique(
                times[ends[0]:ends[2]].astype(np.int64) // seconds, return_index=True,
            )
            selected[uid] = DailyEvaluationUser(
                uid, _mapped_ids(cohort.catalog_item_ids, items[ends[0]:ends[2]]),
                times[ends[0]:ends[2]].astype(np.uint32, copy=True),
            )
    if set(selected) != set(trains):
        raise ValueError("source is missing users from the fixed training cohort")
    users = tuple(selected[user.user_id] for user in cohort.users)
    anchors = []
    for user in users:
        days, indices = selected_daily_offsets[user.user_id]
        for day, index in zip(days, indices):
            items, times = _past_context(trains[user.user_id], user, int(index), length)
            anchors.append(DailyEvaluationAnchor(
                user.user_id, f"{user.user_id}:{int(day)}", int(day), int(index),
                int(user.future_items[index]), int(user.future_timestamps[index]),
                daily_context_sha256(user.user_id, items, times),
            ))
    summary = {
        "daily_evaluation_format_version": 1, "timestamp_unit": "seconds", "day_seconds": seconds,
        "anchor_selection": "first_chronological_event_per_user_per_day_before_catalog_mapping",
        "unknown_context_policy": "mapped_zero_embedding_keep_observed_slots",
        "catalog_source": "unchanged_fixed_training_cohort_catalogue",
        "train_end_timestamp": train_end, "validation_end_timestamp": validation_end,
        "test_end_timestamp": test_end, "reference_context_length": length,
        "source_path": str(source), "source_size_bytes": source.stat().st_size,
        "source_users_scanned": scanned, "source_train_prefix_exact_matches": len(users),
        "source_selected_users_time_sorted": reordered, "validation_period_events": validation_events,
        "test_period_events": test_events, "all_listens_retained": True,
        "reference_context_min_length": min((min(length, trains[a.user_id].train.size + a.future_index)
                                              for a in anchors), default=0),
        "reference_context_max_length": max((min(length, trains[a.user_id].train.size + a.future_index)
                                              for a in anchors), default=0),
        "model_weights_frozen_after_training": True, "target_timestamps_are_diagnostics_only": True,
        **{key: cohort.summary[key] for key in COHORT_SIGNATURES},
        **{key: cohort.summary[key] for key in ("source_dataset", "source_revision", "source_file") if key in cohort.summary},
    }
    data = DailyEvaluationData(users, tuple(anchors), summary)
    summary.update(_content_summary(data))
    _validate_content(data, cohort)
    return data


def iter_daily_evaluation_examples(
    cohort: LongHistoryCohort, data: DailyEvaluationData, *, split_name: str, max_seq_len: int,
) -> Iterator[DailyEvaluationExample]:
    """Yield one next-item prediction per occupied day with immutable model weights.

    The reference hash always identifies the full reference window (normally
    16K), including when a Recent128 model receives only its last 128 slots.
    All previous observed evaluation events enter context; no target or later
    event enters its own context. Unknown item IDs remain occupied zero slots.
    """
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be validation or test")
    length = _positive_int(max_seq_len, "max_seq_len")
    if [user.user_id for user in data.users] != [user.user_id for user in cohort.users]:
        raise ValueError("daily evaluation users do not match the fixed training cohort")
    low = data.summary["train_end_timestamp" if split_name == "validation" else "validation_end_timestamp"]
    high = data.summary["validation_end_timestamp" if split_name == "validation" else "test_end_timestamp"]
    trains = {user.user_id: user for user in cohort.users}
    users = {user.user_id: user for user in data.users}
    for anchor in data.anchors:
        if low <= anchor.target_timestamp < high:
            items, times = _past_context(trains[anchor.user_id], users[anchor.user_id], anchor.future_index, length)
            yield DailyEvaluationExample(
                anchor.user_id, anchor.anchor_id, anchor.day, items, times,
                anchor.target, anchor.target_timestamp, anchor.reference_context_hash,
            )


def _future_schema() -> Any:
    import pyarrow as pa
    return pa.schema([pa.field("uid", pa.int64()), pa.field("future_items", pa.list_(pa.int32())),
                      pa.field("future_timestamps", pa.list_(pa.uint32()))])


def _anchor_schema() -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field("user_id", pa.int64()), pa.field("anchor_id", pa.string()), pa.field("day", pa.int32()),
        pa.field("future_index", pa.int32()), pa.field("target", pa.int32()),
        pa.field("target_timestamp", pa.uint32()), pa.field("reference_context_hash", pa.string()),
    ])


def _feature_schema() -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field("user_id", pa.int64()), pa.field("anchor_id", pa.string()), pa.field("day", pa.int32()),
        pa.field("split", pa.string()), pa.field("reference_context_hash", pa.string()),
        *[pa.field(name, pa.float64()) for name in HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS],
    ], metadata={b"feature_policy": b"past_only_train_catalogue_counts_only_no_targets_or_target_timestamps",
                 b"unknown_item_policy": b"zero_id_occupied_observed_chronological_slot"})


def write_prepared_daily_evaluation(
    data: DailyEvaluationData, cohort: LongHistoryCohort, directory: str | Path,
) -> dict[str, Any]:
    """Save future events once, compact anchor records and 20 past-only features."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    _validate_content(data, cohort)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    future_path, anchor_path, feature_path = (root / name for name in (
        "daily-future.parquet", "daily-anchors.parquet", "history-features.parquet"))
    pq.write_table(pa.Table.from_pylist([
        {"uid": user.user_id, "future_items": user.future_items, "future_timestamps": user.future_timestamps}
        for user in data.users], schema=_future_schema()), future_path, compression="zstd", use_dictionary=False)
    pq.write_table(pa.Table.from_pylist([vars(anchor) for anchor in data.anchors], schema=_anchor_schema()),
                   anchor_path, compression="zstd", use_dictionary=False)
    counts = np.zeros(cohort.num_items + 1, dtype=np.int64)
    for user in cohort.users:
        counts += np.bincount(user.train, minlength=counts.size)
    if counts[0] != 0 or int(counts.sum()) != cohort.train_events or np.any(counts[1:] <= 0):
        raise ValueError("training catalogue frequency counts disagree with fixed cohort")
    names = HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS
    train_counts = {user.user_id: int(user.train.size) for user in cohort.users}
    rows = []
    for split in ("validation", "test"):
        for example in iter_daily_evaluation_examples(
            cohort, data, split_name=split, max_seq_len=data.summary["reference_context_length"],
        ):
            values = history_features(example.context, example.timestamps,
                                      train_event_count=train_counts[example.user_id], catalogue_counts=counts)
            if tuple(values) != names or not all(np.isfinite(value) for value in values.values()):
                raise ValueError("daily history feature schema or finite-value contract failed")
            rows.append({"user_id": example.user_id, "anchor_id": example.anchor_id, "day": example.day,
                         "split": split, "reference_context_hash": example.reference_context_hash, **values})
    pq.write_table(pa.Table.from_pylist(rows, schema=_feature_schema()), feature_path,
                   compression="zstd", use_dictionary=False)
    summary = {**data.summary, "prepared_files": {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in (future_path, anchor_path, feature_path)}, "history_features": {
            "rows": len(rows), "feature_fields": list(names),
            "diagnostic_fields": ["user_id", "anchor_id", "day", "split", "reference_context_hash"],
            "targets_or_target_timestamps_in_features": False,
            "catalogue_frequencies_scope": "all_eligible_pre_cutoff_training_user_events_only",
            "catalogue_count_vector_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
            "reference_context_length": data.summary["reference_context_length"],
        }}
    (root / "daily-evaluation-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_prepared_daily_evaluation(directory: str | Path, cohort: LongHistoryCohort) -> DailyEvaluationData:
    """Verify byte signatures, exact daily anchors and their past-context hashes."""
    import pyarrow.parquet as pq
    root = Path(directory)
    summary = json.loads((root / "daily-evaluation-summary.json").read_text(encoding="utf-8"))
    files = summary.get("prepared_files")
    names = {"daily-future.parquet", "daily-anchors.parquet", "history-features.parquet"}
    if not isinstance(files, dict) or set(files) != names:
        raise ValueError("daily evaluation manifest must name the three prepared parquet files")
    for name in sorted(names):
        path, expected = root / name, files[name]
        if (not isinstance(expected, dict) or path.stat().st_size != expected.get("bytes")
                or _sha256_file(path) != expected.get("sha256")):
            raise ValueError(f"daily evaluation file size or SHA256 mismatch: {name}")
    future, anchors = pq.read_table(root / "daily-future.parquet"), pq.read_table(root / "daily-anchors.parquet")
    if not future.schema.equals(_future_schema(), check_metadata=False) or not anchors.schema.equals(_anchor_schema(), check_metadata=False):
        raise ValueError("prepared daily evaluation schema mismatch")
    users = []
    for row in future.to_pylist():
        if (row["uid"] is None or row["future_items"] is None or row["future_timestamps"] is None
                or any(value is None for value in row["future_items"] + row["future_timestamps"])):
            raise ValueError("prepared daily user/event values must not be null")
        users.append(DailyEvaluationUser(row["uid"], np.asarray(row["future_items"], dtype=np.int32),
                                         np.asarray(row["future_timestamps"], dtype=np.uint32)))
    records = anchors.to_pylist()
    if any(any(value is None for value in row.values()) for row in records):
        raise ValueError("prepared daily anchor values must not be null")
    data = DailyEvaluationData(tuple(users), tuple(DailyEvaluationAnchor(**row) for row in records), summary)
    _validate_content(data, cohort)
    feature_path = root / "history-features.parquet"
    if not pq.read_schema(feature_path).equals(_feature_schema(), check_metadata=True):
        raise ValueError("prepared daily history feature schema mismatch")
    feature_manifest = summary.get("history_features")
    if (not isinstance(feature_manifest, dict) or feature_manifest.get("rows") != len(data.anchors)
            or feature_manifest.get("feature_fields") != list(HISTORY_FEATURE_FIELDS + CATALOGUE_FREQUENCY_FEATURE_FIELDS)
            or feature_manifest.get("targets_or_target_timestamps_in_features") is not False
            or feature_manifest.get("reference_context_length") != summary["reference_context_length"]):
        raise ValueError("prepared daily history feature manifest mismatch")
    identifiers = ["user_id", "anchor_id", "day", "split", "reference_context_hash"]
    actual = pq.read_table(feature_path, columns=identifiers).to_pylist()
    expected = []
    for split, low, high in (
        ("validation", summary["train_end_timestamp"], summary["validation_end_timestamp"]),
        ("test", summary["validation_end_timestamp"], summary["test_end_timestamp"]),
    ):
        expected.extend({"user_id": a.user_id, "anchor_id": a.anchor_id, "day": a.day,
                         "split": split, "reference_context_hash": a.reference_context_hash}
                        for a in data.anchors if low <= a.target_timestamp < high)
    if actual != expected:
        raise ValueError("prepared daily history features must align with every anchor exactly")
    return data
