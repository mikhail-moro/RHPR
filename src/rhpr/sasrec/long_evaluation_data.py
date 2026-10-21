"""Fixed global-time next-item anchors for an existing train-only cohort.

The training cohort and catalogue are immutable. The first event in each
evaluation period is selected before checking catalogue support. Unseen targets
map to zero and must receive zero full-catalogue ranking credit; unseen observed
context items also map to zero, preserving their occupied chronological slots.
Their slot validity comes from observed sequence length, not item identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from rhpr.sasrec.data import resolve_yambda_parquet_path
from rhpr.sasrec.long_history_data import LongHistoryCohort


COHORT_SIGNATURES = (
    "cohort_uid_length_sha256", "cohort_train_timestamps_sha256",
    "cohort_mapped_train_items_sha256", "catalog_sha256",
)


@dataclass(frozen=True)
class LongEvaluationUser:
    user_id: int
    validation_target: int | None
    validation_timestamp: int | None
    test_target: int | None
    test_timestamp: int | None
    # All observed post-training events before the first test-period event.
    test_history: np.ndarray
    test_history_timestamps: np.ndarray


@dataclass(frozen=True)
class LongEvaluationData:
    users: tuple[LongEvaluationUser, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class LongEvaluationExample:
    user_id: int
    context: np.ndarray
    timestamps: np.ndarray
    target: int  # 0 means an unsupported external item, never a candidate.
    target_timestamp: int  # Diagnostic only; never supplied to the model.


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapped_ids(catalog: np.ndarray, external_ids: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(catalog, external_ids)
    known = indices < catalog.size
    valid = np.flatnonzero(known)
    known[valid] = catalog[indices[valid]] == external_ids[valid]
    result = np.zeros(external_ids.size, dtype=np.int32)
    result[known] = (indices[known] + 1).astype(np.int32)
    return result


def _content_summary(users: tuple[LongEvaluationUser, ...]) -> dict[str, Any]:
    digest = hashlib.sha256()
    result: dict[str, Any] = {
        "cohort_users": len(users), "validation_users": 0, "test_users": 0,
        "validation_oov_targets": 0, "test_oov_targets": 0,
        "both_periods_users": 0, "test_history_events": 0,
        "test_history_oov_events": 0,
    }
    for user in users:
        fields = [user.user_id, user.test_history.size,
                  -1 if user.validation_target is None else user.validation_target,
                  -1 if user.validation_timestamp is None else user.validation_timestamp,
                  -1 if user.test_target is None else user.test_target,
                  -1 if user.test_timestamp is None else user.test_timestamp]
        digest.update(np.asarray(fields, dtype="<i8").tobytes())
        digest.update(np.asarray(user.test_history, dtype="<i4").tobytes())
        digest.update(np.asarray(user.test_history_timestamps, dtype="<u4").tobytes())
        for name in ("validation", "test"):
            target = getattr(user, name + "_target")
            result[name + "_users"] += int(target is not None)
            result[name + "_oov_targets"] += int(target == 0)
        result["both_periods_users"] += int(user.validation_target is not None and user.test_target is not None)
        result["test_history_events"] += int(user.test_history.size)
        result["test_history_oov_events"] += int(np.count_nonzero(user.test_history == 0))
    for name in ("validation", "test"):
        count = result[name + "_users"]
        result[name + "_warm_targets"] = count - result[name + "_oov_targets"]
        result[name + "_target_support_fraction"] = result[name + "_warm_targets"] / count if count else 0.0
        result[name + "_evaluable_user_fraction"] = count / len(users) if users else 0.0
    result["evaluation_content_sha256"] = digest.hexdigest()
    return result


def _validate_content(data: LongEvaluationData, cohort: LongHistoryCohort) -> None:
    summary = data.summary
    if summary.get("evaluation_format_version") != 1:
        raise ValueError("unsupported long evaluation format version")
    for key in COHORT_SIGNATURES:
        if summary.get(key) != cohort.summary.get(key):
            raise ValueError(f"evaluation cohort mismatch: {key}")
    train_end = _positive_int(summary.get("train_end_timestamp"), "train_end_timestamp")
    validation_end = _positive_int(summary.get("validation_end_timestamp"), "validation_end_timestamp")
    test_end = _positive_int(summary.get("test_end_timestamp"), "test_end_timestamp")
    if train_end != cohort.summary.get("train_end_timestamp") or not train_end < validation_end < test_end:
        raise ValueError("evaluation boundaries must follow the existing training cutoff")
    if summary.get("anchor_selection") != "first_chronological_event_in_period_before_catalog_mapping":
        raise ValueError("unsupported evaluation anchor selection")
    if summary.get("unknown_context_policy") != "mapped_zero_embedding_keep_observed_slots":
        raise ValueError("unsupported unknown context policy")
    if [user.user_id for user in data.users] != [user.user_id for user in cohort.users]:
        raise ValueError("evaluation must retain every fixed training-cohort user in order")
    for user in data.users:
        for name, low, high in (("validation", train_end, validation_end), ("test", validation_end, test_end)):
            target, timestamp = getattr(user, name + "_target"), getattr(user, name + "_timestamp")
            if (target is None) != (timestamp is None):
                raise ValueError("missing target and timestamp must agree")
            if target is not None and (type(target) is not int or not 0 <= target <= cohort.num_items
                                       or type(timestamp) is not int or not low <= timestamp < high):
                raise ValueError("evaluation target or timestamp is outside its domain")
        items, timestamps = user.test_history, user.test_history_timestamps
        if items.ndim != 1 or items.dtype != np.dtype("int32"):
            raise ValueError("evaluation context history must be one-dimensional int32")
        if timestamps.ndim != 1 or timestamps.dtype != np.dtype("uint32") or timestamps.shape != items.shape:
            raise ValueError("evaluation context timestamps must be aligned uint32")
        if items.size:
            if (items.min() < 0 or items.max() > cohort.num_items or timestamps[0] < train_end
                    or timestamps[-1] >= validation_end or np.any(timestamps[1:] < timestamps[:-1])):
                raise ValueError("test context bridge must contain chronological validation-period events only")
            if user.test_target is None or user.validation_target is None:
                raise ValueError("stored context bridge requires both evaluation anchors")
            if int(items[0]) != user.validation_target or int(timestamps[0]) != user.validation_timestamp:
                raise ValueError("context bridge must begin with the first validation anchor")
        elif user.test_target is not None and user.validation_target is not None:
            raise ValueError("test context must include the observed validation anchor")
    for key, value in _content_summary(data.users).items():
        if summary.get(key) != value:
            raise ValueError(f"evaluation content disagrees with summary: {key}")


def prepare_long_evaluation_data(
    path: str | Path, cohort: LongHistoryCohort, *, validation_end_timestamp: int,
    test_end_timestamp: int, batch_size: int = 32,
) -> LongEvaluationData:
    """Stream only readable UID/item/timestamp columns and verify train identity.

    Validation is the first event in [train_end, validation_end); test is the
    first in [validation_end, test_end). The model remains trained only on the
    fixed pre-train_end cohort. Observed intervening validation-period events
    update test context, including unseen item slots, without updating weights.
    All source listens and repeated items are retained without positive filtering.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long evaluation preparation requires pyarrow") from exc
    train_end = _positive_int(cohort.summary.get("train_end_timestamp"), "train_end_timestamp")
    validation_end = _positive_int(validation_end_timestamp, "validation_end_timestamp")
    test_end = _positive_int(test_end_timestamp, "test_end_timestamp")
    if not train_end < validation_end < test_end <= np.iinfo(np.uint32).max + 1:
        raise ValueError("evaluation boundaries must be ordered supported seconds after training")
    batch_rows = _positive_int(batch_size, "batch_size")
    source = resolve_yambda_parquet_path(path)
    parquet = pq.ParquetFile(source)
    names = parquet.schema_arrow.names
    if {"uid", "item_id", "timestamp"}.issubset(names):
        columns = ["uid", "item_id", "timestamp"]
    elif {"uid", "item_ids", "timestamps"}.issubset(names):
        columns = ["uid", "item_ids", "timestamps"]
    else:
        raise ValueError("expected sequential UID/item/timestamp list columns")
    if not pa.types.is_integer(parquet.schema_arrow.field("uid").type):
        raise ValueError("uid must be an integer column")
    for name in columns[1:]:
        typ = parquet.schema_arrow.field(name).type
        if not (pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ)):
            raise ValueError("item/timestamp must be list columns")
    train_users = {user.user_id: user for user in cohort.users}
    selected: dict[int, LongEvaluationUser] = {}
    scanned = reordered_count = validation_events = test_events = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns, use_threads=False):
        for row in range(batch.num_rows):
            uid_cell = batch.column(0)[row]
            if not uid_cell.is_valid:
                raise ValueError("source uid must not be null")
            uid = int(uid_cell.as_py())
            scanned += 1
            if uid not in train_users:
                continue
            if uid in selected:
                raise ValueError("selected source user IDs must be distinct")
            cells = [batch.column(index)[row] for index in (1, 2)]
            if any(not cell.is_valid or cell.values.null_count for cell in cells):
                raise ValueError("selected item/timestamp cells and elements must not be null")
            items, times = (cell.values.to_numpy(zero_copy_only=False) for cell in cells)
            if items.shape != times.shape:
                raise ValueError("selected item and timestamp lists must align")
            for array in (items, times):
                if (array.ndim != 1 or not np.issubdtype(array.dtype, np.integer)
                        or (array.size and (array.min() < 0 or array.max() > np.iinfo(np.uint32).max))):
                    raise ValueError("selected source events must be nonnegative uint32-compatible integers")
            if np.any(times[1:] < times[:-1]):
                order = np.argsort(times, kind="stable")
                items, times = items[order], times[order]
                reordered_count += 1
            ends = np.searchsorted(times, [train_end, validation_end, test_end], side="left")
            train = train_users[uid]
            if (int(ends[0]) != train.train.size or not np.array_equal(times[:ends[0]], train.timestamps)
                    or not np.array_equal(items[:ends[0]], cohort.catalog_item_ids[train.train - 1])):
                raise ValueError(f"source training prefix differs from fixed prepared cohort for UID {uid}")
            validation_events += int(ends[1] - ends[0])
            test_events += int(ends[2] - ends[1])
            val_exists, test_exists = ends[0] < ends[1], ends[1] < ends[2]
            val_target = (int(_mapped_ids(cohort.catalog_item_ids, items[ends[0]:ends[0] + 1])[0])
                          if val_exists else None)
            test_target = (int(_mapped_ids(cohort.catalog_item_ids, items[ends[1]:ends[1] + 1])[0])
                           if test_exists else None)
            bridge = _mapped_ids(cohort.catalog_item_ids, items[ends[0]:ends[1]]) if test_exists else np.empty(0, dtype=np.int32)
            bridge_times = times[ends[0]:ends[1]].astype(np.uint32, copy=True) if test_exists else np.empty(0, dtype=np.uint32)
            selected[uid] = LongEvaluationUser(
                uid, val_target, int(times[ends[0]]) if val_exists else None,
                test_target, int(times[ends[1]]) if test_exists else None, bridge, bridge_times,
            )
    if set(selected) != set(train_users):
        raise ValueError("source is missing users from the fixed training cohort")
    users = tuple(selected[user.user_id] for user in cohort.users)
    summary = {
        "evaluation_format_version": 1, "timestamp_unit": "seconds",
        "anchor_selection": "first_chronological_event_in_period_before_catalog_mapping",
        "unknown_context_policy": "mapped_zero_embedding_keep_observed_slots",
        "catalog_source": "unchanged_fixed_training_cohort_catalogue",
        "train_end_timestamp": train_end, "validation_end_timestamp": validation_end,
        "test_end_timestamp": test_end, "source_path": str(source),
        "source_size_bytes": source.stat().st_size, "source_users_scanned": scanned,
        "source_train_prefix_exact_matches": len(users), "source_selected_users_time_sorted": reordered_count,
        "validation_period_events": validation_events, "test_period_events": test_events,
        "all_listens_retained": True, "model_weights_frozen_after_training": True,
        **{key: cohort.summary[key] for key in COHORT_SIGNATURES},
        **{key: cohort.summary[key] for key in ("source_dataset", "source_revision", "source_file") if key in cohort.summary},
        **_content_summary(users),
    }
    data = LongEvaluationData(users, summary)
    _validate_content(data, cohort)
    return data


def write_prepared_long_evaluation(
    data: LongEvaluationData, cohort: LongHistoryCohort, directory: str | Path,
) -> dict[str, Any]:
    """Write small supplementary private artifacts, reusing the train dataset."""
    _validate_content(data, cohort)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long evaluation serialization requires pyarrow") from exc
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("uid", pa.int64()), pa.field("validation_target", pa.int32()),
        pa.field("validation_timestamp", pa.uint32()), pa.field("test_target", pa.int32()),
        pa.field("test_timestamp", pa.uint32()), pa.field("test_history", pa.list_(pa.int32())),
        pa.field("test_history_timestamps", pa.list_(pa.uint32())),
    ], metadata={b"rhpr_long_evaluation_format": b"1", b"unknown_id": b"0_zero_embedding_occupied_event_slot"})
    records = [{
        "uid": user.user_id, "validation_target": user.validation_target,
        "validation_timestamp": user.validation_timestamp, "test_target": user.test_target,
        "test_timestamp": user.test_timestamp, "test_history": user.test_history,
        "test_history_timestamps": user.test_history_timestamps,
    } for user in data.users]
    path = root / "evaluation.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd", use_dictionary=False)
    summary = {**data.summary, "prepared_file": {"name": path.name, "bytes": path.stat().st_size,
                                                "sha256": _sha256_file(path)}}
    (root / "evaluation-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_prepared_long_evaluation(directory: str | Path, cohort: LongHistoryCohort) -> LongEvaluationData:
    """Check byte integrity, train identity, domains and canonical signatures."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("long evaluation loading requires pyarrow") from exc
    root = Path(directory)
    summary = json.loads((root / "evaluation-summary.json").read_text(encoding="utf-8"))
    manifest = summary.get("prepared_file")
    if not isinstance(manifest, dict) or manifest.get("name") != "evaluation.parquet":
        raise ValueError("evaluation manifest must name evaluation.parquet")
    path = root / "evaluation.parquet"
    if path.stat().st_size != manifest.get("bytes") or _sha256_file(path) != manifest.get("sha256"):
        raise ValueError("evaluation file size or SHA256 mismatch")
    table = pq.read_table(path)
    expected = {"uid": pa.int64(), "validation_target": pa.int32(), "validation_timestamp": pa.uint32(),
                "test_target": pa.int32(), "test_timestamp": pa.uint32(), "test_history": pa.list_(pa.int32()),
                "test_history_timestamps": pa.list_(pa.uint32())}
    if table.column_names != list(expected) or any(table.schema.field(name).type != dtype for name, dtype in expected.items()):
        raise ValueError("prepared evaluation schema mismatch")
    users = []
    for row in table.to_pylist():
        if row["uid"] is None or row["test_history"] is None or row["test_history_timestamps"] is None:
            raise ValueError("prepared evaluation UID and history lists must not be null")
        if any(value is None for value in row["test_history"] + row["test_history_timestamps"]):
            raise ValueError("prepared evaluation history elements must not be null")
        users.append(LongEvaluationUser(
            row["uid"], row["validation_target"], row["validation_timestamp"], row["test_target"], row["test_timestamp"],
            np.asarray(row["test_history"], dtype=np.int32), np.asarray(row["test_history_timestamps"], dtype=np.uint32),
        ))
    data = LongEvaluationData(tuple(users), summary)
    _validate_content(data, cohort)
    return data


def iter_long_evaluation_examples(
    cohort: LongHistoryCohort, data: LongEvaluationData, *, split_name: str, max_seq_len: int,
) -> Iterator[LongEvaluationExample]:
    """Yield cropped past-only contexts; target timestamps remain diagnostic.

    Every observed event retains a slot, even if its external item is absent
    from the fixed training catalogue (mapped ID 0). Consumers must keep these
    occupied slots valid, using the frozen zero item-embedding row. Padding is
    defined by sequence length; ``context != 0`` is not an observed-slot mask
    when unknown items are present.
    """
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be validation or test")
    length = _positive_int(max_seq_len, "max_seq_len")
    if [user.user_id for user in data.users] != [user.user_id for user in cohort.users]:
        raise ValueError("evaluation users do not match fixed training cohort")
    for train, user in zip(cohort.users, data.users, strict=True):
        target = getattr(user, split_name + "_target")
        if target is None:
            continue
        if split_name == "test" and user.test_history.size:
            keep_train = max(0, length - user.test_history.size)
            items = np.concatenate((train.train[-keep_train:] if keep_train else train.train[:0],
                                    user.test_history[-length:]))
            times = np.concatenate((train.timestamps[-keep_train:] if keep_train else train.timestamps[:0],
                                    user.test_history_timestamps[-length:]))
        else:
            items, times = train.train[-length:].copy(), train.timestamps[-length:].copy()
        yield LongEvaluationExample(user.user_id, items, times, target, getattr(user, split_name + "_timestamp"))
