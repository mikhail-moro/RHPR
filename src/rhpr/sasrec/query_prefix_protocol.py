"""Fixed last-block SASRec supervision with an old-suffix-conditioned memory.

For each user's complete train prefix T, X=T[:-1], Y=T[1:].  The compressor
reads H=X[:-128] and obtains queries from H[-128:], while R=X[-128:] supplies
128 causal next-item losses.  Passes reshuffle users and resample negatives;
they deliberately repeat the same final targets rather than sampling windows.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch

from rhpr.sasrec.long_history_data import LongHistoryCohort
from rhpr.sasrec.long_quality_schedule import QualityWindow, quality_negative_ids
from rhpr.sasrec.model import SASRecModel, TailOnlySequenceCompressor


PILOT_KIND = "yambda_sasrec_prefix_query_pilot"
SCHEDULE_VERSION = "sasrec-full-prefix-fixed-last128-v1"
QUERY_KINDS = ("query_cross_attention", "query_ssm")


def prefix_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    k = config["summary_budgets"][0]
    return [{"name": "recent128", "kind": "recent", "summary_tokens": 0}] + [
        {"name": f"{kind}_m{k}", "kind": kind, "summary_tokens": k} for kind in QUERY_KINDS
    ]


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _digest(domain: str, *identifiers: int) -> bytes:
    return hashlib.sha256(json.dumps([SCHEDULE_VERSION, domain, *identifiers],
                                    separators=(",", ":")).encode("ascii")).digest()


def validate_prefix_config(config: dict[str, Any]) -> None:
    """Fail closed on protocol drift before touching GPU or prepared data."""
    if config.get("kind") != PILOT_KIND:
        raise ValueError(f"kind must be {PILOT_KIND}")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name is required")
    _positive_integer(config.get("seed"), "seed", minimum=0)
    data, dims, train, compressor, bench = [config.get(key, {}) for key in
                                           ("data", "model", "training", "compressor", "benchmark")]
    for section, keys in ((dims, ("d_model", "n_heads", "n_layers", "ffn_multiplier")),
                          (train, ("raw_context_length", "serving_context_length", "raw_tail_size",
                                   "supervised_targets", "query_context_size", "effective_batch_size",
                                   "num_negatives", "min_passes", "max_passes", "validation_interval",
                                   "early_stopping_patience", "early_stopping_min_passes", "cpu_threads")),
                          (compressor, ("d_inner", "d_state", "gate_rank", "query_context_size")),
                          (bench, ("evaluation_batch_size", "item_chunk_size", "iterations"))):
        for key in keys:
            _positive_integer(section.get(key), key)
    if config.get("summary_budgets") != [4]:
        raise ValueError("first full-prefix pilot fixes K=4")
    if (train["raw_context_length"] != 32768 or train["serving_context_length"] != 32768
            or train.get("max_seq_len", 32768) != 32768 or "old_prefix_length" in train):
        raise ValueError("full prefixes use a 32768 safety cap without fixed old_prefix_length")
    if any(train[key] != 128 for key in ("raw_tail_size", "supervised_targets", "query_context_size")):
        raise ValueError("fixed last128 input states/targets and old-suffix query128 are required")
    if compressor["query_context_size"] != 128 or compressor.get("query_source") != "old_suffix":
        raise ValueError("queries must read the last128 events of the old prefix")
    if (dims["d_model"] != 128 or dims["n_layers"] != 2 or dims["n_heads"] != 4
            or dims["ffn_multiplier"] != 4 or dims.get("use_time_features") is not False):
        raise ValueError("first pilot fixes SASRec d128/L2/heads4/FFN4 and Time OFF")
    if train.get("device") != "cuda" or type(train.get("amp")) is not bool:
        raise ValueError("native CUDA and explicit boolean AMP are required")
    for section, key in ((dims, "dropout"), (compressor, "dropout")):
        value = section.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError("dropout must lie in [0,1)")
    if compressor.get("dt_min") != 1e-4 or compressor.get("dt_max") != .1:
        raise ValueError("the native SSM initialization must remain pinned at1e-4..0.1")
    for key in ("learning_rate", "grad_clip_norm", "initial_loss_scale", "budget_seconds", "budget_safety_factor"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    value = train.get("weight_decay")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if train["budget_safety_factor"] < 1 or not 30 <= train["min_passes"] <= train["max_passes"] <= 100:
        raise ValueError("the bounded pilot requires30..100 passes and safety_factor>=1")
    if not train["min_passes"] <= train["early_stopping_min_passes"] <= train["max_passes"]:
        raise ValueError("early stopping must respect the registered minimum training passes")
    for values, cap in ((train.get("physical_batch_sizes"), train["effective_batch_size"]),
                        (bench.get("batch_sizes"), None)):
        if (not isinstance(values, list) or not values or len(values) != len(set(values))
                or any(type(value) is not int or value < 1 or cap and cap % value for value in values)):
            raise ValueError("batch sizes must be unique positive integers and training divisors")
    if (type(bench.get("warmup")) is not int or bench["warmup"] < 0
            or bench.get("quality_cutoffs") != [10, 100]
            or not any(1 < batch <= bench["evaluation_batch_size"] for batch in bench["batch_sizes"])):
        raise ValueError("full-catalogue @10/@100 and a parallel benchmark batch are required")
    if data.get("evaluation_protocol") != "first_listen_per_user_day_v1":
        raise ValueError("the existing globally split daily evaluation protocol must stay unchanged")
    for key in ("prepared_path", "evaluation_path"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} is required")
    for key in ("evaluation_content_sha256", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256"):
        value = data.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"data.{key} must be a SHA256 digest")
    for key in ("train_end_timestamp", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        _positive_integer(data.get(key), key)
    if not data["train_end_timestamp"] < data["validation_end_timestamp"] < data["test_end_timestamp"]:
        raise ValueError("global daily split boundaries must increase")
    expected_sources = ["carasik/rhpr-yambda500m-train16k-profile", "carasik/rhpr-yambda500m-daily-evaluation",
                        "carasik/rhpr-cp313-torch210-mamba-runtime"]
    if config.get("kaggle", {}).get("dataset_sources") != expected_sources:
        raise ValueError("the three previously verified private dataset sources must remain pinned")
    runtime = config.get("runtime", {})
    if (runtime.get("target_torch_version") != "2.10.0+cu128" or runtime.get("target_python_minor") != "3.13"
            or runtime.get("manifest_sha256") != "cb1cd1421931548762bd02636771e9927ae7e6993a4dfe7df51810a874baa643"
            or runtime.get("requirements_sha256") != "7403aec00eb0ce74b03e06b062a4e382d54e59e2b9fff94f9f84f949fc6521cb"):
        raise ValueError("the previously verified native Mamba wheel runtime must remain pinned")
    if config.get("variants") != [variant["name"] for variant in prefix_variants(config)]:
        raise ValueError("exactly Recent128, Cross Attention and QSSM are required in fixed order")


def build_prefix_model(config: dict[str, Any], variant: dict[str, Any], num_items: int) -> SASRecModel:
    """Initialize the common SASRec before attaching specially initialized SSM.

    SASRec's blanket Linear initializer would erase the SSM's inverse-softplus
    dt bias if the compressor were passed to its constructor.  Attaching later
    also keeps both common weights and the following dropout RNG identical.
    """
    from rhpr.sasrec.query_ssm import make_query_compressor

    if variant["kind"] not in ("recent", *QUERY_KINDS):
        raise ValueError("unregistered full-prefix pilot variant")
    torch.manual_seed(config["seed"])
    dims, train = config["model"], config["training"]
    model = SASRecModel(num_items=num_items, max_seq_len=train["raw_context_length"],
        **{key: dims[key] for key in ("d_model", "n_heads", "n_layers", "ffn_multiplier", "dropout")},
        use_time_features=False)
    if variant["kind"] == "recent":
        model.compressor = TailOnlySequenceCompressor(train["raw_tail_size"])
    else:
        compressor = dict(config["compressor"])
        compressor["query_source"] = "old_suffix"
        model.compressor = make_query_compressor(kind=variant["kind"], d_model=dims["d_model"],
            summary_tokens=variant["summary_tokens"], tail_size=train["raw_tail_size"],
            n_heads=dims["n_heads"], **compressor)
    model.compressor_type = "custom"
    return model


def fixed_prefix_schedule(cohort: LongHistoryCohort, *, seed: int, epoch: int) -> tuple[QualityWindow, ...]:
    """Keep the entire X=T[:-1]; change only user order and negative draws."""
    seed = _positive_integer(seed, "seed", minimum=0)
    epoch = _positive_integer(epoch, "epoch", minimum=0)
    if not cohort.users or cohort.num_items < 1:
        raise ValueError("a nonempty cohort and catalogue are required")
    ordered, seen = [], set()
    for index, user in enumerate(cohort.users):
        uid = _positive_integer(user.user_id, "user_id", minimum=0)
        if uid in seen:
            raise ValueError("training user IDs must be unique")
        seen.add(uid)
        if user.train.ndim != 1 or len(user.train) < 257:
            raise ValueError(f"user {uid} needs old128 + raw128 + final target")
        end = len(user.train) - 1
        negative_seed = int.from_bytes(_digest("negatives", seed, epoch, uid, end)[:8], "little")
        window = QualityWindow(index, uid, end, epoch, negative_seed)
        ordered.append((_digest("order", seed, epoch, uid), window))
    ordered.sort(key=lambda pair: (pair[0], pair[1].user_id))
    return tuple(window for _, window in ordered)


def prefix_batch_arrays(cohort: LongHistoryCohort, windows: Sequence[QualityWindow],
                        config: dict[str, Any], variant: dict[str, Any]) -> dict[str, np.ndarray]:
    """Right-pad complete histories; masks describe occupied slots, not IDs.

    The Recent control crops before lookup.  Its source positions and aligned
    labels are local to that crop, while its 128 supervised labels and identity-
    bound negative draws equal those of the full-prefix models exactly.
    """
    if not windows:
        raise ValueError("a nonempty physical batch is required")
    if variant["kind"] not in ("recent", *QUERY_KINDS):
        raise ValueError("unregistered full-prefix pilot variant")
    train = config["training"]
    cap = _positive_integer(train["raw_context_length"], "raw_context_length")
    tail = _positive_integer(train["raw_tail_size"], "raw_tail_size")
    rows = []
    for window in windows:
        if not 0 <= window.user_index < len(cohort.users):
            raise ValueError("window user_index is outside the cohort")
        user = cohort.users[window.user_index]
        if user.user_id != window.user_id or window.end != len(user.train) - 1:
            raise ValueError("full-prefix window must end at the final train-only input")
        if window.end > cap:
            raise ValueError("complete train prefix exceeds safety cap; truncation is forbidden")
        if window.end < tail + train["query_context_size"]:
            raise ValueError("train prefix must contain the complete raw tail and old query suffix")
        width = tail if variant["kind"] == "recent" else window.end
        start = window.end - width
        rows.append((user.train[start:window.end], user.train[start + 1:window.end + 1], window))
    width = max(len(row[0]) for row in rows)
    shape = (len(rows), width)
    item_ids = np.zeros(shape, dtype=np.int64)
    targets = np.zeros(shape, dtype=np.int64)
    valid = np.zeros(shape, dtype=np.bool_)
    negative_ids = []
    for index, (inputs, labels, window) in enumerate(rows):
        size = len(inputs)
        item_ids[index, :size], targets[index, :size], valid[index, :size] = inputs, labels, True
        negative_ids.append(quality_negative_ids(window, num_items=cohort.num_items,
                                                 num_negatives=train["num_negatives"]))
    return {"item_ids": item_ids, "valid_mask": valid, "targets": targets,
            "negative_ids": np.asarray(negative_ids, dtype=np.int64)}


def prefix_schedule_summary(cohort: LongHistoryCohort, windows: Sequence[QualityWindow],
                            config: dict[str, Any]) -> dict[str, Any]:
    """Count fixed unique target positions without writing private item arrays."""
    if not windows:
        raise ValueError("a nonempty schedule is required")
    rows = [[window.epoch, window.user_id, window.end, window.negative_seed] for window in windows]
    def fingerprint(records: list[list[int]]) -> str:
        return hashlib.sha256(json.dumps([SCHEDULE_VERSION, records], separators=(",", ":")).encode("ascii")).hexdigest()
    lengths = np.asarray([window.end for window in windows], dtype=np.int64)
    tail = config["training"]["raw_tail_size"]
    target_rows = sorted({(window.user_id, window.end) for window in windows})
    target_sha = hashlib.sha256(json.dumps(target_rows, separators=(",", ":")).encode("ascii")).hexdigest()
    return {"schedule_version": SCHEDULE_VERSION, "schedule_sha256": fingerprint(sorted(rows)),
            "ordered_schedule_sha256": fingerprint(rows), "fixed_target_positions_sha256": target_sha,
            "windows": len(windows), "unique_users": len({window.user_id for window in windows}),
            "epochs": sorted({window.epoch for window in windows}), "unique_endpoints": len(target_rows),
            "supervised_targets_per_window": tail, "target_presentations": len(windows) * tail,
            "unique_target_positions": len(target_rows) * tail,
            "training_input_length_min": int(lengths.min()), "training_input_length_max": int(lengths.max()),
            "training_input_length_mean": float(lengths.mean()),
            "training_old_prefix_length_min": int(lengths.min()) - tail,
            "training_old_prefix_length_max": int(lengths.max()) - tail,
            "training_old_prefix_length_mean": float(lengths.mean()) - tail,
            "training_raw_tail_size": tail, "training_raw_context_min": 1, "training_raw_context_max": tail,
            "query_source": "old_suffix", "full_train_prefix_retained": True,
            "training_targets_are_fixed_last_block": True, "train_is_exhaustive_event_epoch": False,
            "available_train_transitions": sum(len(user.train) - 1 for user in cohort.users)}
