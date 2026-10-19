"""Matched train-only rotating blocks for the selected QSSM and native FuXi.

A pass retains one 1152-input window and 128 causal next-item losses per user.
The first pass is the historical latest block; later passes visit disjoint
128-target blocks in a deterministic user-specific order without replacement.
Only the training endings change. Daily validation and catalogue remain pinned.
"""
from __future__ import annotations

from collections.abc import Sequence
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rhpr.sasrec.long_quality_schedule import QualityWindow, quality_negative_ids
from rhpr.sasrec.query_history_checkpoints import (
    load_history_source_weights, prepare_history_source,
)
from rhpr.sasrec.query_history_protocol import validate_history_config
from rhpr.sasrec.query_prefix_protocol import _digest as _prefix_digest


ROTATING_KIND = "yambda_query_rotating_windows"
SCHEDULE_VERSION = "qssm-fuxi-rotating-block128-v1"
_COMMON = ("benchmark", "compressor", "data", "model", "parent", "runtime", "seed")
_SOURCE_DIGESTS = {
    "qssm": "e91882fc9a684e192dfdbc6accb5678380668d344c8cc49148cac03d877de5fa",
    "fuxi": "ac4ab26c461ded020a7f6d17ca678a1eab115487ad49a450f9126115a47dd020",
}
_VARIANTS = (
    {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32},
    {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0},
)
_TRIALS = tuple({"name": name, "backbone_lr": .001, "compressor_lr": .001}
                for name in ("historical_fixed", "rotating"))
_SCHEDULE = {
    "version": SCHEDULE_VERSION,
    "training_input_tokens": 1152,
    "old_history_tokens": 1024,
    "target_block_size": 128,
    "first_pass": "latest_block",
    "remaining_blocks": "uid_seed_hash_permutation_without_replacement",
    "subsequent_cycles": "new_uid_seed_cycle_hash_permutation",
    "train_only": True,
}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_rotating_config(config: dict[str, Any]) -> None:
    """Reject changes to the registered schedule, models, controls or budget."""
    fields = {*_COMMON, "name", "kind", "history_sources", "summary_budgets",
              "training", "hpo", "variants", "rotating_batches", "schedule", "kaggle"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("the exact registered rotating-window configuration fields are required")
    if (config["kind"] != ROTATING_KIND
            or config["name"] != "yambda500m-query-rotating-windows"):
        raise ValueError("the named QSSM/FuXi rotating-window experiment is required")
    sources = config["history_sources"]
    if not isinstance(sources, dict) or set(sources) != set(_SOURCE_DIGESTS):
        raise ValueError("the exact qssm and fuxi historical controls are required")
    for family, digest in _SOURCE_DIGESTS.items():
        validate_history_config(sources[family])
        if sources[family]["family"] != family or _canonical(sources[family]) != digest:
            raise ValueError("the complete immutable selected history-source registrations are required")
        for key in _COMMON:
            if config[key] != sources[family][key]:
                raise ValueError(f"rotating windows changed source {key}")
    if (config["variants"] != [variant["name"] for variant in _VARIANTS]
            or config["summary_budgets"] != [32]):
        raise ValueError("exactly selected K32/slow QSSM and unchanged native FuXi are registered")
    expected_hpo = {"proxy_old_length": 1024, "min_passes": 20, "max_passes": 60,
        "validation_interval": 10, "patience": 2, "bootstrap_draws": 5000,
        "physical_batch_size": 32, "full_length_validation": False, "test_evaluations": 0,
        "trials": [dict(trial) for trial in _TRIALS]}
    if config["hpo"] != expected_hpo:
        raise ValueError("H1024/min20/max60/validation10/patience2 and two historical/fresh aliases are required")
    original = config["parent"]["experiment_config"]
    expected_training = dict(original["training"], min_passes=20, max_passes=60,
        budget_seconds=10800, early_stopping_min_passes=20, early_stopping_patience=2)
    if config["training"] != expected_training:
        raise ValueError("only registered window supervision and stopping may change parent optimization")
    expected_batches = {variant["name"]: {"training": 32 if index == 0 else 16, "proxy": 32}
                        for index, variant in enumerate(_VARIANTS)}
    if config["rotating_batches"] != expected_batches or config["schedule"] != _SCHEDULE:
        raise ValueError("the exact common block schedule and Q32/F16/evaluation32 batches are required")
    expected_kaggle = dict(original["kaggle"], kernel_sources=[config["parent"]["kernel"],
        sources["qssm"]["selected_source"]["kernel"], sources["fuxi"]["selected_source"]["kernel"]])
    if config["kaggle"] != expected_kaggle:
        raise ValueError("only pinned data/runtime, original pilot and the two selected controls are required")


def rotating_variants(_config: dict[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(list(_VARIANTS))


def _registered(variant: dict[str, Any] | str) -> dict[str, Any]:
    for row in _VARIANTS:
        if variant == row or variant == row["name"]:
            return dict(row)
    raise ValueError("a registered selected QSSM or FuXi variant is required")


def rotating_trials(_config: dict[str, Any], variant: dict[str, Any] | str) -> list[dict[str, Any]]:
    _registered(variant)
    return [dict(trial) for trial in _TRIALS]


def _old_length(value: Any) -> None:
    if type(value) is not int or value != 1024:
        raise ValueError("rotating windows require H1024; no full-history inference is registered")


def rotating_evaluation_batch(config: dict[str, Any], variant: dict[str, Any] | str,
                              old_length: int | None) -> int:
    registered = _registered(variant)
    _old_length(old_length)
    return config["rotating_batches"][registered["name"]]["proxy"]


def _block_digest(seed: int, uid: int, length: int, cycle: int, block: int) -> bytes:
    payload = [SCHEDULE_VERSION, "block", seed, uid, length, cycle, block]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("ascii")).digest()


@lru_cache(maxsize=16384)
def _block_permutation(seed: int, uid: int, length: int, cycle: int) -> tuple[int, ...]:
    """Cache small descriptors only, never private item or timestamp arrays."""
    count = (length - 1025) // 128
    first = 1 if cycle == 0 else 0
    order = sorted(range(first, count),
                   key=lambda block: (_block_digest(seed, uid, length, cycle, block), block))
    return tuple(([0] if cycle == 0 else []) + order)


def rotating_schedule(cohort: Any, *, seed: int, epoch: int) -> tuple[QualityWindow, ...]:
    """One full H1024/R128 input per user; no repeated target block per cycle.

    The initial latest block and the order/negative hash domains exactly retain
    the historical first pass. Subsequent endings alone change the task.
    """
    seed = _integer(seed, "seed")
    epoch = _integer(epoch, "epoch", minimum=1)
    if not cohort.users or cohort.num_items < 1:
        raise ValueError("a nonempty cohort and catalogue are required")
    ordered, seen = [], set()
    for index, user in enumerate(cohort.users):
        uid = _integer(user.user_id, "user_id")
        if uid in seen:
            raise ValueError("training user IDs must be unique")
        seen.add(uid)
        if user.train.ndim != 1 or len(user.train) < 1153:
            raise ValueError(f"user {uid} needs H1024 + R128 + final target")
        count = (len(user.train) - 1025) // 128
        cycle, offset = divmod(epoch - 1, count)
        block = _block_permutation(seed, uid, len(user.train), cycle)[offset]
        end = len(user.train) - 1 - 128 * block
        negative_seed = int.from_bytes(_prefix_digest("negatives", seed, epoch, uid, end)[:8], "little")
        window = QualityWindow(index, uid, end, epoch, negative_seed)
        ordered.append((_prefix_digest("order", seed, epoch, uid), window))
    ordered.sort(key=lambda row: (row[0], row[1].user_id))
    return tuple(window for _, window in ordered)


def _checked_user(cohort: Any, window: QualityWindow) -> Any:
    index = _integer(window.user_index, "window user_index")
    if index >= len(cohort.users):
        raise ValueError("window user_index is outside the cohort")
    user = cohort.users[index]
    end = _integer(window.end, "window end", minimum=1152)
    if user.user_id != window.user_id:
        raise ValueError("window user ID disagrees with the cohort index")
    if user.train.ndim != 1 or end >= len(user.train):
        raise ValueError("window endpoint must remain inside the user's train-only history")
    if (len(user.train) - 1 - end) % 128:
        raise ValueError("window endpoint must belong to the registered disjoint target-block grid")
    _integer(window.epoch, "window epoch", minimum=1)
    _integer(window.negative_seed, "negative_seed")
    return user


def rotating_batch_arrays(cohort: Any, windows: Sequence[QualityWindow], config: dict[str, Any],
                          variant: dict[str, Any] | str, *, old_length: int | None) -> dict[str, np.ndarray]:
    """Read only a 1152-input train slice; both architectures see identical IDs."""
    _registered(variant)
    _old_length(old_length)
    if not windows:
        raise ValueError("a nonempty physical batch is required")
    inputs, labels, negatives = [], [], []
    for window in windows:
        user = _checked_user(cohort, window)
        start = window.end - 1152
        inputs.append(user.train[start:window.end])
        labels.append(user.train[start + 1:window.end + 1])
        negatives.append(quality_negative_ids(window, num_items=cohort.num_items,
                                              num_negatives=config["training"]["num_negatives"]))
    item_ids = np.asarray(inputs, dtype=np.int64)
    return {"item_ids": item_ids, "targets": np.asarray(labels, dtype=np.int64),
            "valid_mask": np.ones(item_ids.shape, dtype=np.bool_),
            "negative_ids": np.asarray(negatives, dtype=np.int64)}


def rotating_schedule_summary(cohort: Any, windows: Sequence[QualityWindow],
                              _config: dict[str, Any]) -> dict[str, Any]:
    """Exact target-position coverage without expanding raw item arrays."""
    if not windows:
        raise ValueError("a nonempty schedule is required")
    identities, slots, user_lengths = [], set(), {}
    for window in windows:
        user = _checked_user(cohort, window)
        identity = (window.epoch, window.user_id)
        if identity in slots:
            raise ValueError("a pass may contain only one window per user")
        slots.add(identity)
        user_lengths[window.user_id] = len(user.train)
        identities.append([window.epoch, window.user_id, window.end, window.negative_seed])
    def fingerprint(records: Any) -> str:
        return hashlib.sha256(json.dumps([SCHEDULE_VERSION, records],
                                        separators=(",", ":")).encode("ascii")).hexdigest()
    endpoints = sorted({(window.user_id, window.end) for window in windows})
    available_blocks = sum((length - 1025) // 128 for length in user_lengths.values())
    transitions = sum(length - 1 for length in user_lengths.values())
    ends = np.asarray([window.end for window in windows], dtype=np.int64)
    age = np.asarray([user_lengths[window.user_id] - 1 - window.end for window in windows], dtype=np.int64)
    return {"schedule_version": SCHEDULE_VERSION,
        "schedule_sha256": fingerprint(sorted(identities)),
        "ordered_schedule_sha256": fingerprint(identities),
        "target_positions_sha256": fingerprint(endpoints),
        "windows": len(windows), "unique_users": len(user_lengths),
        "epochs": sorted({window.epoch for window in windows}),
        "unique_endpoints": len(endpoints), "repeated_endpoint_presentations": len(windows) - len(endpoints),
        "supervised_targets_per_window": 128, "target_presentations": 128 * len(windows),
        "unique_target_positions": 128 * len(endpoints),
        "available_train_transitions": transitions,
        "available_full_target_blocks": available_blocks,
        "available_target_positions_in_full_blocks": 128 * available_blocks,
        "unavailable_initial_context_transitions": 1024 * len(user_lengths),
        "unavailable_block_remainder_transitions": transitions - 1024 * len(user_lengths) - 128 * available_blocks,
        "sampled_end_min": int(ends.min()), "sampled_end_max": int(ends.max()),
        "sampled_end_mean": float(ends.mean()),
        "target_block_events_behind_latest_min": int(age.min()),
        "target_block_events_behind_latest_max": int(age.max()),
        "target_block_events_behind_latest_mean": float(age.mean()),
        "training_input_length_min": 1152, "training_input_length_max": 1152,
        "training_input_length_mean": 1152.,
        "training_old_prefix_length_min": 1024, "training_old_prefix_length_max": 1024,
        "training_old_prefix_length_mean": 1024.,
        "training_raw_tail_size": 128, "training_raw_context_min": 1, "training_raw_context_max": 128,
        "query_source": "old_suffix", "all_targets_inside_train_history": True,
        "first_pass_matches_historical_fixed_schedule": True,
        "training_targets_are_fixed_last_block": False, "full_train_prefix_retained": False,
        "train_is_exhaustive_event_epoch": False}


def load_rotating_controls(config: dict[str, Any], evaluation: Any,
                           output_dir: Path) -> dict[str, Any]:
    """Verify and reuse selected fixed-block QSSM/FuXi without retraining them.

    CPU strict loading checks finite actual weights, preserving the caller's
    RNG. The FuXi historical execution used its older mixed-precision reads;
    current FP32 recurrence therefore prevents causal isolation of its schedule.
    """
    import torch

    from rhpr.sasrec.query_baseline_protocol import _resolve_kernel_root
    from rhpr.sasrec.query_history_ablation import _verified_file
    from rhpr.sasrec.query_history_models import build_history_model

    validate_rotating_config(config)
    controls = {}
    for family, variant in zip(("qssm", "fuxi"), _VARIANTS):
        history = config["history_sources"][family]
        metadata = prepare_history_source(history, evaluation, output_dir)
        source = history["selected_source"]
        root = _resolve_kernel_root(source["kernel"], source["checkpoint_root"])
        report_name = "query-qssm-search.json" if family == "qssm" else "query-baseline-screen.json"
        report = json.loads(_verified_file(root, report_name, source["documents"][report_name]["sha256"]).read_text())
        result = copy.deepcopy(report["trials"][variant["name"]][source["trial"]["name"]])
        num_items = metadata["source_best_validation"].get("num_items")
        if type(num_items) is not int or num_items <= 0:
            raise ValueError("verified control metrics must identify a positive integer catalogue size")
        # CPU model factories call torch.manual_seed(), which also resets
        # already initialized CUDA generators. Preserve those states without
        # initializing a GPU solely to verify historical CPU weight bytes.
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=rng_devices):
            model = build_history_model(history, num_items)
            proof = load_history_source_weights(metadata, model, num_items=num_items)
            del model
        alias = rotating_trials(config, variant)[0]
        provenance = dict(metadata["provenance"], original_trial=copy.deepcopy(source["trial"]),
            registered_trial_alias=copy.deepcopy(alias), **proof)
        if family == "fuxi":
            provenance.update(historical_execution_precision="original mixed-precision FuXi recurrence/reads",
                current_execution_precision="FP32 FuXi recurrence/reads under the common AMP setting",
                schedule_effect_is_causally_isolated=False)
        record = dict(metadata["checkpoint_record"], trial=copy.deepcopy(alias),
            original_trial=copy.deepcopy(source["trial"]), provenance=copy.deepcopy(provenance))
        for field in ("best_full_validation", "full_validation_rank_file", "full_validation_wall_seconds",
                      "full_validation_is_frozen_1k_checkpoint", "resumable_checkpoint", "failure"):
            result.pop(field, None)
        result.update(status="success", trial=copy.deepcopy(alias), trial_name=alias["name"],
            reused=True, reused_source_trial=copy.deepcopy(source["trial"]),
            training_performed_in_this_run=False, resume_checkpoint_retained=False,
            best_checkpoint=record,
            best_validation_rank_file=metadata["validation_artifacts"]["validation_ranks"]["file"],
            rank_file_condition="H1024+R128 daily validation only; no full-H/test ranks read",
            full_transfer_not_requested=True, full_length_training_performed=False,
            reused_provenance=provenance, stopping_policy_replay=metadata["stopping_policy_replay"],
            source_preflight=copy.deepcopy(report.get("preflight", {}).get(
                f"{variant['name']}-{source['trial']['name']}", {})))
        controls[variant["name"]] = result
    return controls
