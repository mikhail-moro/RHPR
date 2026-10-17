"""Matched short-history learning-rate proxy for the full-prefix compressors.

History length changes only a CPU-side left crop. Original window identities,
last128 targets, negative draws and the 32768-position model initializer stay
fixed. Proxy trials choose optimization settings, never the held-out test set.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rhpr.sasrec.daily_evaluation_data import DailyEvaluationExample
from rhpr.sasrec.long_history_data import LongHistoryCohort
from rhpr.sasrec.long_quality_schedule import QualityWindow, quality_negative_ids
from rhpr.sasrec.query_prefix_protocol import (
    build_prefix_model, prefix_batch_arrays, prefix_variants, validate_prefix_config,
)


HPO_KIND = "yambda_query_prefix_proxy_hpo"
_PARENT_KERNEL = "carasik/rhpr-sasrec-prefix-query-pilot"
_PARENT_SOURCE = "660c7705ed3e005f35b339313c8c865255e36ede"
_PINNED_PARENT_HASHES = {
    "experiment_config": "ec32270c734a89e10cf70c85831518477a7276293d594ec8e68df9fae546c0ad",
    "baselines": "80db3faf42768f5a363f4b01bc8d7300f251fd07352f57255fe610720d0998ed",
    "checkpoints": "fcc9ecb02d245df44414dcd1df6e736784cd421b18995e3c24a8d7b71626bce7",
}
_TRIALS = (
    {"name": "current_lr", "backbone_lr": .001, "compressor_lr": .001},
    {"name": "lower_lr", "backbone_lr": .0003, "compressor_lr": .0003},
    {"name": "faster_compressor", "backbone_lr": .001, "compressor_lr": .003},
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_proxy_hpo_config(config: dict[str, Any]) -> None:
    """Fail closed on dataset, initialization, trial or compute-budget drift."""
    if config.get("kind") != HPO_KIND or not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("a named query-prefix proxy HPO configuration is required")
    parent = config.get("parent", {})
    original = parent.get("experiment_config")
    if not isinstance(original, dict):
        raise ValueError("parent.experiment_config is required")
    validate_prefix_config(original)
    for section, expected in _PINNED_PARENT_HASHES.items():
        if _canonical_hash(parent.get(section)) != expected:
            raise ValueError(f"the exact completed parent {section} must remain pinned")
    if (parent.get("kernel") != _PARENT_KERNEL or parent.get("source_code_sha") != _PARENT_SOURCE
            or parent.get("manifest_sha256") != "7962c22fec6079c6fd1dfa5bd082caa28274a2dc1f89b2caff21dbc20f9a4774"
            or parent.get("context_manifest_sha256") != "e33bdbf050e95d32cb88b40a6ea8b8d46c8ed26a4025b59b1543101528635110"
            or Path(parent.get("checkpoint_root", "")).name != _PARENT_KERNEL.split("/")[1]):
        raise ValueError("the completed private parent source, manifests and mount must remain pinned")
    for section in ("seed", "data", "model", "compressor", "runtime", "benchmark", "summary_budgets"):
        if config.get(section) != original.get(section):
            raise ValueError(f"proxy HPO changed parent {section}")
    expected_training = dict(original["training"], max_passes=30, budget_seconds=21600)
    if config.get("training") != expected_training:
        raise ValueError("HPO preserves parent optimization and fixes full30 passes and a21600-second budget")
    if config.get("variants") != [variant["name"] for variant in hpo_variants(config)]:
        raise ValueError("exactly Cross Attention K4 and QSSM K4 are registered in that order")
    expected_hpo = {"proxy_old_length": 4096, "diagnostic_old_lengths": [1024, 2048, 4096],
                    "proxy_passes": 20, "final_passes": 30, "validation_interval": 10,
                    "bootstrap_draws": 5000, "physical_batch_size": 32,
                    "trials": [dict(trial) for trial in _TRIALS]}
    if config.get("hpo") != expected_hpo:
        raise ValueError("HPO requires six4K/20-pass trials, six frozen diagnostics and twofull/30-pass finalists")
    kaggle = config.get("kaggle", {})
    if kaggle.get("kernel_sources") != [_PARENT_KERNEL]:
        raise ValueError("the completed private pilot must be the sole kernel source")
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        if kaggle.get(key) != original["kaggle"].get(key):
            raise ValueError(f"parent Kaggle sources changed: {key}")


def hpo_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [variant for variant in prefix_variants(config["parent"]["experiment_config"])
            if variant["kind"] != "recent"]


def hpo_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(trial) for trial in config["hpo"]["trials"]]


def _check_variant(config: dict[str, Any], variant: dict[str, Any]) -> None:
    if variant not in hpo_variants(config):
        raise ValueError("the proxy requires a registered Cross Attention or QSSM K4 variant")


def build_hpo_model(config: dict[str, Any], variant: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Use the exact parent initializer, including its full32768 PE table and RNG."""
    _check_variant(config, variant)
    return build_prefix_model(config["parent"]["experiment_config"], variant, num_items)


def _old_length(value: int | None, query_size: int = 128) -> int | None:
    if value is not None and (type(value) is not int or value < query_size):
        raise ValueError("old_length must be None or an integer retaining the entire query suffix")
    return value


def hpo_batch_arrays(cohort: LongHistoryCohort, windows: Sequence[QualityWindow],
                     config: dict[str, Any], variant: dict[str, Any], *, old_length: int | None
                     ) -> dict[str, np.ndarray]:
    """Crop only H before embedding lookup; retain final labels and negative identities."""
    _check_variant(config, variant)
    original = config["parent"]["experiment_config"]
    train = original["training"]
    old_length = _old_length(old_length, train["query_context_size"])
    if old_length is None:
        return prefix_batch_arrays(cohort, windows, original, variant)
    if not windows:
        raise ValueError("a nonempty physical batch is required")
    rows = []
    for window in windows:
        if not 0 <= window.user_index < len(cohort.users):
            raise ValueError("window user_index is outside the cohort")
        user = cohort.users[window.user_index]
        if user.user_id != window.user_id or window.end != len(user.train) - 1:
            raise ValueError("HPO window must end at the original final train-only input")
        if window.end > train["raw_context_length"]:
            raise ValueError("complete train prefix exceeds safety cap; truncation is forbidden")
        if window.end < train["raw_tail_size"] + train["query_context_size"]:
            raise ValueError("train prefix must contain the complete raw tail and old query suffix")
        start = max(0, window.end - train["raw_tail_size"] - old_length)
        rows.append((user.train[start:window.end], user.train[start + 1:window.end + 1], window))
    shape = (len(rows), max(len(inputs) for inputs, _, _ in rows))
    item_ids, targets = (np.zeros(shape, dtype=np.int64) for _ in range(2))
    valid = np.zeros(shape, dtype=np.bool_)
    negatives = []
    for index, (inputs, labels, window) in enumerate(rows):
        size = len(inputs)
        item_ids[index, :size], targets[index, :size], valid[index, :size] = inputs, labels, True
        negatives.append(quality_negative_ids(window, num_items=cohort.num_items,
                                              num_negatives=train["num_negatives"]))
    return {"item_ids": item_ids, "targets": targets, "valid_mask": valid,
            "negative_ids": np.asarray(negatives, dtype=np.int64)}


def hpo_optimizer(model: torch.nn.Module, training: dict[str, Any],
                  trial: dict[str, Any]) -> torch.optim.Optimizer:
    """Split learning rates while preserving parent AdamW decay on every parameter.

The parent helper does not exempt biases, norms or SSM parameters bearing a
_no_weight_decay attribute. Deliberately retain that convention here so the
current_lr arm is an exact optimizer control.
"""
    for key in ("backbone_lr", "compressor_lr"):
        value = trial.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"trial.{key} must be finite and positive")
    groups = {"backbone": [], "compressor": []}
    for name, parameter in model.named_parameters():
        groups["compressor" if name.startswith("compressor.") else "backbone"].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError("both backbone and compressor parameters are required")
    return torch.optim.AdamW([
        {"params": groups[name], "lr": trial[f"{name}_lr"], "component": name}
        for name in ("backbone", "compressor")],
        lr=training["learning_rate"], weight_decay=training["weight_decay"], foreach=False)


def transform_hpo_example(example: DailyEvaluationExample, old_length: int | None
                          ) -> DailyEvaluationExample:
    """Suffix crop observed past only; keep the full reference hash and target identity."""
    old_length = _old_length(old_length)
    if old_length is None:
        return example
    if example.context.ndim != 1 or example.timestamps.shape != example.context.shape:
        raise ValueError("daily context and timestamps must be aligned one-dimensional arrays")
    if len(example.context) < 256:
        raise ValueError("daily context must retain query128 and raw128")
    width = old_length + 128
    return replace(example, context=example.context[-width:].copy(),
                   timestamps=example.timestamps[-width:].copy())
