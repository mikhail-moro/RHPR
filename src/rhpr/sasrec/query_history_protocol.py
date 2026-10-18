"""Registered paired continuation isolating the effect of observed history length.

Each private kernel contains one model family and two arms. Both begin with the
same selected short-history weights, reset optimizer/scaler/RNG, and consume the
same twenty additional schedules of users, final128 labels and negatives. Only
the amount of observed H changes. This is continuation, not from-scratch HPO.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from rhpr.sasrec.query_prefix_protocol import validate_prefix_config
from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants


HISTORY_KIND = "yambda_query_history_continuation"
_PARENT_DIGEST = "867436ea189c6ac9a5024568ea2f3de49342b23b2a949a517752905f2c3986e8"
_SOURCE_DIGESTS = {
    "qssm": "1f0fbba463258ba127d2adc0000855a9e6f4dedbaa6106fde60841f7faf2c282",
    "fuxi": "221bb7e2f4d94bdc6041535afaa977011b5e1e4126685b550ecd485606cb60a0",
}
_ARCHITECTURES = {
    "qssm": {"d_inner": 64, "d_state": 16, "gate_rank": 32,
             "dt_min": .0001, "dt_max": .01},
    "fuxi": {"chunk_size": 128, "temporal_heads": 8, "positional_dim": 32},
}
_POLICY = {
    "additional_passes": 20, "validation_passes": [0, 10, 20],
    "early_stopping": False, "primary_metric": "endpoint20_ndcg10",
    "secondary_metrics": ["endpoint20_ndcg100", "best_within_budget_ndcg10"],
    "optimizer_state": "reset", "scaler_state": "reset",
    "rng_state": "reset_seed42_per_arm",
    "training_schedule": "matched_original_final_prefix",
    "schedule_epoch_offset": 60,
    "test_evaluations": 0, "bootstrap_draws": 5000,
    "preflight_effective_updates": 2, "preflight_longest_row": True,
}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _family(config: dict[str, Any]) -> str:
    family = config.get("family")
    if not isinstance(family, str) or family not in _ARCHITECTURES:
        raise ValueError("exactly the registered qssm or fuxi family is required")
    return family


def validate_history_config(config: dict[str, Any]) -> None:
    """Reject changes to the registered causal comparison or provenance pins."""
    fields = {"name", "kind", "family", "architecture", "benchmark", "compressor",
              "data", "model", "parent", "runtime", "seed", "summary_budgets",
              "training", "continuation", "selected_source", "kaggle"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("the exact registered paired-history configuration fields are required")
    family = _family(config)
    if (config["kind"] != HISTORY_KIND
            or config["name"] != f"yambda500m-history-continuation-{family}"):
        raise ValueError("the named paired-history continuation is required")
    parent = config["parent"]
    if _canonical(parent) != _PARENT_DIGEST:
        raise ValueError("the original full-prefix pilot and all data/anchor pins must remain exact")
    original = parent["experiment_config"]
    validate_prefix_config(original)
    for field in ("benchmark", "compressor", "data", "model", "runtime", "seed"):
        if config[field] != original[field]:
            raise ValueError(f"paired-history continuation changed parent {field}")
    if (config["architecture"] != _ARCHITECTURES[family]
            or config["summary_budgets"] != ([32] if family == "qssm" else [0])):
        raise ValueError("only the selected QSSM K32/slow or unchanged native FuXi is registered")
    if _canonical(config["selected_source"]) != _SOURCE_DIGESTS[family]:
        raise ValueError("the completed selected source, weights, trial and validation artifacts must remain pinned")
    expected_training = {key: value for key, value in original["training"].items()
                         if not key.startswith("early_stopping_")}
    expected_training.update(min_passes=20, max_passes=20,
        budget_seconds=21600 if family == "qssm" else 28800,
        physical_batch_sizes=[1, 32 if family == "qssm" else 16])
    if config["training"] != expected_training:
        raise ValueError("twenty fixed paired passes retain original targets, negatives, effective batch and optimization")
    if config["continuation"] != _POLICY:
        raise ValueError("the initial/10/20 validation, reset and fixed-endpoint policy must remain exact")
    sources = dict(original["kaggle"], kernel_sources=[parent["kernel"], config["selected_source"]["kernel"]])
    if config["kaggle"] != sources:
        raise ValueError("only pinned runtime/data, original pilot and selected family kernel are required")


def history_variant(config: dict[str, Any]) -> dict[str, Any]:
    """Return the sole family under study; history length is an arm, not HPO."""
    if _family(config) == "qssm":
        return {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
    return {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0}


def history_trial(config: dict[str, Any]) -> dict[str, Any]:
    _family(config)
    return copy.deepcopy(config["selected_source"]["trial"])


def history_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Short/full use independent reset training with a shared initialization."""
    family = _family(config)
    return [
        {"name": "short", "old_length": 1024,
         "physical_batch_size": 32 if family == "qssm" else 16,
         "evaluation_batch_size": 32, "context_length": 1152},
        {"name": "full", "old_length": None, "physical_batch_size": 1,
         "evaluation_batch_size": 4, "context_length": 32768},
    ]


def _registered_arm(config: dict[str, Any], arm: dict[str, Any] | str) -> dict[str, Any]:
    for registered in history_arms(config):
        if arm == registered or arm == registered["name"]:
            return registered
    raise ValueError("a registered short or full-history continuation arm is required")


def history_context_cap(config: dict[str, Any], arm: dict[str, Any] | str) -> int:
    return _registered_arm(config, arm)["context_length"]


def history_evaluation_batch(config: dict[str, Any], arm: dict[str, Any] | str) -> int:
    return _registered_arm(config, arm)["evaluation_batch_size"]


def history_batch_arrays(cohort: Any, windows: Any, config: dict[str, Any],
                         arm: dict[str, Any] | str) -> dict[str, Any]:
    """Keep schedule, final labels and negatives; crop only the old input H.

    This delegates to the original audited input implementation. For full H it
    retains every train-only input, rejecting any sequence exceeding the safety
    cap rather than silently truncating it. FuXi consumes these same raw arrays;
    the original QSSM registration here is solely a data adapter.
    """
    registered = _registered_arm(config, arm)
    original_variant = next(variant for variant in hpo_variants(config)
                            if variant["kind"] == "query_ssm")
    return hpo_batch_arrays(cohort, windows, config, original_variant,
                            old_length=registered["old_length"])
