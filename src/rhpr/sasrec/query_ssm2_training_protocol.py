"""Registered one-arm state128 SSD training with existing rotating windows."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from rhpr.sasrec.query_rotating_protocol import rotating_batch_arrays


SSM2_TRAINING_KIND = "yambda_query_ssm2_training"
VARIANT = {"name": "query_ssm2_m32_s128", "kind": "query_ssm", "summary_tokens": 32}
TRIAL = {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}
ARCHITECTURE = {"d_inner": 64, "d_state": 128, "gate_rank": 32, "head_dim": 64,
    "chunk_size": 64, "c_conditioning": "query", "dt_min": .0001, "dt_max": .01,
    "query_source": "old_suffix", "summary_tokens": 32}
COMPRESSOR = {key: value for key, value in ARCHITECTURE.items() if key != "summary_tokens"}
COMPRESSOR.update(query_context_size=128, dropout=0.)
_Q1_VARIANT = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
_INITIALIZER_SHA = "e91882fc9a684e192dfdbc6accb5678380668d344c8cc49148cac03d877de5fa"
_SCHEDULE = {"version": "qssm-fuxi-rotating-block128-v1", "training_input_tokens": 1152,
    "old_history_tokens": 1024, "target_block_size": 128, "first_pass": "latest_block",
    "remaining_blocks": "uid_seed_hash_permutation_without_replacement",
    "subsequent_cycles": "new_uid_seed_cycle_hash_permutation", "train_only": True}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_ssm2_training_config(config: dict[str, Any]) -> None:
    fields = {"kind", "name", "seed", "initializer", "parent", "data", "runtime", "model",
        "compressor", "architecture", "training", "benchmark", "hpo", "variants", "summary_budgets", "schedule", "kaggle"}
    if (set(config) != fields or config["kind"] != SSM2_TRAINING_KIND
            or config["name"] != "yambda500m-query-ssm2-state128-training" or config["seed"] != 42):
        raise ValueError("the named seed42 one-arm SSD state128 training registration is required")
    source = config["initializer"]
    if _canonical(source) != _INITIALIZER_SHA:
        raise ValueError("the selected original fresh QSSM initializer registration is immutable")
    for key in ("parent", "data", "runtime", "model", "benchmark"):
        if config[key] != source[key]:
            raise ValueError(f"state128 training must preserve registered {key}")
    expected_training = dict(source["parent"]["experiment_config"]["training"], min_passes=20,
        max_passes=60, early_stopping_min_passes=20, early_stopping_patience=2, budget_seconds=7200)
    if config["training"] != expected_training:
        raise ValueError("min20/max60/validation10/patience2 and the original optimization with a 7200s ceiling are required")
    hpo = {"proxy_old_length": 1024, "min_passes": 20, "max_passes": 60,
        "validation_interval": 10, "patience": 2, "bootstrap_draws": 5000,
        "physical_batch_size": 32, "full_length_validation": False, "test_evaluations": 0,
        "trials": [dict(TRIAL)]}
    if (config["hpo"] != hpo or config["architecture"] != ARCHITECTURE or config["compressor"] != COMPRESSOR
            or config["variants"] != [VARIANT["name"]] or config["summary_budgets"] != [32]
            or config["schedule"] != _SCHEDULE):
        raise ValueError("exactly K32/inner64/state128/head64/chunk64 and unchanged rotating supervision are required")
    expected_kaggle = dict(source["parent"]["experiment_config"]["kaggle"],
                          kernel_sources=[source["parent"]["kernel"]])
    if config["kaggle"] != expected_kaggle:
        raise ValueError("only original context metadata plus pinned cohort/evaluation/runtime sources are required")


def ssm2_training_variants(_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(VARIANT)]


def ssm2_training_trials(_config: dict[str, Any], variant: dict[str, Any]) -> list[dict[str, Any]]:
    if variant != VARIANT:
        raise ValueError("the registered SSD state128 variant is required")
    return [dict(TRIAL)]


def ssm2_training_batch_arrays(cohort: Any, windows: Any, config: dict[str, Any],
                              variant: dict[str, Any], *, old_length: int | None) -> dict[str, Any]:
    if variant != VARIANT:
        raise ValueError("the registered SSD state128 variant is required")
    return rotating_batch_arrays(cohort, windows, config, _Q1_VARIANT, old_length=old_length)


def ssm2_training_evaluation_batch(_config: dict[str, Any], variant: dict[str, Any],
                                  old_length: int | None) -> int:
    if variant != VARIANT or old_length != 1024 or type(old_length) is not int:
        raise ValueError("state128 evaluation requires the registered H1024+R128 variant")
    return 32


def ssm2_training_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    return {"training": "one fresh QSSM2 state128 model H1024+R128, min20/max60, validation10/patience2",
        "targets": "unchanged seed42 rotating 128-target train blocks; first latest block then per-user shuffled blocks without replacement",
        "architecture": "SASRec2x128 + query-conditioned native Mamba2 SSD; K32 inner64 head64 state128 chunk64; C(q)",
        "proxy_selection": "earliest strictly best full-catalogue macro-user validation NDCG10",
        "family_selection": "one fixed registered recipe; no hyperparameter or historical-control model selection",
        "full_history": "not registered: no full-H training, profiling, inference transfer or selection",
        "control_reuse": "none; original pilot context hashes/manifests and saved validation ranks are read only as evaluation provenance",
        "initialization": "fresh selected original SASRec/catalogue/32768PE initializer; only compressor replaced; no checkpoint weights loaded",
        "comparison": "same cohort, catalogue, daily validation anchors, windows, negatives and optimizer as previous rotating run; historical comparisons remain cross-run exploratory",
        "test": "no test ranks deserialized; no test inference, evaluation or selection",
        "state_dimension": "d_state128 is the B/C state width; head_dim remains64, one scalar-A head per query"}
