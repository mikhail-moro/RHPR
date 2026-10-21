"""Independent Mamba-mechanism ablations of the selected H1024 Query SSM.

The historical K32/slow control is verified and reused without GPU training.
Each fresh recipe changes one compressor mechanism; original users, labels,
negative identities, predictor initialization and stopping policy stay fixed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from rhpr.sasrec.query_history_checkpoints import (
    load_history_source_weights, prepare_history_source,
)
from rhpr.sasrec.query_history_protocol import validate_history_config
from rhpr.sasrec.query_qssm_protocol import (
    qssm_batch_arrays, select_qssm_candidate, validate_qssm_config,
)


MAMBA_KIND = "yambda_query_mamba_ablation"
_COMMON = ("benchmark", "compressor", "data", "model", "parent", "runtime", "seed")
_CONTROL_HISTORY_DIGEST = "e91882fc9a684e192dfdbc6accb5678380668d344c8cc49148cac03d877de5fa"
_ARCHITECTURE = {"d_inner": 64, "d_state": 16, "gate_rank": 32,
                 "dt_min": .0001, "dt_max": .01}
_VARIANT = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
_TRIALS = tuple({"name": name, "feature": feature,
                 "backbone_lr": .001, "compressor_lr": .001}
                for name, feature in (("base", "none"), ("conv4", "conv4"),
                                      ("output_gate", "output_gate"), ("skip_d", "skip_d")))


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mamba_source_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an independent copy of the exact completed K/dt search config."""
    return copy.deepcopy(config["source_search_config"])


def validate_mamba_config(config: dict[str, Any]) -> None:
    """Reject unregistered features, policy changes, lengths and provenance."""
    required = {*_COMMON, "name", "kind", "architecture", "summary_budgets",
                "training", "hpo", "variants", "mamba_batches", "kaggle",
                "source_search_config", "control_history_config"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("the exact registered Mamba-ablation configuration fields are required")
    if (config["kind"] != MAMBA_KIND
            or config["name"] != "yambda500m-query-mamba-ablation"):
        raise ValueError("the named independent Query SSM Mamba ablation is required")
    control = config["control_history_config"]
    validate_history_config(control)
    if control["family"] != "qssm" or _canonical(control) != _CONTROL_HISTORY_DIGEST:
        raise ValueError("the exact immutable selected K32/slow history-source registration is required")
    source = config["source_search_config"]
    validate_qssm_config(source)
    if _canonical(source) != control["selected_source"]["experiment_config_sha256"]:
        raise ValueError("the completed QSSM search configuration must match its pinned source digest")
    for field in _COMMON:
        if config[field] != source[field] or config[field] != control[field]:
            raise ValueError(f"Mamba ablation changed source {field}")
    if config["architecture"] != _ARCHITECTURE or config["summary_budgets"] != [32]:
        raise ValueError("only the selected K32/inner64/state16/rank32/slow-dt recipe is registered")
    if config["variants"] != [_VARIANT["name"]]:
        raise ValueError("exactly the selected query_ssm_m32 variant is required")
    expected_hpo = dict(source["hpo"], full_length_validation=False,
                        trials=[dict(trial) for trial in _TRIALS])
    if config["hpo"] != expected_hpo:
        raise ValueError("independent base/conv4/output_gate/skip_d trials retain H1024/min20/max60/validation10/patience2")
    if config["training"] != dict(source["training"], budget_seconds=10800):
        raise ValueError("Mamba ablation preserves optimization and registers a three-hour ceiling")
    if config["mamba_batches"] != {_VARIANT["name"]: {"training": 32, "proxy": 32}}:
        raise ValueError("the exact physical32/evaluation32 batches are required")
    expected_sources = dict(control["kaggle"])
    if config["kaggle"] != expected_sources:
        raise ValueError("only the original pilot, selected QSSM source and pinned runtime/data are required")


def mamba_variants(_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [copy.deepcopy(_VARIANT)]


def _registered(variant: dict[str, Any] | str) -> dict[str, Any]:
    if variant != _VARIANT and variant != _VARIANT["name"]:
        raise ValueError("the registered query_ssm_m32 variant is required")
    return copy.deepcopy(_VARIANT)


def mamba_trials(_config: dict[str, Any], variant: dict[str, Any] | str) -> list[dict[str, Any]]:
    _registered(variant)
    return [dict(trial) for trial in _TRIALS]


def _old_length(old_length: int | None) -> None:
    if type(old_length) is not int or old_length != 1024:
        raise ValueError("Mamba ablation is registered for H1024 only; no full-H inference")


def mamba_evaluation_batch(config: dict[str, Any], variant: dict[str, Any] | str,
                           old_length: int | None) -> int:
    registered = _registered(variant)
    _old_length(old_length)
    return config["mamba_batches"][registered["name"]]["proxy"]


def mamba_batch_arrays(cohort: Any, windows: Any, config: dict[str, Any],
                       variant: dict[str, Any] | str, *, old_length: int | None) -> dict[str, Any]:
    """Retain source window/negative identities and crop only the old input H."""
    registered = _registered(variant)
    _old_length(old_length)
    return qssm_batch_arrays(cohort, windows, config["source_search_config"], registered,
                             old_length=old_length)


def select_mamba_candidate(trials: Any, results: Any) -> dict[str, Any] | None:
    """Select finite H1024 NDCG10; exact ties retain registration order."""
    return select_qssm_candidate(trials, results)


def load_mamba_controls(config: dict[str, Any], evaluation: Any,
                         output_dir: Path) -> dict[str, Any]:
    """Reuse selected H1024 weights/ranks after strict CPU weight verification.

    Only H1024 rank pointers are retained. No full-history or test-rank payload
    is opened, copied or evaluated, and no short ranks are mislabeled as full.
    """
    import torch

    from rhpr.sasrec.query_baseline_protocol import _resolve_kernel_root
    from rhpr.sasrec.query_history_ablation import _verified_file
    from rhpr.sasrec.query_history_models import build_history_model

    validate_mamba_config(config)
    control = config["control_history_config"]
    metadata = prepare_history_source(control, evaluation, output_dir)
    if metadata["source_config"] != config["source_search_config"]:
        raise ValueError("prepared control configuration differs from the registered completed search")
    source = control["selected_source"]
    root = _resolve_kernel_root(source["kernel"], source["checkpoint_root"])
    report = json.loads(_verified_file(root, "query-qssm-search.json",
        source["documents"]["query-qssm-search.json"]["sha256"]).read_text())
    result = copy.deepcopy(report["trials"][_VARIANT["name"]][source["trial"]["name"]])
    num_items = metadata["source_best_validation"].get("num_items")
    if type(num_items) is not int or num_items <= 0:
        raise ValueError("verified control metrics must identify a positive integer catalogue size")
    with torch.random.fork_rng(devices=[]):
        model = build_history_model(control, num_items)
        proof = load_history_source_weights(metadata, model, num_items=num_items)
        del model
    base = mamba_trials(config, _VARIANT)[0]
    provenance = dict(metadata["provenance"], original_trial=copy.deepcopy(source["trial"]),
        registered_trial_alias=copy.deepcopy(base), **proof)
    record = dict(metadata["checkpoint_record"], trial=copy.deepcopy(base),
        original_trial=copy.deepcopy(source["trial"]), provenance=copy.deepcopy(provenance))
    for field in ("best_full_validation", "full_validation_rank_file", "full_validation_wall_seconds",
                  "full_validation_is_frozen_1k_checkpoint", "resumable_checkpoint"):
        result.pop(field, None)
    ranks = metadata["validation_artifacts"]["validation_ranks"]["file"]
    result.update(status="success", trial=copy.deepcopy(base), trial_name=base["name"],
        reused=True, reused_source_trial=copy.deepcopy(source["trial"]),
        training_performed_in_this_run=False, resume_checkpoint_retained=False,
        best_checkpoint=record, best_validation_rank_file=ranks,
        rank_file_condition="H1024+R128 only; no full-H inference",
        full_transfer_not_requested=True, full_length_training_performed=False,
        reused_provenance=provenance, stopping_policy_replay=metadata["stopping_policy_replay"],
        source_preflight=copy.deepcopy(report.get("preflight", {}).get(
            f"{_VARIANT['name']}-{source['trial']['name']}", {})))
    return {_VARIANT["name"]: result}
