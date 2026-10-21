"""Registered QSSM summary-count by memory-initialization search.

Only the compression recipe changes. All examples, shifted targets, catalogue
negatives, SASRec initialization and validation anchors remain parent-pinned.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from numbers import Real
from pathlib import Path
from typing import Any

from rhpr.sasrec.query_baseline_protocol import (
    BASELINE_KIND, _FIXED_TRIAL, _NAMES, _STANDALONE,
    load_baseline_controls, validate_baseline_config,
)
from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants


QSSM_KIND = "yambda_query_qssm_search"
_BUDGETS = (4, 8, 16, 32)
_TRIALS = tuple({"name": name, "backbone_lr": .001, "compressor_lr": .001,
    "d_inner": 64, "d_state": 16, "gate_rank": 32, "dt_min": .0001, "dt_max": maximum}
    for name, maximum in (("dt_default", .1), ("dt_slow", .01)))
_COMMON_FIELDS = ("benchmark", "compressor", "data", "model", "parent", "runtime", "seed")
_FULL_TRANSFER = {"new_configurations": 2, "selection_metric": "short_validation_ndcg10",
    "include_reused_control": False, "failure_is_optional": True, "reserve_seconds": 2400}


def _baseline_control_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact registered reference adapter without another config copy.

    The baseline validator verifies the complete canonical reference including
    artifact hashes. Only then may the loader filter to QSSM before weight I/O.
    """
    reference = config["reference_search"]
    original = reference["experiment_config"]
    projected = {field: copy.deepcopy(original[field]) for field in _COMMON_FIELDS}
    projected.update(kind=BASELINE_KIND, name="yambda500m-matched-query-baselines",
        summary_budgets=[4], training=dict(original["training"], budget_seconds=21600),
        hpo=dict(original["hpo"], trials=[dict(_FIXED_TRIAL)]), variants=list(_NAMES),
        standalone_models=copy.deepcopy(_STANDALONE), reference_search=copy.deepcopy(reference),
        baseline_batches={name: {"training": 16 if name == "fuxi_linear" else 32, "proxy": 32,
            "full": 4 if name == "fuxi_linear" else 8 if name == "mamba4rec" else 32}
            for name in _NAMES},
        kaggle=dict(original["kaggle"], kernel_sources=[original["parent"]["kernel"],
            "carasik/rhpr-query-prefix-1k-search", reference["kernel"]]))
    validate_baseline_config(projected)
    return projected


def validate_qssm_config(config: dict[str, Any]) -> None:
    """Fail closed on any unregistered architecture, data, policy or source drift."""
    required = {*_COMMON_FIELDS, "name", "kind", "summary_budgets", "training", "hpo",
        "variants", "qssm_batches", "full_transfer", "reference_search", "kaggle"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("the registered QSSM configuration fields are required exactly")
    if config["kind"] != QSSM_KIND or config["name"] != "yambda500m-query-qssm-search":
        raise ValueError("the named QSSM summary-count and memory-initialization search is required")
    baseline = _baseline_control_config(config)
    original = config["reference_search"]["experiment_config"]
    for field in _COMMON_FIELDS:
        if config[field] != original[field]:
            raise ValueError(f"QSSM search changed parent {field}")
    if config["summary_budgets"] != list(_BUDGETS) or config["variants"] != [
            variant["name"] for variant in qssm_variants(config)]:
        raise ValueError("exactly QSSM K4/K8/K16/K32 are registered in that order")
    if config["training"] != dict(original["training"], budget_seconds=21600):
        raise ValueError("QSSM search preserves optimization with a six-hour budget")
    if config["hpo"] != dict(original["hpo"], trials=[dict(trial) for trial in _TRIALS]):
        raise ValueError("QSSM search requires the exact two fixed-LR dt trials and min20/max60 stopping policy")
    if config["qssm_batches"] != {variant["name"]: {"training": 32, "proxy": 32, "full": 4}
            for variant in qssm_variants(config)}:
        raise ValueError("registered physical32/proxy32/full4 batches are required")
    if config["full_transfer"] != _FULL_TRANSFER:
        raise ValueError("only the two best fresh configurations receive optional frozen full-H transfer")
    sources = dict(baseline["kaggle"], kernel_sources=[config["parent"]["kernel"],
        config["reference_search"]["kernel"]])
    if config["kaggle"] != sources:
        raise ValueError("the pinned runtime/data, original pilot and QSSM architecture source are required")


def qssm_variants(_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": f"query_ssm_m{budget}", "kind": "query_ssm", "summary_tokens": budget}
            for budget in _BUDGETS]


def _registered(config: dict[str, Any], variant: dict[str, Any] | str) -> dict[str, Any]:
    for registered in qssm_variants(config):
        if variant == registered or variant == registered["name"]:
            return registered
    raise ValueError("a registered QSSM K4/K8/K16/K32 variant is required")


def qssm_trials(config: dict[str, Any], variant: dict[str, Any] | str) -> list[dict[str, Any]]:
    _registered(config, variant)
    return copy.deepcopy(config["hpo"]["trials"])


def _old_length(old_length: int | None) -> None:
    if old_length is not None and (type(old_length) is not int or old_length != 1024):
        raise ValueError("QSSM search supports H1024 or frozen full-history inference only")


def qssm_evaluation_batch(config: dict[str, Any], variant: dict[str, Any], old_length: int | None) -> int:
    registered = _registered(config, variant)
    _old_length(old_length)
    return config["qssm_batches"][registered["name"]]["full" if old_length is None else "proxy"]


def qssm_batch_arrays(cohort: Any, windows: Any, config: dict[str, Any], variant: dict[str, Any],
                     *, old_length: int | None) -> dict[str, Any]:
    _registered(config, variant)
    _old_length(old_length)
    # Input arrays are independent of summary count. Reuse the original QSSM
    # registration for the audited crop and unchanged negative identities.
    original = next(row for row in hpo_variants(config) if row["kind"] == "query_ssm")
    return hpo_batch_arrays(cohort, windows, config, original, old_length=old_length)


def select_qssm_candidate(trials: Sequence[dict[str, Any]],
                          results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Select finite primary NDCG10; optional full transfer never chooses a trial."""
    if (not isinstance(trials, Sequence) or isinstance(trials, (str, bytes))
            or not isinstance(results, Sequence) or isinstance(results, (str, bytes))):
        raise ValueError("trials and results must be sequences of registered records")
    names = []
    for trial in trials:
        name = trial.get("name") if isinstance(trial, Mapping) else None
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("registered trial names must be nonempty and unique")
        names.append(name)
    by_name = {}
    for result in results:
        name = result.get("trial_name") if isinstance(result, Mapping) else None
        if name not in names or name in by_name:
            raise ValueError("results must identify unique registered trials from one QSSM variant")
        by_name[name] = result
    selected, best = None, -math.inf
    for trial in trials:
        result = by_name.get(trial["name"], {})
        if (result.get("trial") != trial or result.get("complete") is not True
                or result.get("status") not in ("training_completed", "success", "full_transfer_failed")):
            continue
        metrics = result.get("best_validation")
        score = metrics.get("ndcg10") if isinstance(metrics, Mapping) else None
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            continue
        if score > best:
            selected, best = copy.deepcopy(trial), float(score)
    return selected


def load_qssm_controls(config: dict[str, Any], evaluation: Any, output_dir: Path) -> dict[str, Any]:
    """Reuse only the exact completed K4/inner64/default-dt control, without CA I/O."""
    validate_qssm_config(config)
    controls = load_baseline_controls(_baseline_control_config(config), evaluation, output_dir,
        model_names=("query_ssm_m4",))
    result = controls["query_ssm_m4"]
    original_trial = copy.deepcopy(result["trial"])
    registered_trial = qssm_trials(config, "query_ssm_m4")[0]
    result.update(trial=copy.deepcopy(registered_trial), trial_name=registered_trial["name"],
        reused_source_trial=original_trial)
    result["best_checkpoint"]["trial"] = copy.deepcopy(registered_trial)
    result["reused_provenance"].update(original_trial=original_trial,
        registered_trial_alias=copy.deepcopy(registered_trial))
    return controls
