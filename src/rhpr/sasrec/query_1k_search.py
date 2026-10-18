"""Validation-only 1K LR search with frozen-weight full-length transfer checks.

Six independent short-history models are trained from the parent initializer.
Their best 1K-validation weights are retained, then evaluated once on full H.
No full-history optimization or test-rank deserialization occurs in this run.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import gzip
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable

import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import iter_daily_evaluation_examples, load_prepared_daily_evaluation
from rhpr.sasrec.daily_quality import _aggregate, evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _cleanup, _json, _registered_device, _scaler
from rhpr.sasrec.query_1k_protocol import select_1k_candidate, validate_1k_config
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_numerics import capture_training_rng, diagnose_nonfinite_batch
from rhpr.sasrec.query_prefix_pilot import (
    _checkpoint_record, _profile_daily_rate, _representative_windows, _save_best,
    _save_resume, _validate_complete_histories, train_prefix_pass,
)
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_schedule_summary
from rhpr.sasrec.query_profile import audit_native_query_scan
from rhpr.sasrec.query_proxy_diagnostics import audited_hpo_examples, load_proxy_parent_artifacts
from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_batch_arrays, hpo_optimizer, hpo_trials, hpo_variants
from rhpr.sasrec.query_restart_state import ValidationEarlyStopping, reset_restart_rng


def conservative_1k_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                           users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price all six maximum-length runs; early stopping never funds the grid."""
    hpo, train = config["hpo"], config["training"]
    work = 0.0
    for variant in hpo_variants(config):
        row = profiles[variant["name"]]
        rates = (row["training"]["windows_per_second"],
                 row["proxy_evaluation"]["anchors_per_second"],
                 row["full_evaluation"]["anchors_per_second"])
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
               or not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("positive finite measured preflight rates are required")
        updates = hpo["max_passes"] * users / rates[0]
        proxy = (hpo["max_passes"] // hpo["validation_interval"]
                 * evaluation_summary["validation_anchors"] / rates[1])
        full = evaluation_summary["validation_anchors"] / rates[2]
        work += len(hpo_trials(config)) * (updates + proxy + full)
    # FP32 best writes, rolling complete Adam/RNG saves, hashing and paired
    # summaries. This allowance is charged for every trial, not only winners.
    overhead = 120.0 * len(hpo_variants(config)) * len(hpo_trials(config))
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "prices_all_trials_at_maximum_passes": True, "auto_reduces_trials_or_passes": False,
        "full_training_models": 0, "test_evaluations": 0}


def _preflight_1k(config: dict[str, Any], cohort: Any, evaluation: Any,
                   variant: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Two real short effective updates, then short/full inference with Adam live."""
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    # LR does not affect arithmetic or Adam-state capacity. Use the registered
    # stable control for disposable timing; risky recipes belong inside their
    # isolated trial handlers and never gate the entire grid at preflight.
    trial = next(row for row in hpo_trials(config) if row["name"] == "current_lr")
    model = build_hpo_model(config, variant, cohort.num_items).to(device)
    optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
    try:
        windows = _representative_windows(fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1),
                                          train["effective_batch_size"])
        builder = partial(hpo_batch_arrays, old_length=hpo["proxy_old_length"])
        updates = []
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=hpo["physical_batch_size"], device=device, optimizer=optimizer,
                scaler=scaler, batch_builder=builder)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("1K preflight requires two successful effective Adam updates without AMP retries")
            updates.append(row)
        evaluations = {}
        for phase, cap in (("proxy_evaluation", hpo["proxy_old_length"] + train["raw_tail_size"]),
                           ("full_evaluation", train["serving_context_length"])):
            evaluations[phase] = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
                item_chunk_size=bench["item_chunk_size"])
        return {"warmup": updates[0], "training": updates[1], **evaluations,
            "old_length": hpo["proxy_old_length"], "profile_trial": trial,
            "physical_batch_size": hpo["physical_batch_size"],
            "effective_batch_size": train["effective_batch_size"],
            "profile_weights_reused": False, "quality_used_for_selection": False,
            "approves_stability_of_other_recipes": False,
            "optimizer_state_allocated_during_all_measurements": True}
    finally:
        del model, optimizer, scaler
        _cleanup()


def _failure(error: BaseException, *, stage: str, completed: int, attempted: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"exception_type": type(error).__name__, "message": str(error),
        "stage": stage, "last_completed_training_pass": completed,
        "partial_pass_state_checkpointed": False, "automatic_precision_or_lr_fallback": False}
    if attempted is not None:
        result["attempted_pass"] = attempted
    return result


def _finite_validation(result: dict[str, Any]) -> None:
    for key, value in result["metrics"].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise FloatingPointError(f"nonfinite validation metric: {key}")
    score = result["metrics"].get("ndcg10")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise FloatingPointError("validation NDCG10 must be finite")


class _OptionalFullTransferTimeout(TimeoutError):
    """The registered secondary diagnostic allowance was exhausted."""


@dataclass(frozen=True)
class QuerySearchDefinition:
    """Explicit differences between a registered LR grid and architecture grid.

    The training/data/evaluation engine remains shared. Callbacks keep the
    original 1K public API and its registration independent of newer searches.
    """
    validate: Callable[..., Any]
    trials: Callable[..., list[dict[str, Any]]]
    build_model: Callable[..., torch.nn.Module]
    preflight: Callable[..., dict[str, Any]]
    budget: Callable[..., dict[str, Any]]
    report_file: str = "query-1k-search.json"
    phase: str = "one_k"
    checkpoint_stage: str = "one_k_search"
    profile_each_trial: bool = False
    load_controls: Callable[..., dict[str, Any]] | None = None
    isolate_preflight_failures: bool = False
    variants: Callable[..., list[dict[str, Any]]] | None = None
    batch_arrays: Callable[..., dict[str, Any]] | None = None
    optimizer: Callable[..., torch.optim.Optimizer] | None = None
    training_batch: Callable[..., int] | None = None
    evaluation_examples: Callable[..., Any] | None = None
    evaluation_batch: Callable[..., int] | None = None
    full_transfer_required: Callable[..., bool] | None = None
    protocol_metadata: Callable[..., dict[str, Any]] | None = None
    select_candidate: Callable[..., dict[str, Any] | None] | None = None
    paired_short_validation: bool = False
    hpo_performed: bool = True
    optional_full_transfer: bool = False
    full_transfer_candidates: Callable[..., set[tuple[str, str]]] | None = None
    retained_trial_resumes: int = 0
    full_transfer_budget_seconds: int | None = None


def run_query_1k_search(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    definition = QuerySearchDefinition(validate=validate_1k_config,
        trials=lambda current, _variant: hpo_trials(current),
        build_model=lambda current, variant, _trial, items: build_hpo_model(current, variant, items),
        preflight=lambda current, cohort, evaluation, variant, _trial, device:
            _preflight_1k(current, cohort, evaluation, variant, device),
        budget=conservative_1k_budget)
    return run_registered_query_search(config, output_dir, definition=definition)


def run_registered_query_search(config: dict[str, Any], output_dir: Path, *,
        definition: QuerySearchDefinition) -> dict[str, int | float | str | bool]:
    """Run the shared, validation-only short-history search with explicit recipes."""
    definition.validate(config)
    started, device = time.monotonic(), _registered_device()
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    variants = (definition.variants or hpo_variants)(config)
    trials_by_family = {variant["name"]: definition.trials(config, variant) for variant in variants}
    total_configurations = sum(map(len, trials_by_family.values()))
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "seed": config["seed"],
        "quality_complete": False, "hpo_performed": definition.hpo_performed, "proxy_training_models": total_configurations,
        "full_training_models": 0, "test_evaluations": 0, "train_is_exhaustive_event_epoch": False,
        "exploratory_previously_inspected_holdout": True}
    report: dict[str, Any] = {"complete": False, "config": config,
        "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "protocol": {"training": f"H1024+R128 models, minimum{hpo['min_passes']} maximum{hpo['max_passes']} passes, validation every{hpo['validation_interval']}",
            "stopping": f"strict macro-user NDCG10 improvement; patience{hpo['patience']} checks; no stop before{hpo['min_passes']}",
            "proxy_selection": "earliest strictly best 1K validation weights within each trial",
            "family_selection": "all trials complete before choosing finite full-H validation winner; ties keep registered order",
            "full_history": "one validation-only frozen-weight inference-length transfer per completed short trial; no full training",
            "test": "no test ranks deserialized; no test evaluation or selection",
            "targets": "parent last128 labels, endpoints, users, negative seeds and PE32768 initializer retained",
            "failures": "record numerical/OOM failure and continue; no automatic LR or precision changes",
            "resume": "one rolling Adam/scaler/RNG resume per family; validation boundaries and final completed pass only",
            "frozen_diagnostics_repeated": False},
        "preflight": {}, "trials": {}, "selected_trials": {}, "cases": {}}
    manifest: dict[str, Any] = {"format_version": 3, "source_code_sha": report["source_code_sha"],
        "config": config, "optimizer_state_saved": True, "best_checkpoint_optimizer_state_saved": False,
        "proxy_checkpoints_saved": True, "checkpoints": [], "resume_checkpoints": [],
        "resume_retention": "one last successful completed-pass checkpoint per family; active_trial identifies ownership"}
    if definition.retained_trial_resumes:
        policy = (f"top{definition.retained_trial_resumes} completed fresh trials by primary short NDCG10; "
                  "one additional active trial; exact Adam/scaler/RNG state at completed-pass boundary")
        report["protocol"]["resume"] = manifest["resume_retention"] = policy
    optional_transfer_deadline: float | None = None

    def persist() -> None:
        _json(output_dir / definition.report_file, report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if optional_transfer_deadline is not None and time.monotonic() > optional_transfer_deadline:
            raise _OptionalFullTransferTimeout("registered optional full-transfer allowance exhausted; primary results retained")
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("query search exceeded registered budget; partial outputs retained")

    def record_checkpoint(key: str, path: Path, **metadata: Any) -> dict[str, Any]:
        record = _checkpoint_record(path, output_dir, **metadata)
        manifest[key] = [row for row in manifest[key] if row["path"] != record["path"]] + [record]
        return record

    persist()
    audit = audit_native_query_scan(device, query_source="old_suffix")
    report["native_scan_audit"] = audit
    _json(output_dir / "native-query-scan-audit.json", audit)
    metrics["native_scan_audit_passed"] = True
    guard()
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_daily_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    for key in ("evaluation_content_sha256", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if config["data"][key] != evaluation.summary[key]:
            raise ValueError(f"registered daily evaluation mismatch: {key}")
    _validate_complete_histories(cohort, evaluation, train["raw_context_length"], train["serving_context_length"])
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    # This helper validates manifests/context metadata and validation ranks only.
    # It does not open or deserialize any parent's test-anchor rank payload.
    _, contexts, validation_baselines, _ = load_proxy_parent_artifacts(config, evaluation)
    report["saved_baseline_validation"] = {name: _aggregate(rows)[0] for name, rows in validation_baselines.items()}
    metrics.update(cohort_users=len(cohort.users), num_items=cohort.num_items,
        unique_supervised_target_positions_per_model=len(cohort.users) * train["supervised_targets"])
    controls = (definition.load_controls(config, evaluation, output_dir)
                if definition.load_controls is not None else {})
    for name, control in controls.items():
        registered = trials_by_family.get(name, [])
        if (control.get("status") != "success"
                or control.get("trial") not in registered
                or control.get("trial_name") != control["trial"]["name"]):
            raise ValueError("reused architecture control does not match a successful registered recipe")
        report["trials"].setdefault(name, {})[control["trial_name"]] = control
        manifest["checkpoints"].append(control["best_checkpoint"])
        if definition.load_controls is not None:
            with gzip.open(output_dir / control["full_validation_rank_file"], "rt", encoding="utf-8") as stream:
                validation_baselines[f"reused_1k_{name}"] = json.load(stream)
    reused_count = len(controls)
    if definition.load_controls is not None:
        metrics.update(registered_configurations=total_configurations,
            proxy_training_models=total_configurations - reused_count, reused_controls=reused_count,
            backbone_lr=.001, compressor_lr=.001)
        report["configuration_counts"] = {"total": total_configurations,
            "new_training_configurations": total_configurations - reused_count, "reused_controls": reused_count}
        report["protocol"]["control_reuse"] = "verified completed current_lr trials; no repeated training or full validation"
        report["protocol"]["architecture"] = "SASRec and K4 fixed; only registered compressor shapes change"
    if definition.protocol_metadata is not None:
        report["protocol"].update(definition.protocol_metadata(config))
    for variant in variants:
        name = variant["name"]
        pending = [trial for trial in trials_by_family[name]
                   if trial["name"] not in report["trials"].get(name, {})]
        for trial in (pending if definition.profile_each_trial else pending[:1]):
            guard()
            label = f"{name}-{trial['name']}" if definition.profile_each_trial else name
            print(json.dumps({"phase": definition.phase + "_preflight_start", "model": name,
                "trial": trial["name"]}), flush=True)
            try:
                with _preserve_rng(device):
                    row = definition.preflight(config, cohort, evaluation, variant, trial, device)
                if name in controls:
                    reference = controls[name]
                    relative = {"reference": "verified reused control",
                        "timing_comparison": "different runs; descriptive, not controlled latency evidence"}
                    if reference.get("compressor_parameters") and row.get("compressor_parameters"):
                        relative["compressor_parameter_ratio"] = row["compressor_parameters"] / reference["compressor_parameters"]
                    prior = reference.get("source_preflight", {})
                    for phase, rate in (("training", "windows_per_second"),
                                        ("proxy_evaluation", "anchors_per_second"),
                                        ("full_evaluation", "anchors_per_second")):
                        old_rate, new_rate = prior.get(phase, {}).get(rate), row.get(phase, {}).get(rate)
                        if (isinstance(old_rate, (float, int)) and not isinstance(old_rate, bool)
                                and math.isfinite(old_rate) and old_rate > 0
                                and isinstance(new_rate, (float, int)) and not isinstance(new_rate, bool)
                                and math.isfinite(new_rate) and new_rate > 0):
                            relative[phase + "_seconds_per_example_ratio"] = old_rate / new_rate
                    row["relative_to_reused_control"] = relative
                report["preflight"][label] = row
                print(json.dumps({"phase": definition.phase + "_preflight", "model": name,
                    "trial": trial["name"], **row}), flush=True)
            except (FloatingPointError, torch.OutOfMemoryError) as error:
                if not definition.isolate_preflight_failures:
                    raise
                status = "preflight_numerical_failed" if isinstance(error, FloatingPointError) else "preflight_out_of_memory"
                report["trials"].setdefault(name, {})[trial["name"]] = {
                    "status": status, "complete": False, "trial_name": trial["name"], "trial": dict(trial),
                    "passes": [], "passes_completed": 0, "from_scratch": True,
                    "test_evaluations": 0, "failure": _failure(error, stage="preflight", completed=0)}
                report["preflight"][label] = {"status": status, "trial": dict(trial)}
                print(json.dumps({"phase": definition.phase + "_trial_failed", "model": name,
                    "trial": trial["name"], "status": status}), flush=True)
                _cleanup()
            persist()
    report["budget"] = definition.budget(config, report["preflight"], time.monotonic() - started,
        len(cohort.users), evaluation.summary)
    persist()
    if not report["budget"]["fits"]:
        raise RuntimeError("measured cost cannot support all registered maximum-pass trials and full-H validations")
    schedules = [fixed_prefix_schedule(cohort, seed=config["seed"], epoch=epoch)
                 for epoch in range(1, hpo["max_passes"] + 1)]
    _json(output_dir / "window-schedules.json", {
        "original_parent_schedules": [prefix_schedule_summary(cohort, windows,
            config["parent"]["experiment_config"]) for windows in schedules],
        "proxy_input_old_length_cap": hpo["proxy_old_length"], "proxy_retains_all_old_events": False,
        "full_length_training_performed": False, "raw_tail_size": train["raw_tail_size"]})

    def evaluate(model: torch.nn.Module, variant: dict[str, Any], old: int | None,
                 label: str) -> dict[str, Any]:
        guard()
        rows: list[dict[str, Any]] = []
        examples = iter_daily_evaluation_examples(cohort, evaluation, split_name="validation",
            max_seq_len=train["serving_context_length"])
        audited = (definition.evaluation_examples(examples, old, variant,
            contexts["validation"]["rows"], rows, guard)
            if definition.evaluation_examples is not None else
            audited_hpo_examples(examples, old, contexts["validation"]["rows"], rows, guard))
        evaluation_batch = (definition.evaluation_batch(config, variant, old)
            if definition.evaluation_batch is not None else bench["evaluation_batch_size"])
        before = time.monotonic()
        try:
            result = evaluate_daily_quality(model, audited, device=device, amp=train["amp"],
                batch_size=evaluation_batch, item_chunk_size=bench["item_chunk_size"])
        except RuntimeError as error:
            if str(error) in ("evaluation produced nonfinite final states", "evaluation produced nonfinite catalogue scores"):
                raise FloatingPointError(str(error)) from error
            raise
        _finite_validation(result)
        result["wall_seconds"] = time.monotonic() - before
        result["evaluation_batch_size"] = evaluation_batch
        _gzip_json(output_dir / f"{label}-input-contexts.json.gz", rows)
        _gzip_json(output_dir / f"{label}-anchors.json.gz", result["per_anchor"])
        _json(output_dir / f"{label}-users.json", result["per_user"])
        guard()
        return result

    def fit(variant: dict[str, Any], trial: dict[str, Any]) -> None:
        guard()
        name, trial_name = variant["name"], trial["name"]
        label = f"{name}-{trial_name}"
        physical_batch = (definition.training_batch(config, variant)
            if definition.training_batch is not None else hpo["physical_batch_size"])
        result: dict[str, Any] = {"status": "training", "complete": False, "trial_name": trial_name,
            "trial": dict(trial), "old_length": hpo["proxy_old_length"], "passes": [], "passes_completed": 0,
            "from_scratch": True, "physical_batch_size": physical_batch,
            "full_length_training_performed": False, "test_evaluations": 0}
        report["trials"].setdefault(name, {})[trial_name] = result
        persist()
        model = optimizer = scaler = None
        last: dict[str, Any] = {}
        attempted, stage = 0, "model_initialization"
        best = checkpoints / f"{label}.pt"
        resume = checkpoints / f"{label if definition.retained_trial_resumes else name}-resume.pt"
        checkpoint_config = {**config, "active_trial": dict(trial), "active_stage": definition.checkpoint_stage,
            "active_old_length": hpo["proxy_old_length"], "active_pass_cap": hpo["max_passes"]}
        stopping = ValidationEarlyStopping(-1., minimum_additional_passes=hpo["min_passes"],
            patience=hpo["patience"], validation_interval=hpo["validation_interval"],
            max_additional_passes=hpo["max_passes"])

        def batch_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
            arrays = (definition.batch_arrays or hpo_batch_arrays)(*args, **kwargs,
                old_length=hpo["proxy_old_length"])
            last.update(arrays=arrays, rng=capture_training_rng(device))
            return arrays

        def save_resume(epoch: int) -> None:
            _save_resume(resume, model=model, optimizer=optimizer, scaler=scaler, config=checkpoint_config,
                variant=variant, num_items=cohort.num_items, epoch=epoch, stopping=stopping, device=device)
            result["resumable_checkpoint"] = record_checkpoint("resume_checkpoints", resume,
                training_model=name, epoch=epoch, next_epoch=epoch + 1, trial=dict(trial),
                active_old_length=hpo["proxy_old_length"], optimizer_state_saved=True)
            if definition.retained_trial_resumes:
                result["resume_checkpoint_retained"] = True
            else:
                for other in report["trials"][name].values():
                    if "resumable_checkpoint" in other:
                        other["resume_checkpoint_retained"] = other is result

        try:
            model = definition.build_model(config, variant, trial, cohort.num_items).to(device)
            optimizer, scaler = (definition.optimizer or hpo_optimizer)(model, train, trial), _scaler(device, train)
            reset_restart_rng(config["seed"])
            compressor = getattr(model, "compressor", None)
            result.update(optimizer_groups=[{key: group[key] for key in ("component", "lr", "weight_decay")}
                for group in optimizer.param_groups], total_parameters=sum(p.numel() for p in model.parameters()),
                compressor_parameters=(sum(p.numel() for p in compressor.parameters())
                    if compressor is not None else 0))
            if definition.variants is not None:
                result["model_metadata"] = {"architecture": getattr(model, "architecture", "sasrec"),
                    "upstream_revision": getattr(model, "upstream_revision", None),
                    "native_runtime": getattr(model, "native_runtime", None),
                    "implementation": getattr(model, "attention_implementation", None),
                    "baseline_metadata": getattr(model, "baseline_metadata", {}),
                    "has_compressor": compressor is not None,
                    "initialization": "original matched SASRec initializer" if variant["kind"] in
                        ("recent", "query_only", "query_cross_attention", "query_ssm") else
                        "native backbone initialization; seed matched, architecture weights not shared"}
            for attempted, windows in enumerate(schedules, 1):
                guard()
                stage = "training"
                row = {"epoch": attempted, **train_prefix_pass(model, cohort, windows, config=config,
                    variant=variant, physical_batch=physical_batch, device=device,
                    optimizer=optimizer, scaler=scaler, batch_builder=batch_builder)}
                result["passes"].append(row)
                result["passes_completed"] = attempted
                if attempted % hpo["validation_interval"] == 0:
                    stage = "proxy_validation"
                    with _preserve_rng(device):
                        validation = evaluate(model, variant, hpo["proxy_old_length"], f"{label}-p{attempted}-validation")
                    decision = stopping.observe(attempted, validation["metrics"]["ndcg10"])
                    row.update(validation=validation["metrics"], selection=decision,
                        validation_wall_seconds=validation["wall_seconds"])
                    if decision["improved"]:
                        result.update(best_epoch=attempted, best_validation=validation["metrics"],
                            best_validation_rank_file=f"{label}-p{attempted}-validation-anchors.json.gz")
                        _save_best(best, model=model, config=checkpoint_config, variant=variant,
                            num_items=cohort.num_items, epoch=attempted, score=decision["best_score"])
                        result["best_checkpoint"] = record_checkpoint("checkpoints", best,
                            training_model=name, trial=dict(trial), best_epoch=attempted,
                            active_old_length=hpo["proxy_old_length"], precision="float32")
                    stage = "completed_pass_checkpoint"
                    save_resume(attempted)
                print(json.dumps({"phase": definition.phase + "_search_train", "model": name, "trial": trial_name, **row}), flush=True)
                persist()
                if stopping.stopped:
                    break
            if result.get("resumable_checkpoint", {}).get("epoch") != result["passes_completed"]:
                stage = "completed_pass_checkpoint"
                save_resume(result["passes_completed"])
            stopped_early = result["passes_completed"] < hpo["max_passes"]
            result.update(status="training_completed", complete=True,
                stopped_early=stopped_early, stopping_reason="patience" if stopped_early else "max_passes",
                total_labels=sum(row["labels"] for row in result["passes"]),
                successful_updates=sum(row["successful_updates"] for row in result["passes"]),
                amp_retried_update_attempts=sum(row["amp_skipped_update_attempts"] for row in result["passes"]),
                mean_training_pass_seconds=sum(row["seconds"] for row in result["passes"]) / result["passes_completed"])
        except (FloatingPointError, torch.OutOfMemoryError) as error:
            result.update(status="numerical_failed" if isinstance(error, FloatingPointError) else "out_of_memory",
                failure=_failure(error, stage=stage, completed=result["passes_completed"], attempted=attempted))
            if stage == "training" and model is not None and last:
                try:
                    result["failure_diagnostics"] = diagnose_nonfinite_batch(model, last["arrays"], device,
                        amp=train["amp"], rng_state=last["rng"])
                except Exception as diagnostic_error:
                    result["failure_diagnostics"] = {"complete": False,
                        "exception_type": type(diagnostic_error).__name__, "message": str(diagnostic_error),
                        "original_trial_failure_preserved": True}
            print(json.dumps({"phase": definition.phase + "_trial_failed", "model": name, "trial": trial_name,
                "status": result["status"], "failure": result["failure"]}), flush=True)
        finally:
            last.clear()
            del model, optimizer, scaler
            _cleanup()
            if definition.retained_trial_resumes:
                from rhpr.sasrec.query_search_retention import prune_trial_resumes
                manifest["resume_checkpoints"] = prune_trial_resumes(output_dir,
                    report["trials"], manifest["resume_checkpoints"], definition.retained_trial_resumes)
            persist()

    # Every short training job is resolved before the full-length transfer stage.
    for variant in variants:
        for trial in trials_by_family[variant["name"]]:
            if trial["name"] not in report["trials"].get(variant["name"], {}):
                fit(variant, trial)
    report["all_short_training_trials_resolved_before_full_validation"] = True
    full_candidates = (definition.full_transfer_candidates(config, variants, trials_by_family, report)
                       if definition.full_transfer_candidates is not None else None)
    if full_candidates is not None:
        registered_pairs = {(variant["name"], trial["name"]) for variant in variants
                            for trial in trials_by_family[variant["name"]]}
        if not full_candidates <= registered_pairs:
            raise ValueError("full-transfer candidates must be registered trials")
        report["full_transfer_candidates"] = [list(pair) for pair in sorted(full_candidates)]
    persist()
    print(json.dumps({"phase": definition.phase + "_full_validation_start"}), flush=True)
    if definition.full_transfer_budget_seconds is not None:
        if not definition.optional_full_transfer or definition.full_transfer_budget_seconds <= 0:
            raise ValueError("a positive full-transfer allowance requires optional secondary evaluation")
        optional_transfer_deadline = time.monotonic() + definition.full_transfer_budget_seconds
    for variant in variants:
        for trial in trials_by_family[variant["name"]]:
            result = report["trials"][variant["name"]][trial["name"]]
            if result["status"] != "training_completed":
                continue
            if full_candidates is not None and (variant["name"], trial["name"]) not in full_candidates:
                result.update(status="success", full_transfer_not_requested=True,
                    full_transfer_selection_metric="short_validation_ndcg10")
                persist()
                continue
            label = f"{variant['name']}-{trial['name']}"
            if (definition.full_transfer_required is not None
                    and not definition.full_transfer_required(config, variant)):
                # The model's exact input is already R128 or Q128+R128 in both
                # conditions. Reuse the selected native validation ranks, not
                # the final pass ranks, and avoid an identical second forward.
                result.update(status="success", best_full_validation=dict(result["best_validation"]),
                    full_validation_rank_file=result["best_validation_rank_file"],
                    full_validation_wall_seconds=0., full_validation_reuses_best_proxy=True,
                    effective_model_input_unchanged=True,
                    full_validation_is_frozen_1k_checkpoint=True)
                persist()
                continue
            profile_label = label if definition.profile_each_trial else variant["name"]
            full_preflight = report["preflight"].get(profile_label, {}).get("full_evaluation", {})
            if (definition.optional_full_transfer and full_preflight.get("status") in
                    ("out_of_memory", "numerical_failed")):
                result.update(status="full_transfer_failed", full_transfer_failure={
                    "stage": "preflight_full_length_evaluation", "preflight_result": full_preflight,
                    "primary_short_validation_retained": True, "full_evaluation_attempted": False})
                print(json.dumps({"phase": definition.phase + "_full_validation_skipped",
                    "model": variant["name"], "trial": trial["name"],
                    "reason": "registered full-shape preflight failed; primary result retained"}), flush=True)
                persist()
                continue
            model = None
            try:
                guard()
                model = definition.build_model(config, variant, trial, cohort.num_items).to(device)
                saved = torch.load(checkpoints / f"{label}.pt", map_location="cpu", weights_only=True)
                if (saved["best_epoch"] != result["best_epoch"]
                        or saved["experiment_config"]["active_trial"] != trial
                        or saved["experiment_config"]["active_old_length"] != hpo["proxy_old_length"]):
                    raise ValueError("1K checkpoint trial, input length or selected epoch mismatch")
                model.load_state_dict(saved["state_dict"], strict=True)
                del saved
                print(json.dumps({"phase": definition.phase + "_full_validation_trial_start",
                    "model": variant["name"], "trial": trial["name"]}), flush=True)
                full = evaluate(model, variant, None, f"{label}-full-validation")
                result.update(status="success", best_full_validation=full["metrics"],
                    full_validation_rank_file=f"{label}-full-validation-anchors.json.gz",
                    full_validation_wall_seconds=full["wall_seconds"],
                    full_validation_is_frozen_1k_checkpoint=True)
                print(json.dumps({"phase": definition.phase + "_full_validation_trial_complete",
                    "model": variant["name"], "trial": trial["name"], "metrics": full["metrics"],
                    "wall_seconds": full["wall_seconds"]}), flush=True)
            except (FloatingPointError, torch.OutOfMemoryError, _OptionalFullTransferTimeout) as error:
                result.update(status="full_transfer_failed", failure=_failure(error,
                    stage="full_length_validation", completed=result["passes_completed"]))
                print(json.dumps({"phase": definition.phase + "_trial_failed", "model": variant["name"],
                    "trial": trial["name"], "status": result["status"], "failure": result["failure"]}), flush=True)
            finally:
                del model
                _cleanup()
                persist()
    optional_transfer_deadline = None
    print(json.dumps({"phase": definition.phase + "_full_validation_complete"}), flush=True)
    missing = []
    print(json.dumps({"phase": definition.phase + "_paired_statistics_start"}), flush=True)
    for variant in variants:
        name = variant["name"]
        trials = trials_by_family[name]
        results = [report["trials"][name][trial["name"]] for trial in trials]
        selected = (definition.select_candidate or select_1k_candidate)(trials, results)
        if selected is None:
            missing.append(name)
            continue
        result = report["trials"][name][selected["name"]]
        report["selected_trials"][name] = dict(selected)
        report["cases"][name] = {key: result[key] for key in ("trial", "best_epoch", "best_validation",
            "best_full_validation", "best_validation_rank_file", "full_validation_rank_file", "best_checkpoint")
            if key in result}
        print(json.dumps({"phase": definition.phase + "_paired_statistics_family_start",
            "model": name, "trial": selected["name"]}), flush=True)
        if "full_validation_rank_file" in result:
            with gzip.open(output_dir / result["full_validation_rank_file"], "rt", encoding="utf-8") as stream:
                ranks = json.load(stream)
            report["cases"][name]["paired_parent_full_validation_contrasts"] = [
                {"reference": reference, "training_seed_uncertainty_included": False,
                 "selection_uncertainty_included": False, "exploratory_validation_selection": True,
                 "cutoffs": {str(k): paired_daily_ndcg_bootstrap(baseline, ranks, k=k,
                     seed=config["seed"], draws=hpo["bootstrap_draws"]) for k in (10, 100)}}
                for reference, baseline in validation_baselines.items()]
        if definition.paired_short_validation:
            with gzip.open(output_dir / result["best_validation_rank_file"], "rt", encoding="utf-8") as stream:
                short_ranks = json.load(stream)
            contrasts = []
            for reference, control in controls.items():
                with gzip.open(output_dir / control["best_validation_rank_file"], "rt", encoding="utf-8") as stream:
                    baseline = json.load(stream)
                contrasts.append({"reference": reference, "training_seed_uncertainty_included": False,
                    "selection_uncertainty_included": False, "exploratory_validation_selection": True,
                    "evaluation_condition": "registered native short input for both arms",
                    "cutoffs": {str(k): paired_daily_ndcg_bootstrap(baseline, short_ranks, k=k,
                        seed=config["seed"], draws=hpo["bootstrap_draws"]) for k in (10, 100)}})
            report["cases"][name]["paired_matched_short_validation_contrasts"] = contrasts
            report["cases"][name]["full_transfer_status"] = (
                "not_requested" if result.get("full_transfer_not_requested") else
                "success" if "best_full_validation" in result else result["status"])
        for prefix, quality in (("proxy", result["best_validation"]), ("full_transfer", result.get("best_full_validation", {}))):
            metrics.update({f"{name}_{prefix}_{key}": value for key, value in quality.items()})
        metrics[f"{name}_best_epoch"] = result["best_epoch"]
        metrics[f"{name}_passes_completed"] = result["passes_completed"]
        print(json.dumps({"phase": definition.phase + "_paired_statistics_family_complete",
            "model": name}), flush=True)
        persist()
    print(json.dumps({"phase": definition.phase + "_paired_statistics_complete"}), flush=True)
    report["families_without_finite_short_candidate" if definition.select_candidate is not None
        else "families_without_finite_full_candidate"] = missing
    _json(output_dir / "final-selection.json", {"selected_trials": report["selected_trials"],
        "selection_split": "validation", "selection_input": (
            "registered_native_short_context" if definition.select_candidate is not None else "full_history"),
        "checkpoint_training_input": ("per-family native short input; long arms H1024+R128"
            if definition.variants is not None else "H1024+R128"), "test_evaluations": 0})
    metrics.update(successful_trials=sum(row["status"] == "success" for family in report["trials"].values() for row in family.values()),
        failed_trials=sum(row["status"] != "success" for family in report["trials"].values() for row in family.values()))
    if definition.optional_full_transfer:
        metrics.update(primary_successful_families=len(report["selected_trials"]),
            full_transfer_failed_trials=sum(row["status"] == "full_transfer_failed"
                for family in report["trials"].values() for row in family.values()))
    persist()
    if missing:
        condition = "native short-context" if definition.select_candidate is not None else "full-H"
        raise RuntimeError("no finite " + condition + " validation candidate for family: " + ", ".join(missing))
    guard()
    report["complete"] = True
    metrics.update(quality_complete=True, suite_seconds=time.monotonic() - started)
    persist()
    return validate_metrics(metrics)
