"""Validation-only 1K LR search with frozen-weight full-length transfer checks.

Six independent short-history models are trained from the parent initializer.
Their best 1K-validation weights are retained, then evaluated once on full H.
No full-history optimization or test-rank deserialization occurs in this run.
"""
from __future__ import annotations

from functools import partial
import gzip
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

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


def run_query_1k_search(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    validate_1k_config(config)
    started, device = time.monotonic(), _registered_device()
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    variants, trials = hpo_variants(config), hpo_trials(config)
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "seed": config["seed"],
        "quality_complete": False, "hpo_performed": True, "proxy_training_models": len(variants) * len(trials),
        "full_training_models": 0, "test_evaluations": 0, "train_is_exhaustive_event_epoch": False,
        "exploratory_previously_inspected_holdout": True}
    report: dict[str, Any] = {"complete": False, "config": config,
        "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "protocol": {"training": "six fresh H1024+R128 models, minimum30 maximum60 passes, validation every10",
            "stopping": "strict macro-user NDCG10 improvement; patience3 checks; no stop before30",
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

    def persist() -> None:
        _json(output_dir / "query-1k-search.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("1K search exceeded registered four-hour budget; partial outputs retained")

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
    for variant in variants:
        guard()
        with _preserve_rng(device):
            row = _preflight_1k(config, cohort, evaluation, variant, device)
        report["preflight"][variant["name"]] = row
        print(json.dumps({"phase": "one_k_preflight", "model": variant["name"], **row}), flush=True)
        persist()
    report["budget"] = conservative_1k_budget(config, report["preflight"], time.monotonic() - started,
        len(cohort.users), evaluation.summary)
    persist()
    if not report["budget"]["fits"]:
        raise RuntimeError("measured cost cannot support all six maximum60-pass trials and six full-H validations")
    schedules = [fixed_prefix_schedule(cohort, seed=config["seed"], epoch=epoch)
                 for epoch in range(1, hpo["max_passes"] + 1)]
    _json(output_dir / "window-schedules.json", {
        "original_parent_schedules": [prefix_schedule_summary(cohort, windows,
            config["parent"]["experiment_config"]) for windows in schedules],
        "proxy_input_old_length_cap": hpo["proxy_old_length"], "proxy_retains_all_old_events": False,
        "full_length_training_performed": False, "raw_tail_size": train["raw_tail_size"]})

    def evaluate(model: torch.nn.Module, old: int | None, label: str) -> dict[str, Any]:
        guard()
        rows: list[dict[str, Any]] = []
        examples = iter_daily_evaluation_examples(cohort, evaluation, split_name="validation",
            max_seq_len=train["serving_context_length"])
        audited = audited_hpo_examples(examples, old, contexts["validation"]["rows"], rows, guard)
        before = time.monotonic()
        try:
            result = evaluate_daily_quality(model, audited, device=device, amp=train["amp"],
                batch_size=bench["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        except RuntimeError as error:
            if str(error) in ("evaluation produced nonfinite final states", "evaluation produced nonfinite catalogue scores"):
                raise FloatingPointError(str(error)) from error
            raise
        _finite_validation(result)
        result["wall_seconds"] = time.monotonic() - before
        _gzip_json(output_dir / f"{label}-input-contexts.json.gz", rows)
        _gzip_json(output_dir / f"{label}-anchors.json.gz", result["per_anchor"])
        _json(output_dir / f"{label}-users.json", result["per_user"])
        guard()
        return result

    def fit(variant: dict[str, Any], trial: dict[str, Any]) -> None:
        guard()
        name, trial_name = variant["name"], trial["name"]
        label = f"{name}-{trial_name}"
        result: dict[str, Any] = {"status": "training", "complete": False, "trial_name": trial_name,
            "trial": dict(trial), "old_length": hpo["proxy_old_length"], "passes": [], "passes_completed": 0,
            "from_scratch": True, "physical_batch_size": hpo["physical_batch_size"],
            "full_length_training_performed": False, "test_evaluations": 0}
        report["trials"].setdefault(name, {})[trial_name] = result
        persist()
        model = optimizer = scaler = None
        last: dict[str, Any] = {}
        attempted, stage = 0, "model_initialization"
        best = checkpoints / f"{label}.pt"
        resume = checkpoints / f"{name}-resume.pt"
        checkpoint_config = {**config, "active_trial": dict(trial), "active_stage": "one_k_search",
            "active_old_length": hpo["proxy_old_length"], "active_pass_cap": hpo["max_passes"]}
        stopping = ValidationEarlyStopping(-1., minimum_additional_passes=hpo["min_passes"],
            patience=hpo["patience"], validation_interval=hpo["validation_interval"],
            max_additional_passes=hpo["max_passes"])

        def batch_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
            arrays = hpo_batch_arrays(*args, **kwargs, old_length=hpo["proxy_old_length"])
            last.update(arrays=arrays, rng=capture_training_rng(device))
            return arrays

        def save_resume(epoch: int) -> None:
            _save_resume(resume, model=model, optimizer=optimizer, scaler=scaler, config=checkpoint_config,
                variant=variant, num_items=cohort.num_items, epoch=epoch, stopping=stopping, device=device)
            result["resumable_checkpoint"] = record_checkpoint("resume_checkpoints", resume,
                training_model=name, epoch=epoch, next_epoch=epoch + 1, trial=dict(trial),
                active_old_length=hpo["proxy_old_length"], optimizer_state_saved=True)
            for other in report["trials"][name].values():
                if "resumable_checkpoint" in other:
                    other["resume_checkpoint_retained"] = other is result

        try:
            model = build_hpo_model(config, variant, cohort.num_items).to(device)
            optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
            reset_restart_rng(config["seed"])
            result.update(optimizer_groups=[{key: group[key] for key in ("component", "lr", "weight_decay")}
                for group in optimizer.param_groups], total_parameters=sum(p.numel() for p in model.parameters()),
                compressor_parameters=sum(p.numel() for p in model.compressor.parameters()))
            for attempted, windows in enumerate(schedules, 1):
                guard()
                stage = "training"
                row = {"epoch": attempted, **train_prefix_pass(model, cohort, windows, config=config,
                    variant=variant, physical_batch=hpo["physical_batch_size"], device=device,
                    optimizer=optimizer, scaler=scaler, batch_builder=batch_builder)}
                result["passes"].append(row)
                result["passes_completed"] = attempted
                if attempted % hpo["validation_interval"] == 0:
                    stage = "proxy_validation"
                    with _preserve_rng(device):
                        validation = evaluate(model, hpo["proxy_old_length"], f"{label}-p{attempted}-validation")
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
                print(json.dumps({"phase": "one_k_search_train", "model": name, "trial": trial_name, **row}), flush=True)
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
            print(json.dumps({"phase": "one_k_trial_failed", "model": name, "trial": trial_name,
                "status": result["status"], "failure": result["failure"]}), flush=True)
        finally:
            last.clear()
            del model, optimizer, scaler
            _cleanup()
            persist()

    # Every short training job is resolved before the full-length transfer stage.
    for variant in variants:
        for trial in trials:
            fit(variant, trial)
    report["all_short_training_trials_resolved_before_full_validation"] = True
    persist()
    for variant in variants:
        for trial in trials:
            result = report["trials"][variant["name"]][trial["name"]]
            if result["status"] != "training_completed":
                continue
            guard()
            label = f"{variant['name']}-{trial['name']}"
            model = None
            try:
                model = build_hpo_model(config, variant, cohort.num_items).to(device)
                saved = torch.load(checkpoints / f"{label}.pt", map_location="cpu", weights_only=True)
                if (saved["best_epoch"] != result["best_epoch"]
                        or saved["experiment_config"]["active_trial"] != trial
                        or saved["experiment_config"]["active_old_length"] != hpo["proxy_old_length"]):
                    raise ValueError("1K checkpoint trial, input length or selected epoch mismatch")
                model.load_state_dict(saved["state_dict"], strict=True)
                del saved
                full = evaluate(model, None, f"{label}-full-validation")
                result.update(status="success", best_full_validation=full["metrics"],
                    full_validation_rank_file=f"{label}-full-validation-anchors.json.gz",
                    full_validation_wall_seconds=full["wall_seconds"],
                    full_validation_is_frozen_1k_checkpoint=True)
            except (FloatingPointError, torch.OutOfMemoryError) as error:
                result.update(status="full_transfer_failed", failure=_failure(error,
                    stage="full_length_validation", completed=result["passes_completed"]))
            finally:
                del model
                _cleanup()
                persist()
    missing = []
    for variant in variants:
        name = variant["name"]
        results = [report["trials"][name][trial["name"]] for trial in trials]
        selected = select_1k_candidate(trials, results)
        if selected is None:
            missing.append(name)
            continue
        result = report["trials"][name][selected["name"]]
        report["selected_trials"][name] = dict(selected)
        report["cases"][name] = {key: result[key] for key in ("trial", "best_epoch", "best_validation",
            "best_full_validation", "best_validation_rank_file", "full_validation_rank_file", "best_checkpoint")}
        with gzip.open(output_dir / result["full_validation_rank_file"], "rt", encoding="utf-8") as stream:
            ranks = json.load(stream)
        report["cases"][name]["paired_parent_full_validation_contrasts"] = [
            {"reference": reference, "training_seed_uncertainty_included": False,
             "selection_uncertainty_included": False, "exploratory_validation_selection": True,
             "cutoffs": {str(k): paired_daily_ndcg_bootstrap(baseline, ranks, k=k,
                 seed=config["seed"], draws=hpo["bootstrap_draws"]) for k in (10, 100)}}
            for reference, baseline in validation_baselines.items()]
        for prefix, quality in (("proxy", result["best_validation"]), ("full_transfer", result["best_full_validation"])):
            metrics.update({f"{name}_{prefix}_{key}": value for key, value in quality.items()})
        metrics[f"{name}_best_epoch"] = result["best_epoch"]
        metrics[f"{name}_passes_completed"] = result["passes_completed"]
    report["families_without_finite_full_candidate"] = missing
    _json(output_dir / "final-selection.json", {"selected_trials": report["selected_trials"],
        "selection_split": "validation", "selection_input": "full_history",
        "checkpoint_training_input": "H1024+R128", "test_evaluations": 0})
    metrics.update(successful_trials=sum(row["status"] == "success" for family in report["trials"].values() for row in family.values()),
        failed_trials=sum(row["status"] != "success" for family in report["trials"].values() for row in family.values()))
    persist()
    if missing:
        raise RuntimeError("no finite full-H validation candidate for family: " + ", ".join(missing))
    guard()
    report["complete"] = True
    metrics.update(quality_complete=True, suite_seconds=time.monotonic() - started)
    persist()
    return validate_metrics(metrics)
