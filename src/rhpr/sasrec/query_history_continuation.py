"""Paired short/full-history continuation from one selected checkpoint.

Every arm starts from identical family weights with fresh optimization state.
The fixed 20-pass endpoint is primary. Intermediate validation chooses only a
secondary best-within-budget checkpoint and never changes the training budget.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import iter_daily_evaluation_examples, load_prepared_daily_evaluation
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _cleanup, _json, _registered_device, _scaler
from rhpr.sasrec.query_1k_search import _failure, _finite_validation
from rhpr.sasrec.query_history_checkpoints import prepare_history_source, load_history_source_weights
from rhpr.sasrec.query_history_models import (
    build_history_model, conservative_history_budget, history_optimizer, preflight_history,
)
from rhpr.sasrec.query_history_protocol import (
    history_arms, history_batch_arrays, history_variant, validate_history_config,
)
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_numerics import capture_training_rng, diagnose_nonfinite_batch
from rhpr.sasrec.query_prefix_pilot import (
    _checkpoint_record, _save_best, _save_resume, _validate_complete_histories, train_prefix_pass,
)
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_schedule_summary
from rhpr.sasrec.query_profile import audit_native_query_scan
from rhpr.sasrec.query_proxy_diagnostics import audited_hpo_examples, load_proxy_parent_artifacts
from rhpr.sasrec.query_restart_state import ValidationEarlyStopping, reset_restart_rng


def run_history_continuation(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    validate_history_config(config)
    started, device = time.monotonic(), _registered_device()
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    train, bench, policy = config["training"], config["benchmark"], config["continuation"]
    variant, arms = history_variant(config), history_arms(config)
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "family": config["family"], "seed": config["seed"],
        "quality_complete": False, "hpo_performed": False, "test_evaluations": 0,
        "training_models": 2, "additional_passes_per_arm": policy["additional_passes"],
        "train_is_exhaustive_event_epoch": False, "exploratory_previously_inspected_holdout": True,
    }
    report: dict[str, Any] = {"complete": False, "config": config,
        "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "protocol": {"training": "paired weight-only continuation; fixed20 additional passes in each arm",
            "source": "one selected short-history family checkpoint, identical initial weights in both arms",
            "optimization": "fresh Adam, scaler and seed42 per arm; same schedules/targets/negative identities",
            "primary_selection": "fixed additional-pass20 endpoint; no early stopping or epoch selection",
            "secondary_selection": "best validation NDCG10 among passes0/10/20, earliest tie",
            "full_history": "actual full observed-prefix optimization in full arm; no frozen-1K transfer label",
            "position_semantics": "QSSM/SASRec recency positions; FuXi native sequence-indexed positions",
            "precision": "native QSSM CUDA scan; FuXi FP32 attention accumulation/products/reads in BOTH arms",
            "comparison": "within-family length adaptation; source checkpoints have different prior training budgets",
            "test": "validation only; test ranks and predictions are not read"},
        "preflight": {}, "arms": {}}
    manifest: dict[str, Any] = {"format_version": 3, "source_code_sha": report["source_code_sha"],
        "config": config, "checkpoints": [], "resume_checkpoints": [],
        "optimizer_state_saved": True, "best_checkpoint_optimizer_state_saved": False,
        "resume_retention": "one final completed-pass Adam/scaler/RNG checkpoint for each arm"}

    def persist() -> None:
        _json(output_dir / "history-continuation.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("history continuation exceeded its registered budget; partial outputs retained")

    def record(key: str, path: Path, **metadata: Any) -> dict[str, Any]:
        row = _checkpoint_record(path, output_dir, **metadata)
        manifest[key] = [old for old in manifest[key] if old["path"] != row["path"]] + [row]
        return row

    persist()
    if config["family"] == "qssm":
        audit = audit_native_query_scan(device, query_source="old_suffix")
        _json(output_dir / "native-query-scan-audit.json", audit)
        report["native_scan_audit"] = audit
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
    _, contexts, _, _ = load_proxy_parent_artifacts(config, evaluation)
    source = prepare_history_source(config, evaluation, output_dir)
    report["source_checkpoint"] = source
    manifest["checkpoints"].append(source["checkpoint_record"])
    metrics.update(cohort_users=len(cohort.users), num_items=cohort.num_items,
        source_best_epoch=config["selected_source"]["checkpoint"]["best_epoch"])

    def loaded_model() -> torch.nn.Module:
        guard()
        model = build_history_model(config, cohort.num_items)
        load_history_source_weights(source, model, num_items=cohort.num_items)
        model.history_source_checkpoint_loaded = True
        return model.to(device)

    for arm in arms:
        guard()
        print(json.dumps({"phase": "history_preflight_start", "family": config["family"], "arm": arm}), flush=True)
        try:
            with _preserve_rng(device):
                report["preflight"][arm["name"]] = preflight_history(config, cohort, evaluation,
                    arm, device, load_model=loaded_model)
        except (FloatingPointError, torch.OutOfMemoryError) as error:
            report["preflight"][arm["name"]] = {"status": "numerical_failed" if isinstance(error, FloatingPointError)
                else "out_of_memory", "failure": _failure(error, stage="preflight", completed=0)}
        print(json.dumps({"phase": "history_preflight", "arm": arm["name"],
            **report["preflight"][arm["name"]]}), flush=True)
        persist()
    if any(row.get("status") != "success" for row in report["preflight"].values()):
        raise RuntimeError("both registered history shapes must pass loaded-weight preflight before any quality training")
    report["budget"] = conservative_history_budget(config, report["preflight"],
        time.monotonic() - started, len(cohort.users), evaluation.summary)
    persist()
    if not report["budget"]["fits"]:
        raise RuntimeError("measured full-history cost exceeds paired20 budget; no automatic trial/pass reduction")
    schedules = [fixed_prefix_schedule(cohort, seed=config["seed"], epoch=policy["schedule_epoch_offset"] + extra)
                 for extra in range(1, policy["additional_passes"] + 1)]
    _json(output_dir / "window-schedules.json", {"shared_between_arms": True,
        "schedule_epoch_offset": policy["schedule_epoch_offset"],
        "full_history_training_registered": True, "full_arm_retains_all_observed_old_events": True,
        "schedules": [prefix_schedule_summary(cohort, rows, config["parent"]["experiment_config"])
                      for rows in schedules]})

    def evaluate(model: torch.nn.Module, arm: dict[str, Any], extra: int) -> dict[str, Any]:
        guard()
        label = f"{arm['name']}-p{extra}-validation"
        rows: list[dict[str, Any]] = []
        examples = iter_daily_evaluation_examples(cohort, evaluation, split_name="validation",
            max_seq_len=train["serving_context_length"])
        audited = audited_hpo_examples(examples, arm["old_length"], contexts["validation"]["rows"], rows, guard)
        def actual_examples():
            for example in audited:
                if config["family"] == "fuxi":
                    rows[-1]["retained_position_recencies_preserved"] = False
                    rows[-1]["native_sequence_indexed_positions"] = True
                yield example
        before = time.monotonic()
        try:
            result = evaluate_daily_quality(model, actual_examples(), device=device, amp=train["amp"],
                batch_size=arm["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        except RuntimeError as error:
            if str(error) in ("evaluation produced nonfinite final states", "evaluation produced nonfinite catalogue scores"):
                raise FloatingPointError(str(error)) from error
            raise
        _finite_validation(result)
        _gzip_json(output_dir / f"{label}-anchors.json.gz", result["per_anchor"])
        _gzip_json(output_dir / f"{label}-input-contexts.json.gz", rows)
        _json(output_dir / f"{label}-users.json", result["per_user"])
        return {"metrics": result["metrics"], "rank_file": f"{label}-anchors.json.gz",
            "contexts_file": f"{label}-input-contexts.json.gz", "wall_seconds": time.monotonic() - before,
            "evaluation_batch_size": arm["evaluation_batch_size"], "training_old_length": arm["old_length"]}

    for arm in arms:
        name = arm["name"]
        result: dict[str, Any] = {"status": "training", "complete": False, "arm": arm,
            "passes": [], "passes_completed": 0, "evaluations": {}, "weight_only_continuation": True,
            "optimizer_reset": True, "scaler_reset": True, "rng_reset": True,
            "full_length_training_performed": False, "test_evaluations": 0}
        report["arms"][name] = result
        model = optimizer = scaler = None
        attempted, stage, last = 0, "model_initialization", {}
        active_config = {**config, "active_arm": arm, "active_stage": "paired_history_continuation",
            "full_length_training_performed": False,
            "source_best_epoch": config["selected_source"]["checkpoint"]["best_epoch"]}
        try:
            model = loaded_model()
            optimizer, scaler = history_optimizer(model, train), _scaler(device, train)
            if optimizer.state:
                raise ValueError("paired continuation must start with empty Adam state")
            reset_restart_rng(config["seed"])
            result["initial_optimizer_state_entries"] = len(optimizer.state)
            result["optimizer_groups"] = [{key: group[key] for key in ("component", "lr", "weight_decay")}
                                          for group in optimizer.param_groups]
            stage = "initial_validation"
            with _preserve_rng(device):
                initial = evaluate(model, arm, 0)
            result["evaluations"]["0"] = initial
            stopping = ValidationEarlyStopping(initial["metrics"]["ndcg10"], minimum_additional_passes=20,
                patience=10, validation_interval=10, max_additional_passes=20)
            best = checkpoints / f"{name}-best.pt"

            def save_best(extra: int, score: float) -> None:
                _save_best(best, model=model, config=active_config, variant=variant,
                    num_items=cohort.num_items, epoch=extra, score=score)
                result["best_checkpoint"] = record("checkpoints", best, training_model=name,
                    checkpoint_role="secondary_best_within_budget", best_epoch=extra,
                    training_old_length=arm["old_length"],
                    full_length_training_performed=extra > 0 and arm["old_length"] is None)
                result.update(best_epoch=extra, best_validation=score)

            save_best(0, initial["metrics"]["ndcg10"])

            def batch_builder(cohort_arg, windows_arg, _config, _variant):
                guard()
                arrays = history_batch_arrays(cohort_arg, windows_arg, config, arm)
                last.update(arrays=arrays, rng=capture_training_rng(device))
                return arrays

            for attempted, windows in enumerate(schedules, 1):
                guard()
                stage = "training"
                row = {"additional_pass": attempted, "schedule_epoch": policy["schedule_epoch_offset"] + attempted,
                    **train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                        physical_batch=arm["physical_batch_size"], device=device, optimizer=optimizer,
                        scaler=scaler, batch_builder=batch_builder)}
                result["passes"].append(row)
                result["passes_completed"] = attempted
                result["full_length_training_performed"] = arm["old_length"] is None
                active_config["full_length_training_performed"] = arm["old_length"] is None
                if attempted in policy["validation_passes"]:
                    stage = "validation"
                    with _preserve_rng(device):
                        validation = evaluate(model, arm, attempted)
                    result["evaluations"][str(attempted)] = validation
                    decision = stopping.observe(attempted, validation["metrics"]["ndcg10"])
                    if decision["improved"]:
                        save_best(attempted, validation["metrics"]["ndcg10"])
                    resume = checkpoints / f"{name}-resume.pt"
                    _save_resume(resume, model=model, optimizer=optimizer, scaler=scaler, config=active_config,
                        variant=variant, num_items=cohort.num_items, epoch=attempted, stopping=stopping, device=device)
                    result["resumable_checkpoint"] = record("resume_checkpoints", resume, training_model=name,
                        epoch=attempted, next_epoch=attempted + 1, optimizer_state_saved=True,
                        absolute_schedule_epoch=policy["schedule_epoch_offset"] + attempted,
                        full_length_training_performed=arm["old_length"] is None)
                    row["validation"] = validation["metrics"]
                print(json.dumps({"phase": "history_train", "family": config["family"], "arm": name, **row}), flush=True)
                persist()
            endpoint = result["evaluations"][str(policy["additional_passes"])]["metrics"]
            path = checkpoints / f"{name}-endpoint.pt"
            _save_best(path, model=model, config=active_config, variant=variant, num_items=cohort.num_items,
                epoch=policy["additional_passes"], score=endpoint["ndcg10"])
            result["endpoint_checkpoint"] = record("checkpoints", path, training_model=name,
                checkpoint_role="primary_fixed_endpoint", epoch=policy["additional_passes"],
                training_old_length=arm["old_length"], full_length_training_performed=arm["old_length"] is None)
            result.update(status="success", complete=True, endpoint_validation=endpoint,
                total_labels=sum(row["labels"] for row in result["passes"]),
                successful_updates=sum(row["successful_updates"] for row in result["passes"]),
                mean_training_pass_seconds=sum(row["seconds"] for row in result["passes"]) / len(result["passes"]))
            metrics.update({f"{name}_endpoint_{key}": value for key, value in endpoint.items()})
        except (FloatingPointError, torch.OutOfMemoryError) as error:
            result.update(status="numerical_failed" if isinstance(error, FloatingPointError) else "out_of_memory",
                failure=_failure(error, stage=stage, completed=result["passes_completed"], attempted=attempted))
            if stage == "training" and last:
                try:
                    result["failure_diagnostics"] = diagnose_nonfinite_batch(model, last["arrays"], device,
                        amp=train["amp"], rng_state=last["rng"])
                except Exception as diagnostic_error:
                    result["failure_diagnostics"] = {"complete": False,
                        "exception_type": type(diagnostic_error).__name__, "message": str(diagnostic_error),
                        "original_trial_failure_preserved": True}
        finally:
            last.clear()
            del model, optimizer, scaler
            _cleanup()
            persist()
    if any(row.get("status") != "success" for row in report["arms"].values()):
        raise RuntimeError("paired history comparison requires both complete20-pass arms; partial results retained")

    def ranks(name: str, extra: int) -> list[dict[str, Any]]:
        with gzip.open(output_dir / report["arms"][name]["evaluations"][str(extra)]["rank_file"], "rt") as stream:
            return json.load(stream)
    report["paired_contrasts"] = {str(extra): {str(k): paired_daily_ndcg_bootstrap(
        ranks("short", extra), ranks("full", extra), k=k, seed=config["seed"], draws=policy["bootstrap_draws"])
        for k in (10, 100)} for extra in (0, policy["additional_passes"])}
    for k in (10, 100):
        delta = report["paired_contrasts"][str(policy["additional_passes"])][str(k)]["delta_ndcg"]
        initial = report["paired_contrasts"]["0"][str(k)]["delta_ndcg"]
        metrics[f"full_minus_short_endpoint_ndcg{k}"] = delta
        metrics[f"length_adaptation_difference_in_differences_ndcg{k}"] = delta - initial
    _json(output_dir / "final-selection.json", {"selection_split": "validation",
        "primary": "fixed20-pass endpoint in each arm", "secondary": "best within0/10/20",
        "source_checkpoint": config["selected_source"]["checkpoint"], "test_evaluations": 0})
    guard()
    report["complete"] = True
    metrics.update(quality_complete=True, suite_seconds=time.monotonic() - started)
    persist()
    return validate_metrics(metrics)
