"""Bounded validation-only 4K LR screening, then fresh full-history finalists.

Proxy weights are deliberately discarded: their validation ranks choose a
learning-rate recipe, never initialize a full-history model. Both full-history
checkpoint selections are frozen before either test set is evaluated.
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
from rhpr.sasrec.query_history_ablation import _verified_file
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_only_control import _read_baselines
from rhpr.sasrec.query_prefix_pilot import (
    _checkpoint_record, _profile_daily_rate, _representative_windows, _save_best,
    _save_resume, _validate_complete_histories, train_prefix_pass,
)
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_schedule_summary
from rhpr.sasrec.query_profile import audit_native_query_scan
from rhpr.sasrec.query_proxy_diagnostics import (
    audited_hpo_examples, load_proxy_parent_artifacts, run_proxy_diagnostics,
)
from rhpr.sasrec.query_proxy_protocol import (
    build_hpo_model, hpo_batch_arrays, hpo_optimizer, hpo_trials, hpo_variants,
    validate_proxy_hpo_config,
)
from rhpr.sasrec.query_restart_state import ValidationEarlyStopping, reset_restart_rng


def select_proxy_trial(trials: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Strict validation improvement; an exact tie keeps configuration order."""
    if len(trials) != len(results) or not trials:
        raise ValueError("every registered trial must finish before proxy selection")
    selected, best = None, -math.inf
    for trial, result in zip(trials, results, strict=True):
        score = result.get("best_validation", {}).get("ndcg10")
        if (not result.get("complete") or result.get("trial") != trial or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("proxy selection requires complete matching trials and finite validation scores")
        if score > best:
            selected, best = dict(trial), score
    assert selected is not None
    return selected


def conservative_hpo_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                            users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price all fixed trials and finalists prospectively; never shorten the grid."""
    hpo, train = config["hpo"], config["training"]
    work = 0.0
    for variant in hpo_variants(config):
        for phase, passes, copies in (("proxy", hpo["proxy_passes"], len(hpo_trials(config))),
                                      ("full", hpo["final_passes"], 1)):
            row = profiles[variant["name"]][phase]
            rate, daily = row["training"]["windows_per_second"], row["daily_evaluation"]["anchors_per_second"]
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or x <= 0 for x in (rate, daily)):
                raise ValueError("positive finite measured preflight rates are required")
            anchors = passes // hpo["validation_interval"] * evaluation_summary["validation_anchors"]
            if phase == "full":
                anchors += evaluation_summary["test_anchors"]
            work += copies * (passes * users / rate + anchors / daily)
    # Two best weights + two complete Adam resumes, hashing, paired statistics,
    # and B1/B8/B32 inference benchmarks. Diagnostics/preflights are in elapsed.
    overhead = 240.0 * len(hpo_variants(config))
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_benchmark_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "auto_reduces_trials_or_passes": False}


def _preflight_shape(config: dict[str, Any], cohort: Any, evaluation: Any,
                     variant: dict[str, Any], old_length: int | None,
                     device: torch.device) -> dict[str, Any]:
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    trial = max(hpo_trials(config), key=lambda row: (row["compressor_lr"], row["backbone_lr"]))
    model = build_hpo_model(config, variant, cohort.num_items).to(device)
    optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
    try:
        schedule = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
        windows = _representative_windows(schedule, train["effective_batch_size"])
        batch = partial(hpo_batch_arrays, old_length=old_length)
        rows = []
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=hpo["physical_batch_size"], device=device, optimizer=optimizer,
                scaler=scaler, batch_builder=batch)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("HPO preflight requires warmup and measured successful effective Adam updates without AMP retries")
            rows.append(row)
        cap = train["serving_context_length"] if old_length is None else old_length + train["raw_tail_size"]
        daily = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
            batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
            item_chunk_size=bench["item_chunk_size"])
        return {"warmup": rows[0], "training": rows[1], "daily_evaluation": daily,
            "old_length": old_length, "profile_trial": trial, "physical_batch_size": hpo["physical_batch_size"],
            "effective_batch_size": train["effective_batch_size"], "profile_weights_reused": False,
            "quality_used_for_selection": False, "optimizer_state_allocated_during_measurement": True}
    finally:
        del model, optimizer, scaler
        _cleanup()


def _contrasts(references: dict[str, Any], rows: list[dict[str, Any]], *, name: str,
               config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"reference": reference, "candidate": name,
        "training_seed_uncertainty_included": False, "exploratory_previously_inspected_holdout": True,
        "cutoffs": {str(k): paired_daily_ndcg_bootstrap(baseline, rows, k=k, seed=config["seed"],
            draws=config["hpo"]["bootstrap_draws"]) for k in (10, 100)}}
        for reference, baseline in references.items()]


def run_query_proxy_hpo(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_prefix_benchmark import benchmark_prefix_quality

    validate_proxy_hpo_config(config)
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
        "full_training_models": len(variants), "test_evaluations": 0, "train_is_exhaustive_event_epoch": False,
        "exploratory_previously_inspected_holdout": True}
    report: dict[str, Any] = {"complete": False, "config": config,
        "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "protocol": {"screening": "six fresh 4K-old + R128 models,20 fixed passes, validation10/20",
            "selection": "earliest strictly best macro-user NDCG10; trial ties keep configuration order",
            "finalists": "one chosen LR recipe per family; fresh parent initialization, full H,30 fixed passes",
            "targets": "same fixed last128 labels, original endpoints, user order and identity-bound negatives",
            "test": "both final checkpoint choices fixed before two exploratory tests; no test-based tuning",
            "proxy_weights": "not saved or transferred; ranks and selection statistics retained",
            "serving": "uncached full-history benchmark; cache quality/cost requires a later serving experiment"},
        "preflight": {}, "proxy": {}, "selected_trials": {}, "finalists": {}, "cases": {}}
    manifest: dict[str, Any] = {"format_version": 3, "source_code_sha": report["source_code_sha"],
        "config": config, "optimizer_state_saved": True, "best_checkpoint_optimizer_state_saved": False,
        "proxy_checkpoints_saved": False, "checkpoints": [], "resume_checkpoints": []}

    def persist() -> None:
        _json(output_dir / "query-proxy-hpo.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("proxy HPO exceeded the registered6h budget; partial outputs retained")

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
    root, contexts, validation_baselines, _ = load_proxy_parent_artifacts(config, evaluation)
    # Hash test artifacts now, but neither deserialize nor inspect their ranks.
    baseline_paths = {name: {split: _verified_file(root, spec["file"], spec["sha256"])
        for split, spec in splits.items()} for name, splits in config["parent"]["baselines"].items()}
    report["saved_baseline_validation"] = {name: _aggregate(rows)[0] for name, rows in validation_baselines.items()}
    metrics.update(cohort_users=len(cohort.users), num_items=cohort.num_items,
        unique_supervised_target_positions_per_model=len(cohort.users) * train["supervised_targets"])
    guard()
    with _preserve_rng(device):
        report["frozen_diagnostics"] = run_proxy_diagnostics(config, cohort, evaluation, device, output_dir, guard)
    persist()
    for variant in variants:
        report["preflight"][variant["name"]] = {}
        for phase, old in (("proxy", hpo["proxy_old_length"]), ("full", None)):
            guard()
            with _preserve_rng(device):
                row = _preflight_shape(config, cohort, evaluation, variant, old, device)
            report["preflight"][variant["name"]][phase] = row
            print(json.dumps({"phase": "proxy_hpo_preflight", "model": variant["name"], "context": phase, **row}), flush=True)
            persist()
    guard()
    report["budget"] = conservative_hpo_budget(config, report["preflight"], time.monotonic() - started,
        len(cohort.users), evaluation.summary)
    persist()
    if not report["budget"]["fits"]:
        raise RuntimeError("measured cost cannot support the complete registered HPO grid and both fresh finalists")
    schedules = [fixed_prefix_schedule(cohort, seed=config["seed"], epoch=epoch)
                 for epoch in range(1, max(hpo["proxy_passes"], hpo["final_passes"]) + 1)]
    _json(output_dir / "window-schedules.json", {
        "original_parent_schedules": [prefix_schedule_summary(cohort, windows,
            config["parent"]["experiment_config"]) for windows in schedules],
        "proxy_input_old_length_cap": hpo["proxy_old_length"], "proxy_retains_all_old_events": False,
        "finalist_retains_all_old_events": True, "raw_tail_size": train["raw_tail_size"]})

    def evaluate(model: torch.nn.Module, split: str, old: int | None, label: str) -> dict[str, Any]:
        guard()
        rows: list[dict[str, Any]] = []
        examples = iter_daily_evaluation_examples(cohort, evaluation, split_name=split,
            max_seq_len=train["serving_context_length"])
        audited = audited_hpo_examples(examples, old, contexts[split]["rows"], rows, guard)
        before = time.monotonic()
        result = evaluate_daily_quality(model, audited, device=device, amp=train["amp"],
            batch_size=bench["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        result["wall_seconds"] = time.monotonic() - before
        _gzip_json(output_dir / f"{label}-input-contexts.json.gz", rows)
        _gzip_json(output_dir / f"{label}-anchors.json.gz", result["per_anchor"])
        guard()
        return result

    def fit(variant: dict[str, Any], trial: dict[str, Any], *, old: int | None,
            passes: int, label: str, final: bool) -> dict[str, Any]:
        guard()
        model = build_hpo_model(config, variant, cohort.num_items).to(device)
        optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
        reset_restart_rng(config["seed"])
        stopping = ValidationEarlyStopping(-1., minimum_additional_passes=passes, patience=3,
            validation_interval=hpo["validation_interval"], max_additional_passes=passes)
        result: dict[str, Any] = {"complete": False, "trial": trial, "old_length": old,
            "passes": [], "from_scratch": True, "physical_batch_size": hpo["physical_batch_size"],
            "optimizer_groups": [{key: group[key] for key in ("component", "lr", "weight_decay")}
                                 for group in optimizer.param_groups],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters())}
        destination = report["finalists"] if final else report["proxy"].setdefault(variant["name"], {})
        destination[variant["name"] if final else trial["name"]] = result
        batch_builder = partial(hpo_batch_arrays, old_length=old)
        best, resume = checkpoints / f"{variant['name']}.pt", checkpoints / f"{variant['name']}-resume.pt"
        checkpoint_config = {**config, "active_trial": dict(trial), "active_stage": "full_finalist",
            "active_old_length": old, "active_pass_cap": passes}
        try:
            for epoch, windows in enumerate(schedules[:passes], 1):
                guard()
                row = {"epoch": epoch, **train_prefix_pass(model, cohort, windows, config=config,
                    variant=variant, physical_batch=hpo["physical_batch_size"], device=device,
                    optimizer=optimizer, scaler=scaler, batch_builder=batch_builder)}
                if epoch % hpo["validation_interval"] == 0:
                    with _preserve_rng(device):
                        validation = evaluate(model, "validation", old, f"{label}-p{epoch}-validation")
                    decision = stopping.observe(epoch, validation["metrics"]["ndcg10"])
                    row.update(validation=validation["metrics"], selection=decision,
                        validation_wall_seconds=validation["wall_seconds"])
                    if decision["improved"]:
                        result.update(best_epoch=epoch, best_validation=validation["metrics"],
                            best_validation_rank_file=f"{label}-p{epoch}-validation-anchors.json.gz")
                        if final:
                            _save_best(best, model=model, config=checkpoint_config, variant=variant,
                                num_items=cohort.num_items, epoch=epoch, score=decision["best_score"])
                            _gzip_json(output_dir / f"{variant['name']}-validation-anchors.json.gz", validation["per_anchor"])
                            _json(output_dir / f"{variant['name']}-validation-users.json", validation["per_user"])
                result["passes"].append(row)
                print(json.dumps({"phase": "full_finalist_train" if final else "proxy_lr_train",
                    "model": variant["name"], "trial": trial["name"], **row}), flush=True)
                persist()
            result.update(complete=True, passes_completed=passes,
                total_labels=sum(row["labels"] for row in result["passes"]),
                successful_updates=sum(row["successful_updates"] for row in result["passes"]),
                amp_retried_update_attempts=sum(row["amp_skipped_update_attempts"] for row in result["passes"]),
                mean_training_pass_seconds=sum(row["seconds"] for row in result["passes"]) / passes)
            if final:
                _save_resume(resume, model=model, optimizer=optimizer, scaler=scaler, config=checkpoint_config,
                    variant=variant, num_items=cohort.num_items, epoch=passes, stopping=stopping, device=device)
                manifest["checkpoints"].append(_checkpoint_record(best, output_dir, training_model=variant["name"],
                    variant=variant, best_epoch=result["best_epoch"], trial=trial, precision="float32"))
                manifest["resume_checkpoints"].append(_checkpoint_record(resume, output_dir,
                    training_model=variant["name"], epoch=passes, next_epoch=passes + 1, trial=trial,
                    optimizer_state_saved=True))
            persist()
            return result
        finally:
            del model, optimizer, scaler
            _cleanup()

    for variant in variants:
        results = [fit(variant, trial, old=hpo["proxy_old_length"], passes=hpo["proxy_passes"],
            label=f"proxy-{variant['name']}-{trial['name']}", final=False) for trial in trials]
        report["selected_trials"][variant["name"]] = select_proxy_trial(trials, results)
        persist()
    for variant in variants:
        fit(variant, report["selected_trials"][variant["name"]], old=None, passes=hpo["final_passes"],
            label=f"full-{variant['name']}", final=True)
    if len(report["finalists"]) != len(variants) or not all(row["complete"] for row in report["finalists"].values()):
        raise RuntimeError("all full-history checkpoint choices must be fixed before test evaluation")
    report["all_final_selections_fixed_before_test"] = True
    for variant in variants:
        name = variant["name"]
        with gzip.open(output_dir / f"{name}-validation-anchors.json.gz", "rt", encoding="utf-8") as stream:
            validation_rows = json.load(stream)
        report["finalists"][name]["paired_parent_validation_contrasts"] = _contrasts(
            validation_baselines, validation_rows, name=name, config=config)
    _json(output_dir / "final-selection.json", {name: {key: result[key]
        for key in ("trial", "best_epoch", "best_validation", "best_validation_rank_file")}
        for name, result in report["finalists"].items()})
    persist()
    # Only now deserialize already observed parent test ranks, never during HPO.
    test_baselines = _read_baselines(baseline_paths, "test")
    report["saved_baseline_test"] = {name: _aggregate(rows)[0] for name, rows in test_baselines.items()}
    candidate_test_rows = {}
    for variant in variants:
        guard()
        name = variant["name"]
        model = build_hpo_model(config, variant, cohort.num_items).to(device)
        try:
            saved = torch.load(checkpoints / f"{name}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"], strict=True)
            del saved
            test = evaluate(model, "test", None, f"{name}-test")
            metrics["test_evaluations"] += 1
            candidate_test_rows[name] = test["per_anchor"]
            _json(output_dir / f"{name}-test-users.json", test["per_user"])
            guard()
            inference = benchmark_prefix_quality(model, cohort, config=config, device=device)
            result = report["finalists"][name]
            report["cases"][name] = {"trial": result["trial"], "best_epoch": result["best_epoch"],
                "validation": result["best_validation"], "test": test["metrics"],
                "test_wall_seconds": test["wall_seconds"], "evaluation_metadata": test["metadata"],
                "trained_inference": inference,
                "paired_parent_test_contrasts": _contrasts(test_baselines, test["per_anchor"], name=name, config=config)}
            metrics.update({f"{name}_test_{key}": value for key, value in test["metrics"].items()})
            metrics.update({f"{name}_{key}": value for key, value in inference["metrics"].items()})
            metrics.update({f"{name}_{key}": result[key] for key in ("best_epoch", "passes_completed",
                "total_labels", "successful_updates", "amp_retried_update_attempts", "mean_training_pass_seconds")})
            persist()
        finally:
            del model
            _cleanup()
    report["paired_finalist_test_contrast"] = _contrasts({variants[0]["name"]: candidate_test_rows[variants[0]["name"]]},
        candidate_test_rows[variants[1]["name"]], name=variants[1]["name"], config=config)
    guard()
    report["complete"] = True
    metrics.update(quality_complete=True, suite_seconds=time.monotonic() - started)
    persist()
    return validate_metrics(metrics)
