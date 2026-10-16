"""Train one short-context control against pinned, already observed pilot ranks.

The four summaries are generated from the 128 events immediately before R128.
No distant-history scan is run. This is a single-seed exploratory comparison
against a previously inspected test set, not a new confirmatory holdout.
"""
from __future__ import annotations

from dataclasses import replace
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
from rhpr.sasrec.daily_quality import _aggregate, evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _cleanup, _json, _optimizer, _registered_device, _scaler
from rhpr.sasrec.query_history_ablation import (
    PARENT_KERNEL, PARENT_SOURCE_SHA, _audited_examples, _digest, _verified_file, resolve_history_parent,
)
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_only_protocol import (
    QUERY_ONLY_KIND, build_query_only_model, query_only_batch_arrays, query_only_variant,
)
from rhpr.sasrec.query_prefix_pilot import (
    _checkpoint_record, _profile_daily_rate, _representative_windows, _save_best, _save_resume,
    _validate_complete_histories, train_prefix_pass,
)
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule, prefix_schedule_summary, validate_prefix_config
from rhpr.sasrec.query_restart_state import ValidationEarlyStopping, reset_restart_rng


BASELINES = ("recent128", "query_cross_attention_m4", "query_ssm_m4")
_LOOP_OVERRIDES = {"min_passes", "max_passes", "physical_batch_sizes", "budget_seconds"}


def validate_query_only_config(config: dict[str, Any]) -> None:
    """Keep optimization, target identities and common backbone matched to the pilot."""
    if config.get("kind") != QUERY_ONLY_KIND or not isinstance(config.get("name"), str):
        raise ValueError("a named query-only control configuration is required")
    parent = config.get("parent", {})
    original = parent.get("experiment_config")
    if not isinstance(original, dict):
        raise ValueError("parent.experiment_config is required")
    validate_prefix_config(original)
    if parent.get("kernel") != PARENT_KERNEL or parent.get("source_code_sha") != PARENT_SOURCE_SHA:
        raise ValueError("the completed SASRec prefix pilot must be pinned")
    for key in ("seed", "data", "model", "compressor", "runtime", "benchmark", "summary_budgets"):
        if config.get(key) != original.get(key):
            raise ValueError(f"query-only control changed parent {key}")
    train = config.get("training", {})
    if ({key: value for key, value in train.items() if key not in _LOOP_OVERRIDES}
            != {key: value for key, value in original["training"].items() if key not in _LOOP_OVERRIDES}):
        raise ValueError("query-only control changed parent training or optimization")
    if (train.get("min_passes") != 30 or train.get("max_passes") != 30
            or train.get("physical_batch_sizes") != [32] or train.get("budget_seconds") != 1200):
        raise ValueError("query-only control fixes30 passes, physical32 and a1200-second budget")
    if config.get("variants") != ["query_only_m4"]:
        raise ValueError("exactly query_only_m4 is registered")
    if config.get("control") != {"input_context_length": 256, "training_passes": 30, "bootstrap_draws": 5000}:
        raise ValueError("query-only control fixes input256, passes30 and bootstrap5000")
    if config.get("kaggle", {}).get("kernel_sources") != [PARENT_KERNEL]:
        raise ValueError("the completed private pilot must be the sole kernel source")
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        if config["kaggle"].get(key) != original["kaggle"].get(key):
            raise ValueError(f"parent Kaggle sources changed: {key}")
    if Path(parent.get("checkpoint_root", "")).name != PARENT_KERNEL.split("/")[1]:
        raise ValueError("checkpoint_root must name the registered private parent kernel")
    if any(not _digest(parent.get(key)) for key in ("manifest_sha256", "context_manifest_sha256")):
        raise ValueError("parent manifest and context SHA256 pins are required")
    if set(parent.get("baselines", {})) != set(BASELINES):
        raise ValueError("all three saved parent baselines are required")
    for name, splits in parent["baselines"].items():
        if set(splits) != {"validation", "test"}:
            raise ValueError("validation and test baseline artifacts are required")
        for split, spec in splits.items():
            if spec.get("file") != f"{name}-{split}-anchors.json.gz" or not _digest(spec.get("sha256")):
                raise ValueError("baseline filename and SHA256 must pin the registered anchor ranks")


def _load_parent_artifacts(config: dict[str, Any], evaluation: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root, parent = resolve_history_parent(config), config["parent"]
    path = _verified_file(root, "checkpoint-manifest.json", parent["manifest_sha256"])
    manifest = json.loads(path.read_text())
    if (manifest.get("source_code_sha") != PARENT_SOURCE_SHA or manifest.get("format_version") != 3
            or manifest.get("config") != parent["experiment_config"]):
        raise ValueError("parent checkpoint manifest source/configuration mismatch")
    path = _verified_file(root, "evaluation-input-contexts.json.gz", parent["context_manifest_sha256"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        contexts = json.load(stream)
    for split in ("validation", "test"):
        if len(contexts[split]["rows"]) != evaluation.summary[f"{split}_anchors"]:
            raise ValueError("parent actual context count mismatch")
    paths = {name: {split: _verified_file(root, spec["file"], spec["sha256"])
                    for split, spec in splits.items()} for name, splits in parent["baselines"].items()}
    return contexts, paths


def _read_baselines(paths: dict[str, Any], split: str) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for name in BASELINES:
        with gzip.open(paths[name][split], "rt", encoding="utf-8") as stream:
            result[name] = json.load(stream)
    return result


def _contrasts(baselines: dict[str, Any], candidate: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"reference": name, "candidate": "query_only_m4", "training_seed_uncertainty_included": False,
        "exploratory_previously_inspected_holdout": True,
        "cutoffs": {str(k): paired_daily_ndcg_bootstrap(baselines[name], candidate,
            k=k, seed=config["seed"], draws=config["control"]["bootstrap_draws"]) for k in (10, 100)}}
        for name in BASELINES]


def _preflight(config: dict[str, Any], cohort: Any, evaluation: Any, device: torch.device) -> dict[str, Any]:
    """A real disposable Adam update and representative full-catalogue evaluation."""
    train, bench, variant = config["training"], config["benchmark"], query_only_variant()
    model = build_query_only_model(config, variant, cohort.num_items).to(device)
    optimizer, scaler = _optimizer(model, train), _scaler(device, train)
    try:
        schedule = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
        windows = _representative_windows(schedule, train["effective_batch_size"])
        warmup = train_prefix_pass(model, cohort, windows, config=config, variant=variant, physical_batch=32,
            device=device, optimizer=optimizer, scaler=scaler, batch_builder=query_only_batch_arrays)
        if warmup["successful_updates"] != 1 or warmup["amp_skipped_update_attempts"] or not optimizer.state:
            raise FloatingPointError("query-only preflight warmup requires a finite successful Adam update")
        row = train_prefix_pass(model, cohort, windows, config=config, variant=variant, physical_batch=32,
            device=device, optimizer=optimizer, scaler=scaler, batch_builder=query_only_batch_arrays)
        if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
            raise FloatingPointError("query-only preflight requires one finite successful Adam update")
        daily = _profile_daily_rate(model, cohort, evaluation, context_length=256,
            batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
            item_chunk_size=bench["item_chunk_size"])
        return {"training": row, "warmup": warmup, "daily_evaluation": daily, "profile_weights_reused": False,
            "quality_used_for_selection": False,
            "estimated_training_pass_seconds": len(cohort.users) / row["windows_per_second"],
            "estimated_validation_seconds": evaluation.summary["validation_anchors"] / daily["anchors_per_second"],
            "estimated_test_seconds": evaluation.summary["test_anchors"] / daily["anchors_per_second"]}
    finally:
        del model, optimizer, scaler
        _cleanup()


def run_query_only_control(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_prefix_benchmark import benchmark_prefix_quality

    validate_query_only_config(config)
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    device, started = _registered_device(), time.monotonic()
    train, bench, variant = config["training"], config["benchmark"], query_only_variant()
    name = variant["name"]
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "seed": config["seed"],
        "quality_complete": False, "training_models": 1, "hpo_performed": False,
        "test_evaluations": 0, "input_context_length": 256, "distant_history_scan": False,
        "train_is_exhaustive_event_epoch": False, "exploratory_previously_inspected_holdout": True}
    report: dict[str, Any] = {"complete": False, "config": config,
        "code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "protocol": {"training": "same fixed last128 labels, original window endpoints/order/negative seeds; crop Q128+R128 before embedding",
            "model": "same initial common SASRec; four learned queries over Q128 plus summary FFN; no long scan",
            "optimization": "from scratch, fresh Adam/scaler, seed42,30 fixed passes; validation at10/20/30",
            "selection": "earliest strictly best validation macro-user NDCG@10",
            "test": "one selected-checkpoint evaluation, after all training and selection",
            "comparison": "cryptographically pinned saved parent ranks; same anchors, catalogue and OOV policy",
            "interpretation": "exploratory single-seed control; the parent test set was already inspected"},
        "passes": [], "cases": {}}
    manifest: dict[str, Any] = {"format_version": 3, "source_code_sha": report["code_sha"], "config": config,
        "optimizer_state_saved": True, "best_checkpoint_optimizer_state_saved": False,
        "checkpoints": [], "resume_checkpoints": []}

    def persist() -> None:
        _json(output_dir / "query-only-control.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("query-only control exceeded the1200-second budget; partial outputs retained")

    persist()
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
    contexts, baseline_paths = _load_parent_artifacts(config, evaluation)
    validation_baselines = _read_baselines(baseline_paths, "validation")
    report["saved_baseline_validation"] = {name: _aggregate(rows)[0] for name, rows in validation_baselines.items()}
    metrics.update(cohort_users=len(cohort.users), num_items=cohort.num_items,
        unique_supervised_target_positions=len(cohort.users) * 128)
    guard()
    with _preserve_rng(device):
        report["preflight"] = _preflight(config, cohort, evaluation, device)
    cost = report["preflight"]
    estimate = time.monotonic() - started + train["budget_safety_factor"] * (
        30 * cost["estimated_training_pass_seconds"] + 3 * cost["estimated_validation_seconds"]
        + cost["estimated_test_seconds"] + 240)
    report["preflight"]["conservative_total_seconds"] = estimate
    persist()
    if estimate > train["budget_seconds"]:
        raise RuntimeError("measured query-only preflight cannot support the registered30-pass budget")

    def evaluate(model: torch.nn.Module, split: str) -> dict[str, Any]:
        guard()
        rows: list[dict[str, Any]] = []
        examples = iter_daily_evaluation_examples(cohort, evaluation, split_name=split,
            max_seq_len=config["parent"]["experiment_config"]["training"]["serving_context_length"])
        audited = _audited_examples(examples, condition="query_only_history", config=config,
            reference_contexts=contexts[split]["rows"], rows=rows, guard=guard)
        result = evaluate_daily_quality(model, audited, device=device, amp=train["amp"],
            batch_size=bench["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        _gzip_json(output_dir / f"{name}-{split}-input-contexts.json.gz", rows)
        return result

    model = build_query_only_model(config, variant, cohort.num_items).to(device)
    report.update(total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        compressor_parameters=sum(parameter.numel() for parameter in model.compressor.parameters()))
    optimizer, scaler = _optimizer(model, train), _scaler(device, train)
    reset_restart_rng(config["seed"])
    stopping = ValidationEarlyStopping(-1., minimum_additional_passes=30, patience=3,
        validation_interval=10, max_additional_passes=30)
    best, resume = checkpoints / f"{name}.pt", checkpoints / f"{name}-resume.pt"
    schedules = []
    for epoch in range(1, 31):
        guard()
        windows = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=epoch)
        schedules.append({"original_parent_schedule": prefix_schedule_summary(
            cohort, windows, config["parent"]["experiment_config"]),
            "actual_model_input_length": 256, "full_train_prefix_retained": False,
            "actual_model_old_context_length": 128})
        row = {"epoch": epoch, **train_prefix_pass(model, cohort, windows, config=config, variant=variant,
            physical_batch=32, device=device, optimizer=optimizer, scaler=scaler,
            batch_builder=query_only_batch_arrays)}
        if epoch % 10 == 0:
            with _preserve_rng(device):
                validation = evaluate(model, "validation")
            decision = stopping.observe(epoch, validation["metrics"]["ndcg10"])
            row.update(validation=validation["metrics"], selection=decision)
            if decision["improved"]:
                report.update(best_epoch=epoch, best_validation=validation["metrics"])
                _save_best(best, model=model, config=config, variant=variant, num_items=cohort.num_items,
                    epoch=epoch, score=decision["best_score"])
                _json(output_dir / f"{name}-validation-users.json", validation["per_user"])
                _gzip_json(output_dir / f"{name}-validation-anchors.json.gz", validation["per_anchor"])
                best_validation_rows = validation["per_anchor"]
        report["passes"].append(row)
        print(json.dumps({"phase": "query_only_train", **row}), flush=True)
        persist()
    _json(output_dir / "window-schedules.json", schedules)
    _save_resume(resume, model=model, optimizer=optimizer, scaler=scaler, config=config, variant=variant,
        num_items=cohort.num_items, epoch=30, stopping=stopping, device=device)
    manifest["checkpoints"].append(_checkpoint_record(best, output_dir, training_model=name,
        variant=variant, best_epoch=report["best_epoch"], precision="float32"))
    manifest["resume_checkpoints"].append(_checkpoint_record(resume, output_dir, training_model=name,
        epoch=30, next_epoch=31, optimizer_state_saved=True))
    del optimizer, scaler
    _cleanup()
    report["paired_validation_contrasts"] = _contrasts(validation_baselines, best_validation_rows, config)
    saved = torch.load(best, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    # The sole new test result is unavailable until the selected checkpoint is fixed.
    test = evaluate(model, "test")
    metrics["test_evaluations"] = 1
    _json(output_dir / f"{name}-test-users.json", test["per_user"])
    _gzip_json(output_dir / f"{name}-test-anchors.json.gz", test["per_anchor"])
    test_baselines = _read_baselines(baseline_paths, "test")
    report["saved_baseline_test"] = {name: _aggregate(rows)[0] for name, rows in test_baselines.items()}
    report["paired_test_contrasts"] = _contrasts(test_baselines, test["per_anchor"], config)
    guard()
    cropped = replace(cohort, users=tuple(replace(user, train=user.train[-256:], timestamps=user.timestamps[-256:])
        for user in cohort.users))
    inference = benchmark_prefix_quality(model, cropped, config=config, device=device)
    guard()
    inference["metadata"]["context_selection"] = "fixed Q128+R128; crop before embedding; original complete timelines remain data sources"
    inference["metadata"]["uid_sampling"] = "all cropped lengths tie; UID quantiles can differ from the parent's full-length quantiles"
    report["cases"][name] = {"test": test["metrics"], "evaluation_metadata": test["metadata"],
        "validation": report["best_validation"], "best_epoch": report["best_epoch"], "trained_inference": inference}
    metrics.update({f"{name}_test_{key}": value for key, value in test["metrics"].items()})
    metrics.update({f"{name}_{key}": value for key, value in inference["metrics"].items()})
    metrics.update(best_epoch=report["best_epoch"], passes_completed=30,
        training_labels=sum(row["labels"] for row in report["passes"]),
        successful_updates=sum(row["successful_updates"] for row in report["passes"]),
        amp_retried_update_attempts=sum(row["amp_skipped_update_attempts"] for row in report["passes"]),
        mean_training_pass_seconds=sum(row["seconds"] for row in report["passes"]) / 30,
        suite_seconds=time.monotonic() - started, quality_complete=True)
    report["complete"] = True
    persist()
    del model
    _cleanup()
    return validate_metrics(metrics)
