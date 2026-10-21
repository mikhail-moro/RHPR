"""A four-arm LR diagnostic from fixed pass48 weights, with fresh AdamW state.

This is an optimizer restart, not an exact continuation of the parent run.
Only validation chooses checkpoints, stopping times and the two tested LRs.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterator

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import iter_daily_evaluation_examples, load_prepared_daily_evaluation
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.daily_training_profile import _measure_daily_rate
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json, _sha256
from rhpr.sasrec.long_quality import _cleanup, _json, _optimizer, _registered_device, _scaler, train_quality_pass
from rhpr.sasrec.long_quality_schedule import quality_schedule_summary, quality_window_schedule
from rhpr.sasrec.long_training_profile import _train_shape
from rhpr.sasrec.query_pilot import _context_manifest, build_query_model
from rhpr.sasrec.query_profile import _choose_training_batch, _positive_rate
from rhpr.sasrec.query_restart_state import (
    ValidationEarlyStopping, load_parent_checkpoint, reset_restart_rng,
    resolve_parent_checkpoint, restart_arms, validate_restart_config,
)


@contextmanager
def _preserve_rng(device: torch.device) -> Iterator[None]:
    """Validation and timing cannot advance the training dropout stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _load_restart_model(config: dict[str, Any], arm: dict[str, Any], num_items: int,
                        device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_query_model(config, arm, num_items)
    payload = load_parent_checkpoint(config, arm["parent_name"], num_items=num_items, model=model)
    metadata = {key: payload[key] for key in ("source_code_sha", "best_epoch", "selection_score")}
    metadata.update(config["restart"]["checkpoints"][arm["parent_name"]])
    del payload
    return model.to(device), metadata


def _profile_restart_arm(config: dict[str, Any], arm: dict[str, Any], cohort: Any,
                         evaluation: Any, schedule: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    """Profile real parent weights and each LR using disposable fresh optimizers."""
    train = {**config["training"], "learning_rate": arm["learning_rate"]}
    bench = config["benchmark"]
    training_tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    training_context = training_tail if arm["kind"] == "recent" else train["raw_context_length"]
    serving_context = train["raw_tail_size"] if arm["kind"] == "recent" else train["serving_context_length"]
    profile = {**train, "tail_size": train["supervised_targets"], "update_repetitions": 1,
               "warmup_microbatches": 3}
    started, rows = time.monotonic(), []
    for physical in train["physical_batch_sizes"]:
        model, _ = _load_restart_model(config, arm, cohort.num_items, device)
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        try:
            row = _train_shape(model, cohort, schedule, profile, training_context,
                               physical, device, optimizer, scaler)
        except torch.OutOfMemoryError:
            row = {"status": "oom", "physical_batch_size": physical}
        except FloatingPointError as error:
            row = {"status": "nonfinite", "physical_batch_size": physical, "reason": str(error)}
        finally:
            del model, optimizer, scaler
            _cleanup()
        rows.append(row)
    physical, rate = _choose_training_batch(rows)
    model, _ = _load_restart_model(config, arm, cohort.num_items, device)
    model.compressor.tail_size = train["raw_tail_size"]
    try:
        evaluation_row = _measure_daily_rate(model, cohort, evaluation, context_length=serving_context,
            batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
            item_chunk_size=bench["item_chunk_size"], sample_count=128)
    finally:
        del model
        _cleanup()
    if evaluation_row["status"] != "ok" or not _positive_rate(evaluation_row.get("anchors_per_second")):
        raise RuntimeError(f"{arm['name']}: registered parallel evaluation batch did not succeed")
    return {**arm, "training_batch_size": physical, "evaluation_batch_size": bench["evaluation_batch_size"],
        "estimated_train_pass_seconds": len(cohort.users) / rate,
        "estimated_validation_seconds": evaluation.summary["validation_anchors"] / evaluation_row["anchors_per_second"],
        "estimated_test_seconds": evaluation.summary["test_anchors"] / evaluation_row["anchors_per_second"],
        "training": rows, "evaluation": [evaluation_row], "profile_seconds": time.monotonic() - started,
        "profile_weights_reused_for_training": False, "profile_initialization": "verified parent pass48 weights",
        "training_context_length": training_context, "serving_context_length": serving_context}


def _restart_budget(config: dict[str, Any], profiles: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Require room for all four full caps; ES alone may shorten an arm."""
    train = config["training"]
    passes, interval = train["max_additional_passes"], train["validation_every_additional_passes"]
    families = sorted({row["parent_name"] for row in profiles})
    for row in profiles:
        if any(not _positive_rate(row.get(key)) for key in (
                "estimated_train_pass_seconds", "estimated_validation_seconds", "estimated_test_seconds")):
            raise ValueError("restart budget requires finite positive measurements for all arms")
    training = passes * sum(row["estimated_train_pass_seconds"] for row in profiles)
    validation = (passes // interval) * sum(row["estimated_validation_seconds"] for row in profiles)
    # Baseline is evaluated once per family; test is run only on its selected LR.
    fixed_evaluation = sum(max(row["estimated_validation_seconds"] + row["estimated_test_seconds"]
        for row in profiles if row["parent_name"] == family) for family in families)
    fixed_overhead = 60.0 * len(profiles)  # checkpoint reads, saves and final RNG/Adam snapshots
    future = training + validation + fixed_evaluation + fixed_overhead
    conservative = elapsed + train["budget_safety_factor"] * future
    if not math.isfinite(conservative) or conservative > train["budget_seconds"]:
        raise ValueError(f"budget cannot support all four registered {passes}-pass caps: "
                         f"need {conservative:.1f}s, have {train['budget_seconds']:.1f}s")
    return {"max_additional_passes": passes, "models": len(profiles),
        "estimated_training_seconds": training, "estimated_validation_seconds": validation,
        "estimated_fixed_evaluation_seconds": fixed_evaluation, "fixed_overhead_seconds": fixed_overhead,
        "preflight_seconds": elapsed, "safety_factor": train["budget_safety_factor"],
        "estimated_suite_seconds": elapsed + future, "conservative_suite_seconds": conservative,
        "budget_seconds": train["budget_seconds"], "cap_reduced_by_preflight": False,
        "selection_uses_quality_metrics": False}


def _parent_output_directory(config: dict[str, Any], parent_name: str) -> Path:
    checkpoint = resolve_parent_checkpoint(config, parent_name)
    return checkpoint.parents[len(Path(config["restart"]["checkpoints"][parent_name]["file"]).parts) - 1]


def _load_parent_metadata(config: dict[str, Any], contexts: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for family in dict.fromkeys(arm["parent_name"] for arm in restart_arms(config)):
        root = _parent_output_directory(config, family)
        path = root / "query-pilot-quality.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("complete") is not True or value.get("code_sha") != config["restart"]["source_code_sha"]:
            raise ValueError("parent report is incomplete or has an unexpected source revision")
        for split in ("validation", "test"):
            expected = value.get("input_contexts", {}).get(split, {})
            for key in ("anchors", "ordered_serving_context_sha256"):
                if expected.get(key) != contexts[split][key]:
                    raise ValueError(f"parent input context mismatch: {split}.{key}")
        result[family] = {"root": str(root), "report_sha256": _sha256(path), "report": value}
    return result


def _cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_state(item) for item in value)
    return value


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _save_best_checkpoint(path: Path, *, model: torch.nn.Module, config: dict[str, Any],
                          arm: dict[str, Any], parent: dict[str, Any], num_items: int,
                          additional_passes: int, score: float) -> None:
    _atomic_torch_save({"format_version": 2, "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "experiment_config": config, "model_config": config["model"], "variant": arm,
        "num_items": num_items, "parent_checkpoint": parent, "optimizer_restart": True,
        "best_epoch": config["restart"]["parent_epoch"] + additional_passes,
        "best_additional_passes": additional_passes, "selection_score": score,
        "state_dict": _cpu_state(model.state_dict())}, path)


def _save_resume_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                            scaler: Any, config: dict[str, Any], arm: dict[str, Any],
                            parent: dict[str, Any], additional_passes: int, stopping: Any,
                            device: torch.device, num_items: int) -> None:
    numpy_state = np.random.get_state()
    _atomic_torch_save({"format_version": 2, "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "experiment_config": config, "model_config": config["model"], "num_items": num_items,
        "variant": arm, "parent_checkpoint": parent,
        "optimizer_restart": True, "optimizer_class": "torch.optim.AdamW", "optimizer_state_saved": True,
        "torch_version": str(torch.__version__), "cuda_device": str(device),
        "global_epoch": config["restart"]["parent_epoch"] + additional_passes,
        "additional_passes": additional_passes, "next_global_epoch": config["restart"]["parent_epoch"] + additional_passes + 1,
        "model_training": model.training, "compressor_tail_size": model.compressor.tail_size,
        "state_dict": _cpu_state(model.state_dict()), "optimizer_state_dict": _cpu_state(optimizer.state_dict()),
        "scaler_state_dict": scaler.state_dict(), "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "early_stopping": {key: getattr(stopping, key) for key in (
            "best_score", "best_additional_passes", "bad_checks", "last_additional_passes", "stopped")}}, path)


def _checkpoint_record(path: Path, output_dir: Path, **metadata: Any) -> dict[str, Any]:
    return {"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path), **metadata}


def run_query_lr_restart(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    validate_restart_config(config)
    device, started = _registered_device(), time.monotonic()
    train, bench, restart = config["training"], config["benchmark"], config["restart"]
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_daily_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    for key in ("evaluation_content_sha256", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if evaluation.summary[key] != config["data"][key]:
            raise ValueError(f"registered daily evaluation mismatch: {key}")
    if evaluation.summary["reference_context_length"] < train["serving_context_length"]:
        raise ValueError("prepared reference context is shorter than the serving context")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    contexts = _context_manifest(cohort, evaluation, config)
    _gzip_json(output_dir / "evaluation-input-contexts.json.gz", contexts)
    parent_metadata = _load_parent_metadata(config, contexts)
    context_summary = {split: {key: value for key, value in row.items() if key != "rows"}
                       for split, row in contexts.items()}
    del contexts
    arms = restart_arms(config)
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "quality_complete": False,
        "hpo_performed": True, "diagnostic_only": True, "optimizer_restart": True,
        "exact_parent_optimizer_resume": False, "parent_epoch": restart["parent_epoch"],
        "training_models": len(arms), "max_additional_passes": train["max_additional_passes"],
        "cohort_users": len(cohort.users), "num_items": cohort.num_items,
        "gpu_name": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__)}
    report: dict[str, Any] = {"complete": False, "config": config,
        "code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": metrics["gpu_name"]},
        "protocol": {"optimizer": "fresh AdamW, fresh GradScaler and common fresh Python/NumPy/CPU/CUDA RNG",
            "continuation": "warm restart from pass48 weights; parent Adam and RNG were not saved",
            "windows": "global passes49..96; common endpoint/order/negative schedule for all four arms",
            "validation": "additional0 then every8; strict macro-user NDCG@10; baseline0 eligible",
            "early_stopping": "minimum24 additional passes; patience3 validation checks; strict improvements only",
            "lr_selection": "best validation score per family; LR ties keep first registered LR",
            "test_usage": "both LR winners fixed and persisted before test; only those two checkpoints tested",
            "primary_quality": "macro_user_ndcg10", "secondary_quality": "macro_user_ndcg100",
            "training": "old8000 + rawtail159, query-first128, loss on final32 raw states",
            "serving": "old8000 + rawtail128, one final-state full-catalogue prediction per daily anchor",
            "latency": "unchanged architectures; original pilot measurements remain the cost reference",
            "interpretation": "single-seed optimization diagnostic, not final architecture superiority evidence"},
        "input_contexts": context_summary,
        "parent_reports": {family: {key: row[key] for key in ("root", "report_sha256")}
                           for family, row in parent_metadata.items()},
        "preflight": [], "models": {}, "selections": {}, "cases": {}, "schedules": []}
    manifest: dict[str, Any] = {"format_version": 2, "source_code_sha": report["code_sha"],
        "config": config, "optimizer_restart": True, "optimizer_state_saved": True,
        "best_checkpoint_optimizer_state_saved": False, "checkpoints": [], "resume_checkpoints": []}

    def persist() -> None:
        _json(output_dir / "query-lr-restart-quality.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("LR restart exceeded its registered wall-time budget; partial outputs retained")

    first_epoch = restart["parent_epoch"] + 1
    schedules = [quality_window_schedule(cohort, seed=config["seed"], epoch=epoch,
        raw_context_length=train["raw_context_length"])
        for epoch in range(first_epoch, first_epoch + train["max_additional_passes"])]
    report["schedules"] = [quality_schedule_summary(cohort, windows, raw_context_length=train["raw_context_length"],
        raw_tail_size=train["raw_tail_size"], supervised_targets=train["supervised_targets"]) for windows in schedules]
    _json(output_dir / "window-schedules.json", report["schedules"])
    persist()
    for arm in arms:
        guard()
        with _preserve_rng(device):
            row = _profile_restart_arm(config, arm, cohort, evaluation, schedules[0], device)
        report["preflight"].append(row)
        persist()
        print(json.dumps({"phase": "restart_preflight", **row}), flush=True)
    report["budget"] = _restart_budget(config, report["preflight"], time.monotonic() - started)
    persist()
    baseline_cache: dict[str, Any] = {}
    for arm, preflight in zip(arms, report["preflight"], strict=True):
        guard()
        name, family = arm["name"], arm["parent_name"]
        arm_train = {**train, "learning_rate": arm["learning_rate"]}
        arm_config = {**config, "training": arm_train}
        model, parent = _load_restart_model(config, arm, cohort.num_items, device)
        optimizer, scaler = _optimizer(model, arm_train), _scaler(device, arm_train)
        if optimizer.state:
            raise RuntimeError("restart optimizer unexpectedly contains parent moments")
        serving_length = train["raw_tail_size"] if arm["kind"] == "recent" else train["serving_context_length"]

        def evaluate(split: str) -> dict[str, Any]:
            guard()
            model.compressor.tail_size = train["raw_tail_size"]
            with _preserve_rng(device):
                return evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
                    split_name=split, max_seq_len=serving_length), device=device, amp=train["amp"],
                    batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])

        def save_rows(split: str, value: dict[str, Any]) -> None:
            _json(output_dir / f"{name}-{split}-users.json", value["per_user"])
            _gzip_json(output_dir / f"{name}-{split}-anchors.json.gz", value["per_anchor"])

        if family not in baseline_cache:
            baseline_cache[family] = evaluate("validation")
        baseline = baseline_cache[family]
        score = baseline["metrics"]["ndcg10"]
        if abs(score - parent["selection_score"]) > 1e-6:
            raise ValueError(f"{family}: restored baseline validation differs from parent by more than1e-6")
        stopping = ValidationEarlyStopping(score,
            minimum_additional_passes=train["early_stopping_min_additional_passes"],
            patience=train["early_stopping_patience"],
            validation_interval=train["validation_every_additional_passes"],
            max_additional_passes=train["max_additional_passes"])
        checkpoint = checkpoints / f"{name}.pt"
        result: dict[str, Any] = {**arm, "parent_checkpoint": parent, "passes": [],
            "baseline_validation": baseline["metrics"], "best_validation": baseline["metrics"],
            "baseline_validation_delta_from_parent": score - parent["selection_score"],
            "best_additional_passes": 0, "best_epoch": restart["parent_epoch"],
            "physical_batch_size": preflight["training_batch_size"],
            "hstu_normalization_length": model.hstu_normalization_length,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters())}
        report["models"][name] = result
        _save_best_checkpoint(checkpoint, model=model, config=arm_config, arm=arm, parent=parent,
            num_items=cohort.num_items, additional_passes=0, score=score)
        save_rows("validation", baseline)
        reset_restart_rng(config["seed"])
        for additional, windows in enumerate(schedules, 1):
            guard()
            row = {"epoch": restart["parent_epoch"] + additional, "additional_passes": additional,
                **train_quality_pass(model, cohort, windows, config=arm_config, variant=arm,
                    physical_batch=preflight["training_batch_size"], device=device, optimizer=optimizer, scaler=scaler)}
            stop = False
            if additional % train["validation_every_additional_passes"] == 0:
                validation = evaluate("validation")
                row["validation"] = validation["metrics"]
                decision = stopping.observe(additional, validation["metrics"]["ndcg10"])
                row["early_stopping"] = decision
                stop = decision["should_stop"]
                if decision["improved"]:
                    result.update(best_additional_passes=additional, best_epoch=row["epoch"],
                                  best_validation=validation["metrics"])
                    _save_best_checkpoint(checkpoint, model=model, config=arm_config, arm=arm, parent=parent,
                        num_items=cohort.num_items, additional_passes=additional, score=stopping.best_score)
                    save_rows("validation", validation)
                del validation
            result["passes"].append(row)
            persist()
            print(json.dumps({"phase": "restart_train", "model": name, **row}), flush=True)
            if stop:
                break
        result.update(additional_passes_completed=additional, global_epoch_completed=restart["parent_epoch"] + additional,
            stopped_early=additional < train["max_additional_passes"],
            stop_reason="validation_patience" if additional < train["max_additional_passes"] else "registered_cap",
            best_at_last_pass=result["best_additional_passes"] == additional)
        resume = checkpoints / f"{name}-last-resume.pt"
        guard()
        _save_resume_checkpoint(resume, model=model, optimizer=optimizer, scaler=scaler, config=arm_config,
            arm=arm, parent=parent, additional_passes=additional, stopping=stopping, device=device,
            num_items=cohort.num_items)
        record = _checkpoint_record(checkpoint, output_dir, training_model=name,
            best_epoch=result["best_epoch"], best_additional_passes=result["best_additional_passes"], precision="float32")
        result["checkpoint"] = record
        manifest["checkpoints"].append(record)
        manifest["resume_checkpoints"].append(_checkpoint_record(resume, output_dir,
            training_model=name, global_epoch=result["global_epoch_completed"], optimizer_state_saved=True,
            rng_state_saved=True, checkpoint_role="last training state; independent of validation-best weights"))
        for key in ("best_epoch", "best_additional_passes", "additional_passes_completed", "stopped_early"):
            metrics[f"{name}_{key}"] = result[key]
        metrics[f"{name}_best_validation_ndcg10"] = result["best_validation"]["ndcg10"]
        metrics[f"{name}_training_labels"] = sum(row["labels"] for row in result["passes"])
        metrics[f"{name}_successful_updates"] = sum(row["successful_updates"] for row in result["passes"])
        del model, optimizer, scaler
        _cleanup()
        persist()

    # Freeze BOTH LR decisions before constructing any test iterator.
    for family in dict.fromkeys(arm["parent_name"] for arm in arms):
        candidates = [arm["name"] for arm in arms if arm["parent_name"] == family]
        report["selections"][family] = max(candidates, key=lambda name: report["models"][name]["best_validation"]["ndcg10"])
        winner = report["models"][report["selections"][family]]
        metrics[f"{family}_selected_learning_rate"] = winner["learning_rate"]
        metrics[f"{family}_selected_best_global_epoch"] = winner["best_epoch"]
    report["selection_frozen_before_test"] = True
    persist()
    test_rows = {}
    for family, name in report["selections"].items():
        guard()
        result = report["models"][name]
        arm = next(arm for arm in arms if arm["name"] == name)
        preflight = next(row for row in report["preflight"] if row["name"] == name)
        model = build_query_model(config, arm, cohort.num_items).to(device)
        saved = torch.load(output_dir / result["checkpoint"]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        model.compressor.tail_size = train["raw_tail_size"]
        serving_length = train["raw_tail_size"] if arm["kind"] == "recent" else train["serving_context_length"]
        with _preserve_rng(device):
            test = evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
                split_name="test", max_seq_len=serving_length), device=device, amp=train["amp"],
                batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        _json(output_dir / f"{name}-test-users.json", test["per_user"])
        _gzip_json(output_dir / f"{name}-test-anchors.json.gz", test["per_anchor"])
        test_rows[family] = test["per_anchor"]
        case = {"training_model": name, "parent_name": family, "learning_rate": arm["learning_rate"],
            "best_epoch": result["best_epoch"], "best_additional_passes": result["best_additional_passes"],
            "checkpoint": result["checkpoint"], "validation": result["best_validation"],
            "test": test["metrics"], "evaluation_metadata": test["metadata"]}
        if family in parent_metadata:
            parent_path = Path(parent_metadata[family]["root"]) / f"{family}-test-anchors.json.gz"
            if parent_path.is_file():
                with gzip.open(parent_path, "rt", encoding="utf-8") as stream:
                    parent_rows = json.load(stream)
                case["paired_vs_parent_pass48"] = {str(k): paired_daily_ndcg_bootstrap(
                    parent_rows, test["per_anchor"], k=k, seed=config["seed"], draws=5000) for k in (10, 100)}
                del parent_rows
        report["cases"][name] = case
        for key, value in test["metrics"].items():
            metrics[f"{name}_test_{key}"] = value
            metrics[f"{family}_selected_test_{key}"] = value
        persist()
        print(json.dumps({"phase": "restart_selected_test", "case": name, "test": test["metrics"]}), flush=True)
        del model
        _cleanup()
    report["paired_selected_ca_vs_recent"] = {str(k): paired_daily_ndcg_bootstrap(
        test_rows["recent128"], test_rows["query_cross_attention_m4"], k=k, seed=config["seed"], draws=5000)
        for k in (10, 100)}
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"] = time.monotonic() - started
    metrics["test_evaluations"] = len(report["cases"])
    persist()
    return validate_metrics(metrics)
