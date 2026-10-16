"""Three matched SASRec arms with complete old prefixes and 128 causal losses.

The fixed final train block is repeated on each pass.  Only user order, dropout
and identity-bound negative samples change; this is not an exhaustive event
training epoch.  Cost measurements choose a common pass cap before any quality
training, and validation alone chooses each arm's stopping time and checkpoint.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import iter_daily_evaluation_examples, load_prepared_daily_evaluation
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json, _sha256
from rhpr.sasrec.long_quality import _cleanup, _json, _optimizer, _registered_device, _scaler, _sync
from rhpr.sasrec.long_quality_schedule import choose_uniform_pass_budget
from rhpr.sasrec.long_training_profile import _loss
from rhpr.sasrec.query_lr_restart import _atomic_torch_save, _cpu_state, _preserve_rng
from rhpr.sasrec.query_pilot import _context_manifest
from rhpr.sasrec.query_prefix_protocol import (
    build_prefix_model, fixed_prefix_schedule, prefix_batch_arrays,
    prefix_schedule_summary, prefix_variants, validate_prefix_config,
)
from rhpr.sasrec.query_profile import _choose_training_batch, _positive_rate, audit_native_query_scan
from rhpr.sasrec.query_restart_state import ValidationEarlyStopping, reset_restart_rng


def train_prefix_pass(model: torch.nn.Module, cohort: Any, windows: tuple[Any, ...], *,
                      config: dict[str, Any], variant: dict[str, Any], physical_batch: int,
                      device: torch.device, optimizer: torch.optim.Optimizer, scaler: Any,
                      batch_builder: Callable[..., dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """One successful effective update per group, with bounded overflow replay.

    Every microbatch uses exactly the same identity-bound labels and negatives
    independently of the architecture and physical batch.  Replayed overflow
    attempts also restore dropout RNG, so they cannot silently discard data.
    """
    train = config["training"]
    make_batch = prefix_batch_arrays if batch_builder is None else batch_builder
    if not windows or type(physical_batch) is not int or physical_batch < 1:
        raise ValueError("nonempty windows and a positive physical batch are required")
    effective = train["effective_batch_size"]
    if physical_batch > effective or effective % physical_batch:
        raise ValueError("physical batch must divide the effective batch")
    model.compressor.tail_size = train["raw_tail_size"]
    model.train()
    _sync(device)
    started, total_loss, successful, retries = time.perf_counter(), 0.0, 0, 0
    for first in range(0, len(windows), effective):
        group = windows[first:first + effective]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        for attempt in range(5):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for offset in range(0, len(group), physical_batch):
                chunk = group[offset:offset + physical_batch]
                arrays = make_batch(cohort, chunk, config, variant)
                batch = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=train["amp"] and device.type == "cuda"):
                    encoded = model.encode(batch["item_ids"], batch["valid_mask"])
                loss = _loss(model, encoded, batch, train["supervised_targets"], train["num_negatives"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite prefix training loss")
                weight = len(chunk) / len(group)
                group_loss += float(loss.detach()) * weight
                scaler.scale(loss * weight).backward()
                del encoded, loss, batch, arrays
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip_norm"])
            if bool(torch.isfinite(norm)):
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < previous_scale:
                    raise FloatingPointError("unexpected AMP skip after finite gradient check")
                successful += 1
                total_loss += group_loss * len(group)
                break
            retries += 1
            if not scaler.is_enabled() or attempt == 4:
                raise FloatingPointError("nonfinite gradients after bounded prefix AMP recovery")
            # Do not call scaler.step: the aggregate norm can overflow even if
            # GradScaler found no elementwise Inf.  Adam must remain unchanged.
            scaler.update(new_scale=float(scaler.get_scale()) / 2)
        optimizer.zero_grad(set_to_none=True)
    _sync(device)
    seconds = time.perf_counter() - started
    labels = len(windows) * train["supervised_targets"]
    return {"loss": total_loss / len(windows), "windows": len(windows), "labels": labels,
        "successful_updates": successful, "amp_skipped_update_attempts": retries,
        "seconds": seconds, "windows_per_second": len(windows) / seconds,
        "labels_per_second": labels / seconds, "loss_scale": float(scaler.get_scale())}


def _representative_windows(schedule: tuple[Any, ...], count: int) -> tuple[Any, ...]:
    """Deterministic length quantiles include both shortest and longest users."""
    if len(schedule) < count:
        raise ValueError("preflight requires one complete effective batch of users")
    ordered = sorted(schedule, key=lambda window: (window.end, window.user_id))
    selected = {ordered[int(index)].user_id for index in np.linspace(0, len(ordered) - 1, count, dtype=int)}
    # Preserve the real shuffled schedule order. Sorting the measured batch by
    # length would hide the padding cost paid by ordinary training batches.
    return tuple(window for window in schedule if window.user_id in selected)


def _profile_training_shape(config: dict[str, Any], variant: dict[str, Any], cohort: Any,
                            schedule: tuple[Any, ...], physical: int, device: torch.device) -> dict[str, Any]:
    """Warm the maximum history shape, then time a stratified effective update."""
    train = config["training"]
    model = build_prefix_model(config, variant, cohort.num_items).to(device)
    optimizer, scaler = _optimizer(model, train), _scaler(device, train)
    try:
        longest = tuple(sorted(schedule, key=lambda window: (-window.end, window.user_id))[:physical])
        warmup = train_prefix_pass(model, cohort, longest, config=config, variant=variant,
            physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler)
        if not optimizer.state:
            raise FloatingPointError("preflight warmup did not allocate real Adam state")
        measured = _representative_windows(schedule, train["effective_batch_size"])
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        row = train_prefix_pass(model, cohort, measured, config=config, variant=variant,
            physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler)
        lengths = [window.end for window in measured]
        result = {"status": "ok", "physical_batch_size": physical,
            "effective_batch_size": train["effective_batch_size"],
            "accumulation_steps": train["effective_batch_size"] // physical,
            "measured_successful_updates": row["successful_updates"],
            "amp_skipped_updates": row["amp_skipped_update_attempts"],
            "train_windows_per_second_with_prepare_h2d_sync": row["windows_per_second"],
            "train_labels_per_second_with_prepare_h2d_sync": row["labels_per_second"],
            "measured_update_seconds": row["seconds"], "warmup": warmup,
            "sample_input_length_min": min(lengths), "sample_input_length_max": max(lengths),
            "sample_input_length_mean": float(np.mean(lengths)),
            "capacity_warmup_input_length_max": max(window.end for window in longest),
            "sampling": "uniform length quantiles; separate longest-user batch capacity warmup",
            "timing_scope": "CPU collation, H2D, full compression/SASRec, 128-position sampled loss, backward, Adam and synchronization"}
        if device.type == "cuda":
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        return result
    finally:
        del model, optimizer, scaler
        _cleanup()


def _profile_daily_rate(model: torch.nn.Module, cohort: Any, evaluation: Any, *,
                        context_length: int, batch_size: int, device: torch.device,
                        amp: bool, item_chunk_size: int, sample_count: int = 128) -> dict[str, Any]:
    """Time representative daily anchors, warming the longest valid histories."""
    low, high = evaluation.summary["train_end_timestamp"], evaluation.summary["validation_end_timestamp"]
    anchors = [anchor for anchor in evaluation.anchors if low <= anchor.target_timestamp < high]
    if not anchors:
        raise ValueError("validation anchors are required for cost preflight")
    train_lengths = {user.user_id: len(user.train) for user in cohort.users}
    by_length = sorted(anchors, key=lambda anchor: (
        train_lengths[anchor.user_id] + anchor.future_index, anchor.anchor_id))
    selected_ids = {by_length[int(index)].anchor_id for index in np.linspace(
        0, len(by_length) - 1, min(sample_count, len(by_length)), dtype=int)}
    # Match the actual UID/day order after quantile selection; do not profile
    # artificially homogeneous length-sorted batches.
    sample = tuple(anchor for anchor in anchors if anchor.anchor_id in selected_ids)
    warm = tuple(by_length[-min(batch_size, len(by_length)):])

    def examples(selected: tuple[Any, ...]):
        subset = replace(evaluation, anchors=selected)
        return iter_daily_evaluation_examples(cohort, subset, split_name="validation", max_seq_len=context_length)

    evaluate_daily_quality(model, examples(warm), device=device, amp=amp,
                           batch_size=batch_size, item_chunk_size=item_chunk_size)
    _sync(device)
    started = time.perf_counter()
    measured = evaluate_daily_quality(model, examples(sample), device=device, amp=amp,
                                     batch_size=batch_size, item_chunk_size=item_chunk_size)
    _sync(device)
    seconds = time.perf_counter() - started
    count = len(measured["per_anchor"])
    if not _positive_rate(seconds) or count != len(sample):
        raise RuntimeError("daily preflight did not measure every selected anchor")
    lengths = [min(context_length, train_lengths[anchor.user_id] + anchor.future_index) for anchor in sample]
    return {"status": "ok", "batch_size": batch_size, "sample_anchors": count,
        "wall_seconds": seconds, "anchors_per_second": count / seconds,
        "quality_used_for_selection": False,
        "sample_input_length_min": min(lengths), "sample_input_length_max": max(lengths),
        "sample_input_length_mean": float(np.mean(lengths)),
        "capacity_warmup_input_length_max": min(context_length,
            max(train_lengths[anchor.user_id] + anchor.future_index for anchor in warm)),
        "sampling": "validation anchor length quantiles over all users/days; longest-anchor warmup",
        "timing_scope": "lazy observed-context assembly, collation, H2D, full compressor/SASRec, exact catalogue top100, D2H and user aggregation"}


def profile_prefix_variant(config: dict[str, Any], variant: dict[str, Any], cohort: Any,
                           evaluation: Any, schedule: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    train, bench = config["training"], config["benchmark"]
    started, rows = time.monotonic(), []
    for physical in train["physical_batch_sizes"]:
        try:
            row = _profile_training_shape(config, variant, cohort, schedule, physical, device)
        except torch.OutOfMemoryError:
            row = {"status": "oom", "physical_batch_size": physical}
            _cleanup()
        except FloatingPointError as error:
            row = {"status": "nonfinite", "physical_batch_size": physical, "reason": str(error)}
        rows.append(row)
    physical, rate = _choose_training_batch(rows)
    model = build_prefix_model(config, variant, cohort.num_items).to(device)
    serving_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["serving_context_length"]
    try:
        evaluation_row = _profile_daily_rate(model, cohort, evaluation, context_length=serving_length,
            batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
            item_chunk_size=bench["item_chunk_size"])
    finally:
        del model
        _cleanup()
    if not _positive_rate(evaluation_row.get("anchors_per_second")):
        raise RuntimeError("registered parallel evaluation batch did not complete")
    return {**variant, "training_batch_size": physical, "evaluation_batch_size": bench["evaluation_batch_size"],
        "estimated_train_pass_seconds": len(cohort.users) / rate,
        "estimated_validation_seconds": evaluation.summary["validation_anchors"] / evaluation_row["anchors_per_second"],
        "estimated_test_seconds": evaluation.summary["test_anchors"] / evaluation_row["anchors_per_second"],
        "training": rows, "evaluation": [evaluation_row], "profile_seconds": time.monotonic() - started,
        "profile_weights_reused_for_training": False,
        "training_batch_selection": "fastest successful finite effective update with no AMP retries",
        "serving_context_length_cap": serving_length}


def prefix_pass_budget(config: dict[str, Any], profiles: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    """Choose one validation-aligned cap before observing training quality."""
    train = config["training"]
    interval = train["validation_interval"]
    costs = [{"training_pass_seconds": row["estimated_train_pass_seconds"],
              "validation_pass_seconds": row["estimated_validation_seconds"] / interval,
              "evaluation_seconds": row["estimated_test_seconds"]} for row in profiles]
    budget = choose_uniform_pass_budget(costs, budget_seconds=train["budget_seconds"],
        preflight_seconds=elapsed, fixed_overhead_seconds=180 * len(profiles),
        min_passes=train["min_passes"], max_passes=train["max_passes"],
        safety_factor=train["budget_safety_factor"])
    cap = budget["passes"] // interval * interval
    if cap < train["min_passes"]:
        raise ValueError("budget cannot support the minimum common validation-aligned pass cap")
    budget.update(passes=cap, validation_checks_per_model=cap // interval,
        stopping="arm-specific strict validation ES after minimum30; shared prospective maximum cap",
        fixed_overhead_scope="best-checkpoint I/O and hashing, final Adam/scaler/RNG resume snapshots, trained inference benchmark",
        estimated_suite_seconds=elapsed + budget["fixed_overhead_seconds"] + budget["model_fixed_evaluation_seconds"]
            + cap * (budget["training_seconds_per_pass"] + budget["validation_seconds_per_pass"]),
        conservative_suite_seconds=elapsed + train["budget_safety_factor"] * (
            budget["fixed_overhead_seconds"] + budget["model_fixed_evaluation_seconds"]
            + cap * (budget["training_seconds_per_pass"] + budget["validation_seconds_per_pass"])))
    budget["remaining_budget_seconds"] = train["budget_seconds"] - budget["conservative_suite_seconds"]
    return budget


def _validate_complete_histories(cohort: Any, evaluation: Any, train_cap: int, serving_cap: int) -> None:
    lengths = {user.user_id: len(user.train) for user in cohort.users}
    if any(length - 1 > train_cap for length in lengths.values()):
        raise ValueError("complete training history exceeds cap; truncation is forbidden")
    if any(lengths[anchor.user_id] + anchor.future_index > serving_cap for anchor in evaluation.anchors):
        raise ValueError("complete observed serving history exceeds cap; truncation is forbidden")


def _save_best(path: Path, *, model: torch.nn.Module, config: dict[str, Any],
               variant: dict[str, Any], num_items: int, epoch: int, score: float) -> None:
    _atomic_torch_save({"format_version": 3, "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "experiment_config": config, "model_config": config["model"], "variant": variant,
        "num_items": num_items, "best_epoch": epoch, "selection_score": score,
        "state_dict": _cpu_state(model.state_dict())}, path)


def _save_resume(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scaler: Any, config: dict[str, Any], variant: dict[str, Any], num_items: int,
                 epoch: int, stopping: ValidationEarlyStopping, device: torch.device) -> None:
    numpy_state = np.random.get_state()
    _atomic_torch_save({"format_version": 3, "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "experiment_config": config, "model_config": config["model"], "variant": variant,
        "num_items": num_items, "epoch": epoch, "next_epoch": epoch + 1,
        "torch_version": str(torch.__version__), "model_training": model.training,
        "state_dict": _cpu_state(model.state_dict()), "optimizer_class": "torch.optim.AdamW",
        "optimizer_state_saved": True, "optimizer_state_dict": _cpu_state(optimizer.state_dict()),
        "scaler_state_dict": scaler.state_dict(), "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu().clone() if device.type == "cuda" else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
        "early_stopping": {"best_epoch": stopping.best_additional_passes,
            "best_score": stopping.best_score, "bad_checks": stopping.bad_checks,
            "last_validation_epoch": stopping.last_additional_passes, "stopped": stopping.stopped}}, path)


def _checkpoint_record(path: Path, output_dir: Path, **metadata: Any) -> dict[str, Any]:
    return {"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": _sha256(path), **metadata}


def run_query_prefix_pilot(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_prefix_benchmark import benchmark_prefix_quality

    validate_prefix_config(config)
    device, started = _registered_device(), time.monotonic()
    train, bench = config["training"], config["benchmark"]
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = audit_native_query_scan(device, query_source="old_suffix")
    _json(output_dir / "native-query-scan-audit.json", audit)
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_daily_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    for key in ("evaluation_content_sha256", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if evaluation.summary[key] != config["data"][key]:
            raise ValueError(f"registered daily evaluation mismatch: {key}")
    _validate_complete_histories(cohort, evaluation, train["raw_context_length"], train["serving_context_length"])
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    # The prepared reference hash deliberately remains its original16K hash;
    # this separate manifest binds the new full observed forward inputs.
    contexts = _context_manifest(cohort, evaluation, config)
    _gzip_json(output_dir / "evaluation-input-contexts.json.gz", contexts)
    context_summary = {split: {key: value for key, value in row.items() if key != "rows"}
                       for split, row in contexts.items()}
    del contexts
    variants = prefix_variants(config)
    gpu = torch.cuda.get_device_name(device)
    code_sha = os.environ.get("GITHUB_SHA", "local")
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "seed": config["seed"],
        "quality_complete": False, "hpo_performed": False, "cohort_users": len(cohort.users),
        "num_items": cohort.num_items, "training_models": len(variants), "native_scan_audit_passed": True,
        "gpu_name": gpu, "torch_version": str(torch.__version__), "query_source": "old_suffix",
        "supervised_targets_per_user": train["supervised_targets"], "full_old_prefix_retained": True,
        "train_is_exhaustive_event_epoch": False}
    report: dict[str, Any] = {"complete": False, "config": config, "code_sha": code_sha,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": gpu}, "native_scan_audit": audit,
        "protocol": {"predictor": "common independently initialized-and-trained SASRec; identical initial shared weights",
            "training": "T=train-only full sequence; X=T[:-1], Y=T[1:]; H=X[:-128], R=X[-128:], Q=H[-128:]",
            "loss": "one causal forward over summaries+R, next-item loss on all128 raw states only",
            "query": "old-prefix suffix only; summary has zero dependency on every R event",
            "summary_sources": "H_end; every summary is available to all128 supervised raw positions",
            "target_coverage": "fixed last128 targets per user are repeated each pass; all earlier train events are context only",
            "training_raw_context_lengths": "1..128 occupied R states within the causal forward",
            "serving": "entire observed H plus full R128; one final-state full-catalogue Top100 prediction per daily anchor",
            "position_encoding": "standard SASRec raw recency positions within the fixed input window",
            "checkpoint_selection": "earliest strictly best validation macro-user NDCG@10",
            "early_stopping": "validation every10 passes; minimum30; patience3 strict non-improvements",
            "test_usage": "all three validation-selected checkpoints fixed before each is tested once",
            "primary_quality": "macro_user_ndcg10", "secondary_quality": "macro_user_ndcg100",
            "reference_context_hash": "unchanged prepared16K reference; separate actual full-input hashes saved",
            "interpretation": "single-seed bounded pilot; changes backbone/query/training/context relative to previous runs"},
        "input_contexts": context_summary, "preflight": [], "models": {}, "cases": {}, "schedules": []}
    manifest: dict[str, Any] = {"format_version": 3, "source_code_sha": code_sha, "config": config,
        "optimizer_state_saved": True, "best_checkpoint_optimizer_state_saved": False,
        "checkpoints": [], "resume_checkpoints": []}

    def persist() -> None:
        _json(output_dir / "query-prefix-quality.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("prefix query pilot exceeded registered wall-time budget; partial outputs retained")

    first = fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1)
    persist()
    for variant in variants:
        guard()
        with _preserve_rng(device):
            row = profile_prefix_variant(config, variant, cohort, evaluation, first, device)
        report["preflight"].append(row)
        print(json.dumps({"phase": "prefix_preflight", **row}), flush=True)
        persist()
    report["budget"] = prefix_pass_budget(config, report["preflight"], time.monotonic() - started)
    passes = report["budget"]["passes"]
    metrics["common_max_training_passes"] = passes
    schedules = [fixed_prefix_schedule(cohort, seed=config["seed"], epoch=epoch) for epoch in range(1, passes + 1)]
    report["schedules"] = [prefix_schedule_summary(cohort, schedule, config) for schedule in schedules]
    _json(output_dir / "window-schedules.json", report["schedules"])
    metrics["unique_supervised_target_positions_per_model"] = len(cohort.users) * train["supervised_targets"]
    persist()
    for variant, preflight in zip(variants, report["preflight"], strict=True):
        guard()
        name = variant["name"]
        model = build_prefix_model(config, variant, cohort.num_items).to(device)
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        reset_restart_rng(config["seed"])
        stopping = ValidationEarlyStopping(-1., minimum_additional_passes=train["early_stopping_min_passes"],
            patience=train["early_stopping_patience"], validation_interval=train["validation_interval"],
            max_additional_passes=passes)
        checkpoint, resume_path = checkpoints / f"{name}.pt", checkpoints / f"{name}-resume.pt"
        result: dict[str, Any] = {**variant, "passes": [], "best_epoch": None,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters()),
            "physical_batch_size": preflight["training_batch_size"], "selection_metric": "macro_user_ndcg10"}
        report["models"][name] = result
        serving_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["serving_context_length"]
        for epoch, windows in enumerate(schedules, 1):
            guard()
            row = {"epoch": epoch, **train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=preflight["training_batch_size"], device=device, optimizer=optimizer, scaler=scaler)}
            if epoch % train["validation_interval"] == 0:
                guard()
                with _preserve_rng(device):
                    validation = evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
                        split_name="validation", max_seq_len=serving_length), device=device, amp=train["amp"],
                        batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
                row["validation"] = validation["metrics"]
                decision = stopping.observe(epoch, validation["metrics"]["ndcg10"])
                row["early_stopping"] = decision
                if decision["improved"]:
                    result["best_epoch"], result["best_validation"] = epoch, validation["metrics"]
                    _save_best(checkpoint, model=model, config=config, variant=variant, num_items=cohort.num_items,
                        epoch=epoch, score=decision["best_score"])
                    _json(output_dir / f"{name}-validation-users.json", validation["per_user"])
                    _gzip_json(output_dir / f"{name}-validation-anchors.json.gz", validation["per_anchor"])
                del validation
            result["passes"].append(row)
            persist()
            print(json.dumps({"phase": "prefix_train", "model": name, **row}), flush=True)
            if stopping.stopped:
                break
        result.update(passes_completed=epoch, stopped_early=epoch < passes,
            best_at_last_pass=result["best_epoch"] == epoch,
            total_labels=sum(row["labels"] for row in result["passes"]),
            total_successful_updates=sum(row["successful_updates"] for row in result["passes"]),
            amp_skipped_update_attempts=sum(row["amp_skipped_update_attempts"] for row in result["passes"]))
        _save_resume(resume_path, model=model, optimizer=optimizer, scaler=scaler, config=config,
            variant=variant, num_items=cohort.num_items, epoch=epoch, stopping=stopping, device=device)
        result["checkpoint"] = _checkpoint_record(checkpoint, output_dir, training_model=name,
            variant=variant, best_epoch=result["best_epoch"], precision="float32")
        result["resume_checkpoint"] = _checkpoint_record(resume_path, output_dir, training_model=name,
            epoch=epoch, next_epoch=epoch + 1, optimizer_state_saved=True)
        manifest["checkpoints"].append(result["checkpoint"])
        manifest["resume_checkpoints"].append(result["resume_checkpoint"])
        metrics.update({f"{name}_passes_completed": epoch, f"{name}_best_epoch": result["best_epoch"],
            f"{name}_best_at_last_pass": result["best_at_last_pass"], f"{name}_stopped_early": result["stopped_early"],
            f"{name}_training_labels": result["total_labels"],
            f"{name}_successful_updates": result["total_successful_updates"],
            f"{name}_amp_retried_update_attempts": result["amp_skipped_update_attempts"]})
        persist()
        del model, optimizer, scaler
        _cleanup()
    # No test result is available while any arm's checkpoint is being selected.
    test_rows: dict[str, list[dict[str, Any]]] = {}
    for variant, preflight in zip(variants, report["preflight"], strict=True):
        guard()
        name, result = variant["name"], report["models"][variant["name"]]
        model = build_prefix_model(config, variant, cohort.num_items).to(device)
        saved = torch.load(output_dir / result["checkpoint"]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        serving_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["serving_context_length"]
        test = evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
            split_name="test", max_seq_len=serving_length), device=device, amp=train["amp"],
            batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        _json(output_dir / f"{name}-test-users.json", test["per_user"])
        _gzip_json(output_dir / f"{name}-test-anchors.json.gz", test["per_anchor"])
        test_rows[name] = test["per_anchor"]
        guard()
        inference = benchmark_prefix_quality(model, cohort, config=config, device=device)
        report["cases"][name] = {"training_model": name, "summary_tokens": variant["summary_tokens"],
            "best_epoch": result["best_epoch"], "checkpoint": result["checkpoint"],
            "validation": result["best_validation"], "test": test["metrics"],
            "evaluation_metadata": test["metadata"], "trained_inference": inference}
        metrics.update({f"{name}_test_{key}": value for key, value in test["metrics"].items()})
        metrics.update({f"{name}_{key}": value for key, value in inference["metrics"].items()})
        persist()
        print(json.dumps({"phase": "prefix_case_complete", "model": name, "test": test["metrics"]}), flush=True)
        del model
        _cleanup()
    guard()
    report["paired_test_contrasts"] = [{"reference": reference, "candidate": candidate,
        "cutoffs": {str(k): paired_daily_ndcg_bootstrap(test_rows[reference], test_rows[candidate],
            k=k, seed=config["seed"], draws=5000) for k in (10, 100)},
        "training_seed_uncertainty_included": False}
        for reference, candidate in (("recent128", "query_cross_attention_m4"),
            ("recent128", "query_ssm_m4"), ("query_cross_attention_m4", "query_ssm_m4"))]
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"], metrics["test_evaluations"] = time.monotonic() - started, len(test_rows)
    persist()
    return validate_metrics(metrics)
