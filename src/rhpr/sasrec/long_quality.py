"""A bounded, shared-recipe quality pilot for compressed 16K histories."""
from __future__ import annotations

import gc
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.hstu import HSTUModel
from rhpr.sasrec.long_compression import make_prefix_compressor
from rhpr.sasrec.long_evaluation_data import load_prepared_long_evaluation, iter_long_evaluation_examples
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_training_profile import _loss, _train_shape
from rhpr.sasrec.model import TailOnlySequenceCompressor


def quality_variants(summary_budgets: list[int]) -> list[dict[str, Any]]:
    return [{"name": "recent128", "kind": "recent", "summary_tokens": 0}] + [
        {"name": f"{kind}_m{budget}", "kind": kind, "summary_tokens": budget}
        for kind in ("mean", "uniform", "pair_mlp", "qformer") for budget in summary_budgets
    ]


def validate_long_quality_config(config: dict[str, Any]) -> None:
    if config.get("kind") != "yambda_long_compression_quality":
        raise ValueError("kind must be yambda_long_compression_quality")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name is required")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data, model, train, bench = (config.get(k, {}) for k in ("data", "model", "training", "benchmark"))
    for key in ("prepared_path", "evaluation_path", "evaluation_content_sha256",
                "cohort_uid_length_sha256", "catalog_sha256", "cohort_mapped_train_items_sha256"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} is required")
    for section, keys in ((model, ("d_model", "n_heads", "n_layers", "ffn_multiplier")),
                          (train, ("raw_context_length", "raw_tail_size", "supervised_targets",
                                   "effective_batch_size", "num_negatives", "min_passes", "max_passes",
                                   "validation_interval", "cpu_threads")),
                          (bench, ("evaluation_batch_size", "item_chunk_size", "iterations"))):
        if any(type(section.get(k)) is not int or section[k] < 1 for k in keys):
            raise ValueError("positive integer dimensions are required")
    if (model["d_model"] % model["n_heads"] or type(model.get("use_time_features", False)) is not bool
            or model.get("use_time_features", False)):
        raise ValueError("HSTU requires divisible heads and Time OFF")
    if isinstance(model.get("dropout"), bool) or not isinstance(model.get("dropout"), (int, float)) or not 0 <= model["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    if (config.get("summary_budgets") != [16, 8, 4]
            or any(type(v) is not int for v in config["summary_budgets"])
            or type(config.get("group_arity")) is not int or config["group_arity"] != 4):
        raise ValueError("this pilot requires budgets16/8/4 and the four-way PairMLP extension")
    if (train["raw_context_length"] != 16384 or train["raw_tail_size"] != 128
            or train["supervised_targets"] != 32):
        raise ValueError("registered protocol requires total16K, inference128, training32 targets")
    if train["min_passes"] > train["max_passes"]:
        raise ValueError("min_passes exceeds max_passes")
    for key in ("learning_rate", "grad_clip_norm", "initial_loss_scale", "budget_seconds", "budget_safety_factor"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    if train["budget_safety_factor"] < 1:
        raise ValueError("budget_safety_factor must be at least1")
    if (isinstance(train.get("weight_decay"), bool) or not isinstance(train.get("weight_decay"), (int, float))
            or not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0):
        raise ValueError("weight_decay must be nonnegative and finite")
    if type(data.get("train_end_timestamp")) is not int or data["train_end_timestamp"] < 1:
        raise ValueError("data.train_end_timestamp must be a positive integer")
    for key in ("physical_batch_sizes",):
        values = train.get(key)
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(type(v) is not int or v < 1 or train["effective_batch_size"] % v for v in values)):
            raise ValueError("physical batches must be unique positive divisors of effective batch")
    if train.get("device") != "cuda" or type(train.get("amp")) is not bool:
        raise ValueError("published pilot requires explicit CUDA and boolean AMP")
    if (type(bench.get("warmup")) is not int or bench["warmup"] < 0
            or any(type(v) is not int or v < 1 for v in bench.get("batch_sizes", []))
            or not bench.get("batch_sizes")):
        raise ValueError("valid benchmark warmup and batch sizes are required")
    cutoffs = bench.get("quality_cutoffs", [10])
    if (not isinstance(cutoffs, list) or not cutoffs or cutoffs[0] != 10
            or len(set(cutoffs)) != len(cutoffs)
            or any(type(v) is not int or v < 1 for v in cutoffs)):
        raise ValueError("quality_cutoffs must contain unique positive integers with primary10 first")


def build_quality_model(config: dict[str, Any], variant: dict[str, Any], num_items: int) -> HSTUModel:
    """Identical backbone initialization, raw PE table and attention normalization."""
    train, dims = config["training"], config["model"]
    tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    torch.manual_seed(config["seed"])
    if variant["kind"] == "recent":
        compressor = TailOnlySequenceCompressor(tail)
    else:
        compressor = make_prefix_compressor(
            kind=variant["kind"], d_model=dims["d_model"], summary_tokens=variant["summary_tokens"],
            tail_size=tail, group_arity=config["group_arity"], n_heads=dims["n_heads"], dropout=0.0,
        )
    return HSTUModel(num_items=num_items, max_seq_len=train["raw_context_length"],
                     custom_sequence_compressor=compressor,
                     normalization_length=tail + max(config["summary_budgets"]),
                     **{k: dims[k] for k in ("d_model", "n_heads", "n_layers", "ffn_multiplier", "dropout")})


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _registered_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("registered quality pilot requires CUDA")
    return torch.device("cuda:0")


def _optimizer(model: torch.nn.Module, train: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
                             weight_decay=train["weight_decay"], foreach=False)


def _scaler(device: torch.device, train: dict[str, Any]) -> Any:
    return torch.amp.GradScaler(device.type, enabled=train["amp"] and device.type == "cuda",
                               init_scale=train["initial_loss_scale"])


def train_quality_pass(model: torch.nn.Module, cohort: Any, windows: tuple[Any, ...], *,
                       config: dict[str, Any], variant: dict[str, Any], physical_batch: int,
                       device: torch.device, optimizer: torch.optim.Optimizer, scaler: Any,
                       budget_sequence: tuple[int, ...] | None = None) -> dict[str, Any]:
    from rhpr.sasrec.long_quality_schedule import quality_batch_arrays

    train = config["training"]
    model.compressor.tail_size = train["raw_tail_size"] + train["supervised_targets"] - 1
    model.train()
    started, total_loss, successful, skipped = time.perf_counter(), 0.0, 0, 0
    effective = train["effective_batch_size"]
    budget_counts: dict[str, dict[str, int]] = {}
    if budget_sequence is not None:
        from rhpr.sasrec.mixed_budget import select_summary_budget
        if len(budget_sequence) != math.ceil(len(windows) / effective):
            raise ValueError("one Mixed budget is required per effective update")
        choices = model.compressor.mixed_budget_choices
        if any(type(b) is not int or b not in choices for b in budget_sequence):
            raise ValueError("Mixed schedule contains an undeclared budget")
    for first in range(0, len(windows), effective):
        group = windows[first:first + effective]
        if budget_sequence is not None:
            budget = budget_sequence[first // effective]
            select_summary_budget(model, budget)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        for attempt in range(5):
            # Retry the same labels, negatives and dropout after an AMP overflow.
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for offset in range(0, len(group), physical_batch):
                chunk = group[offset:offset + physical_batch]
                arrays = quality_batch_arrays(
                    cohort, chunk, raw_context_length=train["raw_context_length"],
                    raw_tail_size=train["raw_tail_size"], supervised_targets=train["supervised_targets"],
                    num_negatives=train["num_negatives"], recent_only=variant["kind"] == "recent")
                batch = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=train["amp"] and device.type == "cuda"):
                    encoded = model.encode(batch["item_ids"], batch["valid_mask"])
                loss = _loss(model, encoded, batch, train["supervised_targets"], train["num_negatives"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                weight = len(chunk) / len(group)
                group_loss += float(loss.detach()) * weight
                scaler.scale(loss * weight).backward()
                del encoded, loss, batch, arrays
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip_norm"])
            finite = bool(torch.isfinite(norm))
            if finite:
                scaler.step(optimizer)
                scaler.update()
                successful += 1
                total_loss += group_loss * len(group)
                if budget_sequence is not None:
                    counts = budget_counts.setdefault(str(budget), {"updates": 0, "windows": 0, "labels": 0})
                    counts["updates"] += 1
                    counts["windows"] += len(group)
                    counts["labels"] += len(group) * train["supervised_targets"]
                break
            skipped += 1
            if not scaler.is_enabled() or attempt == 4:
                raise FloatingPointError("nonfinite gradients after bounded AMP recovery")
            # A nonfinite aggregate norm can occur even with individually
            # finite gradients. Never let GradScaler mutate Adam in that case.
            scaler.update(new_scale=float(scaler.get_scale()) / 2)
        optimizer.zero_grad(set_to_none=True)
    _sync(device)
    result = {"loss": total_loss / len(windows), "windows": len(windows),
            "labels": len(windows) * train["supervised_targets"], "successful_updates": successful,
            "amp_skipped_update_attempts": skipped, "seconds": time.perf_counter() - started,
            "loss_scale": float(scaler.get_scale())}
    if budget_sequence is not None:
        result["budget_exposures"] = budget_counts
    return result


def _preflight_variant(config: dict[str, Any], variant: dict[str, Any], cohort: Any,
                       evaluation: Any, schedule: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    from rhpr.sasrec.long_quality_benchmark import benchmark_long_quality

    train, bench = config["training"], config["benchmark"]
    tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    context = tail if variant["kind"] == "recent" else train["raw_context_length"]
    profile = {**train, "tail_size": train["supervised_targets"], "update_repetitions": 1,
               "warmup_microbatches": 3}
    training_rows = []
    for physical in train["physical_batch_sizes"]:
        model = build_quality_model(config, variant, cohort.num_items).to(device)
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        try:
            row = _train_shape(model, cohort, schedule, profile, context, physical, device, optimizer, scaler)
        except torch.OutOfMemoryError:
            row = {"status": "oom", "physical_batch_size": physical}
        training_rows.append(row)
        del optimizer, scaler, model
        _cleanup()
    valid = [r for r in training_rows if r["status"] == "ok" and not r["amp_skipped_updates"]]
    if not valid:
        raise RuntimeError(f"{variant['name']}: no successful measured training batch")
    chosen = max(valid, key=lambda r: r["train_windows_per_second_with_prepare_h2d_sync"])
    model = build_quality_model(config, variant, cohort.num_items).to(device)
    model.compressor.tail_size = train["raw_tail_size"]
    examples = iter_long_evaluation_examples(cohort, evaluation, split_name="validation",
                                             max_seq_len=train["raw_tail_size"] if variant["kind"] == "recent" else context)
    from itertools import islice
    contexts = [e.context for e in islice(examples, max(bench["batch_sizes"]))]
    inference = []
    for size in bench["batch_sizes"]:
        try:
            record = benchmark_long_quality(model, contexts, device=device, amp=train["amp"], batch_size=size,
                                             warmup=bench["warmup"], iterations=bench["iterations"],
                                             item_chunk_size=bench["item_chunk_size"])
            case = record["cases"][f"b{size}"]
            inference.append({"status": "ok", "batch_size": size,
                              "users_per_second": case["stages"]["full"]["users_per_second"], **record})
        except torch.OutOfMemoryError:
            inference.append({"status": "oom", "batch_size": size})
            _cleanup()
    del model
    _cleanup()
    return {**variant, "training": training_rows, "inference": inference,
            "selected_physical_batch": chosen["physical_batch_size"],
            "estimated_training_pass_seconds": len(cohort.users) / chosen["train_windows_per_second_with_prepare_h2d_sync"]}


def run_long_quality(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.long_quality_schedule import (
        quality_window_schedule, quality_schedule_summary, choose_uniform_pass_budget,
    )
    from rhpr.sasrec.long_quality_benchmark import evaluate_long_quality, paired_ndcg_bootstrap, benchmark_long_quality

    validate_long_quality_config(config)
    device = _registered_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, bench = config["training"], config["benchmark"]
    extra_cutoffs = tuple(k for k in bench.get("quality_cutoffs", [10]) if k != 10)
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256", "cohort_mapped_train_items_sha256"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_long_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    if evaluation.summary["evaluation_content_sha256"] != config["data"]["evaluation_content_sha256"]:
        raise ValueError("registered evaluation content mismatch")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    variants = quality_variants(config["summary_budgets"])
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "quality_complete": False,
        "hpo_performed": False, "cohort_users": len(cohort.users), "num_items": cohort.num_items,
        "validation_users": evaluation.summary["validation_users"], "test_users": evaluation.summary["test_users"],
        "gpu_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
    }
    report: dict[str, Any] = {"complete": False, "config": config, "code_sha": os.environ.get("GITHUB_SHA", "local"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device)}, "preflight": [], "models": {}, "schedules": []}
    def persist() -> None:
        _json(output_dir / "long-quality.json", report)
        _json(output_dir / "metrics.json", validate_metrics(metrics))
    persist()
    schedule = quality_window_schedule(cohort, seed=config["seed"], epoch=0,
                                       raw_context_length=train["raw_context_length"])
    for variant in variants:
        if time.monotonic() - started > train["budget_seconds"]:
            raise TimeoutError("preflight exceeded the registered execution budget")
        print(json.dumps({"phase": "quality_preflight", **variant}), flush=True)
        row = _preflight_variant(config, variant, cohort, evaluation, schedule, device)
        report["preflight"].append(row)
        persist()

    # Benchmark forward time excludes transfer/preparation; give evaluation a
    # further explicit allowance before the global 25% budget safety factor.
    costs = []
    for row in report["preflight"]:
        ok = [r for r in row["inference"] if r["status"] == "ok" and r["batch_size"] <= bench["evaluation_batch_size"]]
        if not ok:
            raise RuntimeError("no successful inference batch")
        selected = max(ok, key=lambda r: r["users_per_second"])
        row["selected_evaluation_batch"] = selected["batch_size"]
        validation_seconds = 1.5 * evaluation.summary["validation_users"] / selected["users_per_second"]
        test_seconds = 1.5 * evaluation.summary["test_users"] / selected["users_per_second"]
        costs.append({"training_pass_seconds": row["estimated_training_pass_seconds"],
                      "validation_pass_seconds": validation_seconds / train["validation_interval"],
                      "evaluation_seconds": validation_seconds + test_seconds})
    budget = choose_uniform_pass_budget(costs, budget_seconds=train["budget_seconds"],
                                        preflight_seconds=time.monotonic() - started,
                                        fixed_overhead_seconds=60 * len(variants),
                                        min_passes=train["min_passes"], max_passes=train["max_passes"],
                                        safety_factor=train["budget_safety_factor"])
    report["budget"] = budget
    passes = budget["passes"]
    metrics["common_training_passes"] = passes
    metrics["labels_per_model"] = len(cohort.users) * train["supervised_targets"] * passes
    persist()
    print(json.dumps({"phase": "quality_budget", **budget}), flush=True)
    schedules = [quality_window_schedule(cohort, seed=config["seed"], epoch=e,
                                         raw_context_length=train["raw_context_length"]) for e in range(1, passes + 1)]
    report["schedules"] = [quality_schedule_summary(cohort, s, raw_context_length=train["raw_context_length"],
                                                   raw_tail_size=train["raw_tail_size"],
                                                   supervised_targets=train["supervised_targets"]) for s in schedules]
    _json(output_dir / "window-schedules.json", report["schedules"])
    reference: list[dict[str, Any]] | None = None
    # Checkpoints stay outside Kaggle's output tree and are deleted per model.
    with tempfile.TemporaryDirectory(prefix="rhpr-long-quality-") as temp:
        checkpoint = Path(temp) / "best.pt"
        for variant, preflight in zip(variants, report["preflight"], strict=True):
            name = variant["name"]
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            model = build_quality_model(config, variant, cohort.num_items).to(device)
            optimizer, scaler = _optimizer(model, train), _scaler(device, train)
            result: dict[str, Any] = {**variant, "passes": [], "best_epoch": None,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "compressor_parameters": sum(p.numel() for p in model.compressor.parameters()),
                "physical_batch_size": preflight["selected_physical_batch"], "hstu_normalization_length": model.hstu_normalization_length}
            report["models"][name] = result
            best = -1.0
            raw_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["raw_context_length"]
            for epoch, windows in enumerate(schedules, 1):
                if time.monotonic() - started > train["budget_seconds"]:
                    persist()
                    raise TimeoutError("actual wall time exceeded the registered budget; suite is incomplete")
                row = {"epoch": epoch, **train_quality_pass(model, cohort, windows, config=config,
                       variant=variant, physical_batch=preflight["selected_physical_batch"], device=device,
                       optimizer=optimizer, scaler=scaler)}
                if epoch % train["validation_interval"] == 0 or epoch == passes:
                    model.compressor.tail_size = train["raw_tail_size"]
                    val = evaluate_long_quality(model, iter_long_evaluation_examples(
                        cohort, evaluation, split_name="validation", max_seq_len=raw_length), device=device,
                        amp=train["amp"], batch_size=preflight["selected_evaluation_batch"], item_chunk_size=bench["item_chunk_size"],
                        additional_cutoffs=extra_cutoffs)
                    row["validation"] = val["metrics"]
                    if val["metrics"]["ndcg10"] > best:
                        best = val["metrics"]["ndcg10"]
                        result["best_epoch"] = epoch
                        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
                        _json(output_dir / f"{name}-validation-users.json", val["per_user"])
                result["passes"].append(row)
                persist()
                print(json.dumps({"phase": "quality_train", "model": name, **row}), flush=True)
            del optimizer, scaler
            model.zero_grad(set_to_none=True)
            _cleanup()
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.compressor.tail_size = train["raw_tail_size"]
            test = evaluate_long_quality(model, iter_long_evaluation_examples(
                cohort, evaluation, split_name="test", max_seq_len=raw_length), device=device,
                amp=train["amp"], batch_size=preflight["selected_evaluation_batch"], item_chunk_size=bench["item_chunk_size"],
                additional_cutoffs=extra_cutoffs)
            result["test"] = test["metrics"]
            _json(output_dir / f"{name}-test-users.json", test["per_user"])
            from itertools import islice
            serving_contexts = [e.context for e in islice(iter_long_evaluation_examples(
                cohort, evaluation, split_name="validation", max_seq_len=raw_length),
                preflight["selected_evaluation_batch"])]
            result["trained_inference"] = benchmark_long_quality(
                model, serving_contexts, device=device, amp=train["amp"],
                batch_size=preflight["selected_evaluation_batch"], warmup=bench["warmup"],
                iterations=bench["iterations"], item_chunk_size=bench["item_chunk_size"])
            for key, value in result["trained_inference"]["metrics"].items():
                metrics[f"{name}_{key}"] = value
            if reference is None:
                reference = test["per_user"]
            else:
                result["paired_test_ndcg_vs_recent"] = paired_ndcg_bootstrap(reference, test["per_user"], seed=config["seed"])
                for cutoff in extra_cutoffs:
                    result[f"paired_test_ndcg{cutoff}_vs_recent"] = paired_ndcg_bootstrap(
                        reference, test["per_user"], seed=config["seed"], k=cutoff)
            for key, value in test["metrics"].items():
                if isinstance(value, (str, bool, int, float)):
                    metrics[f"{name}_test_{key}"] = value
            metrics[f"{name}_best_epoch"] = result["best_epoch"]
            persist()
            print(json.dumps({"phase": "quality_model_complete", "model": name, "test": test["metrics"]}), flush=True)
            checkpoint.unlink()
            del model
            _cleanup()
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"] = time.monotonic() - started
    persist()
    return validate_metrics(metrics)
