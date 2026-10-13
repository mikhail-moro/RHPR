"""Matched fixed/Mixed-budget HSTU training with daily, user-level evaluation."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import platform
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import load_prepared_daily_evaluation, iter_daily_evaluation_examples
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_quality import (
    _cleanup, _json, _optimizer, _registered_device, _scaler,
    build_quality_model, quality_variants, train_quality_pass, validate_long_quality_config,
)
from rhpr.sasrec.long_quality_benchmark import benchmark_long_quality
from rhpr.sasrec.long_quality_schedule import (
    choose_uniform_pass_budget, quality_window_schedule, quality_schedule_summary,
)
from rhpr.sasrec.mixed_budget import balanced_budget_schedule, select_summary_budget


def validate_mixed_daily_config(config: dict[str, Any]) -> None:
    if config.get("kind") != "yambda_hstu_mixed_budget_daily":
        raise ValueError("kind must be yambda_hstu_mixed_budget_daily")
    # Reuse the registered numerical and causal protocol without changing the
    # previous experiment's accepted configuration.
    shared = copy.deepcopy(config)
    shared.update(kind="yambda_long_compression_quality", summary_budgets=[16, 8, 4])
    validate_long_quality_config(shared)
    if config.get("summary_budgets") != [32, 16, 8, 4]:
        raise ValueError("fixed budgets must be32/16/8/4")
    if config.get("mixed_budgets") != [1, 2, 4, 8, 16, 32]:
        raise ValueError("Mixed budgets must be1/2/4/8/16/32")
    if any(type(v) is not int for v in config["summary_budgets"] + config["mixed_budgets"]):
        raise ValueError("summary budgets must be integers")
    if config["benchmark"].get("quality_cutoffs") != [10, 100]:
        raise ValueError("daily quality requires @10 and @100")
    if config["training"]["min_passes"] != 20 or config["training"]["max_passes"] != 20:
        raise ValueError("all eighteen training models require20 matched passes")
    if config["data"].get("evaluation_protocol") != "first_listen_per_user_day_v1":
        raise ValueError("daily evaluation protocol must be registered")
    if config.get("kaggle", {}).get("dataset_sources") != [
        "carasik/rhpr-yambda500m-train16k-profile", "carasik/rhpr-yambda500m-daily-evaluation",
    ]:
        raise ValueError("Kaggle sources must match the fixed cohort and daily evaluation pins")
    for key in ("evaluation_content_sha256", "cohort_train_timestamps_sha256"):
        digest = config["data"].get(key, "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"data.{key} must be a SHA256 digest")


def daily_training_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    return quality_variants(config["summary_budgets"]) + [{
        "name": "mixed_pair_mlp", "kind": "pair_mlp",
        "summary_tokens": max(config["mixed_budgets"]), "mixed_budgets": config["mixed_budgets"],
    }]


def evaluation_cases(variant: dict[str, Any]) -> list[tuple[str, int]]:
    if "mixed_budgets" in variant:
        return [(f"{variant['name']}_m{k}", k) for k in variant["mixed_budgets"]]
    return [(variant["name"], variant["summary_tokens"])]


def configure_mixed_model(model: torch.nn.Module, variant: dict[str, Any]) -> None:
    if "mixed_budgets" in variant:
        model.compressor.mixed_budget_choices = tuple(variant["mixed_budgets"])


def _gzip_json(path: Path, value: Any) -> None:
    # Fixed mtime makes archived rows reproducible independently of save time.
    with path.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
        zipped.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_mixed_daily_quality(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.daily_training_profile import profile_daily_variant

    validate_mixed_daily_config(config)
    device = _registered_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    train, bench = config["training"], config["benchmark"]
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_daily_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    if evaluation.summary["evaluation_content_sha256"] != config["data"]["evaluation_content_sha256"]:
        raise ValueError("registered daily evaluation content mismatch")
    if evaluation.summary["reference_context_length"] != train["raw_context_length"]:
        raise ValueError("daily paired reference must use the registered raw context length")
    for key in ("validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if evaluation.summary[key] != config["data"][key]:
            raise ValueError(f"registered daily evaluation boundary mismatch: {key}")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    variants = daily_training_variants(config)
    code_sha = os.environ.get("GITHUB_SHA", "local")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "quality_complete": False,
        "hpo_performed": False, "cohort_users": len(cohort.users), "num_items": cohort.num_items,
        "validation_users": evaluation.summary["validation_users"], "test_users": evaluation.summary["test_users"],
        "validation_anchors": evaluation.summary["validation_anchors"], "test_anchors": evaluation.summary["test_anchors"],
        "gpu_name": gpu, "torch_version": str(torch.__version__),
        "training_models": len(variants), "evaluation_cases": sum(len(evaluation_cases(v)) for v in variants),
    }
    report: dict[str, Any] = {
        "complete": False, "config": config, "code_sha": code_sha,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": gpu},
        "protocol": {"primary_quality": "macro_user_ndcg10", "bootstrap_unit": "user",
                     "mixed_checkpoint_selection": "mean validation macro-user NDCG@10 across all six declared budgets",
                     "mixed_budget_assignment": "one balanced random budget per effective optimizer update; same on AMP retries",
                     "checkpoint_selection_tie": "earliest strictly best checkpoint",
                     "test_usage": "all budgets evaluated once after validation selection; no per-user test routing",
                     "inference_latency": "uncached compression+HSTU+full-catalogue Top10, includes stage FLOPs",
                     "quality_ranking": "single full-catalogue Top100 pass for both @10 and @100",
                     "previous_protocol_change": "daily anchors and common HSTU normalization191; compare current models within this suite"},
        "preflight": [], "models": {}, "cases": {}, "schedules": [],
    }
    manifest: dict[str, Any] = {"source_code_sha": code_sha, "config": config,
                              "format_version": 1, "precision": "float32", "checkpoints": []}

    def persist() -> None:
        _json(output_dir / "mixed-daily-quality.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            persist()
            raise TimeoutError("registered wall-time budget exceeded; suite is incomplete")

    persist()
    schedule = quality_window_schedule(cohort, seed=config["seed"], epoch=0,
                                       raw_context_length=train["raw_context_length"])
    for variant in variants:
        guard()
        print(json.dumps({"phase": "daily_preflight", **variant}), flush=True)
        row = profile_daily_variant(config, variant, cohort, evaluation, schedule, device)
        report["preflight"].append(row)
        persist()
    costs = [{"training_pass_seconds": r["estimated_train_pass_seconds"],
              "validation_pass_seconds": r["estimated_validation_seconds"] / train["validation_interval"],
              "evaluation_seconds": r["estimated_validation_seconds"] + r["estimated_test_seconds"]}
             for r in report["preflight"]]
    budget = choose_uniform_pass_budget(costs, budget_seconds=train["budget_seconds"],
        preflight_seconds=time.monotonic() - started, fixed_overhead_seconds=60 * len(variants),
        min_passes=train["min_passes"], max_passes=train["max_passes"], safety_factor=train["budget_safety_factor"])
    report["budget"] = budget
    passes = budget["passes"]
    metrics["common_training_passes"] = passes
    metrics["labels_per_training_model"] = len(cohort.users) * train["supervised_targets"] * passes
    print(json.dumps({"phase": "daily_budget", **budget}), flush=True)
    schedules = [quality_window_schedule(cohort, seed=config["seed"], epoch=e,
                 raw_context_length=train["raw_context_length"]) for e in range(1, passes + 1)]
    report["schedules"] = [quality_schedule_summary(cohort, s, raw_context_length=train["raw_context_length"],
                          raw_tail_size=train["raw_tail_size"], supervised_targets=train["supervised_targets"])
                          for s in schedules]
    budget_schedules = [balanced_budget_schedule(seed=config["seed"], epoch=e,
                        effective_updates=math.ceil(len(s) / train["effective_batch_size"]),
                        budgets=config["mixed_budgets"]) for e, s in enumerate(schedules, 1)]
    _json(output_dir / "window-schedules.json", report["schedules"])
    _json(output_dir / "mixed-budget-schedules.json", budget_schedules)
    reference = None
    mixed_test_rows: dict[int, list[dict[str, Any]]] = {}

    for variant, preflight in zip(variants, report["preflight"], strict=True):
        guard()
        name = variant["name"]
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        model = build_quality_model(config, variant, cohort.num_items).to(device)
        configure_mixed_model(model, variant)
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        checkpoint = checkpoints / f"{name}.pt"
        result: dict[str, Any] = {**variant, "passes": [], "best_epoch": None,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "compressor_parameters": sum(p.numel() for p in model.compressor.parameters()),
            "physical_batch_size": preflight["training_batch_size"],
            "hstu_normalization_length": model.hstu_normalization_length,
            "selection_metric": "mean_declared_budgets_macro_user_ndcg10" if "mixed_budgets" in variant else "macro_user_ndcg10"}
        report["models"][name] = result
        best = -1.0
        raw_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["raw_context_length"]

        def evaluate(split: str) -> dict[str, Any]:
            return evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
                split_name=split, max_seq_len=raw_length), device=device, amp=train["amp"],
                batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])

        def save_rows(case_name: str, split: str, value: dict[str, Any]) -> None:
            _json(output_dir / f"{case_name}-{split}-users.json", value["per_user"])
            _gzip_json(output_dir / f"{case_name}-{split}-anchors.json.gz", value["per_anchor"])

        for epoch, windows in enumerate(schedules, 1):
            guard()
            row = {"epoch": epoch, **train_quality_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=preflight["training_batch_size"], device=device, optimizer=optimizer, scaler=scaler,
                budget_sequence=budget_schedules[epoch - 1] if "mixed_budgets" in variant else None)}
            if epoch % train["validation_interval"] == 0 or epoch == passes:
                model.compressor.tail_size = train["raw_tail_size"]
                validation = {}
                for case_name, k in evaluation_cases(variant):
                    guard()
                    if "mixed_budgets" in variant:
                        select_summary_budget(model, k)
                    validation[case_name] = evaluate("validation")
                row["validation"] = {key: value["metrics"] for key, value in validation.items()}
                score = sum(v["metrics"]["ndcg10"] for v in validation.values()) / len(validation)
                row["selection_score"] = score
                if score > best:
                    best, result["best_epoch"] = score, epoch
                    payload = {"format_version": 1, "source_code_sha": code_sha,
                        "experiment_config": config, "model_config": config["model"], "variant": variant,
                        "num_items": cohort.num_items, "best_epoch": epoch, "selection_score": score,
                        "summary_budgets": variant.get("mixed_budgets", [variant["summary_tokens"]]),
                        "hstu_normalization_length": model.hstu_normalization_length,
                        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                    torch.save(payload, checkpoint)
                    del payload
                    result["best_validation"] = row["validation"]
                    for case_name, value in validation.items():
                        save_rows(case_name, "validation", value)
                del validation
            result["passes"].append(row)
            persist()
            print(json.dumps({"phase": "daily_train", "model": name, **row}), flush=True)
        del optimizer, scaler
        model.zero_grad(set_to_none=True)
        _cleanup()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        del saved
        model.compressor.tail_size = train["raw_tail_size"]
        checkpoint_record = {"path": checkpoint.relative_to(output_dir).as_posix(),
            "size_bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint), "training_model": name,
            "model_config": config["model"], "variant": variant, "best_epoch": result["best_epoch"],
            "summary_budgets": variant.get("mixed_budgets", [variant["summary_tokens"]]), "precision": "float32"}
        manifest["checkpoints"].append(checkpoint_record)
        result["checkpoint"] = checkpoint_record
        for case_name, k in evaluation_cases(variant):
            guard()
            if "mixed_budgets" in variant:
                select_summary_budget(model, k)
            test = evaluate("test")
            save_rows(case_name, "test", test)
            case: dict[str, Any] = {"training_model": name, "summary_tokens": k,
                "best_epoch": result["best_epoch"], "checkpoint": checkpoint_record["path"],
                "checkpoint_sha256": checkpoint_record["sha256"],
                "validation": result["best_validation"][case_name], "test": test["metrics"],
                "evaluation_metadata": test["metadata"]}
            contexts = [e.context for e in islice(iter_daily_evaluation_examples(cohort, evaluation,
                split_name="validation", max_seq_len=raw_length), preflight["evaluation_batch_size"])]
            case["trained_inference"] = benchmark_long_quality(model, contexts, device=device, amp=train["amp"],
                batch_size=preflight["evaluation_batch_size"], warmup=bench["warmup"],
                iterations=bench["iterations"], item_chunk_size=bench["item_chunk_size"])
            if reference is None:
                reference = test["per_anchor"]
            else:
                case["paired_test_vs_recent"] = {str(c): paired_daily_ndcg_bootstrap(
                    reference, test["per_anchor"], seed=config["seed"], k=c) for c in (10, 100)}
            if "mixed_budgets" in variant:
                mixed_test_rows[k] = test["per_anchor"]
            for key, value in test["metrics"].items():
                metrics[f"{case_name}_test_{key}"] = value
            for key, value in case["trained_inference"]["metrics"].items():
                metrics[f"{case_name}_{key}"] = value
            metrics[f"{case_name}_best_epoch"] = result["best_epoch"]
            report["cases"][case_name] = case
            persist()
            print(json.dumps({"phase": "daily_case_complete", "case": case_name, "test": test["metrics"]}), flush=True)
        del model
        _cleanup()
    # Predeclared adjacent-budget comparisons; intervals remain exploratory,
    # unadjusted for multiple comparisons and conditional on a single seed.
    report["mixed_adjacent_budget_test"] = [
        {"reference_budget": a, "candidate_budget": b, "comparisons": {
            str(k): paired_daily_ndcg_bootstrap(mixed_test_rows[a], mixed_test_rows[b], k=k, seed=config["seed"])
            for k in (10, 100)}}
        for a, b in zip(config["mixed_budgets"][:-1], config["mixed_budgets"][1:], strict=True)]
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"] = time.monotonic() - started
    persist()
    return validate_metrics(metrics)
