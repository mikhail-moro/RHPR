"""Bounded query-conditioned history compression with 32 causal targets/scan.

Training scans 8,000 old events using a query from the FIRST 128 of 159 raw
tail events. Only the final 32 raw states are supervised. Serving refreshes the
query from its whole 128-event tail. The stale-query approximation is explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import (
    daily_context_sha256, iter_daily_evaluation_examples, load_prepared_daily_evaluation,
)
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.hstu import HSTUModel
from rhpr.sasrec.long_compression import make_prefix_compressor
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json, _sha256
from rhpr.sasrec.long_quality import (
    _cleanup, _json, _optimizer, _registered_device, _scaler, train_quality_pass,
)
from rhpr.sasrec.long_quality_benchmark import benchmark_long_quality
from rhpr.sasrec.long_quality_schedule import (
    choose_uniform_pass_budget, quality_schedule_summary, quality_window_schedule,
)
from rhpr.sasrec.model import TailOnlySequenceCompressor


PILOT_KIND = "yambda_query_ssm_block_pilot"
QUERY_KINDS = ("query_cross_attention", "query_ssm", "query_ssm_readout")


def query_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    k = config["summary_budgets"][0]
    return [{"name": "recent128", "kind": "recent", "summary_tokens": 0},
            {"name": f"uniform_m{k}", "kind": "uniform", "summary_tokens": k}] + [
        {"name": f"{kind}_m{k}", "kind": kind, "summary_tokens": k} for kind in QUERY_KINDS
    ]


def validate_query_pilot_config(config: dict[str, Any]) -> None:
    """Keep the first pilot small, matched and explicit about every boundary."""
    if config.get("kind") != PILOT_KIND:
        raise ValueError(f"kind must be {PILOT_KIND}")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name is required")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data, model, train, compressor, bench = [config.get(k, {}) for k in (
        "data", "model", "training", "compressor", "benchmark")]
    for section, keys in (
        (model, ("d_model", "n_heads", "n_layers", "ffn_multiplier")),
        (compressor, ("d_inner", "d_state", "gate_rank", "query_context_size")),
        (train, ("raw_context_length", "serving_context_length", "old_prefix_length",
                 "raw_tail_size", "supervised_targets", "query_context_size", "effective_batch_size",
                 "num_negatives", "min_passes", "max_passes", "validation_interval", "cpu_threads")),
        (bench, ("evaluation_batch_size", "item_chunk_size", "iterations")),
    ):
        if any(type(section.get(key)) is not int or section[key] < 1 for key in keys):
            raise ValueError("positive integer dimensions are required")
    if config.get("summary_budgets") != [4] or config.get("group_arity") != 4:
        raise ValueError("this pilot fixes K=4 and group_arity=4")
    if (train["raw_tail_size"] != 128 or train["query_context_size"] != 128
            or compressor["query_context_size"] != 128 or train["supervised_targets"] != 32
            or train["old_prefix_length"] != 8000 or train["raw_context_length"] != 8159
            or train["serving_context_length"] != 8128):
        raise ValueError("registered protocol is old8000 + tail159(train)/128(serve), query-first128, loss32")
    if train["device"] != "cuda" or type(train.get("amp")) is not bool:
        raise ValueError("native CUDA and explicit boolean AMP are required")
    if (model["d_model"] % model["n_heads"] or model.get("use_time_features") is not False
            or model["d_model"] != 128 or model["n_layers"] != 2):
        raise ValueError("first pilot fixes HSTU d128/L2, divisible heads and Time OFF")
    for section, key in ((model, "dropout"), (compressor, "dropout")):
        value = section.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError("dropout must lie in [0,1)")
    if compressor.get("dt_min") != 1e-4 or compressor.get("dt_max") != 0.1:
        raise ValueError("SSM step initialization must span1e-4..0.1 for the8K pilot")
    for key in ("learning_rate", "grad_clip_norm", "initial_loss_scale", "budget_seconds", "budget_safety_factor"):
        value = train.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    wd = train.get("weight_decay")
    if isinstance(wd, bool) or not isinstance(wd, (int, float)) or not math.isfinite(wd) or wd < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if train["min_passes"] < 30 or train["max_passes"] > 60 or train["min_passes"] > train["max_passes"]:
        raise ValueError("all five models require between30 and60 matched passes")
    if train["budget_safety_factor"] < 1:
        raise ValueError("budget_safety_factor must be at least1")
    for values, cap, name in ((train.get("physical_batch_sizes"), train["effective_batch_size"], "physical batches"),
                              (bench.get("batch_sizes"), None, "benchmark batches")):
        if (not isinstance(values, list) or not values or any(type(x) is not int or x < 1 for x in values)
                or len(set(values)) != len(values) or cap and any(cap % x for x in values)):
            raise ValueError(f"invalid {name}")
    if type(bench.get("warmup")) is not int or bench["warmup"] < 0 or bench.get("quality_cutoffs") != [10, 100]:
        raise ValueError("nonnegative warmup and full-catalogue @10/@100 are required")
    if not any(1 < batch <= bench["evaluation_batch_size"] for batch in bench["batch_sizes"]):
        raise ValueError("at least one parallel benchmark batch must fit the evaluation cap")
    if data.get("evaluation_protocol") != "first_listen_per_user_day_v1":
        raise ValueError("registered daily protocol is required")
    for key in ("prepared_path", "evaluation_path"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} is required")
    for key in ("evaluation_content_sha256", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256"):
        value = data.get(key, "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"data.{key} must be a SHA256 digest")
    for key in ("train_end_timestamp", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if type(data.get(key)) is not int or data[key] < 1:
            raise ValueError(f"data.{key} must be a positive integer")
    if not data["train_end_timestamp"] < data["validation_end_timestamp"] < data["test_end_timestamp"]:
        raise ValueError("daily boundaries must increase")
    expected_sources = ["carasik/rhpr-yambda500m-train16k-profile", "carasik/rhpr-yambda500m-daily-evaluation",
                        "carasik/rhpr-cp313-torch210-mamba-runtime"]
    if config.get("kaggle", {}).get("dataset_sources") != expected_sources:
        raise ValueError("Kaggle sources must include the pinned cohort, daily anchors and native runtime")
    runtime = config.get("runtime", {})
    if (runtime.get("target_torch_version") != "2.10.0+cu128" or runtime.get("target_python_minor") != "3.13"
            or runtime.get("manifest_sha256") != "cb1cd1421931548762bd02636771e9927ae7e6993a4dfe7df51810a874baa643"
            or runtime.get("requirements_sha256") != "7403aec00eb0ce74b03e06b062a4e382d54e59e2b9fff94f9f84f949fc6521cb"):
        raise ValueError("the previously verified native Mamba wheel runtime must be pinned")
    if "variants" in config and config["variants"] != [v["name"] for v in query_variants(config)]:
        raise ValueError("variant order must match the five registered controls")


def build_query_model(config: dict[str, Any], variant: dict[str, Any], num_items: int) -> HSTUModel:
    """Share backbone/embedding initialization independently of compressor RNG."""
    from rhpr.sasrec.query_ssm import make_query_compressor

    train, dims = config["training"], config["model"]
    tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    torch.manual_seed(config["seed"])
    if variant["kind"] == "recent":
        compressor = TailOnlySequenceCompressor(tail)
    elif variant["kind"] == "uniform":
        compressor = make_prefix_compressor(kind="uniform", d_model=dims["d_model"],
            summary_tokens=variant["summary_tokens"], tail_size=tail)
    elif variant["kind"] in QUERY_KINDS:
        compressor = make_query_compressor(kind=variant["kind"], d_model=dims["d_model"],
            summary_tokens=variant["summary_tokens"], tail_size=tail, n_heads=dims["n_heads"],
            **config["compressor"])
    else:
        raise ValueError("unregistered query pilot variant")
    return HSTUModel(num_items=num_items, max_seq_len=train["raw_context_length"],
        custom_sequence_compressor=compressor, normalization_length=tail + max(config["summary_budgets"]),
        **{key: dims[key] for key in ("d_model", "n_heads", "n_layers", "ffn_multiplier", "dropout")})


def _context_manifest(cohort: Any, evaluation: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Bind actual 8128/128-event inputs separately from the dataset's 16K hash."""
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        rows = []
        for example in iter_daily_evaluation_examples(cohort, evaluation, split_name=split,
                max_seq_len=config["training"]["serving_context_length"]):
            rows.append({"anchor_id": example.anchor_id,
                "reference_context_hash": example.reference_context_hash,
                "serving_context_length": len(example.context),
                "serving_context_sha256": daily_context_sha256(example.user_id, example.context, example.timestamps),
                "recent_context_sha256": daily_context_sha256(example.user_id, example.context[-128:], example.timestamps[-128:])})
        digest = hashlib.sha256()
        for row in rows:
            digest.update((row["anchor_id"] + ":" + row["serving_context_sha256"] + "\n").encode("ascii"))
        result[split] = {"anchors": len(rows), "ordered_serving_context_sha256": digest.hexdigest(), "rows": rows}
    return result


def _paired_contrasts(rows: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Four predeclared @10 contrasts; user-cluster intervals and Holm sign flips."""
    k = config["summary_budgets"][0]
    candidate = f"query_ssm_m{k}"
    contrasts = []
    for reference in ("recent128", f"uniform_m{k}", f"query_cross_attention_m{k}", f"query_ssm_readout_m{k}"):
        item: dict[str, Any] = {"reference": reference, "candidate": candidate,
            "cutoffs": {str(c): paired_daily_ndcg_bootstrap(rows[reference], rows[candidate],
                k=c, seed=config["seed"], draws=5000) for c in (10, 100)}}
        def means(records: list[dict[str, Any]]) -> dict[int, float]:
            users: dict[int, list[float]] = {}
            for record in records:
                users.setdefault(record["user_id"], []).append(record["ndcg10"])
            return {uid: float(np.mean(values)) for uid, values in users.items()}
        a, b = means(rows[reference]), means(rows[candidate])
        delta = np.asarray([b[uid] - a[uid] for uid in sorted(a)], dtype=np.float64)
        observed, count, draws = abs(float(delta.mean())), 0, 5000
        rng = np.random.default_rng(config["seed"])
        for first in range(0, draws, 64):
            signs = rng.integers(0, 2, size=(min(64, draws - first), len(delta)), dtype=np.int8) * 2 - 1
            count += int(np.count_nonzero(np.abs((signs * delta).mean(axis=1)) >= observed))
        item["sign_flip_two_sided_p10"] = (count + 1) / (draws + 1)
        item["training_seed_uncertainty_included"] = False
        contrasts.append(item)
    previous = 0.0
    for rank, item in enumerate(sorted(contrasts, key=lambda x: x["sign_flip_two_sided_p10"])):
        previous = max(previous, min(1.0, (len(contrasts) - rank) * item["sign_flip_two_sided_p10"]))
        item["holm_p10_family4"] = previous
    return contrasts


def _benchmark_query_serving(model: torch.nn.Module, contexts: list[np.ndarray], *,
                             config: dict[str, Any], preflight: dict[str, Any],
                             device: torch.device) -> dict[str, Any]:
    """Measure the same B1/B8/B32 cases wherever preflight established capacity."""
    bench = config["benchmark"]
    safe = {row["batch_size"] for row in preflight["evaluation"] if row["status"] == "ok"}
    requested = list(dict.fromkeys(bench["batch_sizes"]))
    batches = [batch for batch in requested if batch > 1 and batch in safe]
    if not batches:
        raise RuntimeError("no preflight-safe parallel serving batch")
    result: dict[str, Any] = {"metrics": {}, "cases": {}, "metadata": {}}
    for batch in batches:
        value = benchmark_long_quality(model, contexts, device=device, amp=config["training"]["amp"],
            batch_size=batch, warmup=bench["warmup"], iterations=bench["iterations"],
            item_chunk_size=bench["item_chunk_size"])
        for key, record in value["cases"].items():
            result["cases"].setdefault(key, record)
        for key, record in value["metrics"].items():
            result["metrics"].setdefault(key, record)
        if not result["metadata"]:
            result["metadata"] = value["metadata"]
    result["metadata"].update(requested_batch_sizes=requested,
        measured_batch_sizes=sorted({record["batch_size"] for record in result["cases"].values()}),
        skipped_batch_sizes=[{"batch_size": batch, "reason": "did not complete the parallel daily preflight"}
                             for batch in requested if batch > 1 and batch not in safe])
    result["metadata"].pop("requested_batch_size", None)
    return result


def run_query_pilot(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_profile import audit_native_query_scan, profile_query_variant

    validate_query_pilot_config(config)
    device = _registered_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    train, bench = config["training"], config["benchmark"]
    torch.set_num_threads(train["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    audit = audit_native_query_scan(device)
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
    if evaluation.summary["reference_context_length"] < train["serving_context_length"]:
        raise ValueError("prepared reference context is shorter than the serving context")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    contexts = _context_manifest(cohort, evaluation, config)
    _gzip_json(output_dir / "evaluation-input-contexts.json.gz", contexts)
    variants = query_variants(config)
    code_sha = os.environ.get("GITHUB_SHA", "local")
    gpu = torch.cuda.get_device_name(device)
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "quality_complete": False,
        "hpo_performed": False, "cohort_users": len(cohort.users), "num_items": cohort.num_items,
        "training_models": len(variants), "native_scan_audit_passed": True,
        "gpu_name": gpu, "torch_version": str(torch.__version__),
        "training_context_length": train["raw_context_length"], "serving_context_length": train["serving_context_length"],
        "old_prefix_length": train["old_prefix_length"], "query_max_staleness_events": train["supervised_targets"] - 1,
    }
    report: dict[str, Any] = {"complete": False, "config": config, "code_sha": code_sha,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": gpu}, "native_scan_audit": audit,
        "protocol": {"query": "first128 occupied raw-tail events; independent of last31 teacher-forced inputs",
            "training": "one conditioned old8000 scan; final32 raw states predict32 successive targets",
            "summary_sources": "latest query input; summaries explicitly excluded before loss construction",
            "serving": "fresh query from last128, single final-state prediction per daily anchor",
            "query_staleness": "0..31 events within a training block; zero at serving",
            "checkpoint_selection": "earliest strictly best validation macro-user NDCG@10",
            "test_usage": "one evaluation of the selected checkpoint; no test tuning",
            "primary_quality": "macro_user_ndcg10", "secondary_quality": "macro_user_ndcg100",
            "bootstrap_unit": "user", "paired_family": "QSSM vs Recent, Uniform4, query CA, query-readout SSM",
            "inference_latency": "uncached compression+HSTU+full-catalogue Top10; CPU collation and transfers excluded",
            "comparison_to_previous_run": "new 8K crop, common normalization163 and native torch2.10 runtime; compare within this suite",
            "convergence": "common cost-selected30..60 passes is a bounded pilot, not proof of convergence"},
        "input_contexts": {split: {k: v for k, v in value.items() if k != "rows"} for split, value in contexts.items()},
        "preflight": [], "models": {}, "cases": {}, "schedules": []}
    del contexts
    manifest: dict[str, Any] = {"source_code_sha": code_sha, "config": config,
        "format_version": 1, "precision": "float32", "optimizer_state_saved": False, "checkpoints": []}

    def persist() -> None:
        _json(output_dir / "query-pilot-quality.json", report)
        _json(output_dir / "checkpoint-manifest.json", manifest)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > train["budget_seconds"]:
            raise TimeoutError("query pilot exceeded its registered wall-time budget; partial outputs retained")

    first_schedule = quality_window_schedule(cohort, seed=config["seed"], epoch=1,
        raw_context_length=train["raw_context_length"])
    persist()
    for variant in variants:
        guard()
        row = profile_query_variant(config, variant, cohort, evaluation, first_schedule, device)
        report["preflight"].append(row)
        print(json.dumps({"phase": "query_preflight", **row}), flush=True)
        persist()
    costs = [{"training_pass_seconds": row["estimated_train_pass_seconds"],
              "validation_pass_seconds": row["estimated_validation_seconds"] / train["validation_interval"],
              "evaluation_seconds": row["estimated_validation_seconds"] + row["estimated_test_seconds"]}
             for row in report["preflight"]]
    budget = choose_uniform_pass_budget(costs, budget_seconds=train["budget_seconds"],
        preflight_seconds=time.monotonic() - started, fixed_overhead_seconds=60 * len(variants),
        min_passes=train["min_passes"], max_passes=train["max_passes"], safety_factor=train["budget_safety_factor"])
    passes = budget["passes"]
    report["budget"] = budget
    metrics["common_training_passes"] = passes
    metrics["labels_per_training_model"] = len(cohort.users) * train["supervised_targets"] * passes
    schedules = [quality_window_schedule(cohort, seed=config["seed"], epoch=epoch,
        raw_context_length=train["raw_context_length"]) for epoch in range(1, passes + 1)]
    report["schedules"] = [quality_schedule_summary(cohort, windows, raw_context_length=train["raw_context_length"],
        raw_tail_size=train["raw_tail_size"], supervised_targets=train["supervised_targets"]) for windows in schedules]
    _json(output_dir / "window-schedules.json", report["schedules"])
    test_rows: dict[str, list[dict[str, Any]]] = {}
    for variant, preflight in zip(variants, report["preflight"], strict=True):
        guard()
        name = variant["name"]
        model = build_query_model(config, variant, cohort.num_items).to(device)
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        checkpoint = checkpoints / f"{name}.pt"
        result: dict[str, Any] = {**variant, "passes": [], "best_epoch": None,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "compressor_parameters": sum(p.numel() for p in model.compressor.parameters()),
            "physical_batch_size": preflight["training_batch_size"], "selection_metric": "macro_user_ndcg10",
            "hstu_normalization_length": model.hstu_normalization_length}
        report["models"][name] = result
        best = -1.0
        serving_length = train["raw_tail_size"] if variant["kind"] == "recent" else train["serving_context_length"]

        def evaluate(split: str) -> dict[str, Any]:
            return evaluate_daily_quality(model, iter_daily_evaluation_examples(cohort, evaluation,
                split_name=split, max_seq_len=serving_length), device=device, amp=train["amp"],
                batch_size=preflight["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])

        def save_rows(split: str, value: dict[str, Any]) -> None:
            _json(output_dir / f"{name}-{split}-users.json", value["per_user"])
            _gzip_json(output_dir / f"{name}-{split}-anchors.json.gz", value["per_anchor"])

        for epoch, windows in enumerate(schedules, 1):
            guard()
            row = {"epoch": epoch, **train_quality_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=preflight["training_batch_size"], device=device, optimizer=optimizer, scaler=scaler)}
            if epoch % train["validation_interval"] == 0 or epoch == passes:
                model.compressor.tail_size = train["raw_tail_size"]
                guard()
                validation = evaluate("validation")
                row["validation"] = validation["metrics"]
                score = validation["metrics"]["ndcg10"]
                if score > best:
                    best, result["best_epoch"] = score, epoch
                    payload = {"format_version": 1, "source_code_sha": code_sha, "experiment_config": config,
                        "model_config": config["model"], "variant": variant, "num_items": cohort.num_items,
                        "best_epoch": epoch, "selection_score": score,
                        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                    torch.save(payload, checkpoint)
                    del payload
                    result["best_validation"] = validation["metrics"]
                    save_rows("validation", validation)
                del validation
            result["passes"].append(row)
            persist()
            print(json.dumps({"phase": "query_train", "model": name, **row}), flush=True)
        result["best_at_last_pass"] = result["best_epoch"] == passes
        del optimizer, scaler
        model.zero_grad(set_to_none=True)
        _cleanup()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        del saved
        model.compressor.tail_size = train["raw_tail_size"]
        checkpoint_record = {"path": checkpoint.relative_to(output_dir).as_posix(), "size_bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint), "training_model": name, "variant": variant,
            "best_epoch": result["best_epoch"], "precision": "float32"}
        manifest["checkpoints"].append(checkpoint_record)
        result["checkpoint"] = checkpoint_record
        guard()
        test = evaluate("test")
        save_rows("test", test)
        test_rows[name] = test["per_anchor"]
        case = {"training_model": name, "summary_tokens": variant["summary_tokens"],
            "best_epoch": result["best_epoch"], "best_at_last_pass": result["best_at_last_pass"],
            "checkpoint": checkpoint_record, "validation": result["best_validation"], "test": test["metrics"],
            "evaluation_metadata": test["metadata"]}
        safe_batch = max(row["batch_size"] for row in preflight["evaluation"] if row["status"] == "ok")
        contexts = [example.context for example in islice(iter_daily_evaluation_examples(cohort, evaluation,
            split_name="validation", max_seq_len=serving_length), safe_batch)]
        guard()
        case["trained_inference"] = _benchmark_query_serving(model, contexts, config=config,
            preflight=preflight, device=device)
        for key, value in test["metrics"].items():
            metrics[f"{name}_test_{key}"] = value
        for key, value in case["trained_inference"]["metrics"].items():
            metrics[f"{name}_{key}"] = value
        metrics[f"{name}_best_epoch"] = result["best_epoch"]
        metrics[f"{name}_best_at_last_pass"] = result["best_at_last_pass"]
        report["cases"][name] = case
        persist()
        print(json.dumps({"phase": "query_case_complete", "case": name, "test": test["metrics"]}), flush=True)
        del model
        _cleanup()
    guard()
    report["paired_test_contrasts"] = _paired_contrasts(test_rows, config)
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"] = time.monotonic() - started
    persist()
    return validate_metrics(metrics)
