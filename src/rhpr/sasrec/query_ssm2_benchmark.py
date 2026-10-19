"""Bounded native QSSM1/SSD QSSM2 training-step measurements, without HPO.

Every arm consumes the same real H1024+R128 batches and dense catalogue. Native
kernel/audit failures are results; there is no reference GPU or dtype fallback.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_quality import _cleanup, _json, _scaler, _sync
from rhpr.sasrec.long_training_profile import _loss
from rhpr.sasrec.query_history_models import build_history_model, history_optimizer
from rhpr.sasrec.query_restart_state import reset_restart_rng
from rhpr.sasrec.query_rotating_protocol import rotating_batch_arrays, rotating_schedule


BENCHMARK_KIND = "yambda_query_ssm2_benchmark"
_INITIALIZER_SHA256 = "e91882fc9a684e192dfdbc6accb5678380668d344c8cc49148cac03d877de5fa"
_VARIANT = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}
_ARMS = (
    {"name": "qssm1_native", "backend": "mamba1_selective_scan"},
    {"name": "qssm2_h32", "backend": "mamba2_ssd", "head_dim": 32, "chunk_size": 64},
    {"name": "qssm2_h64", "backend": "mamba2_ssd", "head_dim": 64, "chunk_size": 64},
)
_MEASUREMENT = {"warmup_updates": 2, "measured_updates": 5,
    "compressor_warmup": 2, "compressor_iterations": 5, "physical_batch_size": 32,
    "old_history_tokens": 1024, "budget_seconds": 600, "quality_training": False,
    "validation_evaluations": 0, "test_evaluations": 0, "automatic_fallback": False}
_DATA_KEYS = ("prepared_path", "cohort_uid_length_sha256", "catalog_sha256",
              "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "train_end_timestamp")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def benchmark_arms() -> list[dict[str, Any]]:
    return [dict(arm) for arm in _ARMS]


def validate_ssm2_benchmark_config(config: dict[str, Any]) -> None:
    fields = {"name", "kind", "seed", "initializer", "data", "training", "measurement", "runtime", "kaggle", "arms"}
    if (set(config) != fields or config["kind"] != BENCHMARK_KIND
            or config["name"] != "yambda500m-query-ssm2-benchmark" or config["seed"] != 42):
        raise ValueError("the named seed42 native QSSM2 speed benchmark is required")
    source = config["initializer"]
    if _digest(source) != _INITIALIZER_SHA256:
        raise ValueError("the selected original K32/inner64/state16/slow QSSM initializer must remain pinned")
    if (config["data"] != {key: source["data"][key] for key in _DATA_KEYS}
            or config["training"] != source["parent"]["experiment_config"]["training"]
            or config["runtime"] != source["runtime"]):
        raise ValueError("the actual cohort, full catalogue, original optimization and offline runtime are pinned")
    if config["measurement"] != _MEASUREMENT or config["arms"] != benchmark_arms():
        raise ValueError("exactly three registered native arms with 2 warmup/5 timed updates and a 600s ceiling are required")
    expected_kaggle = {"dataset_sources": ["carasik/rhpr-yambda500m-train16k-profile",
        "carasik/rhpr-cp313-torch210-mamba-runtime"], "kernel_sources": [], "competition_sources": [], "model_sources": []}
    if config["kaggle"] != expected_kaggle:
        raise ValueError("only cohort and offline runtime mounts are required; no checkpoint or evaluation sources")


def build_benchmark_model(config: dict[str, Any], arm: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Rebuild the historical fresh initializer, then replace only the compressor."""
    if arm not in benchmark_arms():
        raise ValueError("an exactly registered benchmark arm is required")
    model = build_history_model(config["initializer"], num_items)
    if arm["backend"] == "mamba2_ssd":
        from rhpr.sasrec.query_ssm2 import make_query_ssm2_compressor
        dims, train = config["initializer"]["model"], config["training"]
        model.compressor = make_query_ssm2_compressor(d_model=dims["d_model"], summary_tokens=32,
            tail_size=train["raw_tail_size"], query_context_size=128, n_heads=dims["n_heads"],
            d_state=16, d_inner=64, gate_rank=32, head_dim=arm["head_dim"], chunk_size=arm["chunk_size"],
            c_conditioning="query", dropout=0., dt_min=1e-4, dt_max=.01, query_source="old_suffix")
    if model.history_source_checkpoint_loaded:
        raise RuntimeError("timing models must never load source weights")
    return model


def prepare_benchmark_batches(config: dict[str, Any], cohort: Any) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Collate each real batch once, outside timing, with identical IDs for all arms."""
    measurement, train = config["measurement"], config["training"]
    count = measurement["warmup_updates"] + measurement["measured_updates"]
    effective = train["effective_batch_size"]
    schedule = rotating_schedule(cohort, seed=config["seed"], epoch=1)
    if len(schedule) < count * effective:
        raise ValueError("benchmark requires enough distinct actual users for seven complete effective updates")
    schedule = schedule[:count * effective]
    batches, signature = [], hashlib.sha256()
    for first in range(0, len(schedule), effective):
        arrays = rotating_batch_arrays(cohort, schedule[first:first + effective], config, _VARIANT, old_length=1024)
        for key in sorted(arrays):
            value = np.ascontiguousarray(arrays[key])
            signature.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode())
            signature.update(value.tobytes())
        batches.append(arrays)
    descriptors = [[w.user_id, w.end, w.negative_seed] for w in schedule]
    return batches, {"batch_arrays_sha256": signature.hexdigest(), "window_descriptors_sha256": _digest(descriptors),
        "same_arrays_for_every_arm": True, "schedule_epoch": 1, "windows": len(schedule),
        "effective_updates": count, "input_tokens": 1152, "supervised_targets_per_window": 128,
        "num_negatives": train["num_negatives"], "distinct_users": len({w.user_id for w in schedule}),
        "collation_outside_measurement": True, "contains_private_item_arrays": False}


def _backbone_hash(model: torch.nn.Module) -> str:
    signature = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith("compressor."):
            continue
        value = parameter.detach().cpu().contiguous()
        signature.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        signature.update(value.numpy().tobytes())
    return signature.hexdigest()


def _deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("registered 600-second benchmark budget exhausted; remaining work is skipped")


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or (array <= 0).any():
        raise FloatingPointError("timing samples must be positive finite values")
    return {"samples": len(values), "mean": float(array.mean()), "p50": float(np.median(array)),
            "p95": float(np.quantile(array, .95)), "min": float(array.min()), "max": float(array.max())}


def _timed(operation: Any, device: torch.device) -> tuple[Any, float, float]:
    """CUDA events give stream elapsed time, not a sum of busy kernel durations."""
    _sync(device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    value = operation()
    end.record()
    _sync(device)
    return value, time.perf_counter() - started, float(begin.elapsed_time(end))


def _critical_gradients(compressor: torch.nn.Module) -> dict[str, float]:
    parameters = dict(compressor.named_parameters())
    norms = {}
    for name in ("write_projection.weight", "delta_projection.weight", "delta_projection.bias",
                 "query_gate.weight", "query_generator.queries", "read_projection.weight", "A_log"):
        gradient = parameters[name].grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"missing or nonfinite critical compressor gradient: {name}")
        norm = float(gradient.detach().float().norm())
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f"zero or nonfinite critical compressor gradient: {name}")
        norms[name] = norm
    return norms


def _full_update(model: torch.nn.Module, arrays: dict[str, np.ndarray], config: dict[str, Any],
                 device: torch.device, optimizer: Any, scaler: Any, *, audit_gradients: bool = False) -> dict[str, Any]:
    """One true effective AdamW update, including dense catalogue gradients/state."""
    train, physical = config["training"], config["measurement"]["physical_batch_size"]
    effective = train["effective_batch_size"]
    if len(arrays["item_ids"]) != effective:
        raise ValueError("every timed update must contain exactly 128 actual users")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.
    for first in range(0, effective, physical):
        batch = {key: torch.from_numpy(value[first:first + physical]).to(device) for key, value in arrays.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=train["amp"] and device.type == "cuda"):
            encoded = model.encode(batch["item_ids"], batch["valid_mask"])
        loss = _loss(model, encoded, batch, train["supervised_targets"], train["num_negatives"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite benchmark training loss")
        total_loss += float(loss.detach()) * physical / effective
        scaler.scale(loss * physical / effective).backward()
        del batch, encoded, loss
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip_norm"])
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("nonfinite benchmark gradient; precision and loss scale remain unchanged")
    critical = _critical_gradients(model.compressor) if audit_gradients else None
    previous = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) < previous or not optimizer.state:
        raise FloatingPointError("a timed update was skipped or failed to allocate actual Adam state")
    optimizer.zero_grad(set_to_none=True)
    result = {"loss": total_loss, "gradient_norm": float(norm), "successful_updates": 1,
              "amp_skipped_updates": 0, "loss_scale": float(scaler.get_scale())}
    if critical is not None:
        result["critical_compressor_gradient_norms"] = critical
    return result


def _memory(device: torch.device) -> dict[str, float]:
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _live_memory(device: torch.device) -> dict[str, float]:
    return {"allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
            "reserved_gib": torch.cuda.memory_reserved(device) / 2**30}


def _compressor_benchmark(model: torch.nn.Module, arrays: dict[str, np.ndarray], config: dict[str, Any],
                          device: torch.device, deadline: float) -> dict[str, Any]:
    physical = config["measurement"]["physical_batch_size"]
    ids = torch.from_numpy(arrays["item_ids"][:physical]).to(device)
    mask = torch.from_numpy(arrays["valid_mask"][:physical]).to(device)
    with torch.no_grad():
        embeddings = model.item_embedding(ids).detach().requires_grad_(True)
    del ids
    def operation() -> float:
        model.compressor.zero_grad(set_to_none=True)
        embeddings.grad = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.compressor(embeddings, mask)[0][:, :32]
        loss = outputs.float().square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite compressor-only loss")
        (loss * config["training"]["initial_loss_scale"]).backward()
        return float(loss.detach())
    measured_wall, measured_gpu, losses, warmup = [], [], [], []
    for index in range(config["measurement"]["compressor_warmup"] + config["measurement"]["compressor_iterations"]):
        _deadline(deadline)
        loss, wall, gpu = _timed(operation, device)
        if index == 0:
            # Gradient audits are outside the compressor timing region.
            if embeddings.grad is None or not bool(torch.isfinite(embeddings.grad).all()):
                raise FloatingPointError("missing or nonfinite compressor-only input gradient")
            _critical_gradients(model.compressor)
        if index >= config["measurement"]["compressor_warmup"]:
            measured_wall.append(wall)
            measured_gpu.append(gpu)
            losses.append(loss)
        else:
            warmup.append({"wall_seconds": wall, "cuda_stream_elapsed_ms": gpu, "loss": loss})
    model.compressor.zero_grad(set_to_none=True)
    return {"status": "success", "physical_batch_size": physical, "wall_seconds": _statistics(measured_wall),
        "cuda_stream_elapsed_ms": _statistics(measured_gpu), "losses": losses, "warmup": warmup,
        "scope": "detached real embeddings; compression and scaled summary loss backward; no optimizer or embedding lookup",
        "optimizer_state_remains_allocated": True,
        "allocation_scope": "absolute whole-process peak, including live dense item table and full model Adam state",
        **_memory(device)}


def benchmark_one_arm(config: dict[str, Any], arm: dict[str, Any], batches: list[dict[str, np.ndarray]],
                      num_items: int, device: torch.device, deadline: float) -> dict[str, Any]:
    model = optimizer = scaler = None
    started = time.perf_counter()
    stage = "native_audit"
    result: dict[str, Any] = {"arm": dict(arm), "status": "running", "automatic_fallback": False}
    try:
        _deadline(deadline)
        if arm["backend"] == "mamba1_selective_scan":
            from rhpr.sasrec.query_profile import audit_native_query_scan
            result["native_audit"] = audit_native_query_scan(device, query_source="old_suffix")
        else:
            from rhpr.sasrec.query_ssm2 import audit_native_query_ssm2
            result["native_audit"] = audit_native_query_ssm2(device, head_dim=arm["head_dim"], chunk_size=arm["chunk_size"])
        result["native_audit_wall_seconds"] = time.perf_counter() - started
        _deadline(deadline)
        stage, initialization = "model_initialization", time.perf_counter()
        model = build_benchmark_model(config, arm, num_items)
        result["fresh_backbone_sha256"] = _backbone_hash(model)
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        result["compressor_parameters"] = sum(p.numel() for p in model.compressor.parameters())
        model = model.to(device).train()
        optimizer = history_optimizer(model, config["training"])
        scaler = _scaler(device, config["training"])
        _sync(device)
        result["initialization_seconds"] = time.perf_counter() - initialization
        reset_restart_rng(config["seed"])
        wall, gpu, updates, warmup = [], [], [], []
        stage = "full_training_step"
        full_live = {}
        for index, arrays in enumerate(batches):
            _deadline(deadline)
            if index == config["measurement"]["warmup_updates"]:
                torch.cuda.reset_peak_memory_stats(device)
                full_live = _live_memory(device)
            row, seconds, milliseconds = _timed(lambda: _full_update(model, arrays, config, device, optimizer, scaler,
                audit_gradients=index == 0), device)
            row.update(wall_seconds=seconds, cuda_stream_elapsed_ms=milliseconds)
            if index < config["measurement"]["warmup_updates"]:
                warmup.append(row)
            else:
                wall.append(seconds)
                gpu.append(milliseconds)
                updates.append(row)
        result["full_training"] = {"wall_seconds": _statistics(wall), "cuda_stream_elapsed_ms": _statistics(gpu),
            "warmup": warmup, "measured_updates": updates, "successful_timed_updates": len(updates),
            "effective_batch_size": 128, "physical_batch_size": 32, "amp_skipped_updates": 0,
            "scope": "precollated CPU arrays, H2D, compressor, SASRec, sampled loss, backward, clipping, dense AdamW, synchronization",
            "windows_per_second": 128 / float(np.mean(wall)), "initialization_excluded": True,
            "optimizer_state_allocated": bool(optimizer.state), "live_memory_before_measurement": full_live,
            "peak_extra_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 - full_live["allocated_gib"],
            **_memory(device)}
        stage = "compressor_only"
        torch.cuda.reset_peak_memory_stats(device)
        comp_live = _live_memory(device)
        result["compressor_only"] = _compressor_benchmark(model, batches[0], config, device, deadline)
        result["compressor_only"].update(live_memory_before_measurement=comp_live,
            peak_extra_allocated_gib=result["compressor_only"]["peak_allocated_gib"] - comp_live["allocated_gib"])
        result["scan_backend"] = model.compressor.last_scan_backend
        result["status"] = "success"
    except Exception as error:
        # Triton compilation failures and numerical audit AssertionErrors are
        # also arm results. KeyboardInterrupt/SystemExit still propagate.
        result.update(status="budget_exhausted" if isinstance(error, TimeoutError) else "failed",
            failure={"stage": stage, "exception_type": type(error).__name__, "message": str(error)[:3000]})
    finally:
        result["arm_wall_seconds"] = time.perf_counter() - started
        del model, optimizer, scaler
        _cleanup()
    return result


def _versions(device: torch.device) -> dict[str, Any]:
    packages = {}
    for package in ("mamba-ssm", "triton", "causal-conv1d"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_installed"
    properties = torch.cuda.get_device_properties(device)
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "packages": packages, "gpu_name": properties.name,
        "gpu_total_memory_gib": properties.total_memory / 2**30,
        "compute_capability": list(torch.cuda.get_device_capability(device)), "gpu_count": torch.cuda.device_count()}


def _comparisons(arms: dict[str, Any]) -> dict[str, Any]:
    baseline = arms.get("qssm1_native", {})
    if baseline.get("status") != "success":
        return {"available": False, "reason": "native1 baseline did not complete"}
    result: dict[str, Any] = {"available": True, "baseline": "qssm1_native", "speed_only_not_quality_equivalence": True}
    for name, candidate in arms.items():
        if name == "qssm1_native" or candidate.get("status") != "success":
            continue
        if candidate["fresh_backbone_sha256"] != baseline["fresh_backbone_sha256"]:
            raise RuntimeError("benchmark arms did not preserve identical fresh SASRec/catalogue weights")
        result[name] = {
            "full_step_speedup": baseline["full_training"]["wall_seconds"]["p50"] / candidate["full_training"]["wall_seconds"]["p50"],
            "compressor_speedup": baseline["compressor_only"]["wall_seconds"]["p50"] / candidate["compressor_only"]["wall_seconds"]["p50"],
            "full_step_peak_allocated_gib_delta": candidate["full_training"]["peak_allocated_gib"] - baseline["full_training"]["peak_allocated_gib"],
            "backbone_initialization_identical": True}
    return result


def run_query_ssm2_benchmark(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    validate_ssm2_benchmark_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("QSSM2 benchmark requires an actual CUDA device; CPU timing fallback is forbidden")
    started = time.perf_counter()
    deadline = started + config["measurement"]["budget_seconds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(config["training"]["cpu_threads"])
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in _DATA_KEYS[1:]:
        if cohort.summary.get(key) != config["data"][key]:
            raise ValueError(f"actual prepared training cohort disagrees with pinned {key}")
    batches, batch_metadata = prepare_benchmark_batches(config, cohort)
    report: dict[str, Any] = {"format_version": 1, "complete": False, "source_code_sha": os.environ.get("GITHUB_SHA", "local"),
        "config": config, "runtime": _versions(device), "cohort_summary": cohort.summary,
        "batches": batch_metadata, "arms": {}, "quality_metrics_computed": False,
        "source_checkpoint_weights_loaded": False, "validation_evaluations": 0, "test_evaluations": 0,
        "model_checkpoints_saved": False, "architecture_equivalence_claimed": False,
        "timing_caveats": ["five samples per arm; sequential arm order; no statistical speed guarantee",
            "CUDA stream elapsed time includes host launch gaps and is not summed kernel busy time",
            "Mamba2 scalar-A-per-head constraints change the architecture; this does not measure quality",
            "C is query-only, preserving the current QSSM readout choice; no query+token ablation"],
        "budget_includes": "data loading, native audits and JIT, initialization, warmup and measurements; offline install precedes runner"}
    report_path = output_dir / "query-ssm2-benchmark.json"
    _json(report_path, report)
    for arm in benchmark_arms():
        print(json.dumps({"phase": "ssm2_benchmark_arm_start", "arm": arm["name"]}), flush=True)
        report["arms"][arm["name"]] = benchmark_one_arm(config, arm, batches, cohort.num_items, device, deadline)
        _json(report_path, report)
        print(json.dumps({"phase": "ssm2_benchmark_arm_finished", "arm": arm["name"],
            "status": report["arms"][arm["name"]]["status"]}), flush=True)
    report.update(complete=True, all_arms_succeeded=all(row["status"] == "success" for row in report["arms"].values()),
        wall_seconds=time.perf_counter() - started, comparisons=_comparisons(report["arms"]))
    _json(report_path, report)
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "benchmark_complete": True,
        "all_arms_succeeded": report["all_arms_succeeded"], "quality_metrics_computed": False,
        "seed": config["seed"], "num_items": cohort.num_items, "wall_seconds": report["wall_seconds"],
        "gpu_name": report["runtime"]["gpu_name"], "automatic_fallback": False}
    lines = ["# QSSM2 native training-step benchmark", "", "Speed and memory only; no recommendation-quality result.", "",
        "| Arm | Status | Full update median, s | Compressor F+B median, s | Peak full step, GiB |", "|---|---|---:|---:|---:|"]
    for name, row in report["arms"].items():
        metrics[f"{name}_status"] = row["status"]
        if row["status"] == "success":
            full, compressor = row["full_training"], row["compressor_only"]
            for metric, value in (("update_p50_seconds", full["wall_seconds"]["p50"]),
                ("compressor_p50_seconds", compressor["wall_seconds"]["p50"]),
                ("peak_allocated_gib", full["peak_allocated_gib"])):
                metrics[f"{name}_{metric}"] = value
            comparison = report["comparisons"].get(name)
            if comparison:
                metrics[f"{name}_full_step_speedup"] = comparison["full_step_speedup"]
            lines.append(f"| {name} | success | {full['wall_seconds']['p50']:.4f} | {compressor['wall_seconds']['p50']:.4f} | {full['peak_allocated_gib']:.3f} |")
        else:
            lines.append(f"| {name} | {row['status']} | — | — | — |")
            lines.extend(["", f"{name}: {row['failure']['stage']}: {row['failure']['message']}", ""])
    lines.extend(["", "The two SSD models change state-transition structure; runtime gains do not imply preserved recommendation accuracy.",
        "The same real item/target/negative arrays and fresh backbone are shared. Collation and initialization are excluded from per-update timing.",
        "No full-catalogue validation or inference benchmark was run. See query-ssm2-benchmark.json for audits, raw samples and provenance."])
    (output_dir / "query-ssm2-benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return validate_metrics(metrics)
