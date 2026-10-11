from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def dense_encoder_flops(
    *, batch_size: int, sequence_length: int, d_model: int, n_layers: int, ffn_multiplier: int
) -> int:
    """Nominal dense forward matmul operations, counting one multiply-add as two."""
    values = (batch_size, sequence_length, d_model, n_layers, ffn_multiplier)
    if any(value < 1 for value in values):
        raise ValueError("encoder FLOP dimensions must be positive")
    return batch_size * n_layers * (
        (8 + 4 * ffn_multiplier) * sequence_length * d_model**2
        + 4 * sequence_length**2 * d_model
    )


@torch.no_grad()
def score_catalogue(
    model: nn.Module,
    states: torch.Tensor,
    *,
    k: int = 10,
    item_chunk_size: int = 65_536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k over items 1..N; seen items remain eligible, scoring is FP32."""
    if k < 1 or item_chunk_size < 1:
        raise ValueError("k and item_chunk_size must be positive")
    num_items = int(model.num_items)
    if num_items < 1 or states.ndim != 2:
        raise ValueError("non-empty catalogue and two-dimensional states are required")
    effective_k = min(k, num_items)
    best_scores = torch.empty((len(states), 0), device=states.device, dtype=torch.float32)
    best_ids = torch.empty((len(states), 0), device=states.device, dtype=torch.long)
    # Explicitly disable an outer encoder autocast as well.
    with torch.autocast(device_type=states.device.type, enabled=False):
        for first in range(1, num_items + 1, item_chunk_size):
            last = min(first + item_chunk_size, num_items + 1)
            scores = states.float() @ model.item_embedding.weight[first:last].float().T
            chunk_scores, chunk_indexes = scores.topk(min(effective_k, last - first), dim=1)
            candidate_scores = torch.cat((best_scores, chunk_scores), dim=1)
            candidate_ids = torch.cat((best_ids, chunk_indexes + first), dim=1)
            best_scores, indexes = candidate_scores.topk(min(effective_k, candidate_scores.shape[1]), dim=1)
            best_ids = candidate_ids.gather(1, indexes)
    return best_scores, best_ids


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _history(model: nn.Module, batch: dict[str, Any], device: torch.device, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        if batch["gaps"] is None:
            return model.last_state(batch["ids"], batch["mask"])
        return model.last_state(batch["ids"], batch["mask"], gap_ids=batch["gaps"])


def _prepare_batch(
    contexts: Sequence[np.ndarray], gaps: Sequence[np.ndarray] | None, device: torch.device
) -> dict[str, Any]:
    lengths = [len(context) for context in contexts]
    padded_length = max(lengths)
    ids = np.zeros((len(contexts), padded_length), dtype=np.int64)
    mask = np.zeros_like(ids, dtype=np.bool_)
    gap_array = np.zeros_like(ids) if gaps is not None else None
    for row, context in enumerate(contexts):
        ids[row, : len(context)] = context
        mask[row, : len(context)] = True
        if gap_array is not None:
            gap_array[row, : len(context)] = gaps[row]
    return {
        "ids": torch.from_numpy(ids).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "gaps": None if gap_array is None else torch.from_numpy(gap_array).to(device),
        "lengths": lengths,
        "batch_size": len(contexts),
        "raw_padded_length": padded_length,
    }


def _profile_shapes(model: nn.Module, batch: dict[str, Any], device: torch.device, amp: bool) -> None:
    """Observe actual evaluated shapes outside the timed loop, including packed pairs."""
    observed: dict[str, int] = {"compressor_flops": 0, "projection_rows": 0}

    def encoder_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        observed["encoder_length"] = int(inputs[0].shape[1])
        observed["d_model"] = int(inputs[0].shape[2])

    def linear_hook(module: nn.Linear, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        rows = math.prod(inputs[0].shape[:-1])
        observed["compressor_flops"] += 2 * rows * module.in_features * module.out_features
        observed["projection_rows"] += rows

    handles = [model.encoder.register_forward_pre_hook(encoder_hook)]
    handles.extend(
        module.register_forward_hook(linear_hook)
        for module in model.compressor.modules()
        if isinstance(module, nn.Linear)
    )
    try:
        batch["states"] = _history(model, batch, device, amp)
    finally:
        for handle in handles:
            handle.remove()
    layer = model.encoder.layers[0]
    d_model = observed["d_model"]
    observed["encoder_flops"] = dense_encoder_flops(
        batch_size=batch["batch_size"],
        sequence_length=observed["encoder_length"],
        d_model=d_model,
        n_layers=len(model.encoder.layers),
        ffn_multiplier=layer.linear1.out_features // d_model,
    )
    observed["scoring_flops"] = 2 * batch["batch_size"] * int(model.num_items) * d_model
    batch["nominal"] = observed


def _wall_sample(call: Callable[[], Any], device: torch.device) -> float:
    _synchronize(device)
    started = time.perf_counter()
    call()
    _synchronize(device)
    return (time.perf_counter() - started) * 1_000.0


def _stage_calls(
    model: nn.Module, batches: Sequence[dict[str, Any]], device: torch.device, amp: bool,
    k: int, item_chunk_size: int,
) -> dict[str, list[Callable[[], Any]]]:
    def full(batch: dict[str, Any]) -> Any:
        return score_catalogue(model, _history(model, batch, device, amp), k=k, item_chunk_size=item_chunk_size)

    return {
        "history": [lambda batch=batch: _history(model, batch, device, amp) for batch in batches],
        "scoring": [
            lambda batch=batch: score_catalogue(model, batch["states"], k=k, item_chunk_size=item_chunk_size)
            for batch in batches
        ],
        "full": [lambda batch=batch: full(batch) for batch in batches],
    }


def _warm(calls: Sequence[Callable[[], Any]], count: int, device: torch.device) -> None:
    for iteration in range(count):
        calls[iteration % len(calls)]()
    _synchronize(device)


def _memory_start(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"supported": False}
    memory = {"supported": True, "baseline_allocated_bytes": torch.cuda.memory_allocated(device),
              "baseline_reserved_bytes": torch.cuda.memory_reserved(device)}
    torch.cuda.reset_peak_memory_stats(device)
    return memory


def _memory_end(device: torch.device, memory: dict[str, Any]) -> dict[str, Any]:
    if device.type == "cuda":
        memory.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
        memory["incremental_peak_allocated_bytes"] = max(
            0, memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"])
    return memory


def _benchmark_case(
    model: nn.Module, contexts: Sequence[np.ndarray], gaps: Sequence[np.ndarray] | None,
    *, device: torch.device, amp: bool, k: int, item_chunk_size: int, batch_size: int,
    iterations: int, throughput_iterations: int, warmup: int,
) -> dict[str, Any]:
    b1 = [_prepare_batch([context], None if gaps is None else [gaps[index]], device)
          for index, context in enumerate(contexts)]
    complete_contexts = (len(contexts) // batch_size) * batch_size
    batches = [
        _prepare_batch(contexts[first:first + batch_size],
                       None if gaps is None else gaps[first:first + batch_size], device)
        for first in range(0, complete_contexts, batch_size)
    ]
    # Synthetic cases still measure the requested physical batch, not B=1 throughput.
    if len(contexts) < batch_size:
        indexes = [index % len(contexts) for index in range(batch_size)]
        batches = [_prepare_batch([contexts[index] for index in indexes],
                                  None if gaps is None else [gaps[index] for index in indexes], device)]
    for batch in [*b1, *batches]:
        _profile_shapes(model, batch, device, amp)
    result: dict[str, Any] = {
        "b1_shapes": [_shape_record(batch) for batch in b1],
        "throughput_shapes": [_shape_record(batch) for batch in batches],
        "throughput_requested_batch": batch_size,
        "repeated_inputs_for_full_batch": len(contexts) < batch_size,
        "throughput_dropped_context_count": len(contexts) - complete_contexts if complete_contexts else 0,
        "throughput_partial_batch_policy": "drop trailing incomplete batch; repeat inputs only when N<B",
        "b1": {}, "throughput": {},
    }
    for stage, calls in _stage_calls(model, b1, device, amp, k, item_chunk_size).items():
        _warm(calls, warmup, device)
        memory = _memory_start(device)
        samples = [_wall_sample(calls[index % len(calls)], device) for index in range(iterations)]
        result["b1"][stage] = {
            "samples_ms": samples, "count": len(samples),
            "p50_ms": float(np.percentile(samples, 50)), "p95_ms": float(np.percentile(samples, 95)),
            "memory": _memory_end(device, memory),
        }
    for stage, calls in _stage_calls(model, batches, device, amp, k, item_chunk_size).items():
        _warm(calls, warmup, device)
        memory = _memory_start(device)
        _synchronize(device)
        started = time.perf_counter()
        for index in range(throughput_iterations):
            calls[index % len(calls)]()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        users = sum(batches[index % len(batches)]["batch_size"] for index in range(throughput_iterations))
        result["throughput"][stage] = {
            "batches": throughput_iterations, "users": users, "elapsed_seconds": elapsed,
            "users_per_second": users / elapsed, "wall_ms_per_batch": elapsed * 1_000 / throughput_iterations,
            "memory": _memory_end(device, memory),
        }
    groups: dict[str, Any] = {}
    for name, minimum, maximum in (("le128", 1, 128), ("129to512", 129, 512), ("513to1028", 513, 1028)):
        member_indexes = {index for index, context in enumerate(contexts) if minimum <= len(context) <= maximum}
        stages = {}
        for stage, stats in result["b1"].items():
            samples = [value for index, value in enumerate(stats["samples_ms"])
                       if index % len(contexts) in member_indexes]
            stages[stage] = {"count": len(samples)}
            if samples:
                stages[stage].update({"p50_ms": float(np.percentile(samples, 50)),
                                      "p95_ms": float(np.percentile(samples, 95))})
        groups[name] = {"context_count": len(member_indexes), "stages": stages}
    result["b1_length_groups"] = groups
    return result


def _shape_record(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: batch[key] for key in ("batch_size", "lengths", "raw_padded_length", "nominal")}


def _flatten_case(name: str, case: dict[str, Any], batch_size: int) -> dict[str, int | float | bool]:
    suffix = "" if name == "real" else "_" + name
    metrics: dict[str, int | float | bool] = {}
    for stage, stats in case["b1"].items():
        prefix = f"inference_{stage}_b1{suffix}"
        metrics.update({f"{prefix}_{key}": stats[key] for key in ("p50_ms", "p95_ms", "count")})
    for stage, stats in case["throughput"].items():
        prefix = f"inference_{stage}_b{batch_size}{suffix}"
        metrics.update({f"{prefix}_{key}": stats[key] for key in ("users_per_second", "wall_ms_per_batch")})
    if name == "real":
        for group, record in case["b1_length_groups"].items():
            metrics[f"inference_b1_length_{group}_context_count"] = record["context_count"]
            for stage, stats in record["stages"].items():
                metrics.update({f"inference_{stage}_b1_length_{group}_{key}": value
                                for key, value in stats.items()})
    for label, stages in (("b1", case["b1"]), (f"b{batch_size}", case["throughput"])):
        for stage, stats in stages.items():
            for key, value in stats["memory"].items():
                if key.endswith("_bytes"):
                    metrics[f"inference_{stage}_{label}{suffix}_gpu_{key.removesuffix('_bytes')}_gib"] = value / 2**30
    for label, shapes in (("b1", case["b1_shapes"]), (f"b{batch_size}", case["throughput_shapes"])):
        calls = case["b1"]["history"]["count"] if label == "b1" else case["throughput"]["history"]["batches"]
        evaluated_shapes = [shapes[index % len(shapes)] for index in range(calls)]
        for stage in ("encoder", "compressor", "history", "scoring", "full"):
            values = []
            weights = []
            for shape in evaluated_shapes:
                nominal = shape["nominal"]
                components = {"encoder": nominal["encoder_flops"], "compressor": nominal["compressor_flops"],
                              "scoring": nominal["scoring_flops"]}
                components["history"] = components["encoder"] + components["compressor"]
                components["full"] = components["history"] + components["scoring"]
                values.append(components[stage] / shape["batch_size"])
                weights.append(shape["batch_size"])
            prefix = f"inference_{stage}_{label}{suffix}_dense_matmul_flops"
            metrics[prefix] = float(np.average(values, weights=weights))
            metrics[prefix + "_p95"] = float(np.percentile(np.repeat(values, weights), 95))
    return metrics


def _metadata(device: torch.device, amp: bool, model: nn.Module) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device), "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(), "cpu_threads": torch.get_num_threads(),
        "encoder_amp": bool(amp and device.type == "cuda"), "scoring_dtype": "float32",
        "item_embedding_dtype": str(model.item_embedding.weight.dtype),
        "num_items": int(model.num_items), "multiply_add_flops": 2,
        "flop_method": "analytic dense-equivalent matmuls using observed padded encoder and projection shapes",
        "flop_excludes": ["embedding lookup", "bias/add", "GELU", "softmax", "LayerNorm", "packing", "masking", "top-k"],
        "timer": "perf_counter wall time, CUDA synchronize immediately before and after measured range",
        "latency_includes": ["Python/host orchestration", "GPU compressor packing", "all selected stage calls"],
        "latency_excludes": ["parquet/DataLoader", "CPU collate", "gap construction", "input H2D transfer",
                             "output D2H transfer", "checkpoint loading", "warmup", "shape profiling", "external serving"],
        "kv_cache": False,
        "model": {name: getattr(model, name, None) for name in
                  ("d_model", "n_heads", "n_layers", "ffn_multiplier", "max_seq_len", "compressor_type",
                   "tail_size", "use_time_features", "num_gap_buckets", "compression_depth")},
        "precision": {"matmul_float32_precision": torch.get_float32_matmul_precision(),
                      "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                      "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)},
        "attention_backend_policy": {"flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                                     "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                     "math_enabled": torch.backends.cuda.math_sdp_enabled()},
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    return metadata


def _optional_profiler(
    model: nn.Module, context: np.ndarray, gap: np.ndarray | None, *, device: torch.device,
    amp: bool, k: int, item_chunk_size: int, output_dir: Path,
) -> None:
    batch = _prepare_batch([context], None if gap is None else [gap], device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, with_flops=True) as profiler:
        score_catalogue(model, _history(model, batch, device, amp), k=k, item_chunk_size=item_chunk_size)
        _synchronize(device)
    rows = [{"operator": event.key, "input_shapes": event.input_shapes,
             "reported_flops": int(event.flops or 0), "self_cpu_time_us": float(event.self_cpu_time_total),
             "self_device_time_us": float(getattr(event, "self_device_time_total", 0.0))}
            for event in profiler.key_averages(group_by_input_shape=True)]
    _write_json(output_dir / "benchmark_profiler.json", {
        "operators": rows, "operators_without_reported_flops": [row["operator"] for row in rows if not row["reported_flops"]],
        "warning": "Zero reported FLOPs do not establish zero work; fused attention/encoder counts can be absent.",
        "potentially_uncounted_fused_ops": [row["operator"] for row in rows if not row["reported_flops"]
                                            and any(part in row["operator"] for part in
                                                    ("scaled_dot_product", "transformer", "multi_head_attention"))],
        "attention_manual_method": "4*B*T*T*d per layer in separate analytic count; do not add profiler total to it.",
    })
    profiler.export_chrome_trace(str(output_dir / "benchmark_profiler_trace.json"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    contexts: Sequence[np.ndarray],
    *,
    device: torch.device,
    amp: bool,
    output_dir: Path,
    k: int = 10,
    item_chunk_size: int = 65_536,
    config: dict[str, Any] | None = None,
    gap_ids: Sequence[np.ndarray] | None = None,
) -> dict[str, int | float | bool]:
    """Independent history/scoring/full wall benchmarks, plus nominal forward FLOPs.

    Options: warmup=30, iterations=200, throughput_iterations=20, batch_size=32,
    real_sample_size=200, fixed_lengths=[128,512,1028], seed=42, profile=False.
    smoke_test=True explicitly permits fewer samples/warmups for local contract tests.
    A throughput OOM raises unless allow_oom_fallback=True; fallback batch is recorded
    and must be matched by rerunning every competing model at that same batch.
    """
    options = dict(config or {})
    if not bool(options.get("enabled", True)):
        return {}
    device = torch.device(device)
    if not contexts:
        raise ValueError("benchmark requires non-empty contexts")
    contexts = [np.asarray(context, dtype=np.int64) for context in contexts]
    if any(context.ndim != 1 or not len(context) or len(context) > model.max_seq_len
           or context.min() < 1 or context.max() > model.num_items for context in contexts):
        raise ValueError("benchmark contexts must contain valid unpadded catalogue ids within max_seq_len")
    if gap_ids is not None:
        gap_ids = [np.asarray(gaps, dtype=np.int64) for gaps in gap_ids]
        if len(gap_ids) != len(contexts) or any(gaps.shape != context.shape for gaps, context in zip(gap_ids, contexts)):
            raise ValueError("benchmark gap ids must align with contexts")
        gap_embedding = getattr(model, "gap_embedding", None)
        if any(np.any(gaps < 0) or (gap_embedding is not None and np.any(gaps >= gap_embedding.num_embeddings))
               for gaps in gap_ids):
            raise ValueError("benchmark gap ids are outside the bucket mapping")
    if getattr(model, "gap_embedding", None) is not None and gap_ids is None:
        raise ValueError("time-feature benchmark requires observed gap ids")
    warmup = int(options.get("warmup", 30))
    iterations = int(options.get("iterations", 200))
    throughput_iterations = int(options.get("throughput_iterations", 20))
    batch_size = int(options.get("batch_size", 32))
    real_sample_size = int(options.get("real_sample_size", 200))
    smoke_test = bool(options.get("smoke_test", False))
    if min(iterations, throughput_iterations, batch_size, real_sample_size) < 1 or warmup < 0:
        raise ValueError("benchmark counts must be positive and warmup non-negative")
    if not smoke_test and (iterations < 200 or warmup < 30):
        raise ValueError("regular benchmark requires at least 200 B1 samples and 30 warmup calls")
    if k < 1 or item_chunk_size < 1:
        raise ValueError("k and item_chunk_size must be positive")
    seed = int(options.get("seed", 42))
    indexes = np.sort(np.random.default_rng(seed).choice(len(contexts), min(real_sample_size, len(contexts)), replace=False))
    real_contexts = [contexts[int(index)] for index in indexes]
    real_gaps = None if gap_ids is None else [gap_ids[int(index)] for index in indexes]
    cases: dict[str, tuple[list[np.ndarray], list[np.ndarray] | None]] = {}
    for length in dict.fromkeys(int(length) for length in options.get("fixed_lengths", [128, 512, 1028])):
        if length < 1:
            raise ValueError("fixed benchmark lengths must be positive")
        if length > model.max_seq_len:
            continue
        context = (np.arange(length, dtype=np.int64) % int(model.num_items)) + 1
        gaps = None
        if gap_ids is not None:
            # Use an observed pattern so buckets satisfy the same mapping as real inputs.
            gaps = [np.resize(real_gaps[0], length)]
        cases[f"n{length}"] = ([context], gaps)
    cases["real"] = (real_contexts, real_gaps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    memory: dict[str, Any] = {"supported": device.type == "cuda"}
    if device.type == "cuda":
        _synchronize(device)
        memory.update({"baseline_allocated_bytes": torch.cuda.memory_allocated(device),
                       "baseline_reserved_bytes": torch.cuda.memory_reserved(device)})
    actual_batch = batch_size
    results: dict[str, Any] = {}
    try:
        while True:
            try:
                results = {name: _benchmark_case(
                    model, inputs, gaps, device=device, amp=amp, k=k, item_chunk_size=item_chunk_size,
                    batch_size=actual_batch, iterations=iterations, throughput_iterations=throughput_iterations, warmup=warmup,
                ) for name, (inputs, gaps) in cases.items()}
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or device.type != "cuda":
                    raise
                if not options.get("allow_oom_fallback", False) or actual_batch == 1:
                    raise RuntimeError(
                        f"inference benchmark OOM at batch {actual_batch}; rerun all variants at a common smaller batch"
                    ) from exc
                actual_batch = max(1, actual_batch // 2)
                results.clear()
                torch.cuda.empty_cache()
        if device.type == "cuda":
            measured = [stats["memory"] for result in results.values()
                        for stages in (result["b1"], result["throughput"]) for stats in stages.values()]
            memory.update({"peak_allocated_bytes": max(stage["peak_allocated_bytes"] for stage in measured),
                           "peak_reserved_bytes": max(stage["peak_reserved_bytes"] for stage in measured),
                           "peak_scope": "maximum of measured stage peaks; resets after each stage warmup"})
            memory["incremental_peak_allocated_bytes"] = max(
                0, memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"])
        metrics: dict[str, int | float | bool] = {
            "inference_requested_batch_size": batch_size, "inference_actual_batch_size": actual_batch,
            "inference_requires_common_batch_rerun": actual_batch != batch_size,
            "inference_benchmark_smoke_test": smoke_test, "inference_real_context_count": len(real_contexts),
            "inference_gpu_memory_available": device.type == "cuda",
        }
        for name, result in results.items():
            metrics.update(_flatten_case(name, result, actual_batch))
        for name, value in memory.items():
            if name.endswith("_bytes"):
                metrics["inference_gpu_" + name.removesuffix("_bytes") + "_gib"] = value / 2**30
        metadata = _metadata(device, amp, model)
        metadata.update({"config": options, "seed": seed, "sample_indexes": indexes.tolist(),
                         "revision": options.get("revision", os.environ.get("GITHUB_SHA", "unknown")),
                         "checkpoint": options.get("checkpoint"),
                         "requested_batch_size": batch_size, "actual_batch_size": actual_batch,
                         "warmup": warmup, "iterations": iterations,
                         "throughput_iterations": throughput_iterations, "memory": memory,
                         "smoke_test": smoke_test,
                         "fixed_inputs": "deterministic cyclic valid item ids; synthetic gaps repeat first sampled observed pattern"})
        _write_json(output_dir / "benchmark_samples.json", results)
        _write_json(output_dir / "benchmark.json", {"metadata": metadata, "metrics": metrics})
        if bool(options.get("profile", False)):
            _optional_profiler(model, real_contexts[0], None if real_gaps is None else real_gaps[0],
                               device=device, amp=amp, k=k, item_chunk_size=item_chunk_size, output_dir=output_dir)
        return metrics
    finally:
        model.train(was_training)
