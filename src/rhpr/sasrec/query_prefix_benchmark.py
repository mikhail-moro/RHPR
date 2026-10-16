"""Uncached serving cost for the full-prefix SASRec query pilot.

Latency executes the trained compressor, SASRec and exact catalogue top-k.
Arithmetic counts are transparent dense-matmul estimates, not profiler claims
about instructions executed by fused attention or a parallel selective scan.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from rhpr.sasrec.benchmark import dense_encoder_flops, score_catalogue
from rhpr.sasrec.long_quality_benchmark import _batch, _encode, _synchronize, _time_call
from rhpr.sasrec.model import TailOnlySequenceCompressor


def _serving_contexts(cohort: Any, size: int, *, recent: bool, tail_size: int,
                      max_seq_len: int) -> tuple[list[np.ndarray], list[int]]:
    """Length-stratify complete observed train prefixes, without reading targets."""
    users = sorted(cohort.users, key=lambda user: (len(user.train), user.user_id))
    if not users:
        raise ValueError("serving measurements require a nonempty cohort")
    if any(len(user.train) < 1 or len(user.train) > max_seq_len for user in users):
        raise ValueError("complete serving histories must fit the safety cap; cropping is forbidden")
    indexes = ([len(users) // 2] if size == 1 else
               np.linspace(0, len(users) - 1, num=size, dtype=np.int64).tolist())
    selected = [users[index] for index in indexes]
    lengths = [len(user.train) for user in selected]
    contexts = [np.asarray(user.train[-tail_size:] if recent else user.train) for user in selected]
    return contexts, lengths


def _observe_prefix_flops(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor],
                          device: torch.device, amp: bool) -> tuple[torch.Tensor, dict[str, int]]:
    """Observe compressor projection shapes and estimate dense SASRec matmuls."""
    observed: dict[str, int] = {"compressor_flops": 0}

    def encoder_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        observed["encoder_length"] = int(inputs[0].shape[1])
        observed["d_model"] = int(inputs[0].shape[2])

    def compressor_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        observed["encoder_valid_tokens"] = int(output[1].sum().item())

    def linear_hook(module: nn.Linear, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        observed["compressor_flops"] += (
            2 * math.prod(inputs[0].shape[:-1]) * module.in_features * module.out_features)

    def attention_hook(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> None:
        query, key, _ = inputs[:3]
        if not module.batch_first or module.kdim != module.embed_dim or module.vdim != module.embed_dim:
            raise ValueError("FLOP accounting requires batch-first equal-width attention")
        size, queries, dimensions = query.shape
        memory = key.shape[1]
        # Q/output projections at query positions, K/V at memory positions;
        # two attention matmuls. Masked and padded positions count as dense work.
        observed["compressor_flops"] += 4 * size * (queries + memory) * dimensions**2
        observed["compressor_flops"] += 4 * size * queries * memory * dimensions

    attentions = [module for module in model.compressor.modules() if isinstance(module, nn.MultiheadAttention)]
    attention_children = {id(child) for module in attentions for child in module.modules()}
    handles = [model.encoder.register_forward_pre_hook(encoder_hook),
               model.compressor.register_forward_hook(compressor_hook)]
    handles.extend(module.register_forward_hook(linear_hook) for module in model.compressor.modules()
                   if isinstance(module, nn.Linear) and id(module) not in attention_children)
    handles.extend(module.register_forward_pre_hook(attention_hook) for module in attentions)
    try:
        states = _encode(model, batch, device, amp)
    finally:
        for handle in handles:
            handle.remove()
    if not bool(torch.isfinite(states).all()):
        raise FloatingPointError("serving benchmark produced nonfinite final states")
    # The native delta projection uses F.linear and bypasses Linear hooks.
    scan_cost = getattr(model.compressor, "scan_compute_flops", None)
    if callable(scan_cost):
        old_width = max(1, int(batch[1].sum(dim=1).max().item()) - int(model.compressor.tail_size))
        scan = scan_cost(old_width, batch_size=len(batch[0]))
        observed["compressor_flops"] += int(scan["unhooked_delta_projection_flops"])
        observed.update({key: int(scan[key]) for key in
                         ("scan_arithmetic_flops", "scan_exp_evaluations", "scan_softplus_evaluations")})
    else:
        observed.update(scan_arithmetic_flops=0, scan_exp_evaluations=0, scan_softplus_evaluations=0)
    first = model.encoder.layers[0]
    observed["encoder_flops"] = dense_encoder_flops(
        batch_size=len(batch[0]), sequence_length=observed["encoder_length"],
        d_model=observed["d_model"], n_layers=len(model.encoder.layers),
        ffn_multiplier=first.linear1.out_features // observed["d_model"])
    observed["scoring_flops"] = 2 * len(batch[0]) * int(model.num_items) * observed["d_model"]
    observed["history_flops"] = observed["compressor_flops"] + observed["encoder_flops"]
    observed["full_flops"] = observed["history_flops"] + observed["scoring_flops"]
    for component in ("compressor", "history", "full"):
        observed[component + "_arithmetic_including_scan_flops"] = (
            observed[component + "_flops"] + observed["scan_arithmetic_flops"])
    return states, observed


@torch.inference_mode()
def benchmark_prefix_quality(model: nn.Module, cohort: Any, *, config: dict[str, Any],
                              device: torch.device) -> dict[str, Any]:
    """Measure B1 and registered parallel batches; compression is always included.

    CPU collation and H2D are excluded here and included separately in the real
    training/evaluation throughput preflight. An unsupported batch is recorded;
    at least one parallel case must succeed without changing history length.
    """
    bench, train = config["benchmark"], config["training"]
    sizes = bench["batch_sizes"]
    if (not sizes or len(set(sizes)) != len(sizes) or 1 not in sizes
            or not any(type(size) is int and size > 1 for size in sizes)
            or any(type(size) is not int or size < 1 for size in sizes)):
        raise ValueError("serving benchmark requires unique B1 and positive parallel batch sizes")
    if (type(bench["warmup"]) is not int or bench["warmup"] < 0
            or type(bench["iterations"]) is not int or bench["iterations"] < 1):
        raise ValueError("warmup must be nonnegative and iterations positive")
    if getattr(model, "architecture", "sasrec") != "sasrec":
        raise ValueError("prefix benchmark requires the SASRec backbone")
    device = torch.device(device)
    amp = bool(train["amp"] and device.type == "cuda")
    recent = isinstance(model.compressor, TailOnlySequenceCompressor)
    tail_size = int(getattr(model.compressor, "tail_size", train["raw_tail_size"]))
    result: dict[str, Any] = {"metrics": {}, "cases": {}, "metadata": {
        "architecture": "sasrec", "cache": False,
        "context_selection": "B1 median; parallel batches evenly length-stratified across complete train-only histories",
        "warmup_per_stage": bench["warmup"], "iterations_per_stage": bench["iterations"],
        "timer": "synchronized perf_counter wall and CUDA events on current stream",
        "cuda_event_scope": "elapsed stream interval, including host-induced gaps between kernels",
        "multiply_add_flops": 2,
        "flop_method": "nominal dense SASRec matmuls and observed compressor Linear/MHA shapes; unhooked native delta projection added explicitly",
        "scan_cost_scope": "analytical sequential recurrence/readout estimate; parallel CUDA scan may execute extra work; exp and softplus are separate operation counts",
        "flop_excludes": ["embedding lookup", "packing", "masking", "reductions", "position sin/cos",
                          "bias/add", "softmax", "GELU", "SiLU", "gating", "LayerNorm", "top-k"],
        "latency_includes": ["uncached compression", "SASRec encoding", "full-catalogue scoring and top-k in full stage"],
        "latency_excludes": ["CPU collation", "input H2D", "output D2H", "checkpoint loading", "external serving"],
        "scoring_dtype": "float32", "k": 100, "item_chunk_size": bench["item_chunk_size"],
        "encoder_amp": amp, "retained_tail_tokens": tail_size,
        "summary_token_budget": int(getattr(model.compressor, "summary_tokens", 0)),
        "oom_policy": "record requested batch as unsupported; require at least one successful parallel case",
    }}
    was_training = model.training
    model.eval()
    try:
        for size in sizes:
            label = f"b{size}"
            contexts, complete_lengths = _serving_contexts(cohort, size, recent=recent,
                tail_size=tail_size, max_seq_len=model.max_seq_len)
            case: dict[str, Any] = {"batch_size": size, "complete_observed_lengths": complete_lengths,
                "input_lengths": [len(context) for context in contexts],
                "repeated_inputs": len(cohort.users) < size, "status": "ok", "stages": {}}
            batch = states = calls = call = None
            try:
                batch = _batch(contexts, model, device)
                states, flops = _observe_prefix_flops(model, batch, device, amp)
                case.update(raw_padded_length=int(batch[0].shape[1]), estimated_flops=flops)
                calls = {
                    "history": lambda: _encode(model, batch, device, amp),
                    "scoring": lambda: score_catalogue(model, states, k=100, item_chunk_size=bench["item_chunk_size"]),
                    "full": lambda: score_catalogue(model, _encode(model, batch, device, amp),
                        k=100, item_chunk_size=bench["item_chunk_size"]),
                }
                for stage, call in calls.items():
                    for _ in range(bench["warmup"]):
                        call()
                    _synchronize(device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    samples = [_time_call(call, device) for _ in range(bench["iterations"])]
                    wall = np.asarray([sample[0] for sample in samples], dtype=np.float64)
                    if not np.isfinite(wall).all() or np.any(wall <= 0):
                        raise FloatingPointError("serving timing samples must be finite and positive")
                    record = {"wall_samples_ms": wall.tolist(), "wall_mean_ms": float(wall.mean()),
                        "wall_p50_ms": float(np.quantile(wall, .5)), "wall_p95_ms": float(np.quantile(wall, .95)),
                        "users_per_second": size * 1000 / float(wall.mean())}
                    if device.type == "cuda":
                        gpu = np.asarray([sample[1] for sample in samples], dtype=np.float64)
                        record.update(gpu_samples_ms=gpu.tolist(), gpu_mean_ms=float(gpu.mean()),
                            gpu_p50_ms=float(np.quantile(gpu, .5)), gpu_p95_ms=float(np.quantile(gpu, .95)),
                            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
                    case["stages"][stage] = record
                # Publish scalar metrics only after every stage of this exact batch succeeds.
                for stage, record in case["stages"].items():
                    result["metrics"].update({f"{stage}_{label}_{key}": value for key, value in record.items()
                                               if not key.endswith("samples_ms")})
                for component in ("compressor", "encoder", "scoring", "history", "full"):
                    result["metrics"][f"{component}_{label}_estimated_matmul_flops_per_user"] = flops[component + "_flops"] / size
                for component in ("compressor", "history", "full"):
                    result["metrics"][f"{component}_{label}_estimated_arithmetic_including_scan_flops_per_user"] = flops[component + "_arithmetic_including_scan_flops"] / size
                for key in ("scan_arithmetic_flops", "scan_exp_evaluations", "scan_softplus_evaluations"):
                    result["metrics"][f"{label}_{key}_per_user"] = flops[key] / size
            except torch.OutOfMemoryError:
                case["status"] = "oom"
                case["stages"] = {}
            finally:
                batch = states = calls = call = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            result["cases"][label] = case
    finally:
        model.train(was_training)
    if result["cases"]["b1"]["status"] != "ok":
        raise RuntimeError("B1 serving case failed without reducing the full history")
    if not any(case["status"] == "ok" and case["batch_size"] > 1 for case in result["cases"].values()):
        raise RuntimeError("no parallel serving case fits the complete history")
    return result
