"""Bounded full-catalogue evaluation and uncached serving measurements.

An observed unknown item has ID zero but still occupies a chronological slot.
Only padding is invalid. Targets unknown to the train-only catalogue receive
zero credit in the primary all-anchor metrics, rather than being dropped.
"""
from __future__ import annotations

from itertools import islice
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from rhpr.sasrec.benchmark import hstu_encoder_flops, score_catalogue


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _context_array(context: np.ndarray, model: nn.Module) -> np.ndarray:
    result = np.asarray(context)
    if (result.ndim != 1 or not result.size or not np.issubdtype(result.dtype, np.integer)
            or result.size > model.max_seq_len or result.min() < 0 or result.max() > model.num_items):
        raise ValueError("contexts must contain occupied integer IDs 0..num_items within max_seq_len")
    return result


def _batch(contexts: Sequence[np.ndarray], model: nn.Module,
           device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = [_context_array(context, model) for context in contexts]
    lengths = np.asarray([len(context) for context in contexts], dtype=np.int64)
    ids = np.zeros((len(contexts), int(lengths.max())), dtype=np.int64)
    for row, context in enumerate(contexts):
        ids[row, :len(context)] = context
    mask = np.arange(ids.shape[1])[None, :] < lengths[:, None]
    return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)


def _encode(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor],
            device: torch.device, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=bool(amp and device.type == "cuda")):
        return model.last_state(*batch)


@torch.inference_mode()
def evaluate_long_quality(model: nn.Module, examples: Iterable[Any], *,
                          device: torch.device, amp: bool = False, batch_size: int = 32,
                          k: int = 10, item_chunk_size: int = 65_536) -> dict[str, Any]:
    """One final-state prediction per anchor; exact top-k over items 1..N.

    ``examples`` supplies ``user_id``, ``context`` and ``target`` attributes.
    It may be a lazy iterator. No target timestamp or future event is read.
    Seen items remain eligible. Returned per-user records contain no item IDs.
    """
    for value, name in ((batch_size, "batch_size"), (k, "k"), (item_chunk_size, "item_chunk_size")):
        _positive_integer(value, name)
    device = torch.device(device)
    iterator = iter(examples)
    was_training = model.training
    records: list[dict[str, Any]] = []
    user_ids: set[int] = set()
    model.eval()
    started = time.perf_counter()
    try:
        while block := list(islice(iterator, batch_size)):
            targets = []
            for example in block:
                if (isinstance(example.user_id, (bool, np.bool_))
                        or not isinstance(example.user_id, (int, np.integer))
                        or isinstance(example.target, (bool, np.bool_))
                        or not isinstance(example.target, (int, np.integer))):
                    raise ValueError("evaluation user IDs and targets must be integers")
                user_id, target = int(example.user_id), int(example.target)
                if user_id in user_ids or user_id < 0:
                    raise ValueError("evaluation anchors must have distinct nonnegative user IDs")
                if not 0 <= target <= model.num_items:
                    raise ValueError("evaluation targets must lie in 0..num_items")
                user_ids.add(user_id)
                targets.append(target)
            states = _encode(model, _batch([example.context for example in block], model, device), device, amp)
            if not bool(torch.isfinite(states).all()):
                raise RuntimeError("evaluation produced nonfinite final states")
            scores, top_ids = score_catalogue(model, states, k=k, item_chunk_size=item_chunk_size)
            if not bool(torch.isfinite(scores).all()):
                raise RuntimeError("evaluation produced nonfinite catalogue scores")
            matches = top_ids == torch.tensor(targets, device=device)[:, None]
            # Zero is never a catalogue candidate, so unknown targets cannot hit.
            hit = matches.any(dim=1)
            ranks = torch.where(hit, matches.long().argmax(dim=1) + 1, 0).cpu().tolist()
            for example, target, rank in zip(block, targets, ranks):
                records.append({"user_id": int(example.user_id), "target_known": target > 0,
                                "rank": int(rank), f"hr{k}": float(rank > 0),
                                f"ndcg{k}": 1.0 / math.log2(rank + 1) if rank else 0.0})
    finally:
        model.train(was_training)
    if not records:
        raise ValueError("evaluation requires at least one anchor")
    warm = [record for record in records if record["target_known"]]
    metrics: dict[str, int | float] = {
        "anchor_count": len(records), "warm_anchor_count": len(warm),
        "oov_target_count": len(records) - len(warm), "num_items": int(model.num_items),
        f"hr{k}": float(np.mean([record[f"hr{k}"] for record in records])),
        f"ndcg{k}": float(np.mean([record[f"ndcg{k}"] for record in records])),
        f"warm_hr{k}": float(np.mean([record[f"hr{k}"] for record in warm])) if warm else 0.0,
        f"warm_ndcg{k}": float(np.mean([record[f"ndcg{k}"] for record in warm])) if warm else 0.0,
        "evaluation_wall_seconds": time.perf_counter() - started,
    }
    return {"metrics": metrics, "per_user": records, "metadata": {
        "prediction": "single final state per chronological anchor",
        "catalogue": "all train-only items 1..num_items; no seen-item filtering",
        "primary_denominator": "all anchors, including unsupported targets with zero credit",
        "warm_denominator": "anchors whose targets belong to the train-only catalogue",
        "zero_warm_count_policy": "warm metrics set to zero; warm_anchor_count records absent support",
        "unknown_context": "zero item embedding in an observed valid chronological slot",
        "batch_size": batch_size, "item_chunk_size": item_chunk_size, "k": k,
        "encoder_amp": bool(amp and device.type == "cuda"), "scoring_dtype": "float32",
    }}


def paired_ndcg_bootstrap(reference_records: Sequence[dict[str, Any]],
                          candidate_records: Sequence[dict[str, Any]], *, seed: int = 42,
                          draws: int = 1000, confidence: float = 0.95,
                          k: int = 10) -> dict[str, int | float]:
    """Paired percentile interval over the same users; never drop unmatched users.

    This quantifies user-sampling uncertainty for one seed, not training-seed
    uncertainty or a correction for selecting among multiple configurations.
    """
    _positive_integer(draws, "draws")
    _positive_integer(k, "k")
    if not 0 < confidence < 1 or type(seed) is not int or seed < 0:
        raise ValueError("confidence must lie in (0,1) and seed must be a nonnegative integer")

    def indexed(records: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        values = {int(record["user_id"]): record for record in records}
        if not values or len(values) != len(records):
            raise ValueError("paired records must be nonempty and have distinct users")
        return values

    reference, candidate = indexed(reference_records), indexed(candidate_records)
    if reference.keys() != candidate.keys():
        raise ValueError("paired records must contain exactly the same users")
    users = sorted(reference)
    if any(reference[user]["target_known"] != candidate[user]["target_known"] for user in users):
        raise ValueError("paired target catalogue support must agree")
    baseline = np.asarray([reference[user][f"ndcg{k}"] for user in users], dtype=np.float64)
    comparison = np.asarray([candidate[user][f"ndcg{k}"] for user in users], dtype=np.float64)
    if (not np.isfinite(baseline).all() or not np.isfinite(comparison).all()
            or np.any((baseline < 0) | (baseline > 1)) or np.any((comparison < 0) | (comparison > 1))):
        raise ValueError("paired NDCG values must be finite and lie in [0,1]")
    differences = comparison - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for first in range(0, draws, 64):
        last = min(first + 64, draws)
        indexes = rng.integers(0, len(users), size=(last - first, len(users)))
        samples[first:last] = differences[indexes].mean(axis=1)
    alpha = (1 - confidence) / 2
    result: dict[str, int | float] = {
        "matched_users": len(users), "draws": draws, "seed": seed, "confidence": confidence,
        "reference_ndcg": float(baseline.mean()), "candidate_ndcg": float(comparison.mean()),
        "delta_ndcg": float(differences.mean()), "ci_lower": float(np.quantile(samples, alpha)),
        "ci_upper": float(np.quantile(samples, 1 - alpha)),
        "bootstrap_fraction_delta_le_zero": float(np.mean(samples <= 0)),
    }
    if baseline.mean() > 0:
        result["relative_delta_percent"] = float(differences.mean() / baseline.mean() * 100)
    return result


def _observe_flops(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor],
                   device: torch.device, amp: bool) -> tuple[torch.Tensor, dict[str, int]]:
    """Observe actual GEMM shapes, including reused hierarchy and MHA calls."""
    observed = {"compressor_flops": 0}

    def encoder_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        embeddings, mask = inputs
        observed["encoder_length"] = int(embeddings.shape[1])
        observed["encoder_valid_tokens"] = int(mask.sum().item())
        observed["d_model"] = int(embeddings.shape[2])

    def linear_hook(module: nn.Linear, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        observed["compressor_flops"] += 2 * math.prod(inputs[0].shape[:-1]) * module.in_features * module.out_features

    def attention_hook(module: nn.MultiheadAttention, inputs: tuple[Any, ...]) -> None:
        query, key, _ = inputs[:3]
        if not module.batch_first or module.kdim != module.embed_dim or module.vdim != module.embed_dim:
            raise ValueError("compressor FLOP accounting requires batch-first equal-width attention")
        batch_size, queries, dimensions = query.shape
        memory = key.shape[1]
        # Q and output projections at Q positions; K/V at memory positions.
        observed["compressor_flops"] += 4 * batch_size * (queries + memory) * dimensions**2
        observed["compressor_flops"] += 4 * batch_size * queries * memory * dimensions

    attention_modules = [module for module in model.compressor.modules() if isinstance(module, nn.MultiheadAttention)]
    attention_children = {id(child) for module in attention_modules for child in module.modules()}
    handles = [model.encoder.register_forward_pre_hook(encoder_hook)]
    handles.extend(module.register_forward_hook(linear_hook) for module in model.compressor.modules()
                   if isinstance(module, nn.Linear) and id(module) not in attention_children)
    handles.extend(module.register_forward_pre_hook(attention_hook) for module in attention_modules)
    try:
        states = _encode(model, batch, device, amp)
    finally:
        for handle in handles:
            handle.remove()
    observed["encoder_flops"] = hstu_encoder_flops(
        batch_size=len(batch[0]), sequence_length=observed["encoder_length"],
        valid_tokens=observed["encoder_valid_tokens"], layers=model.encoder.layers)
    observed["scoring_flops"] = 2 * len(batch[0]) * int(model.num_items) * observed["d_model"]
    observed["history_flops"] = observed["compressor_flops"] + observed["encoder_flops"]
    observed["full_flops"] = observed["history_flops"] + observed["scoring_flops"]
    return states, observed


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(operation: Any, device: torch.device) -> tuple[float, float | None]:
    _synchronize(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    if start_event is not None:
        start_event.record()
    operation()
    if end_event is not None:
        end_event.record()
    _synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000
    gpu_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else None
    return wall_ms, gpu_ms


@torch.inference_mode()
def benchmark_long_quality(model: nn.Module, contexts: Sequence[np.ndarray], *,
                           device: torch.device, amp: bool = False, batch_size: int = 32,
                           warmup: int = 5, iterations: int = 10, k: int = 10,
                           item_chunk_size: int = 65_536) -> dict[str, Any]:
    """B1 and fixed-batch uncached latency; compression is included in history.

    GPU event and synchronized wall samples are separate. CPU collation and
    input transfer are performed outside the measured ranges. Stage timing
    excludes shape observation and warmup. No OOM fallback changes the batch.
    """
    for value, name in ((batch_size, "batch_size"), (iterations, "iterations"),
                        (k, "k"), (item_chunk_size, "item_chunk_size")):
        _positive_integer(value, name)
    if type(warmup) is not int or warmup < 0 or not len(contexts):
        raise ValueError("warmup must be nonnegative and contexts must be nonempty")
    if getattr(model, "architecture", None) != "hstu":
        raise ValueError("this pilot benchmark requires the native HSTU adapter")
    device = torch.device(device)
    was_training = model.training
    model.eval()
    result: dict[str, Any] = {"metrics": {}, "cases": {}, "metadata": {
        "warmup_per_stage": warmup, "iterations_per_stage": iterations,
        "timer": "synchronized perf_counter wall and CUDA events on current stream",
        "cuda_event_scope": "elapsed stream interval, including host-induced gaps between kernels",
        "multiply_add_flops": 2,
        "flop_method": "estimated matmuls from observed executed Linear/MHA shapes and native HSTU dimensions",
        "flop_excludes": ["embedding lookup", "packing", "masking", "reductions", "position sin/cos",
                          "bias/add", "softmax", "GELU", "SiLU", "gating", "LayerNorm", "top-k"],
        "latency_includes": ["compressor", "HSTU encoding", "full-catalogue scoring and top-k in full stage"],
        "latency_excludes": ["CPU collation", "input H2D", "output D2H", "checkpoint loading", "external serving"],
        "cache": False, "scoring_dtype": "float32", "k": k, "item_chunk_size": item_chunk_size,
        "encoder_amp": bool(amp and device.type == "cuda"),
        "retained_tail_tokens": int(getattr(model.compressor, "tail_size", model.tail_size)),
        "summary_token_budget": int(getattr(model.compressor, "summary_tokens", 0)),
        "hstu_normalization_length": int(model.hstu_normalization_length),
        "requested_batch_size": batch_size,
    }}
    try:
        for label, size in (("b1", 1), (f"b{batch_size}", batch_size)):
            if label in result["cases"]:
                continue
            indexes = [index % len(contexts) for index in range(size)]
            selected = [contexts[index] for index in indexes]
            batch = _batch(selected, model, device)
            states, flops = _observe_flops(model, batch, device, amp)
            calls = {
                "history": lambda: _encode(model, batch, device, amp),
                "scoring": lambda: score_catalogue(model, states, k=k, item_chunk_size=item_chunk_size),
                "full": lambda: score_catalogue(model, _encode(model, batch, device, amp),
                                                k=k, item_chunk_size=item_chunk_size),
            }
            case: dict[str, Any] = {"batch_size": size, "input_lengths": [len(context) for context in selected],
                                    "raw_padded_length": int(batch[0].shape[1]), "estimated_flops": flops,
                                    "repeated_inputs": len(contexts) < size, "stages": {}}
            for stage, call in calls.items():
                for _ in range(warmup):
                    call()
                _synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                samples = [_time_call(call, device) for _ in range(iterations)]
                wall = np.asarray([sample[0] for sample in samples], dtype=np.float64)
                record: dict[str, Any] = {"wall_samples_ms": wall.tolist(),
                                          "wall_mean_ms": float(wall.mean()),
                                          "wall_p50_ms": float(np.quantile(wall, .5)),
                                          "wall_p95_ms": float(np.quantile(wall, .95)),
                                          "users_per_second": size * 1000 / float(wall.mean())}
                if device.type == "cuda":
                    gpu = np.asarray([sample[1] for sample in samples], dtype=np.float64)
                    record.update({"gpu_samples_ms": gpu.tolist(), "gpu_mean_ms": float(gpu.mean()),
                                   "gpu_p50_ms": float(np.quantile(gpu, .5)),
                                   "gpu_p95_ms": float(np.quantile(gpu, .95)),
                                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30})
                case["stages"][stage] = record
                result["metrics"].update({f"{stage}_{label}_{key}": value for key, value in record.items()
                                          if not key.endswith("samples_ms")})
            for component in ("compressor", "encoder", "scoring", "history", "full"):
                result["metrics"][f"{component}_{label}_estimated_matmul_flops_per_user"] = flops[component + "_flops"] / size
            result["cases"][label] = case
    finally:
        model.train(was_training)
    return result
