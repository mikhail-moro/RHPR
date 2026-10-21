"""Real-data training cost measurements; no HPO or recommendation-quality claims."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.benchmark import score_catalogue, dense_encoder_flops, hstu_encoder_flops
from rhpr.sasrec.long_history_data import LongHistoryCohort, load_prepared_cohort
from rhpr.sasrec.losses import next_item_loss


ARCHITECTURES = ("sasrec", "hstu", "mamba4rec", "fuxi_linear")


@dataclass(frozen=True)
class ProfileWindow:
    user_index: int
    user_id: int
    end: int  # Exclusive input boundary; the final target is train[end].
    negative_seed: int


def validate_long_profile_config(config: dict[str, Any]) -> None:
    if config.get("kind") != "yambda_long_training_profile":
        raise ValueError("kind must be yambda_long_training_profile")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be nonempty")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    data, profile, model = config.get("data", {}), config.get("profile", {}), config.get("model", {})
    if not isinstance(data.get("prepared_path"), str) or not data["prepared_path"]:
        raise ValueError("data.prepared_path is required")
    for key in ("raw_context_length", "tail_size", "effective_batch_size", "update_repetitions", "num_negatives"):
        if type(profile.get(key)) is not int or profile[key] < 1:
            raise ValueError(f"profile.{key} must be a positive integer")
    if profile["tail_size"] > profile["raw_context_length"]:
        raise ValueError("tail_size exceeds raw_context_length")
    for key in ("warmup_microbatches", "inference_warmup", "inference_iterations"):
        if type(profile.get(key)) is not int or profile[key] < 0:
            raise ValueError(f"profile.{key} must be a nonnegative integer")
    if not profile["inference_iterations"]:
        raise ValueError("inference_iterations must be positive")
    batches = profile.get("physical_batch_sizes")
    if (not isinstance(batches, list) or not batches or len(set(batches)) != len(batches)
            or any(type(b) is not int or b < 1 or profile["effective_batch_size"] % b for b in batches)):
        raise ValueError("physical batch sizes must be unique divisors of effective_batch_size")
    if config.get("architectures", list(ARCHITECTURES)) != list(ARCHITECTURES):
        raise ValueError("all four requested architectures must be profiled in the fixed order")
    for key in ("d_model", "n_layers", "n_heads", "ffn_multiplier"):
        if type(model.get(key)) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be positive")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if type(model.get("use_time_features", False)) is not bool or model.get("use_time_features", False):
        raise ValueError("the timing screen requires Time OFF for all models")
    for key in ("learning_rate", "grad_clip_norm", "initial_loss_scale"):
        v = profile.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"profile.{key} must be positive and finite")
    if type(profile.get("amp")) is not bool or profile.get("device") != "cuda":
        raise ValueError("the published profile requires explicit CUDA and boolean AMP")


def window_schedule(cohort: LongHistoryCohort, *, seed: int, count: int,
                    raw_context_length: int) -> tuple[ProfileWindow, ...]:
    """Independent model RNG; deterministic full contexts and identity-bound negatives."""
    if min(count, raw_context_length) < 1 or not cohort.users:
        raise ValueError("nonempty cohort, positive context and count are required")
    rng = np.random.default_rng(seed)
    eligible = [i for i, u in enumerate(cohort.users) if len(u.train) > raw_context_length]
    if not eligible:
        raise ValueError("no user has a complete shifted window")
    order = rng.permutation(eligible)
    windows = []
    for i in range(count):
        index = int(order[i % len(order)])
        user = cohort.users[index]
        end = int(rng.integers(raw_context_length, len(user.train)))
        digest = hashlib.sha256(f"{seed}:{i}:{user.user_id}:{end}".encode()).digest()
        windows.append(ProfileWindow(index, user.user_id, end, int.from_bytes(digest[:8], "little")))
    return tuple(windows)


def profile_batch(cohort: LongHistoryCohort, windows: tuple[ProfileWindow, ...], *,
                  context_length: int, num_negatives: int) -> dict[str, torch.Tensor]:
    if not windows or min(context_length, num_negatives) < 1:
        raise ValueError("nonempty windows and positive dimensions are required")
    inputs, targets, negatives = [], [], []
    for window in windows:
        user = cohort.users[window.user_index]
        if user.user_id != window.user_id or not context_length <= window.end < len(user.train):
            raise ValueError("window is outside its train-only history")
        inputs.append(user.train[window.end - context_length:window.end])
        targets.append(user.train[window.end - context_length + 1:window.end + 1])
        negatives.append(np.random.default_rng(window.negative_seed).integers(
            1, cohort.num_items + 1, num_negatives, dtype=np.int64))
    item_ids = torch.from_numpy(np.asarray(inputs, dtype=np.int64))
    return {"item_ids": item_ids, "valid_mask": torch.ones_like(item_ids, dtype=torch.bool),
            "targets": torch.from_numpy(np.asarray(targets, dtype=np.int64)),
            "negative_ids": torch.from_numpy(np.asarray(negatives, dtype=np.int64))}


def build_profile_model(architecture: str, num_items: int, context_length: int,
                        model_config: dict[str, Any]) -> torch.nn.Module:
    kwargs = {key: model_config[key] for key in ("d_model", "n_heads", "n_layers", "ffn_multiplier", "dropout")}
    kwargs.update(num_items=num_items, max_seq_len=context_length, use_time_features=False)
    if architecture == "sasrec":
        from rhpr.sasrec.model import SASRecModel
        return SASRecModel(**kwargs)
    if architecture == "hstu":
        from rhpr.sasrec.hstu import HSTUModel
        return HSTUModel(**kwargs)
    if architecture == "mamba4rec":
        from rhpr.sasrec.mamba import Mamba4RecModel
        return Mamba4RecModel(**kwargs)
    if architecture == "fuxi_linear":
        from rhpr.sasrec.fuxi import FuXiLinearModel
        return FuXiLinearModel(**kwargs, chunk_size=128)
    raise ValueError("unsupported profile architecture")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict[str, int | float]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("timing measurements must be nonempty and finite")
    return {"samples": len(a), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "min": float(a.min()), "max": float(a.max())}


def _timed(operation: Any, device: torch.device) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize(device)
    return value, (time.perf_counter() - start) * 1000


def _loss(model: torch.nn.Module, encoded: tuple[torch.Tensor, ...],
          batch: dict[str, torch.Tensor], tail_size: int, num_negatives: int) -> torch.Tensor:
    states, mask, sources = encoded
    if type(tail_size) is not int or tail_size < 1:
        raise ValueError("tail_size must be a positive integer")
    # Sequence-preserving backbones right-pad variable histories, whereas the
    # compressors may place summaries before a common raw tail. Select the last
    # occupied *slots per row*, not the last batch columns or source positions:
    # summary endpoints can overlap scored sources without being scored states.
    # Gather before casting so full-history FP32 state copies are never needed.
    width = min(tail_size, states.shape[1])
    columns = torch.arange(states.shape[1], device=states.device).expand_as(mask)
    selected = columns.masked_fill(~mask, -1).topk(width, dim=1).values.flip(1)
    occupied = selected >= 0
    selected = selected.clamp(min=0)
    states = states.gather(1, selected.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
    mask = mask.gather(1, selected) & occupied
    sources = sources.gather(1, selected).masked_fill(~mask, -1)
    return next_item_loss(name="sampled_softmax", model=model, encoded=states.float(),
        valid_mask=mask, source_positions=sources, targets=batch["targets"],
        original_lengths=batch["valid_mask"].sum(dim=1), loss_tail_size=tail_size,
        num_negatives=num_negatives, negative_ids=batch["negative_ids"])


def _train_shape(model: torch.nn.Module, cohort: LongHistoryCohort,
                 schedule: tuple[ProfileWindow, ...], profile: dict[str, Any],
                 context_length: int, physical_batch: int, device: torch.device,
                 optimizer: torch.optim.Optimizer, scaler: Any) -> dict[str, Any]:
    effective = profile["effective_batch_size"]
    accumulation = effective // physical_batch
    amp = profile["amp"]
    model.train()

    def make_batch(windows: tuple[ProfileWindow, ...]) -> dict[str, torch.Tensor]:
        batch = profile_batch(cohort, windows, context_length=context_length,
                              num_negatives=profile["num_negatives"])
        return {key: value.to(device) for key, value in batch.items()}

    def encode(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            return model.encode(batch["item_ids"], batch["valid_mask"])

    # Allocate real Adam states before resetting the peak. Each warmup performs
    # a physical-batch update and is excluded from effective-128 measurements.
    batch = make_batch(schedule[:physical_batch])
    optimizer.zero_grad(set_to_none=True)
    for _ in range(max(1, profile["warmup_microbatches"])):
        loss = _loss(model, encode(batch), batch, profile["tail_size"], profile["num_negatives"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite warmup loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), profile["grad_clip_norm"])
        if not torch.isfinite(norm):
            # AMP will skip and reduce the scale; permit warmup recovery only.
            pass
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if not optimizer.state:
        raise FloatingPointError("AMP warmup never allocated optimizer state")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    timings = {key: [] for key in ("cpu_prepare_ms", "h2d_ms", "forward_ms", "loss_ms", "backward_ms", "unscale_clip_ms", "optimizer_ms", "zero_grad_ms", "update_wall_ms")}
    losses, norms, skipped = [], [], 0
    for repetition in range(profile["update_repetitions"]):
        update_started = time.perf_counter()
        _, elapsed = _timed(lambda: optimizer.zero_grad(set_to_none=True), device)
        timings["zero_grad_ms"].append(elapsed)
        for j in range(accumulation):
            first = repetition * effective + j * physical_batch
            windows = schedule[first:first + physical_batch]
            prep_started = time.perf_counter()
            cpu_batch = profile_batch(cohort, windows, context_length=context_length,
                                      num_negatives=profile["num_negatives"])
            timings["cpu_prepare_ms"].append((time.perf_counter() - prep_started) * 1000)
            batch, elapsed = _timed(lambda: {k: v.to(device) for k, v in cpu_batch.items()}, device)
            timings["h2d_ms"].append(elapsed)
            encoded, elapsed = _timed(lambda: encode(batch), device)
            timings["forward_ms"].append(elapsed)
            loss, elapsed = _timed(lambda: _loss(model, encoded, batch, profile["tail_size"], profile["num_negatives"]), device)
            timings["loss_ms"].append(elapsed)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite measured loss")
            _, elapsed = _timed(lambda: scaler.scale(loss / accumulation).backward(), device)
            timings["backward_ms"].append(elapsed)
            losses.append(float(loss.detach()))
            del encoded, loss, batch, cpu_batch

        def unscale_clip() -> torch.Tensor:
            scaler.unscale_(optimizer)
            return torch.nn.utils.clip_grad_norm_(model.parameters(), profile["grad_clip_norm"])
        norm, elapsed = _timed(unscale_clip, device)
        timings["unscale_clip_ms"].append(elapsed)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite measured gradient norm; timings cannot claim a successful update")
        norms.append(float(norm))
        old_scale = float(scaler.get_scale())
        def update() -> None:
            scaler.step(optimizer)
            scaler.update()
        _, elapsed = _timed(update, device)
        timings["optimizer_ms"].append(elapsed)
        skipped += int(float(scaler.get_scale()) < old_scale)
        timings["update_wall_ms"].append((time.perf_counter() - update_started) * 1000)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    optimizer.zero_grad(set_to_none=True)
    micro_total = sum(sum(timings[k]) for k in ("forward_ms", "loss_ms", "backward_ms"))
    update_total = sum(sum(timings[k]) for k in ("unscale_clip_ms", "optimizer_ms", "zero_grad_ms"))
    pure_seconds = (micro_total + update_total) / 1000
    windows_count = effective * profile["update_repetitions"]
    return {"status": "ok", "context_length": context_length, "physical_batch_size": physical_batch,
        "effective_batch_size": effective, "accumulation_steps": accumulation,
        "measured_microbatches": len(losses), "measured_update_attempts": profile["update_repetitions"],
        "measured_successful_updates": profile["update_repetitions"] - skipped,
        "amp_skipped_updates": skipped, "warmup_microbatches": max(1, profile["warmup_microbatches"]),
        "warmup_physical_batch_update_attempts": max(1, profile["warmup_microbatches"]),
        "optimizer_state_parameter_entries": len(optimizer.state), "windows": windows_count,
        "supervised_target_presentations": windows_count * profile["tail_size"],
        "timings": {k: _stats(v) for k, v in timings.items()}, "loss": _stats(losses), "gradient_norm": _stats(norms),
        "pure_train_seconds": pure_seconds, "train_windows_per_second": windows_count / pure_seconds,
        "train_windows_per_second_with_prepare_h2d_sync": windows_count / (sum(timings["update_wall_ms"]) / 1000),
        "allocated_before_timed_update_gib": baseline_allocated / 2**30,
        "peak_allocated_gib": peak_allocated / 2**30, "peak_reserved_gib": peak_reserved / 2**30,
        "gpu_increment_peak_gib": max(0, peak_allocated - baseline_allocated) / 2**30,
        "timing_policy": "synchronized_wall_per_component; pure excludes CPU preparation and H2D; optimizer includes scaler update"}


@torch.no_grad()
def _inference(model: torch.nn.Module, cohort: LongHistoryCohort,
               schedule: tuple[ProfileWindow, ...], profile: dict[str, Any],
               context_length: int, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    cpu = profile_batch(cohort, schedule[:batch_size], context_length=context_length,
                        num_negatives=profile["num_negatives"])
    ids, valid = cpu["item_ids"].to(device), cpu["valid_mask"].to(device)
    def history() -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=profile["amp"]):
            return model.last_state(ids, valid)
    for _ in range(profile["inference_warmup"]):
        state = history()
        score_catalogue(model, state, item_chunk_size=profile.get("item_chunk_size", 65536))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    encoder, scoring, totals = [], [], []
    for _ in range(profile["inference_iterations"]):
        started = time.perf_counter()
        state, elapsed = _timed(history, device)
        encoder.append(elapsed)
        scores, elapsed = _timed(lambda: score_catalogue(model, state, item_chunk_size=profile.get("item_chunk_size", 65536)), device)
        scoring.append(elapsed)
        totals.append((time.perf_counter() - started) * 1000)
        if not torch.isfinite(state).all() or not torch.isfinite(scores[0]).all():
            raise FloatingPointError("nonfinite inference state or catalogue scores")
    return {"status": "ok", "batch_size": batch_size, "history_ms": _stats(encoder),
        "catalogue_topk_ms": _stats(scoring), "full_ms": _stats(totals),
        "users_per_second": batch_size / (float(np.mean(totals)) / 1000),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "cache_used": False, "quality_evaluated": False,
        "workload": "full prefill and FP32 exact catalogue top10; no CPU preparation/H2D in timed region"}


def _major_flops(model: torch.nn.Module, architecture: str, length: int) -> dict[str, Any]:
    d, layers = model.d_model, model.n_layers
    if architecture == "sasrec":
        value = dense_encoder_flops(batch_size=1, sequence_length=length, d_model=d,
                                    n_layers=layers, ffn_multiplier=model.ffn_multiplier)
    elif architecture == "hstu":
        value = hstu_encoder_flops(batch_size=1, sequence_length=length, valid_tokens=length,
                                   layers=model.encoder.layers)
    elif architecture == "mamba4rec":
        inner, state, rank = model.mamba_expand * d, model.mamba_d_state, math.ceil(d / 16)
        weights = d * (2 * inner) + inner * (rank + 2 * state) + rank * inner + inner * d + 2 * d * (4 * d)
        value = layers * 2 * length * weights
    elif hasattr(model, "major_matmul_flops"):
        value = model.major_matmul_flops(1, length)
    else:
        return {"available": False, "scope": "native chunkwise FLOP accounting not available; timing remains measured"}
    return {"available": True, "per_user": int(value), "gflops_per_user": value / 1e9,
        "scope": "forward major matrix multiplies only; excludes elementwise scan/conv/norm/activation/packing/embedding lookup/loss/optimizer/catalogue; not training FLOPs"}


def run_long_training_profile(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    validate_long_profile_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("native CUDA required; CPU fallback would invalidate this benchmark")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile, model_config = config["profile"], config["model"]
    device = torch.device("cuda:0")
    torch.set_num_threads(profile.get("cpu_threads", 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config["seed"])
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    if cohort.summary["train_end_timestamp"] != config["data"]["train_end_timestamp"]:
        raise ValueError("prepared cohort cutoff differs from the registered profile")
    if cohort.summary["max_seq_len"] != profile["raw_context_length"]:
        raise ValueError("prepared cohort context differs from the registered profile")
    for field in ("cohort_uid_length_sha256", "catalog_sha256", "cohort_mapped_train_items_sha256"):
        if cohort.summary[field] != config["data"][field]:
            raise ValueError(f"prepared cohort {field} differs from the registered profile")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    count = max(profile["effective_batch_size"] * profile["update_repetitions"], max(profile.get("inference_batch_sizes", [1, 8, 32])))
    schedule = window_schedule(cohort, seed=config["seed"], count=count,
                               raw_context_length=profile["raw_context_length"])
    rows = [{"user_id": w.user_id, "end": w.end, "negative_seed": w.negative_seed} for w in schedule]
    schedule_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    _json(output_dir / "window-schedule.json", {"sha256": schedule_hash, "windows": rows,
        "targets": "next item at each of last tail_size input positions; no held-out targets",
        "negative_policy": "512 catalogue-uniform negatives shared within each window and identical across model/batch choices"})
    properties = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {"complete": False, "quality_evaluated": False, "hpo_performed": False,
        "code_sha": os.environ.get("GITHUB_SHA", "local"), "window_schedule_sha256": schedule_hash,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "gpu_name": properties.name, "gpu_total_memory_gib": properties.total_memory / 2**30,
            "gpu_device": 0, "available_gpu_count": torch.cuda.device_count(),
            "amp_encoder": profile["amp"], "loss_and_catalogue_dtype": "float32", "tf32_matmul": False},
        "config": config, "models": {}}
    metrics: dict[str, int | float | str | bool] = {"experiment": config["name"], "seed": config["seed"],
        "profile_complete": False, "quality_evaluated": False, "hpo_performed": False,
        "cohort_users": len(cohort.users), "train_events": cohort.train_events, "num_items": cohort.num_items,
        "window_schedule_sha256": schedule_hash, "gpu_name": properties.name,
        "torch_version": torch.__version__, "cuda_version": str(torch.version.cuda)}
    def persist() -> None:
        _json(output_dir / "training-profile.json", report)
        _json(output_dir / "metrics.json", validate_metrics(metrics))
    persist()
    started = time.monotonic()
    for architecture in ARCHITECTURES:
        context = profile["tail_size"] if architecture in {"sasrec", "hstu"} else profile["raw_context_length"]
        print(json.dumps({"phase": "profile_model_start", "architecture": architecture, "context": context}), flush=True)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = build_profile_model(architecture, cohort.num_items, context, model_config).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        item_parameters = model.item_embedding.weight.numel()
        result: dict[str, Any] = {"architecture": architecture, "context_length": context,
            "total_parameters": parameters, "item_embedding_parameters": item_parameters,
            "non_item_parameters": parameters - item_parameters,
            "implementation": getattr(model, "attention_implementation", architecture),
            "upstream_revision": getattr(model, "upstream_revision", "rhpr_sasrec"),
            "native_runtime": getattr(model, "native_runtime", {}),
            "forward_major_matmul_flops": _major_flops(model, architecture, context),
            "training": [], "inference": []}
        report["models"][architecture] = result
        metrics[f"{architecture}_total_parameters"] = parameters
        metrics[f"{architecture}_non_item_parameters"] = parameters - item_parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=profile["learning_rate"], weight_decay=profile.get("weight_decay", 1e-5), foreach=False)
        scaler = torch.amp.GradScaler("cuda", enabled=profile["amp"], init_scale=profile["initial_loss_scale"])
        for physical in profile["physical_batch_sizes"]:
            try:
                shape = _train_shape(model, cohort, schedule, profile, context, physical, device, optimizer, scaler)
            except torch.OutOfMemoryError:
                shape = {"status": "oom", "physical_batch_size": physical,
                         "effective_batch_size": profile["effective_batch_size"], "context_length": context}
            result["training"].append(shape)
            metrics[f"{architecture}_b{physical}_training_status"] = shape["status"]
            if shape["status"] == "ok":
                for key in ("train_windows_per_second", "peak_allocated_gib", "peak_reserved_gib", "measured_successful_updates", "amp_skipped_updates"):
                    metrics[f"{architecture}_b{physical}_{key}"] = shape[key]
                metrics[f"{architecture}_b{physical}_forward_p50_ms"] = shape["timings"]["forward_ms"]["p50"]
                metrics[f"{architecture}_b{physical}_backward_p50_ms"] = shape["timings"]["backward_ms"]["p50"]
                metrics[f"{architecture}_b{physical}_optimizer_p50_ms"] = shape["timings"]["optimizer_ms"]["p50"]
                metrics[f"{architecture}_b{physical}_estimated_one_pass_train_seconds"] = len(cohort.users) / shape["train_windows_per_second"]
            persist()
            print(json.dumps({"phase": "profile_training_shape", "architecture": architecture, "batch": physical,
                "status": shape["status"], "train_windows_per_second": shape.get("train_windows_per_second")}), flush=True)
            if shape["status"] == "oom":
                # Do this outside the except block: its traceback may keep the
                # failed graph alive. No partial Adam/scaler state is reused.
                del optimizer, scaler, model
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(config["seed"])
                torch.cuda.manual_seed_all(config["seed"])
                model = build_profile_model(architecture, cohort.num_items, context, model_config).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=profile["learning_rate"], weight_decay=profile.get("weight_decay", 1e-5), foreach=False)
                scaler = torch.amp.GradScaler("cuda", enabled=profile["amp"], init_scale=profile["initial_loss_scale"])
        if not any(row["status"] == "ok" for row in result["training"]):
            raise RuntimeError(f"{architecture}: no physical batch completed a native CUDA training update")
        del optimizer, scaler
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        for batch_size in profile.get("inference_batch_sizes", [1, 8, 32]):
            try:
                inference = _inference(model, cohort, schedule, profile, context, batch_size, device)
            except torch.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                inference = {"status": "oom", "batch_size": batch_size}
            result["inference"].append(inference)
            metrics[f"{architecture}_inference_b{batch_size}_status"] = inference["status"]
            if inference["status"] == "ok":
                metrics[f"{architecture}_inference_b{batch_size}_full_p50_ms"] = inference["full_ms"]["p50"]
                metrics[f"{architecture}_inference_b{batch_size}_users_per_second"] = inference["users_per_second"]
            persist()
        if not any(row["status"] == "ok" for row in result["inference"]):
            raise RuntimeError(f"{architecture}: no native inference batch completed")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report["complete"] = metrics["profile_complete"] = True
    metrics["profile_seconds"] = time.monotonic() - started
    persist()
    return validate_metrics(metrics)
