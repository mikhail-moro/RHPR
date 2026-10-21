"""Loaded-checkpoint models and exact-shape gates for history continuation.

All registered arms use one selected source checkpoint. Timing uses disposable
copies of those actual weights and a fresh AdamW, never an untrained stand-in.
No timing update can become a quality checkpoint.
"""
from __future__ import annotations

from collections.abc import Callable
import math
import time
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler, _sync
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.long_training_profile import _loss, _major_flops
from rhpr.sasrec.query_baseline_models import baseline_optimizer, build_baseline_model
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
from rhpr.sasrec.query_proxy_protocol import build_hpo_model
from rhpr.sasrec.query_ssm import make_query_compressor


_FIXED_TRIAL = {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}
_QSSM_TRIAL = dict(_FIXED_TRIAL, name="dt_slow", d_inner=64, d_state=16,
                   gate_rank=32, dt_min=.0001, dt_max=.01)


def build_history_model(config: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Reconstruct the exact selected architecture before strict weight loading.

    QSSM starts with the original K4 constructor, then replaces its compressor
    exactly as in the completed summary-count search. This preserves the source
    state-dict contract, full positional table and initialization RNG stream.
    """
    family, trial = config["family"], config["selected_source"]["trial"]
    if family == "qssm":
        if trial != _QSSM_TRIAL or any(type(trial.get(key)) is not type(value)
                                     for key, value in _QSSM_TRIAL.items()):
            raise ValueError("history QSSM requires the selected K32/slow inner64 state16 rank32 recipe")
        if config["architecture"] != {key: trial[key] for key in
                                      ("d_inner", "d_state", "gate_rank", "dt_min", "dt_max")}:
            raise ValueError("history QSSM architecture must match its selected source trial")
        model = build_hpo_model(config,
            {"name": "query_ssm_m4", "kind": "query_ssm", "summary_tokens": 4}, num_items)
        parent = config["parent"]["experiment_config"]
        dims, train = parent["model"], parent["training"]
        compressor = dict(parent["compressor"])
        compressor.update({key: trial[key] for key in
                           ("d_inner", "d_state", "gate_rank", "dt_min", "dt_max")})
        compressor["query_source"] = "old_suffix"
        model.compressor = make_query_compressor(kind="query_ssm", d_model=dims["d_model"],
            summary_tokens=32, tail_size=train["raw_tail_size"], n_heads=dims["n_heads"], **compressor)
    elif family == "fuxi":
        if trial != _FIXED_TRIAL or any(type(trial.get(key)) is not type(value)
                                      for key, value in _FIXED_TRIAL.items()):
            raise ValueError("history FuXi requires its selected fixed_lr recipe")
        native = {"chunk_size": 128, "temporal_heads": 8, "positional_dim": 32}
        if config["architecture"] != native:
            raise ValueError("history FuXi requires the selected native two-layer architecture")
        model = build_baseline_model(dict(config, standalone_models={"fuxi_linear": native}),
            {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0},
            dict(_FIXED_TRIAL), num_items)
    else:
        raise ValueError("history continuation supports the selected QSSM or FuXi family")
    model.history_source_checkpoint_loaded = False
    return model


def history_optimizer(model: torch.nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    """Restart registered arms with identical LR/decay and complete parameter ownership."""
    if training["learning_rate"] != .001 or training["weight_decay"] != .00001:
        raise ValueError("history continuation preserves learning_rate=.001 and weight_decay=1e-5")
    return baseline_optimizer(model, training, dict(_FIXED_TRIAL))


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _capacity_backward(model: torch.nn.Module, cohort: Any, windows: tuple[Any, ...], *,
                       config: dict[str, Any], variant: dict[str, Any], device: torch.device,
                       optimizer: torch.optim.Optimizer, batch_builder: Callable[..., Any]) -> dict[str, Any]:
    """Warm the longest real microbatch without changing weights or Adam state."""
    train = config["training"]
    arrays = batch_builder(cohort, windows, config, variant)
    batch = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
    scaler = _scaler(device, train)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    started = time.perf_counter()
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=train["amp"] and device.type == "cuda"):
            encoded = model.encode(batch["item_ids"], batch["valid_mask"])
        loss = _loss(model, encoded, batch, train["supervised_targets"], train["num_negatives"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite longest-history capacity warmup loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip_norm"])
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite longest-history capacity warmup gradients")
        _sync(device)
        return {"status": "success", "seconds": time.perf_counter() - started,
            "physical_batch_size": len(windows), "input_padded_length": int(batch["item_ids"].shape[1]),
            "input_length_min": int(batch["valid_mask"].sum(1).min()),
            "input_length_max": int(batch["valid_mask"].sum(1).max()),
            "forward_and_backward_performed": True, "optimizer_updates": 0,
            "weights_unchanged": True, "optimizer_state_unchanged": not bool(optimizer.state)}
    finally:
        optimizer.zero_grad(set_to_none=True)
        del batch, arrays, scaler


def preflight_history(config: dict[str, Any], cohort: Any, evaluation: Any,
                      arm: dict[str, Any], device: torch.device, *,
                      load_model: Callable[[], torch.nn.Module]) -> dict[str, Any]:
    """Measure the registered short/full shape from verified trained weights.

    ``load_model`` must strictly load the verified selected checkpoint and mark
    ``history_source_checkpoint_loaded=True``. Each call returns a disposable
    model; it is never reused for the quality continuation. A separate longest
    microbatch backward precedes two complete effective Adam updates. RNG from
    profiling is isolated from all actual training arms.
    """
    from rhpr.sasrec.query_history_protocol import history_arms, history_batch_arrays, history_variant

    if arm not in history_arms(config):
        raise ValueError("a registered history-continuation arm is required")
    if not callable(load_model):
        raise ValueError("preflight requires a verified source-checkpoint model loader")
    device = torch.device(device)
    train, bench = config["training"], config["benchmark"]
    variant = history_variant(config)
    physical, cap = arm["physical_batch_size"], arm["context_length"]
    model = optimizer = scaler = None
    with _preserve_rng(device):
        try:
            model = load_model().to(device)
            if getattr(model, "history_source_checkpoint_loaded", False) is not True:
                raise ValueError("preflight must use the verified trained source checkpoint, not fresh weights")
            optimizer = history_optimizer(model, train)
            schedule = fixed_prefix_schedule(cohort, seed=config["seed"],
                epoch=config["continuation"]["schedule_epoch_offset"] + 1)
            longest = tuple(sorted(schedule, key=lambda row: (-row.end, row.user_id))[:physical])
            windows = _representative_windows(schedule, train["effective_batch_size"])
            def builder(batch_cohort: Any, batch_windows: Any, batch_config: dict[str, Any],
                        _variant: dict[str, Any]) -> dict[str, Any]:
                return history_batch_arrays(batch_cohort, batch_windows, batch_config, arm)
            _reset_peak(device)
            capacity = _capacity_backward(model, cohort, longest, config=config, variant=variant,
                device=device, optimizer=optimizer, batch_builder=builder)
            capacity_memory = _peak_memory(device)
            if optimizer.state:
                raise RuntimeError("capacity warmup must not mutate the fresh Adam state")
            scaler = _scaler(device, train)
            updates = []
            _reset_peak(device)
            for _ in range(2):
                row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                    physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler,
                    batch_builder=builder)
                if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                    raise FloatingPointError("history preflight requires two effective Adam updates without AMP retries")
                updates.append(row)
            training_memory = _peak_memory(device)
            _reset_peak(device)
            try:
                daily = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                    batch_size=arm["evaluation_batch_size"], device=device, amp=train["amp"],
                    item_chunk_size=bench["item_chunk_size"])
            except RuntimeError as error:
                if str(error) in ("evaluation produced nonfinite final states",
                                  "evaluation produced nonfinite catalogue scores"):
                    raise FloatingPointError(str(error)) from error
                raise
            daily = dict(daily, **_peak_memory(device), input_context_cap=cap,
                         optimizer_state_allocated=True)
            user = sorted(cohort.users, key=lambda row: (len(row.train), row.user_id))[len(cohort.users) // 2]
            context = user.train[:-1][-cap:]
            model.eval()
            with torch.inference_mode():
                if config["family"] == "qssm":
                    batch = _batch([context], model, device)
                    _, arithmetic = _observe_prefix_flops(model, batch, device, train["amp"])
                    arithmetic = dict(arithmetic, input_tokens=len(context),
                        method="observed_query_ssm_projections_and_dense_SASRec_matmuls_MAC_equals_2")
                    del batch
                else:
                    cost = _major_flops(model, model.architecture, len(context))
                    arithmetic = {"input_tokens": len(context),
                        "forward_major_matmul_flops": cost["per_user"],
                        "catalogue_scoring_matmul_flops": 2 * cohort.num_items * model.d_model,
                        "method": "pinned_native_family_forward_major_matmul_formula_MAC_equals_2",
                        "scope": cost["scope"], "total_forward_flops_available": False}
            lengths = [min(row.end, cap) for row in windows]
            parameters = list(model.named_parameters())
            compressor = getattr(model, "compressor", None)
            return {"status": "success", "arm": arm["name"], "old_length": arm["old_length"],
                "physical_batch_size": physical, "effective_batch_size": train["effective_batch_size"],
                "capacity_warmup": capacity, "capacity_warmup_memory": capacity_memory,
                "warmup": updates[0], "training": updates[1], "training_memory": training_memory,
                "evaluation": daily,
                "sample_input_length_min": min(lengths), "sample_input_length_max": max(lengths),
                "sample_input_length_mean": sum(lengths) / len(lengths),
                "sampling": "uniform length quantiles in actual shuffled schedule order; separate longest backward",
                "total_parameters": sum(parameter.numel() for _, parameter in parameters),
                "compressor_parameters": sum(parameter.numel() for parameter in compressor.parameters())
                    if compressor is not None else 0,
                "non_item_parameters": sum(parameter.numel() for name, parameter in parameters
                                           if not name.startswith("item_embedding.")),
                "processing_parameters": sum(parameter.numel() for name, parameter in parameters
                    if not name.startswith(("item_embedding.", "position_embedding."))
                    and not name.endswith("._channel_p._emb")),
                "representative_b1_arithmetic": arithmetic,
                "arithmetic_excludes": ["embedding_lookup", "packing", "normalization", "activation",
                                        "top_k", "loss", "backward", "optimizer"],
                "source_checkpoint_weights_loaded": True, "fresh_initializer_profiled": False,
                "capacity_warmup_has_no_optimizer_step": True,
                "profile_weights_reused": False, "quality_used_for_selection": False,
                "optimizer_state_allocated_during_measured_updates_and_evaluation": True,
                "approves_stability_of_other_shapes": False,
                "automatic_precision_or_batch_fallback": False}
        finally:
            del model, optimizer, scaler
            _cleanup()


def conservative_history_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                                 users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price all registered fixed-pass arms, including zero-step evaluation."""
    from rhpr.sasrec.query_history_protocol import history_arms

    train, continuation = config["training"], config["continuation"]
    passes = continuation["additional_passes"]
    validations = len(continuation["validation_passes"])
    work = 0.
    for arm in history_arms(config):
        row = profiles[arm["name"]]
        if row.get("status") != "success":
            raise ValueError("all registered history shapes must pass preflight before training")
        rates = row["training"]["windows_per_second"], row["evaluation"]["anchors_per_second"]
        if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("positive finite exact-history-shape preflight rates are required")
        work += passes * users / rates[0] + validations * evaluation_summary["validation_anchors"] / rates[1]
    overhead = 120. * len(history_arms(config))
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": len(history_arms(config)), "passes_per_arm": passes,
        "validation_evaluations_per_arm": validations, "validation_passes_include_source_zero": True,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_arms_at_registered_passes": True,
        "auto_reduces_arms_or_passes": False, "full_training_models": 1, "test_evaluations": 0}
