"""Fresh QSSM2 state128 model and real H1024 training/evaluation cost gate."""
from __future__ import annotations

from functools import partial
import math
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.query_history_models import build_history_model, history_optimizer
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_rotating_models import _peak_memory, _reset_peak
from rhpr.sasrec.query_rotating_protocol import rotating_schedule
from rhpr.sasrec.query_ssm2 import make_query_ssm2_compressor
from rhpr.sasrec.query_ssm2_training_protocol import (
    ARCHITECTURE, COMPRESSOR, TRIAL, VARIANT, ssm2_training_batch_arrays,
)


def _registered(variant: dict[str, Any], trial: dict[str, Any]) -> None:
    if variant != VARIANT or trial != TRIAL:
        raise ValueError("the single registered SSD state128 variant and fixed_lr trial are required")


def build_ssm2_training_model(config: dict[str, Any], variant: dict[str, Any], trial: dict[str, Any],
                              num_items: int) -> torch.nn.Module:
    _registered(variant, trial)
    if config["architecture"] != ARCHITECTURE or config["compressor"] != COMPRESSOR:
        raise ValueError("SSD training requires d_state128 and head_dim64")
    model = build_history_model(config["initializer"], num_items)
    dims, train = config["model"], config["training"]
    model.compressor = make_query_ssm2_compressor(d_model=dims["d_model"],
        tail_size=train["raw_tail_size"], n_heads=dims["n_heads"], summary_tokens=32,
        **config["compressor"])
    if model.history_source_checkpoint_loaded:
        raise RuntimeError("SSD training must begin from a fresh initializer without checkpoint weights")
    model.baseline_metadata = {"architecture": dict(ARCHITECTURE),
        "initialization": "original selected fresh SASRec and catalogue initializer; new SSD compressor",
        "initial_position_table_length": 32768, "source_checkpoint_weights_loaded": False,
        "native_backend": "mamba2_native_triton_ssd", "historical_QSSM_weights_compatible": False}
    return model


def ssm2_training_optimizer(model: torch.nn.Module, training: dict[str, Any],
                            trial: dict[str, Any]) -> torch.optim.Optimizer:
    _registered(VARIANT, trial)
    return history_optimizer(model, training)


def preflight_ssm2_training(config: dict[str, Any], cohort: Any, evaluation: Any,
                            variant: dict[str, Any], trial: dict[str, Any],
                            device: torch.device) -> dict[str, Any]:
    """Time two real effective updates and exact catalogue evaluation with Adam live."""
    _registered(variant, trial)
    device = torch.device(device)
    train, bench = config["training"], config["benchmark"]
    model = optimizer = scaler = None
    handles, critical = [], {}
    with _preserve_rng(device):
        try:
            model = build_ssm2_training_model(config, variant, trial, cohort.num_items).to(device)
            optimizer = ssm2_training_optimizer(model, train, trial)
            scaler = _scaler(device, train)
            windows = _representative_windows(rotating_schedule(cohort, seed=config["seed"], epoch=1),
                                             train["effective_batch_size"])
            builder = partial(ssm2_training_batch_arrays, old_length=1024)
            names = ("write_projection.weight", "delta_projection.weight", "delta_projection.bias",
                     "query_gate.weight", "query_generator.queries", "read_projection.weight", "A_log")
            for name, parameter in model.compressor.named_parameters():
                if name in names:
                    def observe(gradient: torch.Tensor, key: str = name) -> None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError(f"nonfinite preflight critical gradient: {key}")
                        critical[key] = max(critical.get(key, 0.), float(gradient.detach().float().norm()))
                    handles.append(parameter.register_hook(observe))
            _reset_peak(device)
            updates = []
            for index in range(2):
                row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                    physical_batch=config["hpo"]["physical_batch_size"], device=device,
                    optimizer=optimizer, scaler=scaler, batch_builder=builder)
                if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                    raise FloatingPointError("SSD preflight requires two real effective Adam updates without AMP retries")
                updates.append(row)
                if index == 0:
                    for handle in handles:
                        handle.remove()
                    handles.clear()
                    if any(name not in critical or not math.isfinite(critical[name]) or critical[name] <= 0 for name in names):
                        raise FloatingPointError("SSD preflight requires nonzero finite B/delta/query/read/A gradients")
            training_memory = _peak_memory(device)
            _reset_peak(device)
            try:
                daily = _profile_daily_rate(model, cohort, evaluation, context_length=1152, batch_size=32,
                    device=device, amp=train["amp"], item_chunk_size=bench["item_chunk_size"])
            except RuntimeError as error:
                if str(error) in ("evaluation produced nonfinite final states", "evaluation produced nonfinite catalogue scores"):
                    raise FloatingPointError(str(error)) from error
                raise
            daily = dict(daily, **_peak_memory(device), input_context_cap=1152, optimizer_state_allocated=True)
            user = cohort.users[len(cohort.users) // 2]
            model.eval()
            with torch.inference_mode():
                batch = _batch([user.train[:-1][-1152:]], model, device)
                _, arithmetic = _observe_prefix_flops(model, batch, device, train["amp"])
            return {"status": "success", "old_length": 1024, "input_context_cap": 1152,
                "profile_trial": dict(trial), "warmup": updates[0], "training": updates[1],
                "training_memory": training_memory, "proxy_evaluation": daily,
                "full_evaluation": {"status": "not_registered", "profiled": False},
                "physical_batch_size": 32, "effective_batch_size": 128,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "compressor_parameters": sum(p.numel() for p in model.compressor.parameters()),
                "critical_scaled_gradient_norms": critical, "native_backend": model.compressor.last_scan_backend,
                "representative_b1_arithmetic": {"proxy": {"input_tokens": 1152, **arithmetic}},
                "flops_caveat": "major dense projections plus logical SSD recurrence; not actual fused kernel instruction counts",
                "source_checkpoint_weights_loaded": False, "profile_weights_reused": False,
                "quality_used_for_selection": False, "optimizer_state_allocated_during_all_measurements": True,
                "automatic_precision_or_batch_fallback": False, "architecture": dict(ARCHITECTURE)}
        finally:
            for handle in handles:
                handle.remove()
            del model, optimizer, scaler
            _cleanup()


def conservative_ssm2_training_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                                      users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    row = profiles[VARIANT["name"] + "-" + TRIAL["name"]]
    if row.get("status") != "success":
        raise ValueError("the exact registered SSD state128 shape must pass native training/evaluation preflight")
    rates = (row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"])
    if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("positive finite measured training and full-catalogue validation rates are required")
    hpo, train = config["hpo"], config["training"]
    work = (hpo["max_passes"] * users / rates[0] + hpo["max_passes"] // hpo["validation_interval"]
            * evaluation_summary["validation_anchors"] / rates[1])
    total = elapsed + train["budget_safety_factor"] * (work + 120.)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": 120., "safety_factor": train["budget_safety_factor"],
        "conservative_total_seconds": total, "registered_budget_seconds": train["budget_seconds"],
        "fits": total <= train["budget_seconds"], "registered_configurations": 1,
        "new_training_configurations": 1, "reused_controls": 0, "test_evaluations": 0,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0}
