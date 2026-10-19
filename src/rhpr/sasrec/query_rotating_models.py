"""Fresh QSSM/FuXi models and exact H1024+R128 rotating-window profiles.

The canonical history configurations specify architectures and initializers,
not weights to import. Both new arms start from scratch. Profiling consumes
disposable models and isolates its RNG from subsequent quality training.
"""
from __future__ import annotations

from functools import partial
import math
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.long_training_profile import _major_flops
from rhpr.sasrec.query_history_models import build_history_model, history_optimizer
from rhpr.sasrec.query_lr_restart import _preserve_rng
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import (
    _profile_daily_rate, _representative_windows, train_prefix_pass,
)


_FAMILIES = {"query_ssm": "qssm", "fuxi_linear": "fuxi"}
_FAILURES = frozenset(("preflight_numerical_failed", "preflight_out_of_memory"))


def _registered(config: dict[str, Any], variant: dict[str, Any], trial: dict[str, Any]) -> str:
    from rhpr.sasrec.query_rotating_protocol import rotating_trials, rotating_variants

    if variant not in rotating_variants(config) or trial not in rotating_trials(config, variant):
        raise ValueError("a registered rotating-window variant and trial are required")
    if trial.get("name") != "rotating":
        raise ValueError("historical_fixed is a saved comparison, not a fresh training arm")
    if any(type(trial.get(key)) not in (int, float) or trial[key] != .001
           for key in ("backbone_lr", "compressor_lr")):
        raise ValueError("rotating-window arms preserve both learning rates at 0.001")
    try:
        return _FAMILIES[variant["kind"]]
    except KeyError as error:
        raise ValueError("rotating windows support only the selected QSSM and native FuXi") from error


def build_rotating_model(config: dict[str, Any], variant: dict[str, Any],
                         trial: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Preserve each selected architecture's original fresh initializer and RNG.

    Full 32768-entry positional tables are retained exactly. The historical
    checkpoint metadata in ``history_sources`` is never opened or loaded.
    """
    family = _registered(config, variant, trial)
    source = config["history_sources"][family]
    if source["family"] != family or source["seed"] != config["seed"]:
        raise ValueError("the canonical initializer family and seed must match the registered arm")
    model = build_history_model(source, num_items)
    if model.history_source_checkpoint_loaded is not False:
        raise RuntimeError("rotating-window training requires a fresh initializer without source weights")
    model.rotating_metadata = {
        "initialization": "exact_selected_architecture_fresh_initializer_and_post_constructor_rng",
        "initial_position_table_length": source["training"]["raw_context_length"],
        "checkpoint_weights_loaded": False,
        "same_seed_implies_same_cross_architecture_parameters": False,
        "family": family,
        "precision": ("native_query_selective_scan_with_FP32_state_and_FP16_encoder_autocast"
                      if family == "qssm" else
                      "FP32_KV_products_state_recurrence_and_attention_reads_until_channel_norm"),
        "historical_fixed_precision_identical": family == "qssm",
        "historical_fixed_precision_caveat": ("none" if family == "qssm" else
            "saved_fixed_FuXi_used_pre_fix_FP16_attention_arithmetic; historical_delta_is_not_a_pure_schedule_effect"),
    }
    return model


def rotating_optimizer(model: torch.nn.Module, training: dict[str, Any],
                        trial: dict[str, Any]) -> torch.optim.Optimizer:
    """Fresh AdamW at the common fixed LR with every trainable parameter owned."""
    if (trial.get("name") != "rotating"
            or any(type(trial.get(key)) not in (int, float) or trial[key] != .001
                   for key in ("backbone_lr", "compressor_lr"))):
        raise ValueError("the registered rotating trial with fixed learning rates is required")
    return history_optimizer(model, training)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def preflight_rotating(config: dict[str, Any], cohort: Any, evaluation: Any,
                       variant: dict[str, Any], trial: dict[str, Any],
                       device: torch.device) -> dict[str, Any]:
    """Measure two real effective Adam steps and matched 1152-token inference.

    Every registered training row has the same 1152 occupied input positions.
    The first complete effective update warms that exact shape and allocates
    Adam state. The second update measures steady throughput; catalogue timing
    keeps that optimizer state allocated. No full-history transfer is profiled.
    """
    from rhpr.sasrec.query_rotating_protocol import rotating_batch_arrays, rotating_schedule

    family = _registered(config, variant, trial)
    device = torch.device(device)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    batches = config["rotating_batches"][variant["name"]]
    physical, old = batches["training"], hpo["proxy_old_length"]
    cap = old + train["raw_tail_size"]
    if old != 1024 or cap != 1152:
        raise ValueError("the rotating preflight measures exactly H1024+R128")
    model = optimizer = scaler = None
    with _preserve_rng(device):
        try:
            model = build_rotating_model(config, variant, trial, cohort.num_items).to(device)
            optimizer, scaler = rotating_optimizer(model, train, trial), _scaler(device, train)
            schedule = rotating_schedule(cohort, seed=config["seed"], epoch=1)
            windows = _representative_windows(schedule, train["effective_batch_size"])
            builder = partial(rotating_batch_arrays, old_length=old)
            _reset_peak(device)
            updates = []
            for _ in range(2):
                row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                    physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler,
                    batch_builder=builder)
                if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                    raise FloatingPointError("rotating preflight requires two effective Adam updates without AMP retries")
                updates.append(row)
            training_memory = _peak_memory(device)
            _reset_peak(device)
            try:
                daily = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                    batch_size=batches["proxy"], device=device, amp=train["amp"],
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
            if len(context) != cap:
                raise ValueError("rotating preflight requires complete 1152-event histories")
            model.eval()
            with torch.inference_mode():
                if family == "qssm":
                    batch = _batch([context], model, device)
                    _, cost = _observe_prefix_flops(model, batch, device, train["amp"])
                    arithmetic = {"input_tokens": cap, **cost,
                        "method": "observed_query_SSM_projections_dense_SASRec_matmuls_and_analytical_scan_MAC_equals_2"}
                    del batch
                else:
                    cost = _major_flops(model, model.architecture, cap)
                    arithmetic = {"input_tokens": cap,
                        "forward_major_matmul_flops": cost["per_user"],
                        "catalogue_scoring_matmul_flops": 2 * cohort.num_items * model.d_model,
                        "method": "native_FuXi_all_channels_full1152_forward_major_matmuls_MAC_equals_2",
                        "scope": cost["scope"], "total_forward_flops_available": False}
            parameters = list(model.named_parameters())
            compressor = getattr(model, "compressor", None)
            return {"status": "success", "warmup": updates[0], "training": updates[1],
                "training_memory": training_memory, "proxy_evaluation": daily,
                "full_evaluation": {"status": "not_registered", "profiled": False},
                "old_length": old, "input_context_cap": cap, "profile_trial": dict(trial),
                "physical_batch_size": physical, "effective_batch_size": train["effective_batch_size"],
                "total_parameters": sum(parameter.numel() for _, parameter in parameters),
                "compressor_parameters": sum(parameter.numel() for parameter in compressor.parameters())
                    if compressor is not None else 0,
                "non_item_parameters": sum(parameter.numel() for name, parameter in parameters
                                           if not name.startswith("item_embedding.")),
                "processing_parameters": sum(parameter.numel() for name, parameter in parameters
                    if not name.startswith(("item_embedding.", "position_embedding."))
                    and not name.endswith("._channel_p._emb")),
                "representative_b1_arithmetic": {"proxy": arithmetic},
                "arithmetic_excludes": ["embedding_lookup", "packing", "normalization", "activation",
                                        "top_k", "loss", "backward", "optimizer"],
                "cross_family_flop_scopes_identical": False,
                "model_initialization": model.rotating_metadata,
                "source_checkpoint_weights_loaded": False, "fresh_initializer_profiled": True,
                "profile_weights_reused": False, "quality_used_for_selection": False,
                "optimizer_state_allocated_during_all_measurements": True,
                "approves_stability_of_other_shapes": False,
                "automatic_precision_or_batch_fallback": False}
        finally:
            del model, optimizer, scaler
            _cleanup()


def conservative_rotating_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                                  users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price both fresh arms at the registered maximum before quality training."""
    from rhpr.sasrec.query_rotating_protocol import rotating_variants

    train, hpo = config["training"], config["hpo"]
    work, priced, failed = 0., 0, 0
    for variant in rotating_variants(config):
        row = profiles[f"{variant['name']}-rotating"]
        if row.get("status") in _FAILURES:
            failed += 1
            continue
        if row.get("status") != "success":
            raise ValueError("each rotating shape must pass preflight or have an explicit preflight failure")
        rates = row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"]
        if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("positive finite per-family exact-shape preflight rates are required")
        work += (hpo["max_passes"] * users / rates[0]
            + hpo["max_passes"] // hpo["validation_interval"]
            * evaluation_summary["validation_anchors"] / rates[1])
        priced += 1
    overhead = 120. * priced
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": 4, "new_training_configurations": 2,
        "reused_controls": 2, "priced_fresh_shapes": priced, "preflight_failed_shapes": failed,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0, "test_evaluations": 0}
