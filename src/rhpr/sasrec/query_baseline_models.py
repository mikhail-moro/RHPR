"""Native sequence baselines and cheap matched-history cost measurements.

Recent128 and QueryOnly retain the original SASRec initialization. Mamba4Rec
and FuXi use their pinned native block/input initialization; no learned item
table or SASRec weights are copied across architectures. Timing weights are
disposable and are never selected as quality checkpoints.
"""
from __future__ import annotations

from functools import partial
import math
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.long_training_profile import _major_flops
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_prefix_protocol import build_prefix_model, fixed_prefix_schedule


_FIXED_TRIAL = {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}
_FRESH_VARIANTS = (
    {"name": "recent128", "kind": "recent", "summary_tokens": 0},
    {"name": "query_only_m4", "kind": "query_only", "summary_tokens": 4},
    {"name": "mamba4rec", "kind": "mamba4rec", "summary_tokens": 0},
    {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0},
)
_PREFLIGHT_FAILURES = frozenset(("preflight_numerical_failed", "preflight_out_of_memory"))


def _registered_fresh(variant: dict[str, Any], trial: dict[str, Any]) -> None:
    if variant not in _FRESH_VARIANTS:
        raise ValueError("one of the four registered fresh baseline variants is required")
    if trial != _FIXED_TRIAL or any(type(trial.get(key)) not in (int, float)
                                  for key in ("backbone_lr", "compressor_lr")):
        raise ValueError("fresh baselines require the fixed_lr trial with both learning rates 0.001")


def build_baseline_model(config: dict[str, Any], variant: dict[str, Any],
                         trial: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Build one fresh arm, preserving each architecture's existing initializer."""
    _registered_fresh(variant, trial)
    parent = config["parent"]["experiment_config"]
    family = variant["kind"]
    if family == "recent":
        model = build_prefix_model(parent, variant, num_items)
        initialization = "exact_parent_sasrec_initializer_and_full_position_table"
    elif family == "query_only":
        from rhpr.sasrec.query_only_protocol import QUERY_ONLY_KIND, build_query_only_model
        # The helper reads the pinned original parent; the old experiment kind
        # is a narrow view for its constructor, not a change to this run's config.
        model = build_query_only_model(dict(config, kind=QUERY_ONLY_KIND), variant, num_items)
        initialization = "exact_parent_sasrec_plus_matching_query_generator_and_ffn"
    else:
        dims, train = parent["model"], parent["training"]
        kwargs = {key: dims[key] for key in
                  ("d_model", "n_heads", "n_layers", "ffn_multiplier", "dropout")}
        kwargs.update(num_items=num_items, max_seq_len=train["raw_context_length"],
                      use_time_features=False)
        torch.manual_seed(config["seed"])
        if family == "mamba4rec":
            from rhpr.sasrec.mamba import Mamba4RecModel
            native = config["standalone_models"][family]
            model = Mamba4RecModel(**kwargs, mamba_d_state=native["d_state"],
                mamba_d_conv=native["d_conv"], mamba_expand=native["expand"])
            initialization = "native_mamba1_ssm_with_recommendation_embedding_and_ffn_initialization"
        else:
            from rhpr.sasrec.fuxi import FuXiLinearModel
            native = config["standalone_models"][family]
            model = FuXiLinearModel(**kwargs, chunk_size=native["chunk_size"],
                fuxi_temporal_heads=native["temporal_heads"],
                fuxi_positional_dim=native["positional_dim"])
            initialization = "native_fuxi_blocks_sinusoidal_trainable_position_channel_and_recommendation_embedding"
    model.baseline_metadata = {
        "initialization_policy": initialization,
        "shared_item_embedding_copied": False,
        "same_seed_implies_same_cross_architecture_parameters": False,
        "native_special_initialization_preserved": family in ("mamba4rec", "fuxi_linear"),
        "time_features": False,
        "weight_decay_policy": "parent_AdamW_weight_decay_on_all_parameters_including_native_no_weight_decay_flags",
    }
    if family == "fuxi_linear":
        model.baseline_metadata.update(
            temporal_channel="all_three_channels_with_zero_timestamp_payload_no_future_timestamps",
            full_length_transfer="positions_after_training_width_keep_their_untrained_sinusoidal_initialization_except_AdamW_decay",
            paper_temporal_recipe_reproduced=False)
    return model


def baseline_optimizer(model: torch.nn.Module, training: dict[str, Any],
                       trial: dict[str, Any]) -> torch.optim.Optimizer:
    """Use nonempty component groups and the parent's explicit all-parameter decay.

Recent128 and standalone predictors have a single backbone group. QueryOnly
retains separate backbone/compressor groups, both at the same fixed 0.001 LR.
Native ``_no_weight_decay`` flags are deliberately not interpreted: changing
that convention only for a new baseline would change the optimization recipe.
"""
    if trial != _FIXED_TRIAL or any(type(trial.get(key)) not in (int, float)
                                  for key in ("backbone_lr", "compressor_lr")):
        raise ValueError("baseline optimizer requires the registered fixed_lr trial")
    groups: dict[str, list[torch.nn.Parameter]] = {"backbone": [], "compressor": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups["compressor" if name.startswith("compressor.") else "backbone"].append(parameter)
    if not groups["backbone"]:
        raise ValueError("baseline optimizer requires trainable backbone parameters")
    return torch.optim.AdamW([
        {"params": parameters, "lr": trial[f"{component}_lr"], "component": component}
        for component, parameters in groups.items() if parameters],
        lr=training["learning_rate"], weight_decay=training["weight_decay"], foreach=False)


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def preflight_baseline(config: dict[str, Any], cohort: Any, evaluation: Any,
                       variant: dict[str, Any], trial: dict[str, Any],
                       device: torch.device) -> dict[str, Any]:
    """Measure two real effective updates and each distinct inference workload.

Training histories have the same registered H1024+R128 cap, so the first update
warms actual training capacity including Adam. Full-length evaluation separately
warms the longest validation rows at its registered family-specific batch size.
The short controls have identical proxy/full inputs and are measured only once.
"""
    from rhpr.sasrec.query_baseline_protocol import (
        baseline_batch_arrays, baseline_context_cap, baseline_evaluation_batch,
    )

    _registered_fresh(variant, trial)
    device = torch.device(device)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    physical = config["baseline_batches"][variant["name"]]["training"]
    model = optimizer = scaler = None
    try:
        model = build_baseline_model(config, variant, trial, cohort.num_items).to(device)
        optimizer, scaler = baseline_optimizer(model, train, trial), _scaler(device, train)
        windows = _representative_windows(fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1),
                                          train["effective_batch_size"])
        builder = partial(baseline_batch_arrays, old_length=hpo["proxy_old_length"])
        updates = []
        _reset_peak(device)
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler,
                batch_builder=builder)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("baseline preflight requires two effective Adam updates without AMP retries")
            updates.append(row)
        training_memory = _peak_memory(device)
        evaluations: dict[str, Any] = {}
        measured_workloads: dict[tuple[int, int], dict[str, Any]] = {}
        for phase, old_length in (("proxy_evaluation", hpo["proxy_old_length"]), ("full_evaluation", None)):
            cap = baseline_context_cap(config, variant, old_length)
            batch_size = baseline_evaluation_batch(config, variant, old_length)
            key = cap, batch_size
            if key in measured_workloads:
                evaluations[phase] = dict(measured_workloads[key], reused_identical_input_profile=True,
                                          separate_forward_performed=False)
                continue
            _reset_peak(device)
            try:
                try:
                    row = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                        batch_size=batch_size, device=device, amp=train["amp"],
                        item_chunk_size=bench["item_chunk_size"])
                except RuntimeError as error:
                    if str(error) in ("evaluation produced nonfinite final states",
                                      "evaluation produced nonfinite catalogue scores"):
                        raise FloatingPointError(str(error)) from error
                    raise
            except (FloatingPointError, torch.OutOfMemoryError) as error:
                if phase != "full_evaluation":
                    raise
                evaluations[phase] = {
                    "status": "out_of_memory" if isinstance(error, torch.OutOfMemoryError) else "numerical_failed",
                    "input_context_cap": cap, "batch_size": batch_size,
                    "exception_type": type(error).__name__, "message": str(error),
                    "primary_training_and_proxy_evaluation_supported": True,
                    "automatic_precision_or_batch_fallback": False,
                    "quality_used_for_selection": False}
                _cleanup()
                continue
            row = dict(row, **_peak_memory(device), status="success", input_context_cap=cap,
                reused_identical_input_profile=False, separate_forward_performed=True,
                optimizer_state_allocated=True)
            evaluations[phase] = measured_workloads[key] = row
        arithmetic: dict[str, Any] = {}
        user = sorted(cohort.users, key=lambda row: (len(row.train), row.user_id))[len(cohort.users) // 2]
        context = user.train[:-1]
        model.eval()
        with torch.inference_mode():
            for phase, old_length in (("proxy", hpo["proxy_old_length"]), ("full", None)):
                cap = baseline_context_cap(config, variant, old_length)
                if phase == "full" and cap == baseline_context_cap(config, variant, hpo["proxy_old_length"]):
                    arithmetic[phase] = dict(arithmetic["proxy"], reused_identical_input_cost=True)
                    continue
                retained = context[-cap:]
                if getattr(model, "architecture", "sasrec") == "sasrec":
                    batch = _batch([retained], model, device)
                    _, cost = _observe_prefix_flops(model, batch, device, train["amp"])
                    arithmetic[phase] = {"input_tokens": len(retained), **cost,
                        "method": "observed_compressor_linear_attention_shapes_and_dense_sasrec_matmuls"}
                    del batch
                else:
                    cost = _major_flops(model, model.architecture, len(retained))
                    arithmetic[phase] = {"input_tokens": len(retained),
                        "forward_major_matmul_flops": cost["per_user"],
                        "catalogue_scoring_matmul_flops": 2 * cohort.num_items * model.d_model,
                        "method": "pinned_native_family_forward_major_matmul_formula_MAC_equals_2",
                        "scope": cost["scope"], "total_forward_flops_available": False}
        compressor = getattr(model, "compressor", None)
        return {"status": "success", "warmup": updates[0], "training": updates[1],
            "training_memory": training_memory, **evaluations,
            "old_length": hpo["proxy_old_length"], "profile_trial": dict(trial),
            "physical_batch_size": physical, "effective_batch_size": train["effective_batch_size"],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compressor_parameters": sum(parameter.numel() for parameter in compressor.parameters())
                if compressor is not None else 0,
            "non_item_parameters": sum(parameter.numel() for name, parameter in model.named_parameters()
                                       if not name.startswith("item_embedding.")),
            "model_initialization": model.baseline_metadata,
            "native_runtime": getattr(model, "native_runtime", {}),
            "upstream_revision": getattr(model, "upstream_revision", "repository_sasrec"),
            "representative_b1_arithmetic": arithmetic,
            "arithmetic_excludes": ["embedding_lookup", "packing", "normalization", "activation",
                                    "top_k", "loss", "backward", "optimizer"],
            "cross_family_flop_scopes_identical": False,
            "profile_weights_reused": False, "quality_used_for_selection": False,
            "approves_stability_of_other_shapes": False,
            "optimizer_state_allocated_during_all_measurements": True}
    finally:
        del model, optimizer, scaler
        _cleanup()


def conservative_baseline_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                                 users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price four fresh maximum-pass runs and only two distinct full transfers."""
    hpo, train = config["hpo"], config["training"]
    work, priced, failed, full_evaluations, unsupported_full = 0., 0, 0, 0, 0
    for variant in _FRESH_VARIANTS:
        row = profiles[f"{variant['name']}-fixed_lr"]
        if row.get("status", "success") in _PREFLIGHT_FAILURES:
            failed += 1
            continue
        rates = (row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"])
        if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
               or not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("positive finite per-family measured preflight rates are required")
        work += (hpo["max_passes"] * users / rates[0]
            + hpo["max_passes"] // hpo["validation_interval"] * evaluation_summary["validation_anchors"] / rates[1])
        if variant["kind"] in ("mamba4rec", "fuxi_linear"):
            full = row["full_evaluation"]
            if full.get("status") in ("out_of_memory", "numerical_failed"):
                unsupported_full += 1
            else:
                rate = full["anchors_per_second"]
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                    raise ValueError("positive finite supported full-transfer rate is required")
                work += evaluation_summary["validation_anchors"] / rate
                full_evaluations += 1
        priced += 1
    overhead = 120. * priced
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": 6, "new_training_configurations": 4, "reused_controls": 2,
        "priced_fresh_shapes": priced, "preflight_failed_shapes": failed,
        "fresh_full_transfer_evaluations": full_evaluations,
        "unsupported_optional_full_transfers": unsupported_full,
        "short_control_duplicate_full_evaluations": 0,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0, "test_evaluations": 0}
