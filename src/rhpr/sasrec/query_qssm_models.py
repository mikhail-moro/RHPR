"""QSSM summary-budget/memory search with shape-specific proxy cost gates.

Every arm starts from the exact original K4 initializer. Only the compressor
is replaced, matching the completed inner64 control without changing SASRec,
its full position table or the RNG used by subsequent training.
"""
from __future__ import annotations

from functools import partial
import math
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
from rhpr.sasrec.query_proxy_protocol import build_hpo_model, hpo_optimizer
from rhpr.sasrec.query_ssm import make_query_compressor


_CONTROL = ("query_ssm_m4", "dt_default")
_PREFLIGHT_FAILURES = frozenset(("preflight_numerical_failed", "preflight_out_of_memory"))


def _registered(config: dict[str, Any], variant: dict[str, Any], trial: dict[str, Any]) -> None:
    from rhpr.sasrec.query_qssm_protocol import qssm_trials, qssm_variants
    if variant not in qssm_variants(config) or trial not in qssm_trials(config, variant):
        raise ValueError("a registered QSSM variant and memory trial are required")
    if (variant.get("kind") != "query_ssm" or type(variant.get("summary_tokens")) is not int
            or variant["summary_tokens"] not in (4, 8, 16, 32)):
        raise ValueError("QSSM summary budgets must be 4, 8, 16 or 32")
    expected = {"d_inner": 64, "d_state": 16, "gate_rank": 32,
                "dt_min": .0001, "backbone_lr": .001, "compressor_lr": .001}
    for key, value in expected.items():
        actual = trial.get(key)
        numeric = type(actual) in (int, float)
        if not numeric or actual != value or (key in ("d_inner", "d_state", "gate_rank")
                                             and type(actual) is not int):
            raise ValueError(f"QSSM trial fixes {key} at {value}")
    dt_max = {"dt_default": .1, "dt_slow": .01}.get(trial.get("name"))
    if dt_max is None or type(trial.get("dt_max")) not in (int, float) or trial["dt_max"] != dt_max:
        raise ValueError("QSSM requires the registered default or slow dt initialization")


def build_qssm_model(config: dict[str, Any], variant: dict[str, Any],
                     trial: dict[str, Any], num_items: int) -> torch.nn.Module:
    """Keep the completed K4/inner64 control exactly, replacing only its shape.

    Building K4 first is intentional: all arms share the original predictor
    initializer. The replacement uses a CPU RNG fork, so dt ranges with the
    same K differ only in the specially initialized delta bias.
    """
    _registered(config, variant, trial)
    control = {"name": "query_ssm_m4", "kind": "query_ssm", "summary_tokens": 4}
    model = build_hpo_model(config, control, num_items)
    parent = config["parent"]["experiment_config"]
    dims, train = parent["model"], parent["training"]
    compressor = dict(parent["compressor"])
    compressor.update({key: trial[key] for key in
                       ("d_inner", "d_state", "gate_rank", "dt_min", "dt_max")})
    # Preserve the parent constructor's causal query source explicitly.
    compressor["query_source"] = "old_suffix"
    model.compressor = make_query_compressor(kind="query_ssm", d_model=dims["d_model"],
        summary_tokens=variant["summary_tokens"], tail_size=train["raw_tail_size"],
        n_heads=dims["n_heads"], **compressor)
    return model


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def preflight_qssm(config: dict[str, Any], cohort: Any, evaluation: Any,
                   variant: dict[str, Any], trial: dict[str, Any],
                   device: torch.device) -> dict[str, Any]:
    """Measure the exact registered proxy shape with two effective Adam steps.

    Real full-catalogue validation throughput is measured only at H1024+R128.
    Full-history inference is an optional post-selection diagnostic; measuring
    every long shape here would spend its cost before selecting any model.
    """
    from rhpr.sasrec.query_qssm_protocol import qssm_batch_arrays
    _registered(config, variant, trial)
    device = torch.device(device)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    batches = config["qssm_batches"][variant["name"]]
    physical, cap = batches["training"], hpo["proxy_old_length"] + train["raw_tail_size"]
    model = optimizer = scaler = None
    try:
        model = build_qssm_model(config, variant, trial, cohort.num_items).to(device)
        optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
        windows = _representative_windows(fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1),
                                          train["effective_batch_size"])
        builder = partial(qssm_batch_arrays, old_length=hpo["proxy_old_length"])
        updates = []
        _reset_peak(device)
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=physical, device=device, optimizer=optimizer, scaler=scaler,
                batch_builder=builder)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("QSSM preflight requires two effective Adam updates without AMP retries")
            updates.append(row)
        memory = _peak_memory(device)
        _reset_peak(device)
        try:
            proxy = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                batch_size=batches["proxy"], device=device, amp=train["amp"],
                item_chunk_size=bench["item_chunk_size"])
        except RuntimeError as error:
            if str(error) in ("evaluation produced nonfinite final states",
                              "evaluation produced nonfinite catalogue scores"):
                raise FloatingPointError(str(error)) from error
            raise
        proxy = dict(proxy, **_peak_memory(device), input_context_cap=cap,
                     optimizer_state_allocated=True)
        user = sorted(cohort.users, key=lambda row: (len(row.train), row.user_id))[len(cohort.users) // 2]
        model.eval()
        with torch.inference_mode():
            batch = _batch([user.train[:-1][-cap:]], model, device)
            _, cost = _observe_prefix_flops(model, batch, device, train["amp"])
            arithmetic = {"proxy": {"input_tokens": int(batch[0].shape[1]), **cost}}
            del batch
        parameters = list(model.named_parameters())
        return {"status": "success", "warmup": updates[0], "training": updates[1],
            "training_memory": memory, "proxy_evaluation": proxy,
            "full_evaluation": {"status": "deferred_to_selected_finalists", "profiled": False,
                "batch_size": batches["full"], "quality_used_for_selection": False},
            "old_length": hpo["proxy_old_length"], "profile_trial": dict(trial),
            "summary_tokens": variant["summary_tokens"], "physical_batch_size": physical,
            "effective_batch_size": train["effective_batch_size"],
            "total_parameters": sum(parameter.numel() for _, parameter in parameters),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters()),
            "non_item_parameters": sum(parameter.numel() for name, parameter in parameters
                                       if not name.startswith("item_embedding.")),
            "processing_parameters": sum(parameter.numel() for name, parameter in parameters
                if not name.startswith(("item_embedding.", "position_embedding."))),
            "representative_b1_arithmetic": arithmetic,
            "arithmetic_method": "observed Linear/MHA shapes and dense SASRec matmuls; analytical scan recurrence; MAC=2",
            "arithmetic_excludes": ["embedding lookup", "packing", "normalization", "softmax", "activation", "top-k",
                                    "loss", "backward", "optimizer"],
            "scan_exp_and_softplus_reported_separately": True,
            "profile_weights_reused": False, "quality_used_for_selection": False,
            "approves_stability_of_other_shapes": False,
            "optimizer_state_allocated_during_all_measurements": True,
            "automatic_precision_or_batch_fallback": False}
    finally:
        del model, optimizer, scaler
        _cleanup()


def conservative_qssm_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                             users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price all seven fresh shapes at 60 passes plus an explicit full reserve.

    Full transfer has no measured rate before selection. The registered reserve
    is explicitly an allowance, not a throughput prediction or capacity claim;
    the runner must enforce that optional-stage limit without losing primary
    proxy results. No trial or pass is silently removed to make the gate fit.
    """
    from rhpr.sasrec.query_qssm_protocol import qssm_trials, qssm_variants
    hpo, train = config["hpo"], config["training"]
    work, priced, failed, registered = 0., 0, 0, 0
    for variant in qssm_variants(config):
        for trial in qssm_trials(config, variant):
            registered += 1
            if (variant["name"], trial["name"]) == _CONTROL:
                continue
            row = profiles[f"{variant['name']}-{trial['name']}"]
            if row.get("status", "success") in _PREFLIGHT_FAILURES:
                failed += 1
                continue
            rates = (row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"])
            if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError("positive finite exact-shape preflight rates are required")
            work += (hpo["max_passes"] * users / rates[0]
                + hpo["max_passes"] // hpo["validation_interval"] * evaluation_summary["validation_anchors"] / rates[1])
            priced += 1
    reserve = config["full_transfer"]["reserve_seconds"]
    if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
        raise ValueError("a positive finite optional full-transfer reserve is required")
    overhead = 120. * priced
    total = elapsed + train["budget_safety_factor"] * (work + overhead + reserve)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "optional_full_transfer_reserve_seconds": reserve,
        "optional_full_transfer_reserve_is_measured_estimate": False,
        "optional_full_transfer_requires_runtime_budget_guard": True,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": registered, "new_training_configurations": registered - 1,
        "reused_controls": 1, "priced_fresh_shapes": priced, "preflight_failed_shapes": failed,
        "maximum_fresh_full_transfer_evaluations": config["full_transfer"]["new_configurations"],
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0, "test_evaluations": 0}
