"""Fixed-LR compressor architecture search reusing two verified 1K controls.

Only four new architectures train. Every new shape is timed with real Adam
updates and proxy/full catalogue inference before the prospective budget gate.
"""
from __future__ import annotations

from functools import partial
import math
from pathlib import Path
from typing import Any

import torch

from rhpr.sasrec.long_quality import _cleanup, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_optimizer, hpo_variants


_CONTROL_NAMES = frozenset(("ca_base", "q_base"))


def _architecture_trials(config: dict[str, Any], variant: dict[str, Any]) -> list[dict[str, Any]]:
    from rhpr.sasrec.query_architecture_protocol import architecture_trials
    return architecture_trials(config, variant)


def _build_architecture_model(config: dict[str, Any], variant: dict[str, Any],
                              trial: dict[str, Any], num_items: int) -> torch.nn.Module:
    from rhpr.sasrec.query_architecture_models import build_architecture_model
    return build_architecture_model(config, variant, trial, num_items)


def _preflight_architecture(config: dict[str, Any], cohort: Any, evaluation: Any,
                            variant: dict[str, Any], trial: dict[str, Any],
                            device: torch.device) -> dict[str, Any]:
    """Audit this exact shape with two effective updates and both input lengths.

    Best/control weights are never used. FLOPs describe B1 median train history;
    throughput measures real daily validation, including exact catalogue ranking.
    """
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    model = optimizer = scaler = None
    try:
        model = _build_architecture_model(config, variant, trial, cohort.num_items).to(device)
        optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
        windows = _representative_windows(fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1),
                                          train["effective_batch_size"])
        builder = partial(hpo_batch_arrays, old_length=hpo["proxy_old_length"])
        updates = []
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=hpo["physical_batch_size"], device=device, optimizer=optimizer,
                scaler=scaler, batch_builder=builder)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("architecture preflight requires two effective Adam updates without AMP retries")
            updates.append(row)
        evaluations = {}
        for phase, cap in (("proxy_evaluation", hpo["proxy_old_length"] + train["raw_tail_size"]),
                           ("full_evaluation", train["serving_context_length"])):
            try:
                evaluations[phase] = _profile_daily_rate(model, cohort, evaluation, context_length=cap,
                    batch_size=bench["evaluation_batch_size"], device=device, amp=train["amp"],
                    item_chunk_size=bench["item_chunk_size"])
            except RuntimeError as error:
                if str(error) in ("evaluation produced nonfinite final states",
                                  "evaluation produced nonfinite catalogue scores"):
                    raise FloatingPointError(str(error)) from error
                raise
        arithmetic = {}
        user = sorted(cohort.users, key=lambda row: (len(row.train), row.user_id))[len(cohort.users) // 2]
        context = user.train[:-1]
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for phase, cap in (("proxy", hpo["proxy_old_length"] + train["raw_tail_size"]),
                                   ("full", train["serving_context_length"])):
                    batch = _batch([context[-cap:]], model, device)
                    _, cost = _observe_prefix_flops(model, batch, device, train["amp"])
                    arithmetic[phase] = {"input_tokens": int(batch[0].shape[1]), **cost}
                    del batch
        finally:
            model.train(was_training)
        return {"status": "success", "warmup": updates[0], "training": updates[1], **evaluations,
            "old_length": hpo["proxy_old_length"], "profile_trial": dict(trial),
            "physical_batch_size": hpo["physical_batch_size"],
            "effective_batch_size": train["effective_batch_size"],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters()),
            "representative_b1_arithmetic": arithmetic,
            "arithmetic_method": "observed Linear/MHA shapes and nominal dense SASRec matmuls; scan recurrence analytical; MAC=2",
            "arithmetic_excludes": ["embedding lookup", "packing", "normalization", "softmax", "activation", "top-k"],
            "scan_exp_and_softplus_reported_separately": True,
            "profile_weights_reused": False, "quality_used_for_selection": False,
            "approves_stability_of_other_shapes": False,
            "optimizer_state_allocated_during_all_measurements": True}
    finally:
        del model, optimizer, scaler
        _cleanup()


def conservative_architecture_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                                     users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Charge each fresh shape its own maximum60 cost; reused controls cost no training."""
    hpo, train = config["hpo"], config["training"]
    work, priced, preflight_failed = 0., 0, 0
    for variant in hpo_variants(config):
        for trial in _architecture_trials(config, variant):
            if trial["name"] in _CONTROL_NAMES:
                continue
            row = profiles[f"{variant['name']}-{trial['name']}"]
            if row.get("status", "success") in ("preflight_numerical_failed", "preflight_out_of_memory"):
                preflight_failed += 1
                continue
            rates = (row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"],
                     row["full_evaluation"]["anchors_per_second"])
            if any(isinstance(rate, bool) or not isinstance(rate, (int, float))
                   or not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError("positive finite per-shape measured preflight rates are required")
            work += (hpo["max_passes"] * users / rates[0]
                + hpo["max_passes"] // hpo["validation_interval"] * evaluation_summary["validation_anchors"] / rates[1]
                + evaluation_summary["validation_anchors"] / rates[2])
            priced += 1
    overhead = 120. * priced
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": 6, "new_training_configurations": 4, "reused_controls": 2,
        "priced_fresh_shapes": priced, "preflight_failed_shapes": preflight_failed,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0, "test_evaluations": 0}


def run_query_architecture_search(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_architecture_protocol import load_architecture_controls, validate_architecture_config
    definition = QuerySearchDefinition(validate=validate_architecture_config, trials=_architecture_trials,
        build_model=_build_architecture_model, preflight=_preflight_architecture,
        budget=conservative_architecture_budget, report_file="query-compressor-architecture.json",
        phase="architecture", checkpoint_stage="compressor_architecture_search", profile_each_trial=True,
        load_controls=load_architecture_controls, isolate_preflight_failures=True)
    return run_registered_query_search(config, output_dir, definition=definition)
