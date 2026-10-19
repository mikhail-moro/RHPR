"""One-feature-at-a-time QSSM ablations at H1024, with a reused control.

Only the three fresh feature recipes consume training quota.  Every recipe has
its own two-update and catalogue preflight, and every successful feature gets a
paired comparison with the same saved short-history control.
"""
from __future__ import annotations

from functools import partial
import gzip
import json
import math
from pathlib import Path
from typing import Any

import torch

from rhpr.metrics import validate_metrics
from rhpr.sasrec.daily_quality import paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_quality import _cleanup, _json, _scaler
from rhpr.sasrec.long_quality_benchmark import _batch
from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
from rhpr.sasrec.query_prefix_benchmark import _observe_prefix_flops
from rhpr.sasrec.query_prefix_pilot import _profile_daily_rate, _representative_windows, train_prefix_pass
from rhpr.sasrec.query_prefix_protocol import fixed_prefix_schedule
from rhpr.sasrec.query_proxy_protocol import hpo_optimizer


_CONTROL = "base"
_FAILED_PREFLIGHTS = frozenset(("preflight_numerical_failed", "preflight_out_of_memory"))
_REPORT = "query-mamba-ablation.json"


def _peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def preflight_mamba(config: dict[str, Any], cohort: Any, evaluation: Any,
                    variant: dict[str, Any], trial: dict[str, Any],
                    device: torch.device) -> dict[str, Any]:
    """Time this exact fresh shape; never build or infer a full-history arm."""
    from rhpr.sasrec.query_mamba_models import audit_mamba_feature, build_mamba_model
    from rhpr.sasrec.query_mamba_protocol import mamba_batch_arrays, mamba_trials
    if trial not in mamba_trials(config, variant) or trial["name"] == _CONTROL:
        raise ValueError("preflight requires a registered fresh Mamba-feature trial")
    device = torch.device(device)
    train, bench, hpo = config["training"], config["benchmark"], config["hpo"]
    batches = config["mamba_batches"][variant["name"]]
    cap = hpo["proxy_old_length"] + train["raw_tail_size"]
    model = optimizer = scaler = None
    try:
        native_audit = audit_mamba_feature(trial["feature"], device)
        model = build_mamba_model(config, variant, trial, cohort.num_items).to(device)
        optimizer, scaler = hpo_optimizer(model, train, trial), _scaler(device, train)
        windows = _representative_windows(fixed_prefix_schedule(cohort, seed=config["seed"], epoch=1),
                                          train["effective_batch_size"])
        builder = partial(mamba_batch_arrays, old_length=hpo["proxy_old_length"])
        updates = []
        _reset_peak(device)
        for _ in range(2):
            row = train_prefix_pass(model, cohort, windows, config=config, variant=variant,
                physical_batch=batches["training"], device=device, optimizer=optimizer,
                scaler=scaler, batch_builder=builder)
            if row["successful_updates"] != 1 or row["amp_skipped_update_attempts"] or not optimizer.state:
                raise FloatingPointError("Mamba-feature preflight requires two effective Adam updates without AMP retries")
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
            feature_cost = model.compressor.feature_compute_flops(
                old_length=max(1, int(batch[1].sum(dim=1).max()) - train["raw_tail_size"]),
                batch_size=len(batch[0]))
            # Linear hooks already include the optional z projection.  Only
            # convolution and endpoint elementwise arithmetic are missing.
            increment = feature_cost["unhooked_feature_flops"]
            for component in ("compressor", "history", "full"):
                cost[component + "_flops"] += increment
                cost[component + "_arithmetic_including_scan_flops"] += increment
            arithmetic = {"input_tokens": int(batch[0].shape[1]), **cost,
                          "feature_increment": feature_cost}
            del batch
        parameters = list(model.named_parameters())
        return {"status": "success", "native_feature_audit": native_audit,
            "warmup": updates[0], "training": updates[1],
            "training_memory": memory, "proxy_evaluation": proxy,
            "full_evaluation": {"status": "not_registered", "profiled": False},
            "old_length": hpo["proxy_old_length"], "profile_trial": dict(trial),
            "summary_tokens": variant["summary_tokens"], "physical_batch_size": batches["training"],
            "effective_batch_size": train["effective_batch_size"],
            "total_parameters": sum(parameter.numel() for _, parameter in parameters),
            "compressor_parameters": sum(parameter.numel() for parameter in model.compressor.parameters()),
            "non_item_parameters": sum(parameter.numel() for name, parameter in parameters
                                       if not name.startswith("item_embedding.")),
            "processing_parameters": sum(parameter.numel() for name, parameter in parameters
                if not name.startswith(("item_embedding.", "position_embedding."))),
            "representative_b1_arithmetic": {"proxy": arithmetic},
            "arithmetic_method": "observed Linear/MHA shapes and dense SASRec matmuls; analytical scan and explicit feature increments; MAC=2",
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


def conservative_mamba_budget(config: dict[str, Any], profiles: dict[str, Any], elapsed: float,
                              users: int, evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    """Price every fresh shape at maximum60, with no full-history reserve."""
    from rhpr.sasrec.query_mamba_protocol import mamba_trials, mamba_variants
    hpo, train = config["hpo"], config["training"]
    work, priced, failed, registered = 0., 0, 0, 0
    for variant in mamba_variants(config):
        for trial in mamba_trials(config, variant):
            registered += 1
            if trial["name"] == _CONTROL:
                continue
            row = profiles[f"{variant['name']}-{trial['name']}"]
            if row.get("status", "success") in _FAILED_PREFLIGHTS:
                failed += 1
                continue
            rates = (row["training"]["windows_per_second"], row["proxy_evaluation"]["anchors_per_second"])
            if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in rates):
                raise ValueError("positive finite exact-shape preflight rates are required")
            work += (hpo["max_passes"] * users / rates[0]
                + hpo["max_passes"] // hpo["validation_interval"] * evaluation_summary["validation_anchors"] / rates[1])
            priced += 1
    overhead = 120. * priced
    total = elapsed + train["budget_safety_factor"] * (work + overhead)
    return {"measured_elapsed_seconds": elapsed, "estimated_remaining_work_seconds": work,
        "fixed_checkpoint_statistics_overhead_seconds": overhead,
        "safety_factor": train["budget_safety_factor"], "conservative_total_seconds": total,
        "registered_budget_seconds": train["budget_seconds"], "fits": total <= train["budget_seconds"],
        "registered_configurations": registered, "new_training_configurations": registered - 1,
        "reused_controls": 1, "priced_fresh_shapes": priced, "preflight_failed_shapes": failed,
        "prices_each_shape_at_its_measured_rate": True, "prices_all_trials_at_maximum_passes": True,
        "auto_reduces_trials_or_passes": False, "full_training_models": 0,
        "full_history_inference_evaluations": 0, "test_evaluations": 0}


def _protocol_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    return {"training": "H1024+R128, minimum20 maximum60 passes; validation every10; patience2",
        "architecture": "QSSM K32 inner64/state16/rank32 slow-dt; independent conv4, output gate, skip D additions; fixed SASRec",
        "family_selection": "best finite short-validation macro-user NDCG10; ties preserve registered order",
        "full_history": "not registered: no full-history training, profiling or validation",
        "control_reuse": "verified selected K32 slow-dt source; copied H1024 validation ranks, no repeated control training or inference",
        "targets": "original users, endpoints, last128 labels, negative identities, old-suffix query and PE32768 initializer retained",
        "ablation": "each fresh arm adds exactly one feature; no cumulative combinations",
        "comparison": "all three best-validation feature checkpoints compared with saved base; exploratory one-seed validation comparisons",
        "multiple_comparisons": "95% user-bootstrap intervals are unadjusted; 98.333% Bonferroni intervals cover three comparisons within each cutoff, not checkpoint or architecture selection",
        "test": "no test ranks deserialized; no test evaluation or selection"}


def _no_full_transfer(_config: dict[str, Any], _variants: list[dict[str, Any]],
                      _trials: dict[str, list[dict[str, Any]]], report: dict[str, Any]
                      ) -> set[tuple[str, str]]:
    report["full_transfer_selection"] = {"registered": False, "selected": [],
        "full_history_quality_used_for_selection": False, "test_evaluations": 0}
    return set()


def _finish_ablation(config: dict[str, Any], output_dir: Path,
                      metrics: dict[str, Any]) -> dict[str, int | float | str | bool]:
    """Expose every paired feature result, including partial-suite failures."""
    from rhpr.sasrec.query_mamba_protocol import mamba_trials, mamba_variants, select_mamba_candidate
    report = json.loads((output_dir / _REPORT).read_text())
    report.update(complete=False, ablation_complete=False, ablation_statistics_complete=False)
    metrics["quality_complete"] = False
    _json(output_dir / _REPORT, report)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    comparisons, failures, successes = {}, [], 0
    for variant in mamba_variants(config):
        family = report["trials"][variant["name"]]
        control = family[_CONTROL]
        with gzip.open(output_dir / control["best_validation_rank_file"], "rt", encoding="utf-8") as stream:
            base = json.load(stream)
        for trial in mamba_trials(config, variant):
            if trial["name"] == _CONTROL:
                continue
            result = family[trial["name"]]
            key = variant["name"] + "-" + trial["name"]
            if result["status"] != "success" or select_mamba_candidate([trial], [result]) is None:
                failures.append(key)
                comparisons[key] = {"status": result["status"], "comparison_available": False,
                    "failure": result.get("failure", {})}
                continue
            with gzip.open(output_dir / result["best_validation_rank_file"], "rt", encoding="utf-8") as stream:
                candidate = json.load(stream)
            comparisons[key] = {"status": "success", "comparison_available": True,
                "reference": "verified historical K32 slow-dt H1024 control", "feature": trial["feature"],
                "evaluation_condition": "H1024+R128 for both models; identical catalogue and daily anchors",
                "training_seed_uncertainty_included": False, "selection_uncertainty_included": False,
                "exploratory_validation_selection": True,
                "cutoffs": {str(k): {"unadjusted_95": paired_daily_ndcg_bootstrap(base, candidate,
                    k=k, seed=config["seed"], draws=config["hpo"]["bootstrap_draws"]),
                    "bonferroni_three_comparisons": paired_daily_ndcg_bootstrap(base, candidate,
                    k=k, seed=config["seed"], draws=config["hpo"]["bootstrap_draws"], confidence=1 - .05 / 3)}
                    for k in (10, 100)}}
            successes += 1
            for k in (10, 100):
                metrics[f"{trial['name']}_ndcg{k}"] = result["best_validation"][f"ndcg{k}"]
                metrics[f"{trial['name']}_minus_base_ndcg{k}"] = (
                    comparisons[key]["cutoffs"][str(k)]["unadjusted_95"]["delta_ndcg"])
        case = report.get("cases", {}).get(variant["name"], {})
        # Full-history comparisons are never part of this registration.
        case.pop("paired_parent_full_validation_contrasts", None)
    report.update(feature_vs_base=comparisons, fresh_failed_trials=failures,
        runner_completed=True, complete=not failures, ablation_complete=not failures,
        ablation_statistics_complete=True)
    metrics.update(quality_complete=not failures, fresh_successful_trials=successes,
                   fresh_failed_trials=len(failures), full_history_inference_evaluations=0)
    _json(output_dir / _REPORT, report)
    selection_path = output_dir / "final-selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(ablation_complete=not failures, fresh_failed_trials=failures,
        all_feature_comparisons_artifact=_REPORT, full_history_evaluation_registered=False)
    _json(selection_path, selection)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    return validate_metrics(metrics)


def run_query_mamba_ablation(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_mamba_models import build_mamba_model
    from rhpr.sasrec.query_mamba_protocol import (
        load_mamba_controls, mamba_batch_arrays, mamba_evaluation_batch, mamba_trials,
        mamba_variants, select_mamba_candidate, validate_mamba_config,
    )
    definition = QuerySearchDefinition(validate=validate_mamba_config, variants=mamba_variants,
        trials=mamba_trials, build_model=build_mamba_model, preflight=preflight_mamba,
        budget=conservative_mamba_budget, batch_arrays=mamba_batch_arrays,
        training_batch=lambda current, variant: current["mamba_batches"][variant["name"]]["training"],
        evaluation_batch=mamba_evaluation_batch, protocol_metadata=_protocol_metadata,
        select_candidate=select_mamba_candidate, full_transfer_candidates=_no_full_transfer,
        retained_trial_resumes=2, paired_short_validation=True,
        report_file=_REPORT, phase="mamba_ablation", checkpoint_stage="qssm_mamba_feature_ablation",
        profile_each_trial=True, load_controls=load_mamba_controls, isolate_preflight_failures=True,
        hpo_performed=False)
    output_dir = Path(output_dir)
    metrics = run_registered_query_search(config, output_dir, definition=definition)
    return _finish_ablation(config, output_dir, metrics)
