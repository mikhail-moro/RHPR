"""Fresh QSSM/FuXi comparison with shared changing train-only target blocks.

Historical fixed-block results are saved references, never fresh controls. The
primary contrast always compares the two new rotating-window checkpoints even
when a historical checkpoint has a higher validation score within its family.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from rhpr.metrics import validate_metrics
from rhpr.sasrec.daily_quality import _aggregate, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _json
from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
from rhpr.sasrec.query_baseline_screen import select_native_short_candidate


_REPORT = "query-rotating-windows.json"
_COVERAGE = "rotating-window-coverage.json"
_DESCRIPTORS = "rotating-window-descriptors.json.gz"
_FRESH = "rotating"
_CONTROL = "historical_fixed"
_QSSM = "query_ssm_m32"
_FUXI = "fuxi_linear"


def _protocol_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    return {"training": "both fresh models H1024+R128; minimum20 maximum60 passes, validation every10, patience2",
        "targets": "one128-target block/user/pass; first pass is the original latest block; subsequent train-only blocks visit a shared user-specific permutation without replacement within each cycle",
        "architecture": "unchanged selected K32/slow-dt QSSM plus SASRec versus unchanged native FuXi-Linear; distinct capacities and native initializers retained",
        "family_selection": "earliest best native-short macro-user NDCG10; saved historical results can win descriptive family selection but never replace either fresh primary arm",
        "primary_comparison": "new rotating QSSM versus new rotating FuXi using each fresh arm's selected native-short checkpoint",
        "full_history": "not registered: no full-history training, profiling, inference transfer or selection",
        "control_reuse": "verified historical fixed-block QSSM and FuXi results and portable weights; no control training or repeated inference",
        "historical_comparisons": "exploratory saved fixed-block versus fresh rotating per family; FuXi also changes pre-fix mixed-precision recurrence to current FP32 arithmetic",
        "comparison": "matched catalogue, daily anchors, changing target positions, negative draws, optimizer-update budget and stopping; model capacities and throughput differ",
        "uncertainty": "paired95% user-bootstrap intervals; no training-seed or checkpoint-selection uncertainty; secondary slices are exploratory and unadjusted",
        "coverage": "shared registered window descriptors and exact unique target-position coverage; actual completed passes reported separately for each fresh arm",
        "test": "no test ranks deserialized; no test evaluation or selection"}


def _no_full_transfer(_config: dict[str, Any], _variants: list[dict[str, Any]],
                      _trials: dict[str, list[dict[str, Any]]], report: dict[str, Any]
                      ) -> set[tuple[str, str]]:
    report["full_transfer_selection"] = {"registered": False, "selected": [],
        "full_history_quality_used_for_selection": False, "test_evaluations": 0}
    return set()


def _read_ranks(output_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    with gzip.open(output_dir / result["best_validation_rank_file"], "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _slices(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Keep the primary denominator within each subset: macro-user NDCG."""
    earliest: dict[int, int] = {}
    for record in records:
        uid, day = record["user_id"], record["day"]
        earliest[uid] = min(earliest.get(uid, day), day)
    return {"first_anchor": [r for r in records if r["day"] == earliest[r["user_id"]]],
        "later_anchors": [r for r in records if r["day"] != earliest[r["user_id"]]],
        **{f"day_{day}": [r for r in records if r["day"] == day]
           for day in sorted({r["day"] for r in records})}}


def _paired(reference: list[dict[str, Any]], candidate: list[dict[str, Any]],
            config: dict[str, Any]) -> dict[str, Any]:
    return {str(k): paired_daily_ndcg_bootstrap(reference, candidate, k=k,
        seed=config["seed"], draws=config["hpo"]["bootstrap_draws"], confidence=.95)
        for k in (10, 100)}


def _fresh_checkpoint(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in ("trial", "best_epoch", "passes_completed",
        "best_validation", "best_validation_rank_file", "best_checkpoint",
        "resumable_checkpoint", "resume_checkpoint_retained") if key in result}


def _finish_rotating(config: dict[str, Any], output_dir: Path, metrics: dict[str, Any],
                     schedules: list[Any], cohort: Any) -> dict[str, int | float | str | bool]:
    """Expose fresh arms explicitly; historical winners cannot conceal failures."""
    from rhpr.sasrec.query_rotating_protocol import rotating_schedule_summary, rotating_variants

    report = json.loads((output_dir / _REPORT).read_text())
    report.update(complete=False, rotating_comparison_complete=False,
        rotating_statistics_complete=False, fresh_cases={}, fresh_selected_trials={})
    metrics["quality_complete"] = False
    # Persist invalidation before reading ranks or calculating statistics. A
    # stale generic historical-family winner must never look like a new result.
    _json(output_dir / _REPORT, report)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    selection_path = output_dir / "final-selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(rotating_comparison_complete=False, fresh_selected_trials={},
        fresh_checkpoints={}, primary_comparison_available=False)
    _json(selection_path, selection)

    failures, fresh_ranks, historical_comparisons, diagnostics, trajectories = [], {}, {}, {}, {}
    coverage_path = output_dir / _COVERAGE
    coverage = json.loads(coverage_path.read_text())
    coverage["actual_completed_training"] = {}
    coverage["actual_selected_checkpoint"] = {}
    for variant in rotating_variants(config):
        name = variant["name"]
        result = report["trials"][name][_FRESH]
        completed = result.get("passes_completed", 0)
        if type(completed) is not int or not 0 <= completed <= len(schedules):
            raise ValueError("fresh completed passes must match the registered shared schedules")
        windows = tuple(window for row in schedules[:completed] for window in row)
        if windows:
            actual = rotating_schedule_summary(cohort, windows, config)
        else:
            actual = {"windows": 0, "epochs": [], "unique_target_positions": 0,
                "target_presentations": 0, "unique_endpoints": 0,
                "repeated_endpoint_presentations": 0}
        actual.update(passes_completed=completed, trial_name=_FRESH,
            completed_passes_only=True, failed_partial_pass_included=False,
            status=result["status"])
        coverage["actual_completed_training"][name] = actual
        metrics[f"{name}_rotating_unique_supervised_target_positions"] = actual["unique_target_positions"]
        best_epoch = result.get("best_epoch", 0)
        if type(best_epoch) is not int or not 0 <= best_epoch <= completed:
            raise ValueError("fresh selected epoch must lie within completed registered passes")
        selected_windows = tuple(window for row in schedules[:best_epoch] for window in row)
        if selected_windows:
            selected_coverage = rotating_schedule_summary(cohort, selected_windows, config)
        else:
            selected_coverage = {"windows": 0, "epochs": [], "unique_target_positions": 0,
                "target_presentations": 0, "unique_endpoints": 0,
                "repeated_endpoint_presentations": 0}
        selected_updates = sum(row["successful_updates"] for row in result.get("passes", [])
                               if row["epoch"] <= best_epoch)
        selected_coverage.update(selected_checkpoint_available=bool(best_epoch),
            selected_checkpoint_epoch=best_epoch, successful_updates_to_selected_checkpoint=selected_updates,
            eligible_for_primary_comparison=result.get("status") == "success",
            trial_name=_FRESH, completed_training_passes=completed)
        coverage["actual_selected_checkpoint"][name] = selected_coverage
        metrics[f"{name}_rotating_selected_checkpoint_unique_supervised_target_positions"] = (
            selected_coverage["unique_target_positions"])
        trajectories[name] = [{"epoch": row["epoch"], "validation": {
            key: value for key, value in row["validation"].items()
            if key in ("ndcg10", "ndcg100", "first_anchor_ndcg10", "first_anchor_ndcg100")}}
            for row in result.get("passes", []) if "validation" in row]
        trial = result["trial"]
        if result.get("status") != "success" or select_native_short_candidate([trial], [result]) is None:
            failures.append(name + "-" + _FRESH)
            historical_comparisons[name] = {"comparison_available": False,
                "status": result["status"], "failure": result.get("failure", {})}
            continue
        report["fresh_selected_trials"][name] = dict(trial)
        report["fresh_cases"][name] = dict(_fresh_checkpoint(result),
            selected_checkpoint_training_exposure={
                "selected_checkpoint_epoch": best_epoch,
                "successful_updates_to_selected_checkpoint": selected_updates,
                "unique_supervised_target_positions": selected_coverage["unique_target_positions"],
                "target_presentations": selected_coverage["target_presentations"],
                "completed_training_passes": completed})
        ranks = _read_ranks(output_dir, result)
        fresh_ranks[name] = ranks
        slices = _slices(ranks)
        diagnostics[name] = {label: {"available": bool(rows),
            **({"metrics": _aggregate(rows)[0]} if rows else {})}
            for label, rows in slices.items()}
        reference = _read_ranks(output_dir, report["trials"][name][_CONTROL])
        historical_comparisons[name] = {"comparison_available": True,
            "reference": name + "-" + _CONTROL, "candidate": name + "-" + _FRESH,
            "historical_control_retrained": False, "exploratory_validation_selection": True,
            "training_seed_uncertainty_included": False, "selection_uncertainty_included": False,
            "schedule_effect_is_causally_isolated": False,
            "only_registered_training_change_is_schedule": name != _FUXI,
            "arithmetic_matches_source": name != _FUXI,
            "comparison_scope": "total effect of changed schedule and resulting stopping/checkpoint selection under same policy; historical comparison exploratory",
            "precision_caveat": ("saved FuXi used pre-fix FP16 recurrence/reads; fresh uses current FP32 arithmetic"
                                 if name == _FUXI else "same registered arithmetic"),
            "cutoffs": _paired(reference, ranks, config)}
        for k in (10, 100):
            metrics[f"{name}_rotating_ndcg{k}"] = result["best_validation"][f"ndcg{k}"]
            metrics[f"{name}_rotating_minus_historical_ndcg{k}"] = (
                historical_comparisons[name]["cutoffs"][str(k)]["delta_ndcg"])

    primary: dict[str, Any] = {"comparison_available": not failures,
        "reference": _FUXI + "-" + _FRESH, "candidate": _QSSM + "-" + _FRESH,
        "evaluation_condition": "H1024+R128; identical full catalogue and daily anchors",
        "checkpoint_selection": "earliest strictly best native-short macro-user NDCG10 of each fresh arm",
        "historical_family_winners_used": False, "exploratory_validation_selection": True,
        "training_seed_uncertainty_included": False, "selection_uncertainty_included": False}
    primary["fresh_training_exposure"] = {name: row["selected_checkpoint_training_exposure"]
        for name, row in report["fresh_cases"].items()}
    primary["same_per_pass_target_and_update_budget"] = True
    primary["selected_checkpoint_target_coverage_can_differ_under_early_stopping"] = True
    if not failures:
        primary["cutoffs"] = _paired(fresh_ranks[_FUXI], fresh_ranks[_QSSM], config)
        reference_slices, candidate_slices = _slices(fresh_ranks[_FUXI]), _slices(fresh_ranks[_QSSM])
        if reference_slices.keys() != candidate_slices.keys():
            raise ValueError("fresh validation slices must share exactly the same days")
        primary["secondary_slice_contrasts"] = {label: {"comparison_available": bool(reference),
            "exploratory_unadjusted": True,
            **({"cutoffs": _paired(reference, candidate_slices[label], config)} if reference else {})}
            for label, reference in reference_slices.items()}
        for k in (10, 100):
            metrics[f"rotating_qssm_minus_fuxi_ndcg{k}"] = primary["cutoffs"][str(k)]["delta_ndcg"]
    else:
        primary["fresh_failed_trials"] = failures
    report.update(primary_fresh_comparison=primary, fresh_vs_historical=historical_comparisons,
        best_checkpoint_daily_diagnostics=diagnostics, validation_trajectories=trajectories,
        fresh_failed_trials=failures, runner_completed=True, rotating_statistics_complete=True,
        rotating_comparison_complete=not failures, complete=not failures,
        training_coverage_artifact=_COVERAGE, shared_window_descriptors_artifact=_DESCRIPTORS)
    metrics.update(quality_complete=not failures, fresh_successful_trials=len(fresh_ranks),
        fresh_failed_trials=len(failures), full_history_inference_evaluations=0)
    selection.update(fresh_selected_trials=report["fresh_selected_trials"],
        fresh_checkpoints={name: row["best_checkpoint"] for name, row in report["fresh_cases"].items()},
        fresh_cases=report["fresh_cases"], primary_comparison_available=not failures,
        primary_comparison={"reference": primary["reference"], "candidate": primary["candidate"]},
        historical_winners_do_not_replace_fresh_comparison=True,
        rotating_comparison_complete=not failures, fresh_failed_trials=failures,
        full_history_evaluation_registered=False, all_comparisons_artifact=_REPORT)
    _json(coverage_path, coverage)
    _json(output_dir / _REPORT, report)
    _json(selection_path, selection)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    return validate_metrics(metrics)


def run_query_rotating_windows(config: dict[str, Any], output_dir: Path
                               ) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_rotating_models import (
        build_rotating_model, conservative_rotating_budget, preflight_rotating, rotating_optimizer,
    )
    from rhpr.sasrec.query_rotating_protocol import (
        load_rotating_controls, rotating_batch_arrays, rotating_evaluation_batch,
        rotating_schedule, rotating_schedule_summary, rotating_trials, rotating_variants,
        validate_rotating_config,
    )

    output_dir = Path(output_dir)
    schedules, data = [], {}

    def schedule(cohort: Any, *, seed: int, epoch: int) -> Any:
        windows = rotating_schedule(cohort, seed=seed, epoch=epoch)
        data["cohort"] = cohort
        schedules.append(windows)
        return windows

    def summary(cohort: Any, windows: Any, current: dict[str, Any]) -> dict[str, Any]:
        result = rotating_schedule_summary(cohort, windows, current)
        if result["epochs"] == [current["hpo"]["max_passes"]]:
            registered = rotating_schedule_summary(cohort,
                tuple(window for row in schedules for window in row), current)
            _json(output_dir / _COVERAGE, {"registered_maximum_passes": registered,
                "shared_between_fresh_models": True, "actual_completed_training": {},
                "training_is_exhaustive_event_epoch": False,
                "target_blocks_do_not_overlap_within_each_user_cycle": True,
                "descriptors_artifact": _DESCRIPTORS})
            _gzip_json(output_dir / _DESCRIPTORS, {"columns": ["epoch", "user_id", "end", "negative_seed"],
                "descriptors": [[w.epoch, w.user_id, w.end, w.negative_seed]
                                for row in schedules for w in row],
                "end_semantics": "exclusive input boundary; final positive target is train[end]",
                "target_positions": "train[end-127:end+1]", "shared_between_fresh_models": True,
                "contains_raw_item_or_timestamp_arrays": False})
        return result

    definition = QuerySearchDefinition(validate=validate_rotating_config, variants=rotating_variants,
        trials=rotating_trials, build_model=build_rotating_model, optimizer=rotating_optimizer,
        preflight=preflight_rotating, budget=conservative_rotating_budget,
        batch_arrays=rotating_batch_arrays,
        training_batch=lambda current, variant: current["rotating_batches"][variant["name"]]["training"],
        evaluation_batch=rotating_evaluation_batch, protocol_metadata=_protocol_metadata,
        training_schedule=schedule, schedule_summary=summary,
        select_candidate=select_native_short_candidate, full_transfer_candidates=_no_full_transfer,
        retained_trial_resumes=2, paired_short_validation=True, hpo_performed=False,
        report_file=_REPORT, phase="rotating_windows", checkpoint_stage="rotating_train_windows",
        profile_each_trial=True, load_controls=load_rotating_controls, isolate_preflight_failures=True)
    metrics = run_registered_query_search(config, output_dir, definition=definition)
    return _finish_rotating(config, output_dir, metrics, schedules, data["cohort"])
