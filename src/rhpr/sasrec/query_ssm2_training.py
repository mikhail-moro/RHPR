"""One fresh state128 QSSM2 quality run, with unchanged rotating supervision."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rhpr.metrics import validate_metrics
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _json
from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
from rhpr.sasrec.query_baseline_screen import select_native_short_candidate
from rhpr.sasrec.query_rotating_protocol import rotating_schedule, rotating_schedule_summary
from rhpr.sasrec.query_ssm2 import audit_native_query_ssm2
from rhpr.sasrec.query_ssm2_training_models import (
    build_ssm2_training_model, conservative_ssm2_training_budget, preflight_ssm2_training,
    ssm2_training_optimizer,
)
from rhpr.sasrec.query_ssm2_training_protocol import (
    TRIAL, VARIANT, ssm2_training_batch_arrays, ssm2_training_evaluation_batch,
    ssm2_training_metadata, ssm2_training_trials, ssm2_training_variants, validate_ssm2_training_config,
)


_REPORT = "query-ssm2-training.json"
_COVERAGE = "ssm2-training-coverage.json"


def _no_full_transfer(_config: Any, _variants: Any, _trials: Any, report: dict[str, Any]) -> set[Any]:
    report["full_transfer_selection"] = {"registered": False, "selected": [],
        "full_history_quality_used_for_selection": False, "test_evaluations": 0}
    return set()


def _finish_training(config: dict[str, Any], output_dir: Path, metrics: dict[str, Any],
                      cohort: Any, schedules: list[Any]) -> dict[str, int | float | str | bool]:
    report = json.loads((output_dir / _REPORT).read_text())
    # The shared engine completes before this exposure audit. A failed audit
    # must not leave a completed result advertised in persisted scalar output.
    report["complete"] = False
    metrics["quality_complete"] = False
    _json(output_dir / _REPORT, report)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    result = report["trials"][VARIANT["name"]][TRIAL["name"]]
    completed, selected = result["passes_completed"], result["best_epoch"]
    if not 1 <= selected <= completed <= len(schedules):
        raise ValueError("selected and completed training exposure must match registered schedules")
    coverage = {"registered_maximum": rotating_schedule_summary(cohort, tuple(w for row in schedules for w in row), config),
        "actual_completed_training": rotating_schedule_summary(cohort, tuple(w for row in schedules[:completed] for w in row), config),
        "actual_selected_checkpoint": rotating_schedule_summary(cohort, tuple(w for row in schedules[:selected] for w in row), config),
        "passes_completed": completed, "selected_checkpoint_epoch": selected,
        "successful_updates_to_selected_checkpoint": sum(row["successful_updates"] for row in result["passes"] if row["epoch"] <= selected),
        "completed_passes_only": True, "test_evaluations": 0}
    _json(output_dir / _COVERAGE, coverage)
    report.update(training_coverage_artifact=_COVERAGE, selected_model_architecture=config["architecture"],
        source_checkpoint_weights_loaded=False, new_training_configurations=1, reused_controls=0)
    _json(output_dir / _REPORT, report)
    selection_path = output_dir / "final-selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(selected_checkpoint=result["best_checkpoint"], selected_architecture=config["architecture"],
        resumable_checkpoint=result.get("resumable_checkpoint"), training_coverage_artifact=_COVERAGE,
        full_history_evaluation_registered=False, source_checkpoint_weights_loaded=False)
    _json(selection_path, selection)
    metrics.update(quality_complete=True, training_models=1, reused_controls=0, d_state=128, head_dim=64, chunk_size=64,
        full_history_inference_evaluations=0, source_checkpoint_weights_loaded=False,
        actual_unique_supervised_target_positions=coverage["actual_completed_training"]["unique_target_positions"],
        selected_checkpoint_unique_supervised_target_positions=coverage["actual_selected_checkpoint"]["unique_target_positions"])
    report["complete"] = True
    _json(output_dir / _REPORT, report)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    return validate_metrics(metrics)


def run_query_ssm2_training(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    output_dir = Path(output_dir)
    schedules, state = [], {}
    def schedule(cohort: Any, *, seed: int, epoch: int) -> Any:
        windows = rotating_schedule(cohort, seed=seed, epoch=epoch)
        schedules.append(windows)
        state["cohort"] = cohort
        return windows
    def summary(cohort: Any, windows: Any, current: dict[str, Any]) -> dict[str, Any]:
        if windows[0].epoch == current["hpo"]["max_passes"]:
            _gzip_json(output_dir / "ssm2-window-descriptors.json.gz", {
                "columns": ["epoch", "user_id", "end", "negative_seed"],
                "descriptors": [[w.epoch, w.user_id, w.end, w.negative_seed] for row in schedules for w in row],
                "contains_private_item_or_timestamp_arrays": False,
                "same_schedule_as_previous_rotating_experiment": True})
        return rotating_schedule_summary(cohort, windows, current)
    definition = QuerySearchDefinition(validate=validate_ssm2_training_config,
        variants=ssm2_training_variants, trials=ssm2_training_trials,
        build_model=build_ssm2_training_model, optimizer=ssm2_training_optimizer,
        preflight=preflight_ssm2_training, budget=conservative_ssm2_training_budget,
        batch_arrays=ssm2_training_batch_arrays,
        training_batch=lambda current, _variant: current["hpo"]["physical_batch_size"],
        evaluation_batch=ssm2_training_evaluation_batch, protocol_metadata=ssm2_training_metadata,
        training_schedule=schedule, schedule_summary=summary,
        native_audit=lambda device: audit_native_query_ssm2(device, head_dim=64, chunk_size=64, d_state=128),
        select_candidate=select_native_short_candidate, full_transfer_candidates=_no_full_transfer,
        retained_trial_resumes=1, hpo_performed=False, profile_each_trial=True,
        report_file=_REPORT, phase="ssm2_training", checkpoint_stage="ssm2_rotating_state128")
    metrics = run_registered_query_search(config, output_dir, definition=definition)
    return _finish_training(config, output_dir, metrics, state["cohort"], schedules)
