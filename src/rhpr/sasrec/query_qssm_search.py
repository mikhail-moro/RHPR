"""First QSSM capacity/memory grid with validation-only selection.

Seven fresh runs share the original targets, negative identities and SASRec
initializer. Full-history inference is reserved for the two best fresh proxy
recipes and cannot affect the primary ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rhpr.metrics import validate_metrics
from rhpr.sasrec.long_quality import _json
from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search
from rhpr.sasrec.query_qssm_models import (
    build_qssm_model, conservative_qssm_budget, preflight_qssm,
)
from rhpr.sasrec.query_qssm_protocol import (
    load_qssm_controls, qssm_batch_arrays, qssm_evaluation_batch, qssm_trials,
    qssm_variants, select_qssm_candidate, validate_qssm_config,
)


def ranked_qssm_results(config: dict[str, Any], variants: list[dict[str, Any]],
                        trials_by_family: dict[str, list[dict[str, Any]]],
                        report: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank all finite primary results, keeping registration order for ties."""
    rows = []
    for variant in variants:
        for trial in trials_by_family[variant["name"]]:
            result = report["trials"][variant["name"]][trial["name"]]
            if select_qssm_candidate([trial], [result]) is None:
                continue
            rows.append({"variant": dict(variant), "trial": dict(trial),
                "score": float(result["best_validation"]["ndcg10"]),
                "reused": result.get("reused", False) is True})
    return sorted(rows, key=lambda row: -row["score"])


def qssm_full_transfer_candidates(config: dict[str, Any], variants: list[dict[str, Any]],
                                  trials_by_family: dict[str, list[dict[str, Any]]],
                                  report: dict[str, Any]) -> set[tuple[str, str]]:
    """Choose fresh finalists before observing any full-history quality."""
    fresh = [row for row in ranked_qssm_results(config, variants, trials_by_family, report)
             if not row["reused"]]
    selected = fresh[:config["full_transfer"]["new_configurations"]]
    report["full_transfer_selection"] = {"metric": "short_validation_ndcg10",
        "selected": selected, "test_evaluations": 0,
        "reused_control_evaluated_again": False,
        "full_history_quality_used_for_selection": False}
    return {(row["variant"]["name"], row["trial"]["name"]) for row in selected}


def _protocol_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    return {"training": "H1024+R128, minimum20 maximum60 passes; validation every10; patience2",
        "architecture": "QSSM K4/K8/K16/K32 x default/slow dt initialization; inner64/state16/rank32; fixed SASRec",
        "family_selection": "best finite short-validation macro-user NDCG10 within each K; ties preserve registered order",
        "global_selection": "best short-validation NDCG10 across all eight recipes; no test or full-H selection",
        "full_history": "optional frozen-weight inference-length transfer of top2 fresh short-validation recipes only; no full training",
        "control_reuse": "verified architecture-search K4/inner64/default-dt control; weights and validation ranks copied with provenance",
        "memory_initialization": "dt range initializes delta bias; subsequent learned event-dependent steps are not clamped to this range",
        "targets": "original parent users, endpoints, last128 labels, negative identities, query suffix and PE32768 initializer retained",
        "comparison": "exploratory H1024 screen, one training seed; frozen full-H inference does not establish trained long-history quality",
        "test": "no test ranks deserialized; no test evaluation or selection"}


def run_query_qssm_search(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    definition = QuerySearchDefinition(validate=validate_qssm_config, variants=qssm_variants,
        trials=qssm_trials, build_model=build_qssm_model,
        preflight=preflight_qssm, budget=conservative_qssm_budget,
        batch_arrays=qssm_batch_arrays,
        training_batch=lambda current, variant: current["qssm_batches"][variant["name"]]["training"],
        evaluation_batch=qssm_evaluation_batch,
        protocol_metadata=_protocol_metadata, select_candidate=select_qssm_candidate,
        full_transfer_candidates=qssm_full_transfer_candidates,
        full_transfer_budget_seconds=config["full_transfer"]["reserve_seconds"],
        retained_trial_resumes=2, paired_short_validation=True, optional_full_transfer=True,
        report_file="query-qssm-search.json", phase="qssm", checkpoint_stage="qssm_summary_memory_search",
        profile_each_trial=True, load_controls=load_qssm_controls, isolate_preflight_failures=True)
    output_dir = Path(output_dir)
    metrics = run_registered_query_search(config, output_dir, definition=definition)
    path = output_dir / definition.report_file
    report = json.loads(path.read_text())
    variants = qssm_variants(config)
    ranking = ranked_qssm_results(config, variants,
        {variant["name"]: qssm_trials(config, variant) for variant in variants}, report)
    winner = ranking[0]
    result = report["trials"][winner["variant"]["name"]][winner["trial"]["name"]]
    report.update(primary_ranking=ranking, global_selected_configuration=winner)
    metrics.update(selected_summary_tokens=winner["variant"]["summary_tokens"],
        selected_dt_max=winner["trial"]["dt_max"], selected_trial=winner["trial"]["name"],
        selected_ndcg10=winner["score"], selected_ndcg100=result["best_validation"]["ndcg100"],
        selected_is_reused_control=winner["reused"])
    selection_path = output_dir / "final-selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(global_selected_configuration=winner, selection_input="H1024+R128",
        checkpoint_training_input="H1024+R128; identical original last128 labels")
    _json(path, report)
    _json(selection_path, selection)
    _json(output_dir / "metrics.json", validate_metrics(metrics))
    return validate_metrics(metrics)
