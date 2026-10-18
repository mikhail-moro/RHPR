"""Four matched short-input baselines against two immutable compressor controls.

The native short-context validation is primary. Full-history evaluation of the
same selected linear-backbone weights is an inference-length transfer check,
not evidence of training on long histories. No test ranks are read.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rhpr.sasrec.query_1k_search import QuerySearchDefinition, run_registered_query_search


def select_native_short_candidate(trials: list[dict[str, Any]],
                                  results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select only successful training with finite native short validation.

    A failed optional full-H transfer cannot erase the primary short-context
    result. Architecture/data/programming errors still propagate in the engine.
    """
    if len(trials) != len(results):
        raise ValueError("candidate trials and results must have matching support")
    selected, best = None, -1.
    for trial, result in zip(trials, results):
        if (result.get("trial") != trial or result.get("complete") is not True
                or result.get("status") not in ("success", "full_transfer_failed")):
            continue
        score = result.get("best_validation", {}).get("ndcg10")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            continue
        if score > best:
            selected, best = dict(trial), score
    return selected


def _protocol_metadata(_config: dict[str, Any]) -> dict[str, Any]:
    return {"training": "Recent R128; QueryOnly Q128+R128; Mamba4Rec/FuXi H1024+R128; minimum20 maximum60 passes, validation every10",
        "proxy_selection": "earliest strictly best native short-context validation macro-user NDCG10 checkpoint",
        "family_selection": "one fixed recipe per family; primary native short validation, no full-H selection",
        "full_history": "secondary frozen-short-checkpoint inference-length transfer for long-input arms; no full-history training",
        "architecture": "SASRec Recent/QueryOnly/CA/QSSM; native Mamba4Rec and FuXi-Linear predictors; capacities and initialization policies differ",
        "targets": "identical parent last128 labels, endpoints, users and negative seeds; SASRec initializer preserved; linear backbones retain native initialization",
        "control_reuse": "verified completed CA baseline and QSSM inner64; no repeated training or full validation",
        "comparison": "matched 1K screen, not a fully trained long-history benchmark; native input lengths differ for short controls",
        "short_control_full_evaluation": "Recent128 and QueryOnly reuse selected short-validation ranks because effective inputs are identical",
        "test": "no test ranks deserialized; no test evaluation or selection"}


def run_query_baseline_screen(config: dict[str, Any], output_dir: Path
                              ) -> dict[str, int | float | str | bool]:
    from rhpr.sasrec.query_baseline_models import (
        baseline_optimizer, build_baseline_model, conservative_baseline_budget,
        preflight_baseline,
    )
    from rhpr.sasrec.query_baseline_protocol import (
        baseline_batch_arrays, baseline_evaluation_batch, baseline_evaluation_examples,
        baseline_trials, baseline_variants, load_baseline_controls, validate_baseline_config,
    )

    definition = QuerySearchDefinition(validate=validate_baseline_config, variants=baseline_variants,
        trials=baseline_trials, build_model=build_baseline_model, optimizer=baseline_optimizer,
        preflight=preflight_baseline, budget=conservative_baseline_budget,
        batch_arrays=baseline_batch_arrays, evaluation_examples=baseline_evaluation_examples,
        evaluation_batch=baseline_evaluation_batch,
        training_batch=lambda current, variant: current["baseline_batches"][variant["name"]]["training"],
        full_transfer_required=lambda _current, variant: variant["name"] not in ("recent128", "query_only_m4"),
        protocol_metadata=_protocol_metadata, select_candidate=select_native_short_candidate,
        paired_short_validation=True, hpo_performed=False, optional_full_transfer=True,
        report_file="query-baseline-screen.json", phase="baseline", checkpoint_stage="matched_baseline_screen",
        profile_each_trial=True, load_controls=load_baseline_controls, isolate_preflight_failures=True)
    return run_registered_query_search(config, output_dir, definition=definition)
