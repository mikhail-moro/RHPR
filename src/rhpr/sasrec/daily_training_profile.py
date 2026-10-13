"""Cost-only preflight for the matched daily/MixedBudget experiment.

Training uses successful effective-batch updates, including preparation and
transfer. Evaluation timing executes the real lazy daily iterator and exact
full-catalogue top100. Recommendation scores never select a batch size.
"""
from __future__ import annotations

from itertools import islice
import math
import time
from typing import Any

import torch

from rhpr.sasrec.daily_evaluation_data import iter_daily_evaluation_examples
from rhpr.sasrec.daily_quality import evaluate_daily_quality
from rhpr.sasrec.long_quality import _cleanup, _optimizer, _scaler, _sync, build_quality_model
from rhpr.sasrec.long_training_profile import _train_shape
from rhpr.sasrec.mixed_budget import select_summary_budget


def _valid_rate(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _choose_training_batch(rows: list[dict[str, Any]], budgets: tuple[int, ...]) -> tuple[int, float]:
    """A Mixed batch must succeed at both endpoints; rank by its slower arm."""
    options = []
    for physical in sorted({r["physical_batch_size"] for r in rows}):
        group = [r for r in rows if r["physical_batch_size"] == physical]
        usable = {r["summary_tokens"]: r for r in group
                  if r["status"] == "ok" and not r.get("amp_skipped_updates", 0)
                  and _valid_rate(r.get("train_windows_per_second_with_prepare_h2d_sync"))}
        if all(b in usable for b in budgets):
            slowest = min(usable[b]["train_windows_per_second_with_prepare_h2d_sync"] for b in budgets)
            options.append((physical, float(slowest)))
    if not options:
        raise RuntimeError("no physical batch completed a successful update at every profiled budget")
    return max(options, key=lambda pair: pair[1])


def _measure_daily_rate(model: torch.nn.Module, cohort: Any, data: Any, *,
                        context_length: int, batch_size: int, device: torch.device,
                        amp: bool, item_chunk_size: int, sample_count: int = 128) -> dict[str, Any]:
    """Warm one batch; include context construction and aggregation in timing."""
    def examples(count: int):
        return islice(iter_daily_evaluation_examples(
            cohort, data, split_name="validation", max_seq_len=context_length), count)

    # Warm all real stages, including top100, without reusing prepared contexts.
    evaluate_daily_quality(model, examples(batch_size), device=device, amp=amp,
                           batch_size=batch_size, item_chunk_size=item_chunk_size)
    _sync(device)
    started = time.perf_counter()
    measured = evaluate_daily_quality(model, examples(sample_count), device=device, amp=amp,
                                     batch_size=batch_size, item_chunk_size=item_chunk_size)
    _sync(device)
    elapsed = time.perf_counter() - started
    count = len(measured["per_anchor"])
    if count < 1 or not _valid_rate(elapsed):
        raise RuntimeError("daily evaluation produced no usable timing sample")
    return {"status": "ok", "batch_size": batch_size, "sample_anchors": count,
            "wall_seconds": elapsed, "anchors_per_second": count / elapsed,
            "quality_used_for_selection": False,
            "timing_scope": "lazy context assembly, CPU collation, H2D, encoder/compression, exact catalogue top100, D2H and user aggregation"}


def profile_daily_variant(config: dict[str, Any], variant: dict[str, Any], cohort: Any,
                          data: Any, schedule: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    """Choose common batches and forecast all daily arms, never fewer epochs."""
    train, bench = config["training"], config["benchmark"]
    mixed = tuple(variant.get("mixed_budgets", ()))
    all_budgets = mixed or (variant["summary_tokens"],)
    profile_budgets = (min(mixed), max(mixed)) if len(mixed) > 1 else all_budgets
    training_tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    training_context = training_tail if variant["kind"] == "recent" else train["raw_context_length"]
    serving_context = train["raw_tail_size"] if variant["kind"] == "recent" else train["raw_context_length"]
    profile = {**train, "tail_size": train["supervised_targets"],
               "update_repetitions": 1, "warmup_microbatches": 3}
    if len(schedule) < train["effective_batch_size"]:
        raise ValueError("cost preflight requires one complete effective batch of registered windows")
    counts = {split: data.summary[f"{split}_anchors"] for split in ("validation", "test")}
    if any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("daily anchor counts must be positive integers")

    def build_model():
        model = build_quality_model(config, variant, cohort.num_items).to(device)
        if mixed:
            model.compressor.mixed_budget_choices = mixed
        return model

    rows = []
    started = time.monotonic()
    for physical in train["physical_batch_sizes"]:
        for budget in profile_budgets:
            model = build_model()
            if mixed:
                select_summary_budget(model, budget)
            optimizer, scaler = _optimizer(model, train), _scaler(device, train)
            try:
                row = _train_shape(model, cohort, schedule, profile, training_context,
                                   physical, device, optimizer, scaler)
            except torch.OutOfMemoryError:
                row = {"status": "oom", "physical_batch_size": physical}
            except FloatingPointError as error:
                row = {"status": "nonfinite", "physical_batch_size": physical,
                       "reason": str(error)}
            finally:
                del optimizer, scaler, model
                _cleanup()
            rows.append({**row, "summary_tokens": budget})
    training_batch, training_rate = _choose_training_batch(rows, profile_budgets)

    # A common evaluation batch must fit every arm. Candidate rates include
    # context materialization and actual daily macro aggregation, unlike the
    # separate serving benchmark whose timing starts after H2D.
    candidates = [b for b in bench["batch_sizes"] if b <= bench["evaluation_batch_size"]]
    # B1 is retained when it is the only supported candidate. Otherwise profile
    # parallel batches, avoiding thousands of redundant single-user calls.
    parallel = [b for b in candidates if b > 1]
    candidates = parallel or candidates
    if not candidates:
        raise ValueError("no evaluation batch fits the registered cap")
    model = build_model()
    model.compressor.tail_size = train["raw_tail_size"]
    evaluations = []
    try:
        for budget in all_budgets:
            if mixed:
                select_summary_budget(model, budget)
            for batch_size in candidates:
                try:
                    row = _measure_daily_rate(model, cohort, data, context_length=serving_context,
                        batch_size=batch_size, device=device, amp=train["amp"],
                        item_chunk_size=bench["item_chunk_size"])
                except torch.OutOfMemoryError:
                    row = {"status": "oom", "batch_size": batch_size}
                    _cleanup()
                evaluations.append({**row, "summary_tokens": budget})
    finally:
        del model
        _cleanup()

    choices = []
    for batch_size in candidates:
        usable = {r["summary_tokens"]: r for r in evaluations
                  if r["batch_size"] == batch_size and r["status"] == "ok"
                  and _valid_rate(r.get("anchors_per_second"))}
        if all(b in usable for b in all_budgets):
            choices.append((batch_size, min(usable[b]["anchors_per_second"] for b in all_budgets)))
    if not choices:
        raise RuntimeError(f"{variant['name']}: no evaluation batch fits every summary budget")
    evaluation_batch, _ = max(choices, key=lambda pair: pair[1])
    cases = []
    for budget in all_budgets:
        row = next(r for r in evaluations if r["summary_tokens"] == budget
                   and r["batch_size"] == evaluation_batch and r["status"] == "ok")
        cases.append({**row, "estimated_validation_seconds": counts["validation"] / row["anchors_per_second"],
                      "estimated_test_seconds": counts["test"] / row["anchors_per_second"]})
    return {**variant, "training_batch_size": training_batch,
            "evaluation_batch_size": evaluation_batch,
            "estimated_train_pass_seconds": len(cohort.users) / training_rate,
            "estimated_validation_seconds": sum(c["estimated_validation_seconds"] for c in cases),
            "estimated_test_seconds": sum(c["estimated_test_seconds"] for c in cases),
            "training": rows, "evaluation": evaluations, "eval_cases": cases,
            "profiled_training_budgets": list(profile_budgets),
            "training_selection": "fastest worst-arm successful effective update, including prepare/H2D",
            "evaluation_selection": "fastest worst-arm daily top100 throughput; no quality selection",
            "validation_anchors": counts["validation"], "test_anchors": counts["test"],
            "profile_seconds": time.monotonic() - started,
            "hstu_normalization_length": training_tail + max(config["summary_budgets"])}
