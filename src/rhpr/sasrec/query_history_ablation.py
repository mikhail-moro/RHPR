"""Validation-only interventions on verified, frozen SASRec compressor weights.

This diagnostic measures dependence of the existing checkpoints on remote
history. It does not estimate how well a model trained on shortened history
could perform, select a checkpoint, or access test targets.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.offline_runtime import resolve_dataset_directory
from rhpr.sasrec.daily_evaluation_data import (
    daily_context_sha256, iter_daily_evaluation_examples, load_prepared_daily_evaluation,
)
from rhpr.sasrec.daily_quality import _aggregate, evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_history_data import load_prepared_cohort
from rhpr.sasrec.long_mixed_quality import _gzip_json, _sha256
from rhpr.sasrec.long_quality import _cleanup, _json, _registered_device, _sync
from rhpr.sasrec.query_history_interventions import intervene_example, intervention_spec
from rhpr.sasrec.query_prefix_protocol import build_prefix_model, prefix_variants, validate_prefix_config
from rhpr.sasrec.query_profile import audit_native_query_scan


ABLATION_KIND = "yambda_query_history_ablation"
PARENT_KERNEL = "carasik/rhpr-sasrec-prefix-query-pilot"
PARENT_SOURCE_SHA = "660c7705ed3e005f35b339313c8c865255e36ede"
MODEL_NAMES = ("query_ssm_m4", "query_cross_attention_m4")
CONDITIONS = ("full", "query_only_history", "shuffle_remote")
_COMMON_SECTIONS = ("seed", "data", "model", "compressor", "training", "benchmark", "runtime")


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_history_ablation_config(config: dict[str, Any]) -> None:
    """Require the exact parent protocol and a bounded, validation-only design."""
    if config.get("kind") != ABLATION_KIND or not isinstance(config.get("name"), str):
        raise ValueError("a named history-ablation configuration is required")
    parent = config.get("parent", {})
    if parent.get("kernel") != PARENT_KERNEL or parent.get("source_code_sha") != PARENT_SOURCE_SHA:
        raise ValueError("history ablation must pin the completed SASRec prefix pilot")
    original = parent.get("experiment_config")
    if not isinstance(original, dict):
        raise ValueError("the exact parent experiment_config is required")
    validate_prefix_config(original)
    for section in _COMMON_SECTIONS:
        if config.get(section) != original.get(section):
            raise ValueError(f"history ablation changed the parent protocol: {section}")
    if config.get("variants") != list(MODEL_NAMES):
        raise ValueError("exactly frozen QSSM and Cross Attention, in that order, are required")
    if config["model"].get("use_time_features") is not False:
        raise ValueError("history permutation requires the pinned Time-OFF models")
    if config["kaggle"].get("kernel_sources") != [PARENT_KERNEL]:
        raise ValueError("the private parent must be the sole kernel source")
    for key in ("dataset_sources", "competition_sources", "model_sources"):
        if config["kaggle"].get(key) != original["kaggle"].get(key):
            raise ValueError(f"parent Kaggle sources changed: {key}")
    if Path(parent.get("checkpoint_root", "")).name != PARENT_KERNEL.split("/")[1]:
        raise ValueError("checkpoint_root must name the registered private parent kernel")
    for key in ("manifest_sha256", "context_manifest_sha256"):
        if not _digest(parent.get(key)):
            raise ValueError(f"parent.{key} must be a SHA256 digest")
    specs = parent.get("checkpoints", {})
    if set(specs) != set(MODEL_NAMES):
        raise ValueError("the two pinned best checkpoints are required")
    for name, spec in specs.items():
        if spec.get("file") != f"checkpoints/{name}.pt":
            raise ValueError("checkpoint path must be its registered relative checkpoints/*.pt path")
        if (not _digest(spec.get("sha256")) or not _digest(spec.get("validation_anchors_sha256"))
                or type(spec.get("size_bytes")) is not int or spec["size_bytes"] < 1
                or type(spec.get("best_epoch")) is not int or spec["best_epoch"] < 1):
            raise ValueError("checkpoint size, SHA256, validation-anchor SHA256 and best_epoch are required")
        score = spec.get("validation_ndcg10")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("parent validation NDCG@10 must lie in [0,1]")
    ablation = config.get("ablation", {})
    if ablation.get("conditions") != list(CONDITIONS):
        raise ValueError("exactly FULL, query-only history and remote-order permutation are registered")
    for key in ("budget_seconds", "bootstrap_draws", "permutation_draws"):
        if type(ablation.get(key)) is not int or ablation[key] < 1:
            raise ValueError(f"ablation.{key} must be a positive integer")
    if ablation["budget_seconds"] > 2400:
        raise ValueError("the frozen-weight diagnostic is capped at2400 seconds")
    if ablation.get("baseline_metric_tolerance") != 1e-7:
        raise ValueError("baseline metric tolerance must be pinned at1e-7; ranks must match exactly")
    for key in ("equivalence_margin_ndcg10", "equivalence_margin_ndcg100"):
        if ablation.get(key) != .0005:
            raise ValueError("the preregistered absolute NDCG equivalence margin is0.0005")


def resolve_history_parent(config: dict[str, Any], input_root: str | Path = "/kaggle/input") -> Path:
    """Resolve only the known parent slug in bounded Kaggle mount layouts."""
    base = Path(input_root).resolve()
    slug = PARENT_KERNEL.split("/")[1]
    configured = Path(config["parent"]["checkpoint_root"])
    if not configured.is_absolute():
        configured = base / configured
    if configured.name != slug:
        raise ValueError("parent checkpoint root has the wrong slug")
    try:
        configured.resolve().relative_to(base)
    except ValueError as error:
        raise ValueError("parent checkpoint root escapes input_root") from error
    candidates = {configured, base / slug}
    for pattern in (f"*/{slug}", f"*/*/{slug}", f"*/*/*/{slug}"):
        candidates.update(base.glob(pattern))
    matches = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        root = candidate.resolve()
        try:
            root.relative_to(base)
        except ValueError as error:
            raise ValueError("parent kernel mount escapes input_root") from error
        if (root / "checkpoint-manifest.json").is_file():
            matches.add(root)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one mounted parent kernel; found{len(matches)}")
    return next(iter(matches))


def _verified_file(root: Path, filename: str, digest: str, size: int | None = None) -> Path:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("parent artifacts must use bounded relative paths")
    resolved = (root / filename).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("parent artifact escapes its kernel mount") from error
    if not resolved.is_file() or size is not None and resolved.stat().st_size != size:
        raise ValueError(f"parent artifact missing or size mismatch: {filename}")
    if _sha256(resolved) != digest:
        raise ValueError(f"parent artifact SHA256 mismatch: {filename}")
    return resolved


def load_history_checkpoint(config: dict[str, Any], name: str, *, num_items: int,
                            model: torch.nn.Module, root: Path) -> dict[str, Any]:
    """Hash before safe loading and validate every metadata and tensor field."""
    if name not in MODEL_NAMES:
        raise ValueError("unregistered frozen checkpoint")
    spec = config["parent"]["checkpoints"][name]
    checkpoint = _verified_file(root, spec["file"], spec["sha256"], spec["size_bytes"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = config["parent"]["experiment_config"]
    variant = next(row for row in prefix_variants(original) if row["name"] == name)
    expected = {"format_version": 3, "source_code_sha": config["parent"]["source_code_sha"],
        "experiment_config": original, "model_config": original["model"], "variant": variant,
        "num_items": num_items, "best_epoch": spec["best_epoch"]}
    if not isinstance(payload, dict):
        raise ValueError("parent checkpoint must be a dictionary")
    for key, value in expected.items():
        if payload.get(key) != value or key in ("format_version", "num_items", "best_epoch") and type(payload.get(key)) is not int:
            raise ValueError(f"parent checkpoint metadata mismatch: {key}")
    score = payload.get("selection_score")
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or abs(score - spec["validation_ndcg10"]) > 1e-12):
        raise ValueError("parent checkpoint validation selection_score mismatch")
    state, expected_state = payload.get("state_dict"), model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError("parent checkpoint state_dict keys mismatch")
    for key, expected_tensor in expected_state.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected_tensor.shape
                or value.dtype != expected_tensor.dtype):
            raise ValueError(f"parent checkpoint tensor shape or dtype mismatch: {key}")
        if value.is_floating_point() and (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"parent checkpoint floating tensors must be finite FP32: {key}")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return {key: value for key, value in payload.items() if key != "state_dict"}


def verify_full_reproduction(reference: list[dict[str, Any]], observed: dict[str, Any], *,
                             expected_ndcg10: float, tolerance: float) -> dict[str, Any]:
    """Demand identical anchor/support/hash identities and exact top100 ranks."""
    rows = observed["per_anchor"]
    identities = ("user_id", "anchor_id", "day", "reference_context_hash", "target_known", "rank_cutoff")
    identity_mismatches = sum(any(a.get(key) != b.get(key) for key in identities)
                              for a, b in zip(reference, rows)) + abs(len(reference) - len(rows))
    rank_mismatches = sum(any(a.get(key) != b.get(key) for key in ("rank", "rank_at_max_k"))
                          for a, b in zip(reference, rows))
    expected_metrics, _ = _aggregate(reference)
    differences = {key: abs(observed["metrics"][key] - expected_metrics[key])
                   for key in ("ndcg10", "ndcg100", "hr10", "hr100")}
    selection_difference = abs(observed["metrics"]["ndcg10"] - expected_ndcg10)
    passed = (identity_mismatches == rank_mismatches == 0
              and max(differences.values()) <= tolerance and selection_difference <= tolerance)
    return {"passed": passed, "reference_anchors": len(reference), "observed_anchors": len(rows),
        "identity_or_support_mismatches": identity_mismatches,
        "top100_rank_mismatches": rank_mismatches,
        "absolute_metric_differences": differences,
        "absolute_selection_ndcg10_difference": selection_difference,
        "metric_tolerance": tolerance, "ranks_must_match_exactly": True}


def _paired_contrasts(cases: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Four preregistered primary contrasts, paired at the user level."""
    settings, contrasts = config["ablation"], []
    for name in MODEL_NAMES:
        reference = cases[name]["full"]["per_anchor"]
        for condition in CONDITIONS[1:]:
            candidate = cases[name][condition]["per_anchor"]
            row: dict[str, Any] = {"model": name, "reference": "full", "candidate": condition,
                "cutoffs": {}, "training_seed_uncertainty_included": False}
            for cutoff in (10, 100):
                result = paired_daily_ndcg_bootstrap(reference, candidate, k=cutoff,
                    seed=config["seed"], draws=settings["bootstrap_draws"], confidence=.99)
                margin = settings[f"equivalence_margin_ndcg{cutoff}"]
                result["equivalence_margin_absolute"] = margin
                result["fixed_weights_equivalent_within_margin"] = (
                    result["ci_lower"] >= -margin and result["ci_upper"] <= margin)
                result["equivalence_interpretation"] = "frozen-checkpoint intervention only; not trained short-history equivalence"
                row["cutoffs"][str(cutoff)] = result
            def user_means(records: list[dict[str, Any]]) -> dict[int, float]:
                groups: dict[int, list[float]] = {}
                for record in records:
                    groups.setdefault(record["user_id"], []).append(record["ndcg10"])
                return {uid: float(np.mean(values)) for uid, values in groups.items()}
            baseline, changed = user_means(reference), user_means(candidate)
            delta = np.asarray([changed[uid] - baseline[uid] for uid in sorted(baseline)], dtype=np.float64)
            observed, count, draws = abs(float(delta.mean())), 0, settings["permutation_draws"]
            rng = np.random.default_rng(config["seed"])
            for first in range(0, draws, 64):
                signs = rng.integers(0, 2, size=(min(64, draws - first), len(delta)), dtype=np.int8) * 2 - 1
                count += int(np.count_nonzero(np.abs((signs * delta).mean(axis=1)) >= observed))
            row["sign_flip_two_sided_p10"] = (count + 1) / (draws + 1)
            row["sign_flip_draws"] = draws
            contrasts.append(row)
    previous = 0.0
    for rank, row in enumerate(sorted(contrasts, key=lambda item: item["sign_flip_two_sided_p10"])):
        previous = max(previous, min(1., (len(contrasts) - rank) * row["sign_flip_two_sided_p10"]))
        row["holm_p10_family4"] = previous
    return contrasts


def _audited_examples(examples: Iterable[Any], *, condition: str, config: dict[str, Any],
                      reference_contexts: list[dict[str, Any]], rows: list[dict[str, Any]],
                      guard: Callable[[], None]) -> Iterable[Any]:
    """Validate full contexts first and bind each changed input independently."""
    expected = iter(reference_contexts)
    tail, query = config["training"]["raw_tail_size"], config["compressor"]["query_context_size"]
    for example in examples:
        guard()
        reference = next(expected, None)
        if (reference is None or reference["anchor_id"] != example.anchor_id
                or reference["reference_context_hash"] != example.reference_context_hash
                or reference["serving_context_length"] != len(example.context)
                or reference["serving_context_sha256"] != daily_context_sha256(
                    example.user_id, example.context, example.timestamps)):
            raise ValueError("actual full validation context disagrees with pinned parent input")
        changed = intervene_example(example, condition=condition, seed=config["seed"],
                                    tail_size=tail, query_context_size=query)
        suffix = tail + query
        if (not np.array_equal(changed.context[-suffix:], example.context[-suffix:])
                or not np.array_equal(changed.timestamps[-suffix:], example.timestamps[-suffix:])):
            raise RuntimeError("history intervention changed Q or R")
        row = {"anchor_id": example.anchor_id, "reference_context_hash": example.reference_context_hash,
            "original_full_context_sha256": reference["serving_context_sha256"],
            "actual_context_sha256": daily_context_sha256(changed.user_id, changed.context, changed.timestamps),
            "original_context_length": len(example.context), "actual_context_length": len(changed.context),
            "query_context_sha256": daily_context_sha256(example.user_id,
                example.context[-suffix:-tail], example.timestamps[-suffix:-tail]),
            "raw_context_sha256": daily_context_sha256(example.user_id,
                example.context[-tail:], example.timestamps[-tail:]),
            "query_and_raw_tail_preserved": True}
        rows.append(row)
        yield changed
    if next(expected, None) is not None:
        raise ValueError("validation iterator omitted pinned parent contexts")


def run_query_history_ablation(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    """Evaluate both FULL baselines before applying four frozen interventions."""
    validate_history_ablation_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device, started = _registered_device(), time.monotonic()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics: dict[str, int | float | str | bool] = {
        "experiment": config["name"], "seed": config["seed"], "quality_complete": False,
        "training_updates": 0, "training_models": 0, "test_evaluations": 0,
        "validation_evaluations": 0, "frozen_models": len(MODEL_NAMES),
        "native_scan_audit_passed": False, "all_full_baselines_reproduced": False}
    report: dict[str, Any] = {"format_version": 1, "complete": False, "config": config,
        "code_sha": os.environ.get("GITHUB_SHA", "local"),
        "parent_source_sha": config["parent"]["source_code_sha"],
        "protocol": {"split": "validation only", "checkpoint_selection": "none; pinned saved best weights",
            "training": "none; every parameter frozen and inference_mode enabled",
            "forward": "one final-state prediction with intact R128; same full catalogue, anchors and OOV zero credit",
            "reproduction_gate": "both FULL baselines must match parent support, actual contexts and Top100 ranks before any intervention",
            "conditions": [intervention_spec(condition) for condition in CONDITIONS],
            "equivalence": "99% paired-user percentile interval inside absolute0.0005; frozen weights only",
            "multiple_comparisons": "four primary NDCG@10 sign-flip tests with Holm adjustment; @100 secondary",
            "limitations": "one training seed; interventions can shift input distribution; one deterministic shuffle per anchor"},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
        "checkpoint_audits": {}, "baseline_reproduction": {}, "cases": {}}

    def persist() -> None:
        _json(output_dir / "query-history-ablation.json", report)
        _json(output_dir / "metrics.json", validate_metrics(metrics))

    def guard() -> None:
        if time.monotonic() - started > config["ablation"]["budget_seconds"]:
            report["aborted"] = "registered2400-second wall-time budget exceeded; partial outputs retained"
            persist()
            raise TimeoutError(report["aborted"])

    persist()
    audit = audit_native_query_scan(device, query_source="old_suffix")
    _json(output_dir / "native-query-scan-audit.json", audit)
    report["native_scan_audit"] = audit
    metrics["native_scan_audit_passed"] = True
    guard()
    cohort = load_prepared_cohort(resolve_dataset_directory(config["data"]["prepared_path"]))
    for key in ("train_end_timestamp", "cohort_uid_length_sha256", "catalog_sha256",
                "cohort_mapped_train_items_sha256", "cohort_train_timestamps_sha256", "source_revision"):
        if config["data"][key] != cohort.summary[key]:
            raise ValueError(f"registered cohort mismatch: {key}")
    evaluation = load_prepared_daily_evaluation(resolve_dataset_directory(config["data"]["evaluation_path"]), cohort)
    for key in ("evaluation_content_sha256", "validation_end_timestamp", "test_end_timestamp", "day_seconds"):
        if evaluation.summary[key] != config["data"][key]:
            raise ValueError(f"registered daily evaluation mismatch: {key}")
    cap = config["training"]["serving_context_length"]
    train_lengths = {user.user_id: len(user.train) for user in cohort.users}
    low, high = config["data"]["train_end_timestamp"], config["data"]["validation_end_timestamp"]
    if any(train_lengths[anchor.user_id] + anchor.future_index > cap for anchor in evaluation.anchors
           if low <= anchor.target_timestamp < high):
        raise ValueError("full validation context exceeds safety cap; truncation is forbidden")
    _json(output_dir / "cohort-summary.json", cohort.summary)
    _json(output_dir / "evaluation-summary.json", evaluation.summary)
    metrics.update(cohort_users=len(cohort.users), num_items=cohort.num_items)
    root = resolve_history_parent(config)
    parent = config["parent"]
    manifest_path = _verified_file(root, "checkpoint-manifest.json", parent["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("source_code_sha") != PARENT_SOURCE_SHA or manifest.get("format_version") != 3
            or manifest.get("config") != parent["experiment_config"]):
        raise ValueError("parent checkpoint manifest source/configuration mismatch")
    context_path = _verified_file(root, "evaluation-input-contexts.json.gz", parent["context_manifest_sha256"])
    with gzip.open(context_path, "rt", encoding="utf-8") as stream:
        full_context_manifest = json.load(stream)["validation"]
    reference_contexts = full_context_manifest["rows"]
    if len(reference_contexts) != evaluation.summary["validation_anchors"]:
        raise ValueError("pinned parent validation context count mismatch")
    report["parent_artifact_audit"] = {"kernel": PARENT_KERNEL, "manifest_sha256": parent["manifest_sha256"],
        "context_manifest_sha256": parent["context_manifest_sha256"],
        "validation_actual_context_sha256": full_context_manifest["ordered_serving_context_sha256"],
        "validation_anchors": len(reference_contexts), "test_split_evaluated": False}
    variants = {row["name"]: row for row in prefix_variants(parent["experiment_config"])}
    reference_ranks, cases = {}, {name: {} for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        spec = parent["checkpoints"][name]
        records = [row for row in manifest["checkpoints"] if row.get("training_model") == name]
        if len(records) != 1 or any(records[0].get(key) != spec[other] for key, other in (
                ("path", "file"), ("size_bytes", "size_bytes"), ("sha256", "sha256"), ("best_epoch", "best_epoch"))):
            raise ValueError("pinned checkpoint record disagrees with parent manifest")
        path = _verified_file(root, f"{name}-validation-anchors.json.gz", spec["validation_anchors_sha256"])
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            reference_ranks[name] = json.load(stream)

    def load_model(name: str) -> torch.nn.Module:
        guard()
        model = build_prefix_model(parent["experiment_config"], variants[name], cohort.num_items)
        metadata = load_history_checkpoint(config, name, num_items=cohort.num_items, model=model, root=root)
        report["checkpoint_audits"][name] = {"sha256": parent["checkpoints"][name]["sha256"],
            "size_bytes": parent["checkpoints"][name]["size_bytes"], "best_epoch": metadata["best_epoch"],
            "selection_score": metadata["selection_score"], "strict_state_dict": True,
            "finite_fp32_weights": True, "all_parameters_frozen": all(not p.requires_grad for p in model.parameters())}
        return model.to(device)

    def evaluate(name: str, condition: str, model: torch.nn.Module) -> dict[str, Any]:
        guard()
        context_rows: list[dict[str, Any]] = []
        originals = iter_daily_evaluation_examples(cohort, evaluation, split_name="validation", max_seq_len=cap)
        examples = _audited_examples(originals, condition=condition, config=config,
            reference_contexts=reference_contexts, rows=context_rows, guard=guard)
        _sync(device)
        began = time.perf_counter()
        with torch.inference_mode():
            result = evaluate_daily_quality(model, examples, device=device, amp=config["training"]["amp"],
                batch_size=config["benchmark"]["evaluation_batch_size"],
                item_chunk_size=config["benchmark"]["item_chunk_size"])
        _sync(device)
        wall = time.perf_counter() - began
        fingerprint = hashlib.sha256()
        for row in context_rows:
            fingerprint.update((row["anchor_id"] + ":" + row["actual_context_sha256"] + "\n").encode("ascii"))
        result["input_audit"] = {"ordered_actual_context_sha256": fingerprint.hexdigest(),
            "anchors": len(context_rows), "all_queries_and_raw_tails_preserved": True,
            "context_length_min": min(row["actual_context_length"] for row in context_rows),
            "context_length_max": max(row["actual_context_length"] for row in context_rows),
            "context_length_mean": float(np.mean([row["actual_context_length"] for row in context_rows]))}
        result["total_wall_seconds"] = wall
        result["anchors_per_second_with_input_audit"] = len(context_rows) / wall
        _gzip_json(output_dir / f"{name}-{condition}-validation-contexts.json.gz", context_rows)
        _gzip_json(output_dir / f"{name}-{condition}-validation-anchors.json.gz", result["per_anchor"])
        _json(output_dir / f"{name}-{condition}-validation-users.json", result["per_user"])
        cases[name][condition] = result
        report["cases"].setdefault(name, {})[condition] = {
            key: value for key, value in result.items() if key not in ("per_anchor", "per_user")}
        metrics["validation_evaluations"] += 1
        metrics.update({f"{name}_{condition}_validation_{key}": value for key, value in result["metrics"].items()})
        metrics[f"{name}_{condition}_total_wall_seconds"] = wall
        print(json.dumps({"phase": "history_ablation", "model": name, "condition": condition,
                          "metrics": result["metrics"], "total_wall_seconds": wall}), flush=True)
        persist()
        guard()
        return result

    # Establish both matched, intact baselines before observing any intervention.
    for name in MODEL_NAMES:
        model = load_model(name)
        try:
            baseline = evaluate(name, "full", model)
            gate = verify_full_reproduction(reference_ranks[name], baseline,
                expected_ndcg10=parent["checkpoints"][name]["validation_ndcg10"],
                tolerance=config["ablation"]["baseline_metric_tolerance"])
            report["baseline_reproduction"][name] = gate
            persist()
            if not gate["passed"]:
                raise RuntimeError(f"FULL parent validation reproduction failed for{name}; interventions aborted")
        finally:
            del model
            _cleanup()
    metrics["all_full_baselines_reproduced"] = True
    persist()
    for name in MODEL_NAMES:
        model = load_model(name)
        try:
            for condition in CONDITIONS[1:]:
                evaluate(name, condition, model)
        finally:
            del model
            _cleanup()
    guard()
    report["paired_contrasts"] = _paired_contrasts(cases, config)
    for row in report["paired_contrasts"]:
        prefix = f"{row['model']}_{row['candidate']}_minus_full"
        metrics[f"{prefix}_holm_p10_family4"] = row["holm_p10_family4"]
        for cutoff in (10, 100):
            estimate = row["cutoffs"][str(cutoff)]
            for key in ("delta_ndcg", "ci_lower", "ci_upper", "fixed_weights_equivalent_within_margin"):
                metrics[f"{prefix}_ndcg{cutoff}_{key}"] = estimate[key]
    report["complete"] = metrics["quality_complete"] = True
    metrics["suite_seconds"] = time.monotonic() - started
    persist()
    return validate_metrics(metrics)
