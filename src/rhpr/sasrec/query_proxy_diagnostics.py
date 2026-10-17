"""Audited validation-only length controls for frozen prefix compressors.

Shortening a saved checkpoint is an input intervention, not an estimate of the
quality of a separately trained short-history model. Both original FULL ranks
must reproduce before any shortened input is evaluated.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch

from rhpr.sasrec.daily_evaluation_data import daily_context_sha256, iter_daily_evaluation_examples
from rhpr.sasrec.daily_quality import evaluate_daily_quality, paired_daily_ndcg_bootstrap
from rhpr.sasrec.long_mixed_quality import _gzip_json
from rhpr.sasrec.long_quality import _cleanup, _json, _sync
from rhpr.sasrec.query_history_ablation import (
    MODEL_NAMES, PARENT_SOURCE_SHA, _verified_file, load_history_checkpoint,
    resolve_history_parent, verify_full_reproduction,
)
from rhpr.sasrec.query_prefix_protocol import build_prefix_model, prefix_variants
from rhpr.sasrec.query_proxy_protocol import transform_hpo_example


def load_proxy_parent_artifacts(config: dict[str, Any], evaluation: Any,
                                ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pin parent manifests and validation ranks; never deserialize test ranks.

    Prepared evaluation data are independently content-pinned by the runner. Its
    anchor metadata enriches context references so every later iterator checks
    the original target identity as well as its actual past input.
    """
    root, parent = resolve_history_parent(config), config["parent"]
    manifest = json.loads(_verified_file(root, "checkpoint-manifest.json", parent["manifest_sha256"]).read_text())
    if (manifest.get("source_code_sha") != PARENT_SOURCE_SHA or manifest.get("format_version") != 3
            or manifest.get("config") != parent["experiment_config"]):
        raise ValueError("parent checkpoint manifest source/configuration mismatch")
    path = _verified_file(root, "evaluation-input-contexts.json.gz", parent["context_manifest_sha256"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        contexts = json.load(stream)
    for split, low, high in (
        ("validation", "train_end_timestamp", "validation_end_timestamp"),
        ("test", "validation_end_timestamp", "test_end_timestamp"),
    ):
        anchors = {a.anchor_id: a for a in evaluation.anchors
                   if evaluation.summary[low] <= a.target_timestamp < evaluation.summary[high]}
        rows = contexts[split]["rows"]
        if (len(rows) != evaluation.summary[f"{split}_anchors"] or len(anchors) != len(rows)
                or len({row["anchor_id"] for row in rows}) != len(rows)):
            raise ValueError("parent actual context count or unique anchor support mismatch")
        for row in rows:
            anchor = anchors.get(row["anchor_id"])
            if anchor is None or row["reference_context_hash"] != anchor.reference_context_hash:
                raise ValueError("parent context reference disagrees with pinned evaluation anchor")
            row.update(user_id=anchor.user_id, day=anchor.day, target=anchor.target,
                       target_timestamp=anchor.target_timestamp)
    reference_ranks = {}
    for name, spec in parent["checkpoints"].items():
        records = [row for row in manifest["checkpoints"] if row.get("training_model") == name]
        if len(records) != 1 or any(records[0].get(key) != spec[other] for key, other in (
                ("path", "file"), ("size_bytes", "size_bytes"), ("sha256", "sha256"), ("best_epoch", "best_epoch"))):
            raise ValueError("pinned checkpoint record disagrees with parent manifest")
    for name, splits in parent["baselines"].items():
        spec = splits["validation"]
        path = _verified_file(root, spec["file"], spec["sha256"])
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            reference_ranks[name] = json.load(stream)
        ranks = reference_ranks[name]
        rows = contexts["validation"]["rows"]
        if len(ranks) != len(rows) or any(any(record.get(key) != row[key]
                for key in ("user_id", "anchor_id", "day", "reference_context_hash"))
                or record.get("target_known") is not (row["target"] > 0)
                for record, row in zip(ranks, rows)):
            raise ValueError("parent validation rank identity or target support mismatch")
    return root, contexts, reference_ranks, manifest


def audited_hpo_examples(examples: Iterable[Any], old_length: int | None,
                         reference_contexts: list[dict[str, Any]], rows: list[dict[str, Any]],
                         guard: Callable[[], None]) -> Iterable[Any]:
    """Check the original full input/target before a CPU-side suffix crop.

The reference hash always identifies common original support. The actual input
gets a separate hash. Occupied OOV IDs remain real slots, including in Q and R.
    """
    expected = iter(reference_contexts)
    for example in examples:
        guard()
        reference = next(expected, None)
        identity = ("user_id", "anchor_id", "day", "target", "target_timestamp", "reference_context_hash")
        if (reference is None or any(reference.get(key) != getattr(example, key) for key in identity)
                or reference["serving_context_length"] != len(example.context)
                or reference["serving_context_sha256"] != daily_context_sha256(
                    example.user_id, example.context, example.timestamps)):
            raise ValueError("actual full context or target disagrees with pinned parent input")
        changed = transform_hpo_example(example, old_length)
        length = len(example.context) if old_length is None else min(len(example.context), old_length + 128)
        if (len(changed.context) != length
                or not np.array_equal(changed.context, example.context[-length:])
                or not np.array_equal(changed.timestamps, example.timestamps[-length:])
                or any(getattr(changed, key) != getattr(example, key) for key in identity)):
            raise RuntimeError("history crop changed retained events, Q/R or target identity")
        # Cropping on the left preserves recency-relative memory/query PE; the
        # summary source remains the end of H, exactly128 slots before the end.
        rows.append({"user_id": example.user_id, "anchor_id": example.anchor_id,
            "reference_context_hash": example.reference_context_hash,
            "original_full_context_sha256": reference["serving_context_sha256"],
            "actual_context_sha256": daily_context_sha256(changed.user_id, changed.context, changed.timestamps),
            "original_context_length": len(example.context), "actual_context_length": length,
            "query_context_sha256": daily_context_sha256(example.user_id,
                example.context[-256:-128], example.timestamps[-256:-128]),
            "raw_context_sha256": daily_context_sha256(example.user_id,
                example.context[-128:], example.timestamps[-128:]),
            "query_and_raw_tail_preserved": True, "retained_position_recencies_preserved": True})
        yield changed
    if next(expected, None) is not None:
        raise ValueError("evaluation iterator omitted pinned parent contexts")


def run_proxy_diagnostics(config: dict[str, Any], cohort: Any, evaluation: Any,
                          device: torch.device, output_dir: Path, guard: Callable[[], None],
                          ) -> dict[str, Any]:
    """Run two FULL gates, then frozen H1K/H2K/H4K validation interventions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root, contexts, reference_ranks, _ = load_proxy_parent_artifacts(config, evaluation)
    parent, bench = config["parent"], config["benchmark"]
    variants = {row["name"]: row for row in prefix_variants(parent["experiment_config"])}
    cases: dict[str, dict[str, Any]] = {name: {} for name in MODEL_NAMES}
    report: dict[str, Any] = {"complete": False, "split": "validation", "training_updates": 0,
        "test_evaluations": 0, "baseline_reproduction": {}, "cases": {}, "checkpoint_audits": {},
        "parent_artifact_audit": {"source_code_sha": parent["source_code_sha"],
            "manifest_sha256": parent["manifest_sha256"], "context_manifest_sha256": parent["context_manifest_sha256"]},
        "protocol": {"old_lengths_exclude_R128": True, "query_and_raw_tail_preserved": True,
            "retained_position_recencies_preserved": True, "weights_frozen": True,
            "full_gate": "both exact parent Top100 ranks before shortened cases",
            "limitations": "frozen-checkpoint input interventions; not trained-short-model performance; one seed; exploratory length comparisons"}}

    def persist() -> None:
        _json(output_dir / "query-proxy-diagnostics.json", report)

    def load_model(name: str) -> torch.nn.Module:
        guard()
        model = build_prefix_model(parent["experiment_config"], variants[name], cohort.num_items)
        metadata = load_history_checkpoint(config, name, num_items=cohort.num_items, model=model, root=root)
        report["checkpoint_audits"][name] = {"sha256": parent["checkpoints"][name]["sha256"],
            "best_epoch": metadata["best_epoch"], "strict_state_dict": True, "finite_fp32_weights": True,
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters())}
        return model.to(device)

    def evaluate(name: str, length: int | None, model: torch.nn.Module) -> dict[str, Any]:
        guard()
        condition, input_rows = "full" if length is None else f"h{length}", []
        originals = iter_daily_evaluation_examples(cohort, evaluation, split_name="validation",
            max_seq_len=parent["experiment_config"]["training"]["serving_context_length"])
        examples = audited_hpo_examples(originals, length, contexts["validation"]["rows"], input_rows, guard)
        _sync(device)
        began = time.perf_counter()
        with torch.inference_mode():
            result = evaluate_daily_quality(model, examples, device=device, amp=config["training"]["amp"],
                batch_size=bench["evaluation_batch_size"], item_chunk_size=bench["item_chunk_size"])
        _sync(device)
        result["total_wall_seconds"] = time.perf_counter() - began
        fingerprint = hashlib.sha256()
        for row in input_rows:
            fingerprint.update((row["anchor_id"] + ":" + row["actual_context_sha256"] + "\n").encode("ascii"))
        result["input_audit"] = {"anchors": len(input_rows), "ordered_actual_context_sha256": fingerprint.hexdigest(),
            "all_queries_and_raw_tails_preserved": True, "retained_position_recencies_preserved": True,
            "context_length_min": min(row["actual_context_length"] for row in input_rows),
            "context_length_max": max(row["actual_context_length"] for row in input_rows)}
        for suffix, payload in (("anchors", result["per_anchor"]), ("contexts", input_rows)):
            _gzip_json(output_dir / f"{name}-{condition}-validation-{suffix}.json.gz", payload)
        _json(output_dir / f"{name}-{condition}-validation-users.json", result["per_user"])
        cases[name][condition] = result
        report["cases"].setdefault(name, {})[condition] = {
            key: value for key, value in result.items() if key not in ("per_anchor", "per_user")}
        print(json.dumps({"phase": "frozen_length_diagnostic", "model": name, "old_length": length,
            "metrics": result["metrics"], "total_wall_seconds": result["total_wall_seconds"]}), flush=True)
        persist()
        guard()
        return result

    persist()
    for name in MODEL_NAMES:
        model = load_model(name)
        try:
            baseline = evaluate(name, None, model)
            gate = verify_full_reproduction(reference_ranks[name], baseline,
                expected_ndcg10=parent["checkpoints"][name]["validation_ndcg10"], tolerance=1e-7)
            report["baseline_reproduction"][name] = gate
            persist()
            if not gate["passed"]:
                raise RuntimeError(f"FULL parent validation reproduction failed for {name}; shortened cases aborted")
        finally:
            del model
            _cleanup()
    report["all_full_baselines_reproduced"] = True
    for name in MODEL_NAMES:
        model = load_model(name)
        try:
            for length in config["hpo"]["diagnostic_old_lengths"]:
                evaluate(name, length, model)
        finally:
            del model
            _cleanup()
    report["paired_contrasts"] = [{"model": name, "reference": "full", "candidate": f"h{length}",
        "training_seed_uncertainty_included": False,
        "cutoffs": {str(k): paired_daily_ndcg_bootstrap(cases[name]["full"]["per_anchor"],
            cases[name][f"h{length}"]["per_anchor"], k=k, seed=config["seed"],
            draws=config["hpo"]["bootstrap_draws"], confidence=.99) for k in (10, 100)}}
        for name in MODEL_NAMES for length in config["hpo"]["diagnostic_old_lengths"]]
    guard()
    report["complete"] = True
    persist()
    return report
