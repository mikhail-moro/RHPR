"""Pinned weight-only parents for the paired history-length continuation.

Preparation audits validation artifacts before copying one portable checkpoint.
The model factory loads only its parameter tensors; Adam, AMP and RNG state are
always constructed afresh by the continuation runner.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_history_source(config: dict[str, Any], evaluation: Any,
                           output_dir: str | Path) -> dict[str, Any]:
    """Verify the selected source generation, ranks and bytes without loading it.

    Only validation rank/context artifacts are opened. A failed optional full-H
    transfer in the FuXi parent is permitted: its primary short-history quality
    and selected weight checkpoint must still be complete and valid.
    """
    from rhpr.sasrec.query_architecture_protocol import (
        _load_reference_contexts, _policy_replay, _validate_contexts, _validate_rank_metrics,
    )
    from rhpr.sasrec.query_baseline_protocol import _resolve_kernel_root
    from rhpr.sasrec.query_history_ablation import _verified_file
    from rhpr.sasrec.query_history_protocol import history_variant, validate_history_config

    validate_history_config(config)
    source = config["selected_source"]
    root = _resolve_kernel_root(source["kernel"], source["checkpoint_root"])
    documents = {name: json.loads(_verified_file(root, name, pin["sha256"]).read_text())
                 for name, pin in source["documents"].items()}
    report_name = "query-qssm-search.json" if config["family"] == "qssm" else "query-baseline-screen.json"
    source_config = documents["experiment.json"]
    report, manifest = documents[report_name], documents["checkpoint-manifest.json"]
    if (_canonical(source_config) != source["experiment_config_sha256"]
            or report.get("config") != source_config or manifest.get("config") != source_config
            or report.get("source_code_sha") != source["source_code_sha"]
            or manifest.get("source_code_sha") != source["source_code_sha"]
            or manifest.get("format_version") != 3 or report.get("complete") is not True
            or manifest.get("best_checkpoint_optimizer_state_saved") is not False):
        raise ValueError("history source configuration, revision or completed weight-only manifest mismatch")
    variant = history_variant(config)
    trial = source["trial"]
    result = report.get("trials", {}).get(source["model"], {}).get(trial["name"], {})
    allowed_status = ("success", "full_transfer_failed") if config["family"] == "fuxi" else ("success",)
    if (source["model"] != variant["name"] or result.get("status") not in allowed_status
            or result.get("complete") is not True or result.get("trial") != trial
            or result.get("trial_name") != trial["name"]
            or report.get("selected_trials", {}).get(source["model"]) != trial
            or result.get("old_length") != 1024 or result.get("from_scratch") is not True
            or result.get("full_length_training_performed") is not False
            or result.get("test_evaluations") != 0):
        raise ValueError("history source is not the selected completed H1024 validation trial")
    if (result["status"] == "full_transfer_failed"
            and result.get("failure", {}).get("stage") != "full_length_validation"):
        raise ValueError("only a parent optional full-history evaluation failure is permitted")
    if config["family"] == "qssm":
        winner = report.get("global_selected_configuration", {})
        if winner.get("variant") != variant or winner.get("trial") != trial:
            raise ValueError("QSSM source is not the globally selected capacity/memory recipe")
    proof = _policy_replay(source_config, result)
    expected, saved = source["checkpoint"], result.get("best_checkpoint", {})
    if (any(saved.get(key) != expected[other] for key, other in
            (("path", "file"), ("sha256", "sha256"), ("size_bytes", "size_bytes"), ("best_epoch", "best_epoch")))
            or saved.get("training_model") != source["model"] or saved.get("trial") != trial
            or saved.get("active_old_length") != 1024 or saved.get("precision") != "float32"
            or saved.get("best_epoch") != result.get("best_epoch")
            or [row for row in manifest.get("checkpoints", []) if row.get("path") == saved.get("path")] != [saved]):
        raise ValueError("history source checkpoint disagrees with the selected pass or manifest")

    references = _load_reference_contexts(config, evaluation)
    ranks, contexts = source["artifacts"]["validation_ranks"], source["artifacts"]["validation_contexts"]
    if result.get("best_validation_rank_file") != ranks["file"]:
        raise ValueError("history source validation ranks do not identify its selected checkpoint")
    rank_path = _verified_file(root, ranks["file"], ranks["sha256"])
    context_path = _verified_file(root, contexts["file"], contexts["sha256"])
    with gzip.open(rank_path, "rt", encoding="utf-8") as stream:
        rank_rows = json.load(stream)
    with gzip.open(context_path, "rt", encoding="utf-8") as stream:
        context_rows = json.load(stream)
    _validate_rank_metrics(rank_rows, result.get("best_validation", {}), references)
    _validate_contexts(context_rows, references, full=False)
    if contexts["file"] != ranks["file"].replace("-anchors.json.gz", "-input-contexts.json.gz"):
        raise ValueError("history source validation contexts do not identify its selected checkpoint")

    weights = _verified_file(root, expected["file"], expected["sha256"], expected["size_bytes"])
    output_dir = Path(output_dir)
    destination = output_dir / "checkpoints" / f"source-{source['model']}-{trial['name']}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, destination)
    _verified_file(output_dir, destination.relative_to(output_dir).as_posix(),
                   expected["sha256"], expected["size_bytes"])
    copied_artifacts = {}
    for name, path, pin in (("validation_ranks", rank_path, ranks),
                            ("validation_contexts", context_path, contexts)):
        target = output_dir / "source-validation" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied_artifacts[name] = dict(pin, file=target.relative_to(output_dir).as_posix())
    provenance = {"kernel": source["kernel"], "source_code_sha": source["source_code_sha"],
        "original_path": expected["file"], "original_sha256": expected["sha256"],
        "documents": copy.deepcopy(source["documents"]), "selected_best_epoch": expected["best_epoch"],
        "weights_copied": True, "weight_bytes_verified": True,
        "validation_ranks_and_contexts_verified": True, "test_ranks_read": False,
        "optimizer_state_inherited": False, "scaler_state_inherited": False, "rng_state_inherited": False}
    checkpoint_record = dict(saved, path=destination.relative_to(output_dir).as_posix(),
        copied_reference=True, source_code_sha=source["source_code_sha"], provenance=provenance)
    checkpoint_config = dict(source_config, active_trial=copy.deepcopy(trial),
        active_stage="qssm_summary_memory_search" if config["family"] == "qssm" else "matched_baseline_screen",
        active_old_length=1024, active_pass_cap=source_config["hpo"]["max_passes"])
    return {"checkpoint_path": str(destination), "checkpoint_record": checkpoint_record,
        "source_config": copy.deepcopy(source_config), "source_checkpoint_config": checkpoint_config,
        "source_variant": variant, "source_trial": copy.deepcopy(trial),
        "source_best_validation": copy.deepcopy(result["best_validation"]),
        "source_best_epoch": expected["best_epoch"], "provenance": provenance,
        "validation_artifacts": copied_artifacts, "stopping_policy_replay": proof}


def load_history_source_weights(metadata: dict[str, Any], model: Any, *, num_items: int) -> dict[str, Any]:
    """Load finite, exactly matching model tensors without optimizer/RNG restore.

    This preserves the caller's training/eval and requires-grad settings. Both
    continuation arms call it before creating a new optimizer and AMP scaler.
    """
    import torch

    from rhpr.sasrec.query_history_ablation import _verified_file

    path = Path(metadata["checkpoint_path"])
    record = metadata["checkpoint_record"]
    _verified_file(path.parent, path.name, record["sha256"], record["size_bytes"])
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"format_version", "source_code_sha", "experiment_config", "model_config",
        "variant", "num_items", "best_epoch", "selection_score", "state_dict"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("history source must be a model-only best checkpoint without optimizer, scaler or RNG state")
    expected = {"format_version": 3, "source_code_sha": metadata["provenance"]["source_code_sha"],
        "experiment_config": metadata["source_checkpoint_config"],
        "model_config": metadata["source_config"]["model"], "variant": metadata["source_variant"],
        "num_items": num_items, "best_epoch": metadata["source_best_epoch"]}
    for key, value in expected.items():
        if payload.get(key) != value or key in ("format_version", "num_items", "best_epoch") and type(payload.get(key)) is not int:
            raise ValueError(f"history source checkpoint metadata mismatch: {key}")
    score = payload["selection_score"]
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not math.isclose(score, metadata["source_best_validation"]["ndcg10"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("history source checkpoint selection score mismatch")
    state, expected_state = payload["state_dict"], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError("history source checkpoint parameter keys mismatch")
    for key, target in expected_state.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"history source checkpoint parameter shape or dtype mismatch: {key}")
        if value.is_floating_point() and (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"history source checkpoint parameters must be finite FP32: {key}")
    model.load_state_dict(state, strict=True)
    return {"model_parameters_loaded_strictly": True, "num_items": num_items,
        "best_epoch": payload["best_epoch"], "selection_score": score,
        "optimizer_state_inherited": False, "scaler_state_inherited": False, "rng_state_inherited": False}
