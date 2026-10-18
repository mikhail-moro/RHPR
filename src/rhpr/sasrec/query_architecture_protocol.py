"""Registered fixed-LR compressor search and immutable completed-run controls.

The two controls are reused only after checking the actual mounted weights,
validation ranks, source configuration and the new stopping policy. No saved
optimizer ownership or held-out test result is inherited from the earlier run.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path
import shutil
from typing import Any

from rhpr.sasrec.query_1k_protocol import SEARCH_1K_KIND, validate_1k_config

ARCHITECTURE_KIND = "yambda_query_compressor_architecture"
_FAMILIES = ("query_cross_attention_m4", "query_ssm_m4")
_FIXED_LR = {"backbone_lr": .001, "compressor_lr": .001}
_ARCHITECTURE_TRIALS = {
    _FAMILIES[0]: [
        dict(_FIXED_LR, name="ca_base", readout_depth=1, cross_attention_heads=4),
        dict(_FIXED_LR, name="ca_read2", readout_depth=2, cross_attention_heads=4),
        dict(_FIXED_LR, name="ca_heads8", readout_depth=1, cross_attention_heads=8),
    ],
    _FAMILIES[1]: [
        dict(_FIXED_LR, name="q_base", d_inner=128, d_state=16, gate_rank=32),
        dict(_FIXED_LR, name="q_inner64", d_inner=64, d_state=16, gate_rank=32),
        dict(_FIXED_LR, name="q_state32", d_inner=128, d_state=32, gate_rank=32),
    ],
}
_LR_TRIALS = [dict(_FIXED_LR, name="current_lr"),
              {"name": "lower_lr", "backbone_lr": .0003, "compressor_lr": .0003},
              {"name": "faster_compressor", "backbone_lr": .001, "compressor_lr": .003}]

_REUSED_CONTROLS = {'kernel': 'carasik/rhpr-query-prefix-1k-search',
 'source_code_sha': '66a0322b46b17cf9b480952512d49810c1efcd8e',
 'checkpoint_root': '/kaggle/input/rhpr-query-prefix-1k-search',
 'report': {'file': 'query-1k-search.json',
            'sha256': '0827fab7c42d4f61f7ec719ae1aa7802b22569f8745e9c1432e3aab1d5f14786'},
 'manifest': {'file': 'checkpoint-manifest.json',
              'sha256': '07ddc15ec708e4bbadcac56fef50ef496c6cdb3cff10e76f0dcbc21ba0297450'},
 'experiment': {'file': 'experiment.json',
                'sha256': 'a003522b9fcde6bb7c31ae567cbe3794adbfeaaf2b3a50836ef8731b398f8820'},
 'models': {'query_cross_attention_m4': {'original_trial': 'current_lr',
                                         'checkpoint': {'file': 'checkpoints/query_cross_attention_m4-current_lr.pt',
                                                        'sha256': 'f5d774c7c483484037144b78db264f0c3362d70f498e6a53b7773780dd50c6f3',
                                                        'size_bytes': 556857385,
                                                        'best_epoch': 50},
                                         'best_validation': {'file': 'query_cross_attention_m4-current_lr-p50-validation-anchors.json.gz',
                                                             'sha256': 'd964edccff3bc500c11e7344c383e74159a89d4e3ecd0f22559860b7e43358d1'},
                                         'full_validation': {'file': 'query_cross_attention_m4-current_lr-full-validation-anchors.json.gz',
                                                             'sha256': '15c057f694a797bfd5bb987fd9363100661486746c8491e1f6157e0f111a4d10'},
                                         'best_validation_contexts': {'file': 'query_cross_attention_m4-current_lr-p50-validation-input-contexts.json.gz',
                                                                      'sha256': '9142e65efd2737adb12d18ada03a4b636878f6ffe03bd97a15d505e5b7fb013b'},
                                         'full_validation_contexts': {'file': 'query_cross_attention_m4-current_lr-full-validation-input-contexts.json.gz',
                                                                      'sha256': '8dcaa0fdbd88b42a4a016c0b48d0902ca9e15ea643b267a0bffb4c399908ed33'}},
            'query_ssm_m4': {'original_trial': 'current_lr',
                             'checkpoint': {'file': 'checkpoints/query_ssm_m4-current_lr.pt',
                                            'sha256': '466c6924a4e10e81c789bd5fea0582e3bac24b75e3bb41750c25d9732f7c0ff8',
                                            'size_bytes': 556793733,
                                            'best_epoch': 50},
                             'best_validation': {'file': 'query_ssm_m4-current_lr-p50-validation-anchors.json.gz',
                                                 'sha256': 'ab10661bd0e9ca61bd70644fcf431018c2ab14e6ff439aeec23c990163c96172'},
                             'full_validation': {'file': 'query_ssm_m4-current_lr-full-validation-anchors.json.gz',
                                                 'sha256': '3b911757093ee726f059ed3caeb9980c708b7f503353f9a4e405694cf72e77c9'},
                             'best_validation_contexts': {'file': 'query_ssm_m4-current_lr-p50-validation-input-contexts.json.gz',
                                                          'sha256': '9142e65efd2737adb12d18ada03a4b636878f6ffe03bd97a15d505e5b7fb013b'},
                             'full_validation_contexts': {'file': 'query_ssm_m4-current_lr-full-validation-input-contexts.json.gz',
                                                          'sha256': '8dcaa0fdbd88b42a4a016c0b48d0902ca9e15ea643b267a0bffb4c399908ed33'}}}}


def _source_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project only the registered search-plan changes back to the exact source."""
    source = copy.deepcopy(config)
    del source["architecture_trials"], source["reused_controls"]
    source.update(kind=SEARCH_1K_KIND, name="yambda500m-query-prefix-1k-search")
    source["training"].update(min_passes=30, early_stopping_min_passes=30,
                              early_stopping_patience=3)
    source["hpo"].update(min_passes=30, patience=3, trials=copy.deepcopy(_LR_TRIALS))
    source["kaggle"]["kernel_sources"] = [config["parent"]["kernel"]]
    return source


def validate_architecture_config(config: dict[str, Any]) -> None:
    """Reject unregistered architecture, LR, data, policy or provenance changes."""
    if not isinstance(config, dict):
        raise ValueError("architecture configuration must be an object")
    required = {"benchmark", "compressor", "data", "hpo", "kaggle", "kind", "model",
                "name", "parent", "runtime", "seed", "summary_budgets", "training",
                "variants", "architecture_trials", "reused_controls"}
    if set(config) != required:
        raise ValueError("the registered architecture configuration fields are required exactly")
    if (config["kind"] != ARCHITECTURE_KIND
            or config["name"] != "yambda500m-query-compressor-architecture"):
        raise ValueError("the named compressor architecture search is required")
    if config["architecture_trials"] != _ARCHITECTURE_TRIALS:
        raise ValueError("exactly the six fixed-LR architecture configurations are registered")
    if config["reused_controls"] != _REUSED_CONTROLS:
        raise ValueError("completed controls must retain their exact source and artifact pins")
    expected_hpo = {"proxy_old_length": 1024, "min_passes": 20, "max_passes": 60,
        "validation_interval": 10, "patience": 2, "bootstrap_draws": 5000,
        "physical_batch_size": 32, "full_length_validation": True,
        "test_evaluations": 0, "trials": [dict(_FIXED_LR, name="architecture")]}
    if config["hpo"] != expected_hpo:
        raise ValueError("architecture search requires H1024, min20/max60, validation10 and patience2")
    original = config.get("parent", {}).get("experiment_config", {}).get("training")
    if not isinstance(original, dict):
        raise ValueError("pinned parent training configuration is required")
    expected_training = dict(original, min_passes=20, max_passes=60, budget_seconds=14400,
                             early_stopping_min_passes=20, early_stopping_patience=2)
    if config["training"] != expected_training:
        raise ValueError("only the registered architecture stopping policy may change parent optimization")
    if config["kaggle"].get("kernel_sources") != [config["parent"]["kernel"], _REUSED_CONTROLS["kernel"]]:
        raise ValueError("both the original pilot and the completed 1K controls must be mounted")
    validate_1k_config(_source_config(config))


def architecture_trials(config: dict[str, Any], variant: dict[str, Any] | str) -> list[dict[str, Any]]:
    """Return independent per-family trial definitions in registered tie order."""
    family = variant if isinstance(variant, str) else variant.get("name")
    if family not in _FAMILIES:
        raise ValueError("unknown compressor architecture family")
    return copy.deepcopy(config["architecture_trials"][family])


def _resolve_control_root(config: dict[str, Any], input_root: str | Path = "/kaggle/input") -> Path:
    base = Path(input_root).resolve()
    controls = config["reused_controls"]
    slug = controls["kernel"].split("/")[1]
    configured = Path(controls["checkpoint_root"])
    if not configured.is_absolute():
        configured = base / configured
    if configured.name != slug:
        raise ValueError("control checkpoint root has the wrong kernel slug")
    try:
        configured.resolve().relative_to(base)
    except ValueError as error:
        raise ValueError("control checkpoint root escapes input_root") from error
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
            raise ValueError("control kernel mount escapes input_root") from error
        if (root / controls["manifest"]["file"]).is_file():
            matches.add(root)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one mounted 1K control kernel; found {len(matches)}")
    return next(iter(matches))


def _load_reference_contexts(config: dict[str, Any], evaluation: Any) -> list[dict[str, Any]]:
    # Read only context metadata from the original pilot. Its test rank files
    # and the completed search's test artifacts are never opened.
    from rhpr.sasrec.query_history_ablation import _verified_file, resolve_history_parent

    parent = config["parent"]
    path = _verified_file(resolve_history_parent(config), "evaluation-input-contexts.json.gz",
                          parent["context_manifest_sha256"])
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)["validation"]["rows"]
    anchors = {anchor.anchor_id: anchor for anchor in evaluation.anchors
        if evaluation.summary["train_end_timestamp"] <= anchor.target_timestamp
        < evaluation.summary["validation_end_timestamp"]}
    if (len(rows) != evaluation.summary["validation_anchors"] or len(rows) != len(anchors)
            or len({row["anchor_id"] for row in rows}) != len(rows)):
        raise ValueError("control validation context support mismatch")
    result = []
    for row in rows:
        anchor = anchors.get(row["anchor_id"])
        if anchor is None or row["reference_context_hash"] != anchor.reference_context_hash:
            raise ValueError("control context reference disagrees with evaluation anchor")
        result.append(dict(row, user_id=anchor.user_id, day=anchor.day, target=anchor.target))
    return result


def _policy_replay(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    from rhpr.sasrec.query_restart_state import ValidationEarlyStopping

    hpo = config["hpo"]
    stopper = ValidationEarlyStopping(-1.0, minimum_additional_passes=hpo["min_passes"],
        patience=hpo["patience"], validation_interval=hpo["validation_interval"],
        max_additional_passes=hpo["max_passes"])
    passes = result.get("passes")
    if (not isinstance(passes, list)
            or [row.get("epoch") for row in passes] != list(range(1, result.get("passes_completed", 0) + 1))):
        raise ValueError("reused control training passes must be consecutive and complete")
    checks = []
    for row in passes:
        epoch = row["epoch"]
        if epoch % hpo["validation_interval"] == 0:
            metrics = row.get("validation")
            if not isinstance(metrics, dict):
                raise ValueError("reused control lacks a registered validation check")
            observation = stopper.observe(epoch, metrics.get("ndcg10"))
            checks.append(dict(epoch=epoch, score=metrics["ndcg10"], **observation))
            if stopper.stopped:
                break
        elif "validation" in row:
            raise ValueError("reused control contains an off-schedule validation check")
    ended = stopper.last_additional_passes
    if (ended != result["passes_completed"] or stopper.best_additional_passes != result.get("best_epoch")
            or stopper.best_score != result.get("best_validation", {}).get("ndcg10")
            or ended != hpo["max_passes"] and not stopper.stopped):
        raise ValueError("new stopping policy changes the reused control's chosen weights or end pass")
    return {"same_selected_epoch": True, "same_training_end": True,
        "best_epoch": stopper.best_additional_passes, "end_pass": ended,
        "minimum_passes": hpo["min_passes"], "maximum_passes": hpo["max_passes"],
        "patience": hpo["patience"], "validation_interval": hpo["validation_interval"],
        "strict_improvement": True, "checks": checks}


def _validate_rank_metrics(records: Any, expected: dict[str, Any], references: list[dict[str, Any]]) -> None:
    from rhpr.sasrec.daily_quality import _aggregate

    if not isinstance(records, list) or len(records) != len(references) or not records:
        raise ValueError("cached validation rank support mismatch")
    for record, reference in zip(records, references):
        if (any(record.get(key) != reference[key] for key in
                ("user_id", "anchor_id", "day", "reference_context_hash"))
                or record.get("target_known") is not (reference["target"] > 0)):
            raise ValueError("cached validation rank identity or target support mismatch")
        rank = record.get("rank_at_max_k")
        if (type(rank) is not int or not 0 <= rank <= 100 or record.get("rank_cutoff") != 100
                or not record["target_known"] and rank != 0
                or record.get("rank") != (rank if rank <= 10 else 0)):
            raise ValueError("cached validation rank censoring mismatch")
        for cutoff in (10, 100):
            hit = 0 < rank <= cutoff
            for metric, value in ((f"hr{cutoff}", float(hit)),
                                  (f"ndcg{cutoff}", 1 / math.log2(rank + 1) if hit else 0.0)):
                actual = record.get(metric)
                if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                        or not math.isfinite(actual) or not math.isclose(actual, value, rel_tol=0, abs_tol=1e-15)):
                    raise ValueError("cached validation metrics disagree with saved ranks")
    metrics, _ = _aggregate(records)
    if any(key not in expected or isinstance(expected[key], bool)
           or not isinstance(expected[key], (int, float)) or not math.isfinite(expected[key])
           or not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12)
           for key, value in metrics.items()):
        raise ValueError("cached validation ranks do not reproduce recorded aggregate metrics")


def _validate_contexts(rows: Any, references: list[dict[str, Any]], *, full: bool) -> None:
    if not isinstance(rows, list) or len(rows) != len(references):
        raise ValueError("cached validation input context support mismatch")
    for row, reference in zip(rows, references):
        original_length = reference["serving_context_length"]
        if (any(row.get(key) != reference[key] for key in
                ("user_id", "anchor_id", "reference_context_hash"))
                or row.get("original_full_context_sha256") != reference["serving_context_sha256"]
                or row.get("original_context_length") != original_length
                or row.get("actual_context_length") != (original_length if full else min(original_length, 1152))
                or row.get("query_and_raw_tail_preserved") is not True
                or row.get("retained_position_recencies_preserved") is not True
                or full and row.get("actual_context_sha256") != reference["serving_context_sha256"]):
            raise ValueError("cached validation input contexts disagree with pinned original contexts")
        for key in ("actual_context_sha256", "query_context_sha256", "raw_context_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("cached validation context hashes must be SHA256 digests")


def load_architecture_controls(config: dict[str, Any], evaluation: Any, output_dir: Path) -> dict[str, Any]:
    """Verify immutable completed controls, copy ranks and address mounted weights.

    Weights are hashed in place and never deserialized or copied by this loader.
    Only small validation rank/context artifacts are copied into the new output.
    """
    from rhpr.sasrec.query_history_ablation import _verified_file

    validate_architecture_config(config)
    controls, source_config = config["reused_controls"], _source_config(config)
    root = _resolve_control_root(config)
    documents = {key: json.loads(_verified_file(root, controls[key]["file"], controls[key]["sha256"]).read_text())
                 for key in ("report", "manifest", "experiment")}
    report, manifest = documents["report"], documents["manifest"]
    if (documents["experiment"] != source_config or report.get("config") != source_config
            or manifest.get("config") != source_config
            or report.get("source_code_sha") != controls["source_code_sha"]
            or manifest.get("source_code_sha") != controls["source_code_sha"]
            or manifest.get("format_version") != 3 or report.get("complete") is not True):
        raise ValueError("reused control source, configuration or completed manifest mismatch")
    references = _load_reference_contexts(config, evaluation)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for family in _FAMILIES:
        spec = controls["models"][family]
        result = copy.deepcopy(report.get("trials", {}).get(family, {}).get(spec["original_trial"], {}))
        original_trial = dict(_FIXED_LR, name=spec["original_trial"])
        if (result.get("status") != "success" or result.get("complete") is not True
                or result.get("trial") != original_trial or result.get("trial_name") != spec["original_trial"]
                or result.get("old_length") != 1024 or result.get("physical_batch_size") != 32
                or result.get("test_evaluations") != 0 or result.get("from_scratch") is not True
                or result.get("full_length_training_performed") is not False
                or result.get("full_validation_is_frozen_1k_checkpoint") is not True):
            raise ValueError("completed reused control trial or training protocol mismatch")
        proof = _policy_replay(config, result)
        checkpoint = result.get("best_checkpoint", {})
        expected = spec["checkpoint"]
        if (any(checkpoint.get(key) != expected[other] for key, other in
                (("path", "file"), ("sha256", "sha256"), ("size_bytes", "size_bytes"), ("best_epoch", "best_epoch")))
                or checkpoint.get("training_model") != family or checkpoint.get("trial") != original_trial
                or checkpoint.get("active_old_length") != 1024 or checkpoint.get("precision") != "float32"
                or checkpoint.get("best_epoch") != result["best_epoch"]
                or [row for row in manifest.get("checkpoints", []) if row.get("path") == expected["file"]] != [checkpoint]):
            raise ValueError("reused control best checkpoint disagrees with pinned report or manifest")
        weight_path = _verified_file(root, expected["file"], expected["sha256"], expected["size_bytes"])
        context_rows = {}
        for field, metric_field, rank_field in (("best_validation", "best_validation", "best_validation_rank_file"),
                                               ("full_validation", "best_full_validation", "full_validation_rank_file")):
            rank_spec, context_spec = spec[field], spec[field + "_contexts"]
            if result.get(rank_field) != rank_spec["file"]:
                raise ValueError("reused control rank filename disagrees with registered checkpoint")
            path = _verified_file(root, rank_spec["file"], rank_spec["sha256"])
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                records = json.load(stream)
            _validate_rank_metrics(records, result.get(metric_field, {}), references)
            context_path = _verified_file(root, context_spec["file"], context_spec["sha256"])
            with gzip.open(context_path, "rt", encoding="utf-8") as stream:
                rows = json.load(stream)
            _validate_contexts(rows, references, full=field == "full_validation")
            context_rows[field] = rows
            shutil.copyfile(path, output_dir / rank_spec["file"])
            shutil.copyfile(context_path, output_dir / context_spec["file"])
        if any(any(short[key] != full[key] for key in ("query_context_sha256", "raw_context_sha256"))
               for short, full in zip(context_rows["best_validation"], context_rows["full_validation"])):
            raise ValueError("reused control changed query or raw-tail context between proxy and full inference")
        trial = architecture_trials(config, family)[0]
        provenance = {"kernel": controls["kernel"], "source_code_sha": controls["source_code_sha"],
            "original_trial": original_trial, "report_sha256": controls["report"]["sha256"],
            "manifest_sha256": controls["manifest"]["sha256"],
            "experiment_sha256": controls["experiment"]["sha256"],
            "checkpoint_sha256": expected["sha256"], "checkpoint_size_bytes": expected["size_bytes"],
            "weights_verified_in_place": True, "weights_copied": False,
            "validation_ranks_and_contexts_verified": True, "test_ranks_read": False}
        result.update(trial=trial, trial_name=trial["name"], reused=True,
                      source_preflight=copy.deepcopy(report.get("preflight", {}).get(family, {})),
                      reused_provenance=provenance, stopping_policy_replay=proof,
                      resume_checkpoint_retained=False, training_performed_in_this_run=False)
        result.pop("resumable_checkpoint", None)
        result["best_checkpoint"] = dict(checkpoint, path=str(weight_path), trial=trial,
            original_path=expected["file"], original_trial=original_trial,
            source_code_sha=controls["source_code_sha"], reused=True,
            external_kernel=controls["kernel"], verified_mount_path=str(weight_path))
        results[family] = result
    return results
