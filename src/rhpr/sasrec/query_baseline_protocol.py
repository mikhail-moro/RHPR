"""A pinned, validation-only comparison of four matched short-input baselines.

The saved CA/QSSM controls and every fresh model share target identities and
negative samples. Standalone sequence models keep their own native initialization.
Full-history inference is a secondary length-transfer diagnostic after 1K training.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from rhpr.sasrec.daily_evaluation_data import daily_context_sha256
from rhpr.sasrec.query_architecture_protocol import (
    _load_reference_contexts, _policy_replay, _validate_contexts,
    _validate_rank_metrics, validate_architecture_config,
)
from rhpr.sasrec.query_proxy_diagnostics import audited_hpo_examples
from rhpr.sasrec.query_proxy_protocol import hpo_batch_arrays, hpo_variants
from rhpr.sasrec.query_prefix_protocol import prefix_batch_arrays, prefix_variants
from rhpr.sasrec.query_only_protocol import (
    QUERY_ONLY_KIND, query_only_batch_arrays, query_only_variant,
)

BASELINE_KIND = "yambda_query_baseline_screen"
_REFERENCE_DIGEST = "0b7bed3f56489bafdced1fb87deeadba7c6fe988dc181938e523b59f114e0650"
_NAMES = ("query_cross_attention_m4", "query_ssm_m4", "recent128", "query_only_m4", "mamba4rec", "fuxi_linear")
_STANDALONE = {"mamba4rec": {"d_state": 32, "d_conv": 4, "expand": 2},
    "fuxi_linear": {"chunk_size": 128, "temporal_heads": 8, "positional_dim": 32}}
_FIXED_TRIAL = {"name": "fixed_lr", "backbone_lr": .001, "compressor_lr": .001}


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_baseline_config(config: dict[str, Any]) -> None:
    fields = {"name", "kind", "benchmark", "compressor", "data", "model", "parent", "runtime",
        "seed", "summary_budgets", "training", "hpo", "variants", "baseline_batches",
        "standalone_models", "reference_search", "kaggle"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("the registered matched baseline configuration fields are required exactly")
    if config["kind"] != BASELINE_KIND or config["name"] != "yambda500m-matched-query-baselines":
        raise ValueError("the named matched baseline screen is required")
    reference = config["reference_search"]
    if _canonical(reference) != _REFERENCE_DIGEST:
        raise ValueError("the completed architecture search and all reference artifact pins must remain exact")
    original = reference["experiment_config"]
    validate_architecture_config(original)
    for field in ("benchmark", "compressor", "data", "model", "parent", "runtime", "seed", "summary_budgets"):
        if config[field] != original[field]:
            raise ValueError(f"baseline screen changed parent {field}")
    if config["training"] != dict(original["training"], budget_seconds=21600):
        raise ValueError("baseline screen preserves optimization with a registered six-hour budget")
    if config["hpo"] != dict(original["hpo"], trials=[dict(_FIXED_TRIAL)]):
        raise ValueError("baseline screen fixes LR.001, min20/max60, validation10 and patience2")
    if config["variants"] != list(_NAMES) or config["standalone_models"] != _STANDALONE:
        raise ValueError("exactly the two controls and four native baseline architectures are registered")
    batches = {name: {"training": 16 if name == "fuxi_linear" else 32, "proxy": 32,
        "full": 4 if name == "fuxi_linear" else 8 if name == "mamba4rec" else 32} for name in _NAMES}
    if config["baseline_batches"] != batches:
        raise ValueError("registered physical and short/full evaluation batches are required")
    sources = dict(original["kaggle"], kernel_sources=[original["parent"]["kernel"],
        "carasik/rhpr-query-prefix-1k-search", reference["kernel"]])
    if config["kaggle"] != sources:
        raise ValueError("the pinned runtime/data and three immutable parent kernels must be mounted")


def baseline_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    compressed = hpo_variants(config)
    recent = next(v for v in prefix_variants(config["parent"]["experiment_config"]) if v["kind"] == "recent")
    return [*compressed, recent, query_only_variant(),
        {"name": "mamba4rec", "kind": "mamba4rec", "summary_tokens": 0},
        {"name": "fuxi_linear", "kind": "fuxi_linear", "summary_tokens": 0}]


def _registered(config: dict[str, Any], variant: dict[str, Any]) -> None:
    if variant not in baseline_variants(config):
        raise ValueError("a registered matched baseline variant is required")


def baseline_trials(config: dict[str, Any], variant: dict[str, Any]) -> list[dict[str, Any]]:
    _registered(config, variant)
    spec = config["reference_search"]["models"].get(variant["name"])
    return [copy.deepcopy(spec["trial"] if spec else _FIXED_TRIAL)]


def baseline_context_cap(config: dict[str, Any], variant: dict[str, Any], old_length: int | None) -> int:
    _registered(config, variant)
    if old_length is not None and (type(old_length) is not int or old_length != 1024):
        raise ValueError("the matched screen supports H1024 or full-history inference only")
    if variant["kind"] == "recent":
        return 128
    if variant["kind"] == "query_only":
        return 256
    return config["training"]["serving_context_length"] if old_length is None else old_length + 128


def baseline_evaluation_batch(config: dict[str, Any], variant: dict[str, Any], old_length: int | None) -> int:
    _registered(config, variant)
    return config["baseline_batches"][variant["name"]]["full" if old_length is None else "proxy"]


def baseline_batch_arrays(cohort: Any, windows: Any, config: dict[str, Any], variant: dict[str, Any],
                          *, old_length: int | None) -> dict[str, Any]:
    _registered(config, variant)
    baseline_context_cap(config, variant, old_length)
    if variant["kind"] == "recent":
        return prefix_batch_arrays(cohort, windows, config["parent"]["experiment_config"], variant)
    if variant["kind"] == "query_only":
        return query_only_batch_arrays(cohort, windows,
            {"kind": QUERY_ONLY_KIND, "parent": config["parent"]}, variant)
    # Input arrays do not depend on the predictor. Use the existing audited H
    # crop with an original registered variant, retaining original negatives.
    return hpo_batch_arrays(cohort, windows, config, hpo_variants(config)[0], old_length=old_length)


def baseline_evaluation_examples(examples: Any, old_length: int | None, variant: dict[str, Any],
                                 reference_contexts: list[dict[str, Any]], rows: list[dict[str, Any]],
                                 guard: Any) -> Any:
    """Audit the original full input before applying the model's actual cap."""
    # The shared audit records original support, target identity and Q/R hashes.
    # Recent128 additionally drops Q; correct that audit flag explicitly.
    name = variant["name"]
    if name not in _NAMES or old_length is not None and (type(old_length) is not int or old_length != 1024):
        raise ValueError("a registered baseline and H1024/full inference length are required")
    source_length = 128 if name == "query_only_m4" else old_length
    for example in audited_hpo_examples(examples, source_length, reference_contexts, rows, guard):
        if name == "recent128":
            from dataclasses import replace
            example = replace(example, context=example.context[-128:].copy(),
                              timestamps=example.timestamps[-128:].copy())
            rows[-1].update(actual_context_length=len(example.context),
                actual_context_sha256=daily_context_sha256(example.user_id, example.context, example.timestamps),
                query_and_raw_tail_preserved=False, raw_tail_preserved=True,
                query_input_retained=False)
        else:
            rows[-1].update(raw_tail_preserved=True, query_input_retained=True)
        yield example


def _resolve_kernel_root(kernel: str, configured: str, input_root: str | Path = "/kaggle/input") -> Path:
    base = Path(input_root).resolve()
    owner, slug = kernel.split("/")
    candidates = {base / slug, base / owner / slug, base / "notebooks" / owner / slug,
                  base / "kernels" / owner / slug, Path(configured)}
    found = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        root = candidate.resolve()
        try:
            root.relative_to(base)
        except ValueError as error:
            raise ValueError("reference kernel mount escapes input root") from error
        if root.name == slug:
            found.add(root)
    if len(found) != 1:
        raise ValueError(f"expected one mounted reference kernel {kernel}; found {len(found)}")
    return next(iter(found))


def load_baseline_controls(config: dict[str, Any], evaluation: Any, output_dir: Path) -> dict[str, Any]:
    """Verify both source generations; copy selected weights for portable output."""
    from rhpr.sasrec.query_history_ablation import _verified_file
    from rhpr.sasrec.query_prefix_pilot import _checkpoint_record

    validate_baseline_config(config)
    reference = config["reference_search"]
    root = _resolve_kernel_root(reference["kernel"], reference["checkpoint_root"])
    documents = {name: json.loads(_verified_file(root, name, spec["sha256"]).read_text())
                 for name, spec in reference["documents"].items()}
    report, manifest = documents["query-compressor-architecture.json"], documents["checkpoint-manifest.json"]
    expected_config = reference["experiment_config"]
    if (report.get("complete") is not True or report.get("source_code_sha") != reference["source_code_sha"]
            or report.get("config") != expected_config or manifest.get("config") != expected_config
            or manifest.get("source_code_sha") != reference["source_code_sha"]
            or manifest.get("format_version") != 3
            or documents["experiment.json"] != expected_config):
        raise ValueError("reference source, configuration or completion mismatch")
    contexts = _load_reference_contexts(config, evaluation)
    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    controls = {}
    for name, spec in reference["models"].items():
        trial = spec["trial"]
        result = copy.deepcopy(report["trials"][name][trial["name"]])
        if (result.get("status") != "success" or result.get("complete") is not True
                or result.get("trial") != trial or result.get("trial_name") != trial["name"]
                or result.get("old_length") != 1024 or result.get("physical_batch_size") != 32
                or result.get("from_scratch") is not True
                or result.get("full_validation_is_frozen_1k_checkpoint") is not True
                or result.get("test_evaluations") != 0
                or result.get("full_length_training_performed") is not False):
            raise ValueError("reference trial is not a completed matched short-history control")
        proof = _policy_replay(config, result)
        saved = result["best_checkpoint"]
        expected = spec["checkpoint"]
        original_path = saved.get("original_path", saved["path"])
        if (original_path != expected["file"] or any(saved.get(key) != expected[key]
                for key in ("sha256", "size_bytes", "best_epoch"))
                or saved.get("best_epoch") != result["best_epoch"]
                or [row for row in manifest["checkpoints"] if row.get("path") == saved["path"]] != [saved]):
            raise ValueError("reference checkpoint disagrees with pinned manifest or selected epoch")
        weight_root = _resolve_kernel_root(spec["weight_kernel"], spec["weight_root"])
        weights = _verified_file(weight_root, expected["file"], expected["sha256"], expected["size_bytes"])
        context_rows = {}
        for field, metric in (("best_validation_rank_file", "best_validation"),
                              ("full_validation_rank_file", "best_full_validation")):
            artifact, context = spec["artifacts"][field], spec["artifacts"][field + "_contexts"]
            if result[field] != artifact["file"]:
                raise ValueError("reference rank filename disagrees with selected checkpoint")
            path = _verified_file(root, artifact["file"], artifact["sha256"])
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                records = json.load(stream)
            _validate_rank_metrics(records, result[metric], contexts)
            context_path = _verified_file(root, context["file"], context["sha256"])
            with gzip.open(context_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            _validate_contexts(payload, contexts, full=field.startswith("full"))
            context_rows[field] = payload
            shutil.copyfile(path, output_dir / artifact["file"])
            shutil.copyfile(context_path, output_dir / context["file"])
        if any(any(short[key] != full[key] for key in ("query_context_sha256", "raw_context_sha256"))
               for short, full in zip(context_rows["best_validation_rank_file"], context_rows["full_validation_rank_file"])):
            raise ValueError("reference Q/R contexts changed between inference lengths")
        destination = output_dir / "checkpoints" / f"{name}-{trial['name']}.pt"
        shutil.copyfile(weights, destination)
        provenance = {"search_kernel": reference["kernel"], "search_source_code_sha": reference["source_code_sha"],
            "weight_kernel": spec["weight_kernel"], "weight_source_code_sha": spec["weight_source_code_sha"],
            "original_path": expected["file"], "original_sha256": expected["sha256"],
            "weight_bytes_verified": True, "weights_copied": True, "test_ranks_read": False}
        record = _checkpoint_record(destination, output_dir, training_model=name, trial=trial,
            best_epoch=result["best_epoch"], precision="float32", active_old_length=1024,
            copied_reference=True, source_code_sha=spec["weight_source_code_sha"], provenance=provenance)
        if record["sha256"] != expected["sha256"] or record["size_bytes"] != expected["size_bytes"]:
            raise ValueError("copied reference weights do not match pinned bytes")
        result.update(reused=True, training_performed_in_this_run=False, reused_provenance=provenance,
            stopping_policy_replay=proof, best_checkpoint=record, resume_checkpoint_retained=False,
            source_preflight=copy.deepcopy(report.get("preflight", {}).get(f"{name}-{trial['name']}", {})))
        result.pop("resumable_checkpoint", None)
        controls[name] = result
    return controls
