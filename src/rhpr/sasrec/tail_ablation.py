"""Evaluate removal of the old prefix using an immutable MixedDepth checkpoint."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rhpr.metrics import validate_metrics
from rhpr.sasrec.benchmark import benchmark_inference
from rhpr.sasrec.data import YambdaSequences, load_yambda_sequences, resolve_yambda_parquet_path, validation_example
from rhpr.sasrec.diagnostics import context_hash, write_diagnostic_rows
from rhpr.sasrec.experiment import _EvaluationDataset, _collate_evaluation, _full_catalog_ndcg, _loader, _seed_everything
from rhpr.sasrec.model import SASRecModel


MODES = (("depth_0", 0), ("depth_3", 3), ("tail_128", 0))


class TailOnlySequenceCompressor(nn.Module):
    """Retain the raw recent tail with positions in the original full history."""

    def __init__(self, tail_size: int = 128) -> None:
        super().__init__()
        if isinstance(tail_size, bool) or not isinstance(tail_size, int) or tail_size < 1:
            raise ValueError("tail_size must be a positive integer")
        self.tail_size = tail_size

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("embeddings and boolean valid_mask must have matching batch/length shapes")
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("tail compression requires nonempty histories")
        raw_offsets = torch.arange(valid_mask.shape[1], device=embeddings.device)
        if not torch.equal(valid_mask, raw_offsets[None, :] < lengths[:, None]):
            raise ValueError("tail compression requires contiguous histories with right padding")
        tail_lengths = lengths.clamp(max=self.tail_size)
        offsets = torch.arange(int(tail_lengths.max().item()), device=embeddings.device)
        output_mask = offsets[None, :] < tail_lengths[:, None]
        sources = offsets[None, :] + (lengths - tail_lengths)[:, None]
        sources = sources.masked_fill(~output_mask, -1)
        output = torch.gather(embeddings, 1, sources.clamp(min=0).unsqueeze(-1).expand(-1, -1, embeddings.shape[2]))
        return output.masked_fill(~output_mask.unsqueeze(-1), 0), output_mask, sources


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_tail_ablation_config(config: dict[str, Any]) -> None:
    if config.get("kind") != "yambda_tail_ablation":
        raise ValueError("kind must be yambda_tail_ablation")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    if type(config.get("seed")) is not int or config["seed"] != 42:
        raise ValueError("the frozen-checkpoint ablation retains seed42")
    if type(config.get("tail_size")) is not int or config["tail_size"] != 128:
        raise ValueError("the ablation retains raw tail128")
    if "training" in config or "model" in config or "data" in config:
        raise ValueError("read the immutable source model/data recipe; this ablation has no training")
    source = config.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must pin the prior MixedDepth run")
    if not isinstance(source.get("kernel_id"), str) or not re.fullmatch(r"[\w-]+/[\w-]+", source["kernel_id"]):
        raise ValueError("source.kernel_id must identify one private source kernel")
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("source.path must identify the attached output")
    for name, size in (("checkpoint_sha256", 64), ("git_sha", 40)):
        if not isinstance(source.get(name), str) or not re.fullmatch(rf"[0-9a-f]{{{size}}}", source[name]):
            raise ValueError(f"source.{name} must be a pinned hexadecimal digest")
    for name in ("expected_best_epoch", "expected_validation_users"):
        _positive_int(source.get(name), f"source.{name}")
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict) or runtime.get("device") not in ("cpu", "cuda"):
        raise ValueError("runtime.device must explicitly select cpu or cuda")
    if any(type(runtime.get(name)) is not bool for name in ("amp", "deterministic")):
        raise ValueError("runtime amp and deterministic must be boolean")
    if isinstance(runtime.get("workers"), bool) or not isinstance(runtime.get("workers"), int) or runtime["workers"] < 0:
        raise ValueError("runtime.workers must be nonnegative")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("evaluate_test") is not False or evaluation.get("k") != 10:
        raise ValueError("use full-catalog validation top10 and reserve test")
    if evaluation.get("max_validation_batches") is not None:
        raise ValueError("evaluate all validation users")
    for name in ("batch_size", "item_chunk_size"):
        _positive_int(evaluation.get(name), f"evaluation.{name}")
    _positive_int(config.get("bootstrap_samples", 1000), "bootstrap_samples")
    benchmark = config.get("benchmark", {})
    if not isinstance(benchmark, dict) or type(benchmark.get("enabled", False)) is not bool:
        raise ValueError("benchmark.enabled must be boolean")


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(directory: Path, name: str) -> list[dict[str, Any]]:
    path = directory / f"{name}.json"
    if path.is_file():
        rows = _read_json(path)
    else:
        import polars as pl
        rows = pl.read_parquet(directory / f"{name}.parquet").to_dicts()
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"source {name} must contain rows")
    return rows


def _resolve_source(source: dict[str, Any], source_dir: Path | None) -> Path:
    requested = Path(source_dir if source_dir is not None else source["path"])
    if (requested / "best-model.pt").is_file():
        return requested
    if source_dir is not None:
        raise FileNotFoundError(f"source override has no best-model.pt: {requested}")
    slug = source["kernel_id"].split("/", 1)[1]
    candidates = [path.parent for path in Path("/kaggle/input").rglob("best-model.pt")
                  if any(parent.name == slug for parent in path.parents)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one attached {slug} output, found {len(candidates)}")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_source(config: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Check lineage and metadata before deserializing checkpoint tensors."""
    pinned = config["source"]
    checksum = _sha256(directory / "best-model.pt")
    if checksum != pinned["checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA256 mismatch")
    manifest = _read_json(directory / "run-manifest.json")
    recipe = _read_json(directory / "model-config.json")
    metrics = _read_json(directory / "metrics.json")
    reference = _read_json(directory / "diagnostic-reference.json")
    bins = _read_json(directory / "feature_bins.json")
    if manifest.get("git_sha") != pinned["git_sha"]:
        raise ValueError("source git revision mismatch")
    source_config = manifest.get("config", {})
    if source_config.get("kind") != "yambda_mixed_depth" or source_config.get("seed") != config["seed"]:
        raise ValueError("source manifest must describe the seed42 MixedDepth run")
    for field in ("data", "model", "loss", "evaluation", "seed"):
        if source_config.get("base", {}).get(field) != recipe.get(field):
            raise ValueError(f"source manifest/model recipe mismatch for {field}")
    if recipe.get("kind") != "yambda_sasrec" or recipe.get("seed") != config["seed"]:
        raise ValueError("source model recipe must retain SASRec and seed42")
    model = recipe.get("model", {})
    if model.get("compressor_type") != "recursive_pair_mlp" or model.get("use_time_features") is not False:
        raise ValueError("source model must be RecursivePairMLP with Time OFF")
    if model.get("tail_size") != 128 or recipe.get("loss", {}).get("tail_size") != 128:
        raise ValueError("source model and loss must preserve tail128")
    _positive_int(model.get("max_seq_len"), "source max_seq_len")
    if recipe.get("evaluation", {}).get("evaluate_test") is not False or recipe.get("evaluation", {}).get("k") != 10:
        raise ValueError("source must use validation-only full-catalog top10")
    if metrics.get("mixed_depth_complete") is not True or metrics.get("test_evaluated") is not False:
        raise ValueError("source run must be complete with test reserved")
    if metrics.get("seed") != config["seed"] or metrics.get("checkpoint_sha256") != checksum:
        raise ValueError("source metrics seed/checkpoint mismatch")
    if metrics.get("best_epoch") != pinned["expected_best_epoch"]:
        raise ValueError("source selected epoch mismatch")
    users = pinned["expected_validation_users"]
    for field in ("num_users", "validation_examples", "depth_0_validation_examples", "depth_3_validation_examples"):
        if metrics.get(field) != users:
            raise ValueError(f"source {field} mismatch")
    _positive_int(metrics.get("num_items"), "source num_items")
    if metrics.get("epochs_trained", 0) < pinned["expected_best_epoch"]:
        raise ValueError("source selected epoch exceeds trained epochs")
    expected_reference = {"checkpoint": "best-model.pt", "checkpoint_sha256": checksum,
                          "git_sha": pinned["git_sha"], "seed": config["seed"],
                          "best_epoch": pinned["expected_best_epoch"]}
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("source diagnostic reference checkpoint/revision/epoch mismatch")
    if bins.get("fit_split") != "validation" or bins.get("validation_user_count") != users:
        raise ValueError("source bins must be frozen on the matching validation users")
    return {"recipe": recipe, "metrics": metrics, "manifest": manifest, "reference": reference,
            "bins": bins, "features": _read_rows(directory, "history-features"),
            "predictions": _read_rows(directory, "predictions"), "checkpoint_sha256": checksum}


def _join_source(source: dict[str, Any], dataset: YambdaSequences, max_seq_len: int, seed: int
                 ) -> tuple[dict[int, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Require exact users, full contexts, mapped targets and source features."""
    if dataset.num_items != source["metrics"]["num_items"] or len(dataset.splits) != source["metrics"]["num_users"]:
        raise ValueError("source dataset catalogue/user count mismatch")
    observed: dict[int, dict[str, Any]] = {}
    for split in dataset.splits:
        if split.user_id in observed:
            raise ValueError("dataset contains duplicate user IDs")
        context, target = validation_example(split, max_seq_len)
        observed[split.user_id] = {"context_hash": context_hash(context), "context": context,
                                   "target": target, "window_length": len(context),
                                   "observed_length_raw": len(split.train)}
    features: dict[int, dict[str, Any]] = {}
    for row in source["features"]:
        user = row.get("user_id")
        if user not in observed or user in features or row.get("split") != "validation":
            raise ValueError("source feature users/split must exactly match validation")
        expected = observed[user]
        if row.get("context_hash") != expected["context_hash"] or row.get("window_length") != expected["window_length"]:
            raise ValueError("source feature full context/length mismatch")
        old = max(0, expected["window_length"] - 128)
        if row.get("old_length") != old or row.get("tail_length") != min(expected["window_length"], 128):
            raise ValueError("source feature prefix/tail length mismatch")
        if row.get("observed_length_raw") != expected["observed_length_raw"]:
            raise ValueError("source feature observed history length mismatch")
        features[user] = row
    if features.keys() != observed.keys():
        raise ValueError("source feature users do not exactly match the dataset")
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source["predictions"]:
        label, user = row.get("model"), row.get("user_id")
        if label not in {f"depth_{depth}" for depth in range(4)} or user not in observed:
            raise ValueError("source predictions contain unexpected modes/users")
        key = label, user
        if key in predictions:
            raise ValueError("source predictions contain duplicate mode/user keys")
        expected = observed[user]
        if (row.get("split") != "validation" or row.get("seed") != seed or row.get("time_enabled") is not False
                or row.get("compression_depth") != int(label[-1])
                or row.get("context_hash") != expected["context_hash"] or row.get("target") != expected["target"]):
            raise ValueError("source prediction identity/context/target mismatch")
        top10 = row.get("top10")
        if not isinstance(top10, list) or len(top10) != min(10, dataset.num_items) or len(set(top10)) != len(top10):
            raise ValueError("source predictions must contain unique full-catalog top10 item IDs")
        if any(type(item) is not int or not 1 <= item <= dataset.num_items for item in top10):
            raise ValueError("source top10 contains invalid mapped item IDs")
        predictions[key] = row
    expected_keys = {(f"depth_{depth}", user) for depth in range(4) for user in observed}
    if predictions.keys() != expected_keys:
        raise ValueError("source prediction modes/users do not exactly match the dataset")
    return observed, predictions


@torch.no_grad()
def run_tail_ablation(config: dict[str, Any], output_dir: Path, *,
                      dataset: YambdaSequences | None = None, source_dir: Path | None = None
                      ) -> dict[str, int | float | str | bool]:
    """Compare full, depth3 and tail-only inference; never train or read test targets."""
    validate_tail_ablation_config(config)
    output_dir = Path(output_dir)
    source_root = _resolve_source(config["source"], source_dir)
    if output_dir.resolve() == source_root.resolve() or source_root.resolve() in output_dir.resolve().parents:
        raise ValueError("ablation outputs must be separate from the immutable source")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = config["seed"]
    results: dict[str, int | float | str | bool] = {}

    def persist(complete: bool) -> dict[str, int | float | str | bool]:
        metrics = validate_metrics({**results, "experiment": config["name"], "seed": seed,
            "source_checkpoint_sha256": config["source"]["checkpoint_sha256"],
            "source_git_sha": config["source"]["git_sha"], "source_kernel_id": config["source"]["kernel_id"],
            "training_performed": False, "training_epochs": 0, "test_evaluated": False,
            "tail_ablation_complete": complete, "tail_ablation_seconds": time.time() - started})
        _json(output_dir / "metrics.json", metrics)
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "source_path": str(source_root),
            "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "tail_ablation_complete": complete})
        (output_dir / "summary.md").write_text(
            f"# Tail ablation\n\nComplete: {complete}; frozen source checkpoint; no training; test reserved.\n", encoding="utf-8")
        if not complete:
            (output_dir / "comparison.md").write_text(
                "# Tail ablation\n\nIncomplete; do not interpret partial results.\n", encoding="utf-8")
        return metrics

    persist(False)
    print(json.dumps({"phase": "tail_ablation_source_verification", "source": str(source_root)}), flush=True)
    source = _verified_source(config, source_root)
    recipe = source["recipe"]
    if dataset is None:
        data = recipe["data"]
        dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
            timestamp_unit=data.get("timestamp_unit", "seconds"))
    max_seq_len = recipe["model"]["max_seq_len"]
    observed, prior_predictions = _join_source(source, dataset, max_seq_len, seed)
    runtime, evaluation = config["runtime"], config["evaluation"]
    _seed_everything(seed, runtime["deterministic"])
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp = runtime["amp"] and device.type == "cuda"
    # SHA256, metadata and exact dataset identities were checked before loading.
    state = torch.load(source_root / "best-model.pt", map_location="cpu", weights_only=True)
    model = SASRecModel(num_items=dataset.num_items, **recipe["model"])
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device).eval()
    original_compressor = model.compressor
    original_type, original_depth = model.compressor_type, model.compression_depth
    validation_loader = _loader(_EvaluationDataset(dataset.splits, max_seq_len, "validation"),
        batch_size=evaluation["batch_size"], shuffle=False, workers=runtime["workers"], seed=seed,
        collate_fn=_collate_evaluation, device=device)
    predictions: list[dict[str, Any]] = []
    reproduction: dict[str, Any] = {"checkpoint_sha256": source["checkpoint_sha256"],
        "source_git_sha": config["source"]["git_sha"], "identity_match_required": True,
        "bitwise_top10_match_required": False, "modes": {}}
    options = copy.deepcopy(config.get("benchmark", {}))
    contexts: list[np.ndarray] = []
    if options.get("enabled", False):
        indexes = np.random.default_rng(seed).choice(len(dataset.splits),
            size=min(int(options.get("real_sample_size", 200)), len(dataset.splits)), replace=False)
        contexts = [validation_example(dataset.splits[int(index)], max_seq_len)[0] for index in indexes]
    try:
        for label, depth in MODES:
            model.compressor = original_compressor
            model.compressor_type = original_type
            model.set_compression_depth(depth)
            if label == "tail_128":
                model.compressor = TailOnlySequenceCompressor(config["tail_size"])
                model.compressor_type = "tail_only"
            directory = output_dir / "runs" / label
            directory.mkdir(parents=True, exist_ok=True)
            details: dict[str, Any] = {}
            rows: list[dict[str, Any]] = []
            evaluation_started = time.perf_counter()
            print(json.dumps({"phase": "tail_ablation_evaluation", "model": label}), flush=True)
            ndcg, examples = _full_catalog_ndcg(model, validation_loader, device=device, k=evaluation["k"],
                item_chunk_size=evaluation["item_chunk_size"], amp=amp, phase="validation", details=details,
                prediction_rows=rows, model_label=label, seed=seed)
            if examples != len(dataset.splits):
                raise ValueError("ablation must evaluate every source validation user")
            for row in rows:
                expected = observed[row["user_id"]]
                if row["context_hash"] != expected["context_hash"] or row["target"] != expected["target"]:
                    raise ValueError("ablation prediction full context/target mismatch")
                actual = expected["context"][-config["tail_size"]:] if label == "tail_128" else expected["context"]
                row.update({"compression_depth": depth, "reference_context_hash": expected["context_hash"],
                    "reference_window_length": expected["window_length"],
                    "input_context_hash": context_hash(actual), "input_length": len(actual)})
            predictions.extend(rows)
            mode_metrics: dict[str, int | float | str | bool] = {
                "model": label, "compression_depth": depth, "checkpoint": str(source_root / "best-model.pt"),
                "checkpoint_sha256": source["checkpoint_sha256"], "source_best_epoch": config["source"]["expected_best_epoch"],
                "validation_ndcg_at_10": ndcg, "validation_recall_at_10": details["recall_at_10"],
                "validation_examples": examples, "evaluation_seconds": time.perf_counter() - evaluation_started,
                "training_performed": False, "test_evaluated": False, "max_seq_len": max_seq_len,
            }
            if label != "tail_128":
                prior = [prior_predictions[label, row["user_id"]] for row in rows]
                prior_ndcg = sum(row["ndcg_at_10"] for row in prior) / examples
                prior_recall = sum(row["recall_at_10"] for row in prior) / examples
                matched = sum(row["top10"] == old["top10"] for row, old in zip(rows, prior, strict=True))
                reproduction["modes"][label] = {"users": examples, "identity_matches": examples,
                    "top10_equal_count": matched, "top10_changed_count": examples - matched,
                    "source_ndcg_at_10": prior_ndcg, "current_ndcg_at_10": ndcg,
                    "delta_ndcg_at_10": ndcg - prior_ndcg, "source_recall_at_10": prior_recall,
                    "current_recall_at_10": details["recall_at_10"],
                    "delta_recall_at_10": details["recall_at_10"] - prior_recall}
                _json(output_dir / "reproduction-check.json", reproduction)
            if options.get("enabled", False):
                # Clear prior cases' allocator cache outside every timed range,
                # matching the original MixedDepth benchmark protocol.
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                mode_options = {**options, "seed": seed, "checkpoint": str(source_root / "best-model.pt"),
                    "checkpoint_sha256": source["checkpoint_sha256"], "compression_depth": depth,
                    "mode": label, "source_revision": config["source"]["git_sha"],
                    "revision": os.environ.get("GITHUB_SHA", "local")}
                mode_metrics.update(benchmark_inference(model, contexts, device=device, amp=amp,
                    item_chunk_size=evaluation["item_chunk_size"], config=mode_options, output_dir=directory,
                    k=evaluation["k"]))
            _json(directory / "metrics.json", validate_metrics(mode_metrics))
            for key, value in mode_metrics.items():
                results[f"{label}_{key}"] = value
            persist(False)
    finally:
        model.compressor = original_compressor
        model.compressor_type = original_type
        model.set_compression_depth(original_depth)

    current = {(row["model"], row["user_id"]): row for row in predictions}
    short_users = [user for user, row in observed.items() if row["window_length"] <= config["tail_size"]]
    reproduction["short_context_sanity"] = {"user_count": len(short_users), "comparisons": {}}
    for label in ("depth_3", "tail_128"):
        pairs = [(current["depth_0", user], current[label, user]) for user in short_users]
        reproduction["short_context_sanity"]["comparisons"][label] = {
            "top10_equal_count": sum(reference["top10"] == other["top10"] for reference, other in pairs),
            "ndcg_equal_count": sum(reference["ndcg_at_10"] == other["ndcg_at_10"] for reference, other in pairs),
            "max_abs_ndcg_difference": max((abs(reference["ndcg_at_10"] - other["ndcg_at_10"])
                                             for reference, other in pairs), default=0.0),
            "recall_equal_count": sum(reference["recall_at_10"] == other["recall_at_10"]
                                      for reference, other in pairs),
        }
    _json(output_dir / "reproduction-check.json", reproduction)
    write_diagnostic_rows(output_dir, features=source["features"], predictions=predictions)
    _json(output_dir / "source-reference.json", {"source_path": str(source_root),
        "kernel_id": config["source"]["kernel_id"], "checkpoint": "best-model.pt",
        "checkpoint_sha256": source["checkpoint_sha256"], "git_sha": config["source"]["git_sha"],
        "best_epoch": config["source"]["expected_best_epoch"],
        "full_context_hash_preserved": True, "source_features_and_bins_reused": True})
    from rhpr.sasrec.tail_diagnostics import analyze_tail_cohorts
    analyze_tail_cohorts(source["features"], predictions, output_dir, bootstrap_seed=seed,
        bootstrap_samples=config.get("bootstrap_samples", 1000), frozen_bins=source["bins"])
    lines = ["# Tail ablation", "", "One frozen MixedDepth checkpoint; no training; validation only.", "",
        "| Mode | NDCG@10 | Recall@10 | History B1 p50 ms | Full B1 p50 ms | B32 users/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, _ in MODES:
        values = [label, f"{results[label + '_validation_ndcg_at_10']:.6f}",
                  f"{results[label + '_validation_recall_at_10']:.6f}"]
        values.extend(f"{results[label + '_' + name]:.3f}" if label + "_" + name in results else "disabled"
            for name in ("inference_history_b1_p50_ms", "inference_full_b1_p50_ms", "inference_full_b32_users_per_second"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Tail-only drops the old prefix inside the compressor while retaining original raw recency positions.",
        "Full source context hashes define paired comparisons; input_context_hash separately records the retained tail.",
        "This is an inference ablation of jointly trained weights, not a separately trained tail-only model.",
        "Benchmarking includes prefix removal; preprocessing and serving queues follow the existing benchmark exclusions.",
        "Reproduction of depth0/depth3 is reported below; GPU top10 need not be bitwise identical."])
    for label, values in reproduction["modes"].items():
        lines.append(f"- {label}: {values['top10_equal_count']}/{values['users']} identical top10; "
                     f"delta NDCG={values['delta_ndcg_at_10']:.8f}; delta Recall={values['delta_recall_at_10']:.8f}.")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    results.update({"num_users": len(dataset.splits), "num_items": dataset.num_items,
                    "raw_max_seq_len": max_seq_len, "tail_size": config["tail_size"]})
    return persist(True)
