"""One recursive compressor/backbone evaluated at its trained depth budgets."""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rhpr.metrics import validate_metrics
from rhpr.sasrec.data import YambdaSequences, load_yambda_sequences, resolve_yambda_parquet_path, validation_example
from rhpr.sasrec.diagnostics import context_hash, extract_history_features, write_diagnostic_rows
from rhpr.sasrec.experiment import run_yambda_experiment
from rhpr.sasrec.model import RecursivePairMLPSequenceCompressor


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_mixed_depth_config(config: dict[str, Any]) -> None:
    """Retain the common objective and evaluate only depths trained jointly."""
    if config.get("kind") != "yambda_mixed_depth":
        raise ValueError("kind must be yambda_mixed_depth")
    if not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base = config.get("base")
    if not isinstance(base, dict) or base.get("seed") != seed:
        raise ValueError("base.seed must match the experiment seed")
    if base.get("kind") != "yambda_sasrec":
        raise ValueError("base.kind must be yambda_sasrec")
    model, loss = base.get("model", {}), base.get("loss", {})
    if model.get("compressor_type") != "recursive_pair_mlp":
        raise ValueError("MixedDepth requires one recursive_pair_mlp")
    if model.get("use_time_features", False) is not False:
        raise ValueError("the first depth experiment uses Time OFF")
    depth = model.get("compression_depth", 1)
    if model.get("tail_size") != 128 or loss.get("tail_size") != 128:
        raise ValueError("model and loss must preserve the same raw tail128")
    _positive_integer(model.get("max_seq_len"), "max_seq_len")
    if loss.get("name") != "sampled_softmax":
        raise ValueError("MixedDepth retains the sampled_softmax training objective")
    _positive_integer(loss.get("num_negatives"), "num_negatives")
    evaluation = base.get("evaluation", {})
    if evaluation.get("evaluate_test") is not False or evaluation.get("k") != 10:
        raise ValueError("use full-catalog validation top10 and reserve test")
    if evaluation.get("max_validation_batches") is not None:
        raise ValueError("the depth experiment evaluates all validation users")
    training = base.get("training", {})
    epochs = _positive_integer(training.get("max_epochs"), "max_epochs")
    physical = _positive_integer(training.get("physical_batch_size"), "physical_batch_size")
    effective = _positive_integer(training.get("effective_batch_size"), "effective_batch_size")
    if effective % physical:
        raise ValueError("effective batch must be divisible by physical batch")
    if training.get("max_batches_per_epoch") is not None:
        raise ValueError("the depth experiment trains on all training users")
    mixed = training.get("mixed_depth")
    if not isinstance(mixed, dict):
        raise ValueError("training.mixed_depth must configure the depth distribution")
    probabilities = mixed.get("probabilities")
    if not isinstance(probabilities, list) or not 4 <= len(probabilities) <= 7:
        raise ValueError("provide four to seven probabilities for consecutive depths starting at zero")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0
           for p in probabilities) or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("depth probabilities must be finite, positive and sum to one")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in range(len(probabilities)):
        raise ValueError("compression_depth must belong to the trained depth distribution (maximum six)")
    interval = _positive_integer(mixed.get("validation_interval"), "validation_interval")
    if interval > epochs:
        raise ValueError("validation_interval cannot exceed max_epochs")
    budget = config.get("training_budget_seconds", 7200)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("training_budget_seconds must be finite and positive")
    _positive_integer(config.get("bootstrap_samples", 1000), "bootstrap_samples")
    _positive_integer(config.get("diagnostics", {}).get("representation_sample_size", 200), "representation_sample_size")


@torch.no_grad()
def _representation_rows(
    dataset: YambdaSequences, state: dict[str, torch.Tensor], *, max_seq_len: int,
    tail_size: int, sample_size: int, seed: int, depths: tuple[int, ...] = (0, 1, 2, 3),
) -> list[dict[str, Any]]:
    """Inspect old summaries on CPU, outside every inference timer."""
    table = state["item_embedding.weight"]
    compressor = RecursivePairMLPSequenceCompressor(table.shape[1], tail_size=tail_size)
    compressor.load_state_dict({key.removeprefix("compressor."): value
                                for key, value in state.items() if key.startswith("compressor.")})
    compressor.eval()
    indexes = np.sort(np.random.default_rng(seed).choice(len(dataset.splits),
                       size=min(sample_size, len(dataset.splits)), replace=False))
    rows: list[dict[str, Any]] = []
    for index in indexes:
        split = dataset.splits[int(index)]
        items, _ = validation_example(split, max_seq_len)
        n = len(items)
        old = max(0, n - tail_size)
        embeddings = table[torch.from_numpy(items)].unsqueeze(0)
        valid = torch.ones((1, n), dtype=torch.bool)
        for depth in depths:
            output, mask, endpoints, counts = compressor.forward_with_metadata(embeddings, valid, depth=depth)
            sources = endpoints[0, mask[0]]
            spans = counts[0, mask[0]]
            old_output = output[0, mask[0] & (endpoints[0] < old)].double()
            if spans.sum().item() != n or not torch.all(spans >= 1):
                raise RuntimeError("recursive summaries must cover every raw event exactly once")
            expected_tail = torch.arange(old, n)
            if not torch.equal(sources[-len(expected_tail):], expected_tail) or not torch.all(spans[-len(expected_tail):] == 1):
                raise RuntimeError("recursive compression changed the raw tail")
            statistics: dict[str, float | None] = {
                "old_norm_mean": None, "old_norm_p95": None,
                "old_centered_variance": None, "old_angular_spread": None,
            }
            if old_output.numel():
                if not torch.isfinite(old_output).all():
                    raise RuntimeError("non-finite recursive representations")
                norms = old_output.norm(dim=-1)
                statistics["old_norm_mean"] = norms.mean().item()
                statistics["old_norm_p95"] = torch.quantile(norms, .95).item()
                statistics["old_centered_variance"] = (old_output - old_output.mean(dim=0)).square().sum(dim=-1).mean().item()
                units = old_output[norms > 1e-12] / norms[norms > 1e-12, None]
                if len(units):
                    statistics["old_angular_spread"] = float(torch.clamp(1 - units.mean(dim=0).square().sum(), 0, 1).item())
            rows.append({"user_id": split.user_id, "split": "validation", "context_hash": context_hash(items),
                "compression_depth": depth, "window_length": n, "old_length": old,
                "output_length": int(mask.sum()), "old_output_length": len(old_output),
                "source_endpoints": sources.tolist(), "source_counts": spans.tolist(),
                **statistics})
    return rows


def run_mixed_depth(config: dict[str, Any], output_dir: Path, *,
                    dataset: YambdaSequences | None = None) -> dict[str, int | float | str | bool]:
    """Train once, keep one checkpoint, and describe each deterministic mode."""
    validate_mixed_depth_config(config)
    from rhpr.sasrec.depth_diagnostics import analyze_depth_cohorts
    import polars as pl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed = config["seed"]
    candidate = copy.deepcopy(config["base"])
    depths = tuple(range(len(candidate["training"]["mixed_depth"]["probabilities"])))
    depth_range = f"0..{depths[-1]}"
    candidate["name"] = config["name"]
    candidate["training"].update(keep_checkpoint=True,
        deadline_unix_seconds=started + config.get("training_budget_seconds", 7200))
    candidate["diagnostics"] = {"enabled": True}
    _json(output_dir / "training-config.json", candidate)

    def persist(complete: bool, results: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = validate_metrics({**(results or {}), "experiment": config["name"], "seed": seed,
            "mixed_depth_complete": complete, "test_evaluated": False,
            "planned_depths": len(depths), "mixed_depth_seconds": time.time() - started})
        _json(output_dir / "metrics.json", metrics)
        _json(output_dir / "run-manifest.json", {"git_sha": os.environ.get("GITHUB_SHA", "local"),
            "config": config, "started_at": datetime.fromtimestamp(started, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "planned_depths": list(depths),
            "mixed_depth_complete": complete})
        (output_dir / "summary.md").write_text(f"# MixedDepth\n\nComplete: {complete}; one shared checkpoint; test reserved.\n", encoding="utf-8")
        return metrics

    persist(False)
    if dataset is None:
        data = candidate["data"]
        dataset = load_yambda_sequences(resolve_yambda_parquet_path(Path(data["path"])),
            timestamp_unit=data.get("timestamp_unit", "seconds"))
    results = validate_metrics(run_yambda_experiment(candidate, output_dir, dataset=dataset))
    persist(False, results)
    for depth in depths:
        per_depth = json.loads((output_dir / "runs" / f"depth_{depth}" / "metrics.json").read_text())
        if (per_depth.get("checkpoint_sha256") != results.get("checkpoint_sha256")
                or per_depth.get("best_epoch") != results.get("best_epoch")
                or per_depth.get("compression_depth") != depth):
            raise RuntimeError("all trained depth modes must use the same selected checkpoint")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"phase": "mixed_depth_diagnostics_started"}), flush=True)
    state = torch.load(output_dir / "best-model.pt", map_location="cpu", weights_only=True)
    features = extract_history_features(dataset.splits, state["item_embedding.weight"].numpy(),
        max_seq_len=candidate["model"]["max_seq_len"], tail_size=128, split_name="validation",
        num_gap_buckets=candidate["model"].get("num_gap_buckets", 33) - 1,
        time_unit=candidate["data"].get("gap_time_unit", 1.0))
    for row in features:
        for depth in depths:
            row[f"output_length_depth_{depth}"] = row["tail_length"] + (row["old_length"] + 2**depth - 1) // 2**depth
    representation_rows = _representation_rows(dataset, state,
        max_seq_len=candidate["model"]["max_seq_len"], tail_size=128,
        sample_size=config.get("diagnostics", {}).get("representation_sample_size", 200), seed=seed,
        depths=depths)
    del state
    _json(output_dir / "compression-representations.json", {"schema_version": 1,
        "checkpoint": "best-model.pt", "checkpoint_sha256": results["checkpoint_sha256"],
        "selection_seed": seed, "sample_size": len(representation_rows) // len(depths),
        "depths": list(depths),
        "basis": "CPU FP32 compressor outputs before PE and input LayerNorm; old-prefix statistics only",
        "rows": representation_rows})
    pl.DataFrame(representation_rows, infer_schema_length=None,
        schema_overrides={name: pl.Float64 for name in
            ("old_norm_mean", "old_norm_p95", "old_centered_variance", "old_angular_spread")}).write_parquet(
                output_dir / "compression-representations.parquet")
    predictions = pl.read_parquet(output_dir / "predictions.parquet").to_dicts()
    write_diagnostic_rows(output_dir, features=features, predictions=predictions)
    _json(output_dir / "diagnostic-reference.json", {"checkpoint": "best-model.pt", "seed": seed,
        "checkpoint_sha256": results["checkpoint_sha256"],
        "git_sha": os.environ.get("GITHUB_SHA", "local"), "best_epoch": results["best_epoch"],
        "basis": "one shared MixedDepth checkpoint item embedding table before PE",
        "feature_output_length_basis": f"depth1; explicit output_length_depth_{depth_range} provided",
        "target_fields": "prediction-only; excluded from history-features",
        "trained_depths": list(depths),
        "checkpoint_selection": f"unweighted mean validation NDCG@10 of forced depths{depth_range}"})
    analysis = analyze_depth_cohorts(features, predictions, output_dir,
        bootstrap_seed=seed, bootstrap_samples=config.get("bootstrap_samples", 1000), depths=depths)
    metrics = results
    lines = ["# MixedDepth comparison", "", "One jointly trained model; all modes use the same validation-selected checkpoint.",
        f"Checkpoint selection: unweighted mean NDCG@10 at forced depths{depth_range}. Test reserved; one seed.", "",
        "| Depth | Old-prefix reduction | NDCG@10 | Recall@10 | History B1 p50 ms | Full B1 p50 ms | B32 users/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for depth in depths:
        prefix = f"depth_{depth}_"
        values = [str(depth), f"x{2**depth}", f"{metrics[prefix + 'validation_ndcg_at_10']:.6f}",
                  f"{metrics[prefix + 'validation_recall_at_10']:.6f}"]
        values.extend(f"{metrics[prefix + key]:.3f}" if prefix + key in metrics else "disabled"
            for key in ("inference_history_b1_p50_ms", "inference_full_b1_p50_ms", "inference_full_b32_users_per_second"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", f"Shared best epoch: {metrics['best_epoch']}; trained epochs: {metrics['epochs_trained']}.",
        "This screen does not establish convergence or the optimum of separately trained fixed-depth models.",
        "Forced homogeneous inference batches measure each budget. No learned router, local cuts or summary cache.",
        "See depth-cohorts.json for exploratory paired-user differences; a both-miss outcome does not establish safe compression.",
        "See compression-representations.json for actual raw spans, norms and spread; these diagnostics are outside latency timing."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(output_dir / "depth-results.json", {"checkpoint": "best-model.pt", "best_epoch": metrics["best_epoch"],
        "checkpoint_selection": "mean_forced_depth_validation_ndcg_at_10", "depths": list(depths),
        "cohort_count": len(analysis["cohorts"]), "complete": True})
    return persist(True, results)
