from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _run_synthetic(config: dict[str, Any]) -> dict[str, int | float | str]:
    """Run a deterministic synthetic baseline used to verify the whole loop."""
    seed = int(config.get("seed", 42))
    samples = int(config.get("samples", 1000))
    features = int(config.get("features", 8))
    epochs = int(config.get("epochs", 100))
    learning_rate = float(config.get("learning_rate", 0.2))

    if samples < 100 or features < 1 or epochs < 1:
        raise ValueError("samples>=100, features>=1, and epochs>=1 are required")

    rng = random.Random(seed)
    true_weights = [rng.uniform(-1.5, 1.5) for _ in range(features)]
    rows: list[tuple[list[float], int]] = []
    for _ in range(samples):
        x = [rng.gauss(0.0, 1.0) for _ in range(features)]
        score = sum(a * b for a, b in zip(x, true_weights, strict=True))
        score += rng.gauss(0.0, 0.5)
        rows.append((x, int(score > 0.0)))

    rng.shuffle(rows)
    split = int(samples * 0.8)
    train, test = rows[:split], rows[split:]
    weights = [0.0] * features
    bias = 0.0

    for _ in range(epochs):
        gradients = [0.0] * features
        bias_gradient = 0.0
        for x, label in train:
            probability = _sigmoid(sum(a * b for a, b in zip(x, weights, strict=True)) + bias)
            error = probability - label
            for index, value in enumerate(x):
                gradients[index] += error * value
            bias_gradient += error
        scale = learning_rate / len(train)
        weights = [weight - scale * gradient for weight, gradient in zip(weights, gradients, strict=True)]
        bias -= scale * bias_gradient

    correct = 0
    loss = 0.0
    for x, label in test:
        probability = _sigmoid(sum(a * b for a, b in zip(x, weights, strict=True)) + bias)
        correct += int((probability >= 0.5) == bool(label))
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        loss -= label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)

    return {
        "experiment": str(config.get("name", "baseline")),
        "accuracy": correct / len(test),
        "log_loss": loss / len(test),
        "train_samples": len(train),
        "test_samples": len(test),
        "seed": seed,
    }


def run_experiment(
    config: dict[str, Any], output_dir: Path | None = None
) -> dict[str, int | float | str | bool]:
    """Dispatch a versioned experiment while keeping heavyweight imports optional."""

    kind = str(config.get("kind", "synthetic"))
    if kind == "gr_beauty_profile":
        if output_dir is None:
            raise ValueError("output_dir is required for gr_beauty_profile")
        from rhpr.gr.profiling import run_gr_profile
        return run_gr_profile(config, output_dir)
    if kind == "gr_beauty_baseline":
        if output_dir is None:
            raise ValueError("output_dir is required for gr_beauty_baseline")
        from rhpr.gr.experiment import run_gr_baseline
        return run_gr_baseline(config, output_dir)
    if kind == "synthetic":
        return _run_synthetic(config)
    if kind == "yambda_sasrec":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_sasrec")
        from rhpr.sasrec.experiment import run_yambda_experiment

        return run_yambda_experiment(config, output_dir)
    if kind == "yambda_sasrec_search":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_sasrec_search")
        from rhpr.sasrec.search import run_search

        return run_search(config, output_dir)
    if kind == "yambda_compression_pilot":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_compression_pilot")
        from rhpr.sasrec.pilot import run_compression_pilot

        return run_compression_pilot(config, output_dir)
    if kind == "yambda_mixed_depth":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_mixed_depth")
        from rhpr.sasrec.mixed_depth import run_mixed_depth

        return run_mixed_depth(config, output_dir)
    if kind == "yambda_tail_ablation":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_tail_ablation")
        from rhpr.sasrec.tail_ablation import run_tail_ablation

        return run_tail_ablation(config, output_dir)
    if kind == "yambda_recent128":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_recent128")
        from rhpr.sasrec.recent128 import run_recent128

        return run_recent128(config, output_dir)
    if kind == "yambda_backbone_context":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_backbone_context")
        from rhpr.sasrec.backbone_comparison import run_backbone_context

        return run_backbone_context(config, output_dir)
    if kind == "yambda_catalog_profile":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_catalog_profile")
        from rhpr.sasrec.catalog_profile import run_catalog_profile

        return run_catalog_profile(config, output_dir)
    if kind == "yambda_long_training_profile":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_long_training_profile")
        from rhpr.sasrec.long_training_profile import run_long_training_profile

        return run_long_training_profile(config, output_dir)
    if kind == "yambda_long_compression_quality":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_long_compression_quality")
        from rhpr.sasrec.long_quality import run_long_quality

        return run_long_quality(config, output_dir)
    if kind == "yambda_hstu_mixed_budget_daily":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_hstu_mixed_budget_daily")
        from rhpr.sasrec.long_mixed_quality import run_mixed_daily_quality

        return run_mixed_daily_quality(config, output_dir)
    if kind == "yambda_query_ssm_block_pilot":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_ssm_block_pilot")
        from rhpr.sasrec.query_pilot import run_query_pilot

        return run_query_pilot(config, output_dir)
    if kind == "yambda_query_lr_restart":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_lr_restart")
        from rhpr.sasrec.query_lr_restart import run_query_lr_restart

        return run_query_lr_restart(config, output_dir)
    if kind == "yambda_sasrec_prefix_query_pilot":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_sasrec_prefix_query_pilot")
        from rhpr.sasrec.query_prefix_pilot import run_query_prefix_pilot

        return run_query_prefix_pilot(config, output_dir)
    if kind == "yambda_query_history_ablation":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_history_ablation")
        from rhpr.sasrec.query_history_ablation import run_query_history_ablation

        return run_query_history_ablation(config, output_dir)
    if kind == "yambda_query_only_control":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_only_control")
        from rhpr.sasrec.query_only_control import run_query_only_control

        return run_query_only_control(config, output_dir)
    if kind == "yambda_query_prefix_proxy_hpo":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_prefix_proxy_hpo")
        from rhpr.sasrec.query_proxy_hpo import run_query_proxy_hpo

        return run_query_proxy_hpo(config, output_dir)
    if kind == "yambda_query_prefix_1k_search":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_prefix_1k_search")
        from rhpr.sasrec.query_1k_search import run_query_1k_search

        return run_query_1k_search(config, output_dir)
    if kind == "yambda_query_compressor_architecture":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_compressor_architecture")
        from rhpr.sasrec.query_architecture_search import run_query_architecture_search

        return run_query_architecture_search(config, output_dir)
    if kind == "yambda_query_baseline_screen":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_baseline_screen")
        from rhpr.sasrec.query_baseline_screen import run_query_baseline_screen

        return run_query_baseline_screen(config, output_dir)
    if kind == "yambda_query_qssm_search":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_qssm_search")
        from rhpr.sasrec.query_qssm_search import run_query_qssm_search

        return run_query_qssm_search(config, output_dir)
    if kind == "yambda_query_history_continuation":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_history_continuation")
        from rhpr.sasrec.query_history_continuation import run_history_continuation

        return run_history_continuation(config, output_dir)
    if kind == "yambda_query_mamba_ablation":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_mamba_ablation")
        from rhpr.sasrec.query_mamba_ablation import run_query_mamba_ablation

        return run_query_mamba_ablation(config, output_dir)
    if kind == "yambda_query_rotating_windows":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_rotating_windows")
        from rhpr.sasrec.query_rotating_windows import run_query_rotating_windows

        return run_query_rotating_windows(config, output_dir)
    if kind == "yambda_query_ssm2_benchmark":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_ssm2_benchmark")
        from rhpr.sasrec.query_ssm2_benchmark import run_query_ssm2_benchmark

        return run_query_ssm2_benchmark(config, output_dir)
    if kind == "yambda_query_ssm2_training":
        if output_dir is None:
            raise ValueError("output_dir is required for yambda_query_ssm2_training")
        from rhpr.sasrec.query_ssm2_training import run_query_ssm2_training

        return run_query_ssm2_training(config, output_dir)
    raise ValueError(f"unsupported experiment kind: {kind}")
