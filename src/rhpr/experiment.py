from __future__ import annotations

import math
import random
from typing import Any


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def run_experiment(config: dict[str, Any]) -> dict[str, int | float | str]:
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
