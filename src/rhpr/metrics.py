from __future__ import annotations

import math
from collections.abc import Mapping


Scalar = str | int | float | bool


def validate_metrics(metrics: Mapping[str, object]) -> dict[str, Scalar]:
    """Validate the stable, flat metrics.json contract."""
    if not metrics:
        raise ValueError("metrics must not be empty")

    clean: dict[str, Scalar] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int):
            clean[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"metric {key!r} is not finite")
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = value
        else:
            raise ValueError(f"metric {key!r} must be a scalar, got {type(value).__name__}")
    return clean
