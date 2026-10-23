"""Optional squared-distance kernels for GRID's unchanged centroid updates.

The native ``cdist`` reduction avoids GRID's three-dimensional broadcast
temporary. Its matrix-multiplication path is deliberately disabled: subtracting
large squared norms can lose small distances and make self-distances positive.
GRID's initializer continues to use its upstream distance implementation.
"""
from __future__ import annotations

from typing import Any, Mapping


_CDIST_COMPUTE_MODE = "donot_use_mm_for_euclid_dist"


def distance_backend_report(config: Mapping[str, Any]) -> dict[str, str]:
    """Describe the selected assignment kernel without importing PyTorch."""
    backend = config.get("distance_backend", "upstream")
    if backend not in ("upstream", "torch_cdist"):
        raise ValueError("distance_backend must be 'upstream' or 'torch_cdist'")
    return {
        "distance_backend": backend,
        "distance_compute_mode": (
            _CDIST_COMPUTE_MODE if backend == "torch_cdist"
            else "upstream_broadcast_squared_l2"
        ),
        "initialization_distance_backend": "upstream",
    }


def make_grid_distance(grid: Any, config: Mapping[str, Any]) -> Any:
    """Build an assignment-distance object with GRID's ``compute`` interface."""
    backend = distance_backend_report(config)["distance_backend"]
    if backend == "upstream":
        return grid.SquaredEuclideanDistance()
    return TorchCdistSquaredEuclideanDistance()


class TorchCdistSquaredEuclideanDistance:
    """FP32/FP64 squared L2 with a bounded row chunk and no GEMM cancellation.

    This uses the same squared-Euclidean metric as GRID, with ordinary floating
    point rounding differences from its explicit subtract/square/sum kernel.
    It does not change normalization, initialization or cluster-count updates.
    """

    def compute(self, x: Any, y: Any, batch_size: int | None = 256) -> Any:
        import torch

        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError("distance inputs must be 2D with the same number of columns")
        if x.dtype not in (torch.float32, torch.float64) or y.dtype != x.dtype:
            raise ValueError("distance inputs must share a float32 or float64 dtype")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")

        def squared_distance(rows: Any) -> Any:
            return torch.cdist(
                rows, y, p=2, compute_mode=_CDIST_COMPUTE_MODE,
            ).square()

        if batch_size is None or x.shape[0] <= batch_size:
            return squared_distance(x)
        return torch.cat([
            squared_distance(x[start:start + batch_size])
            for start in range(0, x.shape[0], batch_size)
        ], dim=0)
