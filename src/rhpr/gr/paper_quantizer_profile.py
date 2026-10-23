"""Bounded operator timing for GRID's rank-local RK-Means update.

Logical ranks execute sequentially on one device, retaining their separate
running counts. This matches the initialized DDP gradient arithmetic, not the
wall time or communication of eight physical GPUs. Inputs are runtime proxies;
this module does not fit or evaluate a recommendation tokenizer.
"""
from __future__ import annotations

import time


def logical_rank_step(replicas, optimizer, batches, *, residual_prediction_layers=()):
    """Apply one actual-source loss per rank and average gradients before SGD.

    Optional frozen prediction layers measure the rest of the three-stage
    residual forward pass. Every rank observes common pre-update centroids;
    only their cumulative cluster counts remain separate between updates.
    """
    import torch

    if not replicas or len(replicas) != len(batches):
        raise ValueError("logical ranks require one nonempty batch per replica")
    first = replicas[0]
    if any(not model.is_initialized or model.update_manually for model in replicas):
        raise ValueError("logical rank updates require initialized gradient-based GRID models")
    if any(model.centroids.shape != first.centroids.shape for model in replicas):
        raise ValueError("logical rank centroids must share the same shape")
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_parameters) != 1 or optimizer_parameters[0] is not first.centroids:
        raise ValueError("optimizer must update only the first logical rank's centroids")
    with torch.no_grad():
        for replica in replicas[1:]:
            replica.centroids.copy_(first.centroids)
    losses = []
    for replica, batch in zip(replicas, batches):
        if batch.ndim != 2 or not len(batch) or batch.shape[1] != first.centroids.shape[1]:
            raise ValueError("logical rank batch has a different feature shape")
        replica.zero_grad(set_to_none=True)
        assignments, embeddings, loss = replica.model_step(batch)
        if loss is None or not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite logical rank quantizer loss")
        if residual_prediction_layers:
            # These predictions are present even in source layerwise training.
            # Their outputs do not contribute to the current layer's loss.
            with torch.no_grad():
                residual = batch - embeddings
                quantized = torch.zeros_like(batch) + embeddings.detach()
                codes = [assignments]
                for layer in residual_prediction_layers:
                    residual = torch.nn.functional.normalize(residual, dim=-1)
                    ids, selected = layer.predict_step(residual)
                    residual = residual - selected
                    quantized = quantized + selected
                    codes.append(ids)
                torch.stack(codes, dim=-1)
        loss.backward()
        losses.append(loss.detach())
    gradients = [replica.centroids.grad for replica in replicas]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("a logical rank did not produce centroid gradients")
    mean_gradient = torch.stack(gradients).mean(0)
    if not bool(torch.isfinite(mean_gradient).all()):
        raise FloatingPointError("nonfinite logical rank centroid gradients")
    first.centroids.grad = mean_gradient
    optimizer.step()
    with torch.no_grad():
        for replica in replicas[1:]:
            replica.centroids.copy_(first.centroids)
    return {"logical_ranks": len(replicas), "points": sum(len(batch) for batch in batches),
            "loss": float(torch.stack(losses).mean()), "centroid_grad_finite": True}


def benchmark_logical_rank_quantizer(grid_root, config, specification, budget, device):
    """Measure initialized RK arithmetic plus all layerwise prediction stages."""
    import torch
    from rhpr.gr.distances import distance_backend_report, make_grid_distance
    from rhpr.gr.grid import load_grid
    from rhpr.gr.profiling import _clear, _memory, _synchronize

    grid = load_grid(grid_root)
    quantizer = config["quantizer"]
    ranks = quantizer.get("logical_ranks", 8)
    batch_size = quantizer.get("batch_size_per_rank", 2048)
    dimension = quantizer.get("embedding_dim", 2048)
    width = quantizer["codebook_size"]
    levels = quantizer.get("num_hierarchies", 3)
    init_size = quantizer.get("init_buffer_size", 3072)
    updates_per_level = quantizer.get("steps_per_level", 1000)
    iterations, warmup = specification.get("quantizer_iterations", 20), specification.get("warmup", 2)
    for value in (ranks, batch_size, dimension, width, levels, init_size, updates_per_level, iterations, warmup):
        if type(value) is not int or value < 1:
            raise ValueError("quantizer profile dimensions and repetitions must be positive integers")
    if init_size < width:
        raise ValueError("quantizer initialization buffer is smaller than its codebook")
    replicas, trailing_layers, batches = [], [], []
    gpu_devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=gpu_devices):
            torch.manual_seed(config["seed"])
            distance = make_grid_distance(grid, quantizer)
            initializer = grid.KMeansPlusPlusInitInitializer(
                n_clusters=width, distance_function=grid.SquaredEuclideanDistance(),
                initialize_on_cpu=False)
            budget.check()
            initialization_points = torch.nn.functional.normalize(
                torch.randn(init_size, dimension, device=device), dim=-1)
            _synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                initial_centroids = initializer(initialization_points)
            _synchronize(device)
            initialization_seconds = time.perf_counter() - started

            def make_model():
                model = grid.MiniBatchKMeans(n_clusters=width, n_features=dimension,
                    distance_function=distance, initializer=initializer,
                    init_buffer_size=init_size, update_manually=False).to(device)
                with torch.no_grad():
                    model.centroids.copy_(initial_centroids)
                model.cluster_counts = torch.zeros(width, device=device)
                model.is_initialized, model.is_initial_step = True, False
                return model

            replicas = [make_model() for _ in range(ranks)]
            trailing_layers = [make_model() for _ in range(levels - 1)]
            raw_batches = [torch.randn(batch_size, dimension, device=device) for _ in range(ranks)]
            optimizer = torch.optim.SGD(replicas[0].parameters(), lr=0.5)

            def run():
                # The first-stage normalization also belongs to source forward.
                batches = [torch.nn.functional.normalize(batch, dim=-1) for batch in raw_batches]
                return logical_rank_step(replicas, optimizer, batches,
                                         residual_prediction_layers=trailing_layers)

            for _ in range(warmup):
                budget.check()
                run()
            _synchronize(device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(iterations):
                budget.check()
                _synchronize(device)
                started = time.perf_counter()
                result = run()
                _synchronize(device)
                durations.append(time.perf_counter() - started)
            seconds = sum(durations) / len(durations)
            return {"status": "ok", "mode": "sequential_logical_ranks",
                    **distance_backend_report(quantizer),
                    "logical_ranks": ranks, "batch_size_per_rank": batch_size,
                    "points_per_update": ranks * batch_size, "embedding_dim": dimension,
                    "codebook_size": width, "residual_prediction_stages": levels - 1,
                    "total_assignment_stages_per_update": levels,
                    "rank0_kmeanspp_initialization_seconds": initialization_seconds,
                    "seconds_per_logical_update": seconds, "logical_update_seconds": durations,
                    "estimated_all_levels_steady_operator_seconds": levels * updates_per_level * seconds,
                    "centroid_grad_finite": result["centroid_grad_finite"],
                    "synthetic_runtime_embeddings": True, "quality_not_evaluated": True,
                    "matches_initialized_ddp_gradient_arithmetic": True,
                    "matches_eight_gpu_wall_time": False,
                    "estimate_note": "Single-device sequential logical-rank operator proxy; includes all residual prediction stages but excludes data loading, communication, per-level initialization and logging; not a measured complete RK fit",
                    **_memory(device)}
    finally:
        replicas.clear()
        trailing_layers.clear()
        batches.clear()
        _clear(device)
