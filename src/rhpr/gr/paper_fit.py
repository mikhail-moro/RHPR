"""Full catalogue RK fitting with GRID's rank-local update arithmetic."""
from __future__ import annotations

import math
import time


class CatalogueStream:
    """Independent shuffled catalogue cycles for one logical rank."""

    def __init__(self, size, seed):
        import torch
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.position = 0

    def take(self, count):
        import torch
        chunks = []
        while count:
            if self.position == len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
            take = min(count, len(self.order) - self.position)
            chunks.append(self.order[self.position:self.position + take])
            self.position += take
            count -= take
        return torch.cat(chunks)


def fit_paper_rkmeans(embeddings, config, *, seed=42, device="cpu", source_root=None,
                      progress=None):
    import numpy as np
    import torch
    from rhpr.gr.distances import distance_backend_report, make_grid_distance
    from rhpr.gr.grid import load_grid
    from rhpr.gr.paper_quantizer_profile import logical_rank_step
    from rhpr.gr.profiling import _synchronize

    grid = load_grid(source_root)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("RK embeddings must be a finite item-by-dimension matrix")
    width, levels = config["codebook_size"], config["num_hierarchies"]
    ranks, batch_size = config["logical_ranks"], config["batch_size_per_rank"]
    steps, buffer_size = config["steps_per_level"], config["init_buffer_size"]
    if any(type(x) is not int or x < 1 for x in (width, levels, ranks, batch_size, steps, buffer_size)):
        raise ValueError("RK dimensions and budgets must be positive integers")
    init_steps = math.ceil(buffer_size / batch_size)
    if len(array) < width or buffer_size < width or steps <= init_steps:
        raise ValueError("RK budget must cover initialization and at least one update")
    if config.get("normalize_residuals") is not True:
        raise ValueError("paper RK requires residual normalization")
    streams = [CatalogueStream(len(array), seed + rank) for rank in range(ranks)]
    devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    reports, codes, centroids = [], [], []
    _synchronize(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        residual = torch.as_tensor(array, device=device).clone()
        for level in range(levels):
            level_started = time.perf_counter()
            with torch.no_grad():
                residual = torch.nn.functional.normalize(residual, dim=-1)
                # Source consumes whole rank batches while filling rank0's buffer.
                initial = []
                for _ in range(init_steps):
                    batches = [stream.take(batch_size) for stream in streams]
                    initial.append(residual[batches[0].to(device)])
                initializer = grid.KMeansPlusPlusInitInitializer(n_clusters=width,
                    distance_function=grid.SquaredEuclideanDistance(), initialize_on_cpu=False)
                initial_centroids = initializer(torch.cat(initial)[:buffer_size])
            replicas = []
            for _ in range(ranks):
                model = grid.MiniBatchKMeans(n_clusters=width, n_features=array.shape[1],
                    distance_function=make_grid_distance(grid, config), initializer=initializer,
                    init_buffer_size=buffer_size, update_manually=False).to(device)
                with torch.no_grad():
                    model.centroids.copy_(initial_centroids)
                model.cluster_counts = torch.zeros(width, device=device)
                model.is_initialized, model.is_initial_step = True, False
                replicas.append(model)
            optimizer = torch.optim.SGD((replicas[0].centroids,), lr=0.5)
            _synchronize(device)
            initialization_seconds = time.perf_counter() - level_started
            training_started = time.perf_counter()
            for update in range(steps - init_steps):
                batches = [residual[stream.take(batch_size).to(device)] for stream in streams]
                result = logical_rank_step(replicas, optimizer, batches)
                if progress and (update == 0 or (update + 1) % 100 == 0):
                    progress({"level": level, "steady_updates": update + 1, **result})
            _synchronize(device)
            training_seconds = time.perf_counter() - training_started
            with torch.no_grad():
                assignments = torch.cat([replicas[0].predict_step(residual[start:start + batch_size],
                    return_embeddings=False) for start in range(0, len(array), batch_size)])
                centers = replicas[0].centroids.detach().clone()
                if not bool(torch.isfinite(centers).all()):
                    raise FloatingPointError("nonfinite RK centroids")
                expected_count = (steps - init_steps) * batch_size
                if any(int(model.cluster_counts.sum()) != expected_count for model in replicas):
                    raise RuntimeError("RK local cluster counts disagree with the step budget")
                reports.append({"level": level, "initialization_steps": init_steps,
                    "initialization_points_rank0": buffer_size, "steady_updates": steps - init_steps,
                    "training_points_per_rank": expected_count,
                    "rank_cluster_counts": [model.cluster_counts.cpu().tolist() for model in replicas],
                    "used_codes": int(assignments.unique().numel()),
                    "initialization_seconds": initialization_seconds, "training_seconds": training_seconds})
                codes.append(assignments.cpu().numpy().copy())
                centroids.append(centers.cpu().numpy().copy())
                residual = residual - centers[assignments]
            if progress:
                progress({"level_complete": level, **reports[-1]})
    _synchronize(device)
    report = {"algorithm": "GRID source losses/SGD with sequential logical ranks",
        "revision": grid.revision, "python_tree_sha256": grid.python_tree_sha256,
        "logical_ranks": ranks, "batch_size_per_rank": batch_size, "steps_per_level": steps,
        "normalize_residuals": True, "levels": reports, "elapsed_seconds": time.perf_counter() - started,
        "sampling": "independent seeded catalogue shuffle cycles per rank; not TF file/worker order",
        "initialization": "rank0 source KMeans++; initialization batch steps included in each level budget",
        "execution": "cached incoming residuals for frozen prior levels; later predictions omitted because they do not affect the active layer loss",
        "matches_eight_gpu_wall_time": False, **distance_backend_report(config)}
    return np.stack(codes, axis=1), np.stack(centroids), report
