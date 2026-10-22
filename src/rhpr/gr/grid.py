"""Pinned GRID components, with a small adapter for RHPR's data and training loop.

The generator and centroid updates are imported from upstream, not reimplemented.
The only source patch defers a type-only TFRecord import; neither TensorFlow nor
GRID's distributed dataloader is needed to consume our prepared arrays.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping

GRID_REVISION = "2fe3475b2d369580234093f35d52b1a2f54d0472"
GRID_SOURCE_TREE_SHA256 = "79a7c09622842b6d98ca260bac900e78c952b4c42402d6ffe4dbdb58636cd02f"
GRID_PREPARED_TREE_SHA256 = "0177e9c4b094c0b48c043c3a2f285ebf0e03f2f8d5ab99c702567dc6f3f48efd"
_INTERFACE_PATH = Path("src/data/loading/components/interfaces.py")
_MANIFEST_NAME = "rhpr-grid-adapter.json"
_loaded_grid: SimpleNamespace | None = None


def _source_digest(root: Path) -> str:
    rows = []
    for path in sorted((root / "src").rglob("*.py")):
        rows.append(
            path.relative_to(root).as_posix() + "\0"
            + hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
        )
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()


def _verify_source(root: Path, *, prepared: bool = False) -> str:
    expected = GRID_PREPARED_TREE_SHA256 if prepared else GRID_SOURCE_TREE_SHA256
    observed = _source_digest(root)
    if observed != expected:
        raise ValueError(
            f"GRID Python source does not match pinned revision {GRID_REVISION}: "
            f"expected {expected}, got {observed} at {root}"
        )
    return observed


def prepare_grid_source(source_root: str | Path, destination: str | Path | None = None) -> Path:
    """Copy verified upstream Python source and defer its type-only TF import.

    The full Python tree is checked against the pinned commit before and after
    patching. The source checkout is left untouched. An existing destination is
    accepted only if it already contains exactly the expected prepared tree.
    """
    source = Path(source_root).expanduser().resolve()
    source_is_prepared = (source / _MANIFEST_NAME).is_file()
    _verify_source(source, prepared=source_is_prepared)
    if source_is_prepared and destination is None:
        return source
    if destination is None:
        target = Path(tempfile.mkdtemp(prefix="rhpr-grid-"))
    else:
        target = Path(destination).expanduser().resolve()
        if target == source or target.is_relative_to(source):
            raise ValueError("GRID adapter destination must be outside its source checkout")
        if target.exists():
            _verify_source(target, prepared=True)
            return target
        target.mkdir(parents=True)
    shutil.copytree(source / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("LICENSE", "notices.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    interface = target / _INTERFACE_PATH
    before = interface.read_bytes()
    if not source_is_prepared:
        patched = b"from __future__ import annotations\n\n" + before.replace(
            b"from typing import Any, Dict, List, Optional, Union",
            b"from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union",
        ).replace(
            b"from src.data.loading.components.iterators import RawDataIterator",
            b"if TYPE_CHECKING:\n    from src.data.loading.components.iterators import RawDataIterator",
        )
        interface.write_bytes(patched)
    _verify_source(target, prepared=True)
    manifest = {
        "upstream_repository": "https://github.com/snap-research/GRID",
        "upstream_revision": GRID_REVISION,
        "original_python_tree_sha256": GRID_SOURCE_TREE_SHA256,
        "prepared_python_tree_sha256": GRID_PREPARED_TREE_SHA256,
        "patches": [{
            "path": _INTERFACE_PATH.as_posix(),
            "purpose": "Defer type annotations and the annotation-only TFRecord iterator import",
            "algorithm_changes": False,
        }],
    }
    (target / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def load_grid(source_root: str | Path | None = None) -> SimpleNamespace:
    """Load actual pinned GRID classes without replacing its global ``src`` package.

    ``source_root`` can be the checkout, an extracted pinned archive, or a
    prepared directory. If omitted, use ``RHPR_GRID_SOURCE`` or the already
    loaded adapter. A conflicting imported ``src`` package is rejected.
    """
    global _loaded_grid
    if source_root is None:
        source_root = os.environ.get("RHPR_GRID_SOURCE")
    if source_root is None and _loaded_grid is not None:
        return _loaded_grid
    if source_root is None:
        raise ValueError("Supply GRID source_root or set RHPR_GRID_SOURCE")
    source = Path(source_root).expanduser().resolve()
    _verify_source(source, prepared=(source / _MANIFEST_NAME).is_file())
    if _loaded_grid is not None:
        return _loaded_grid
    foreign_modules = [name for name in sys.modules if name == "src" or name.startswith("src.")]
    if foreign_modules:
        raise RuntimeError(
            "GRID requires the top-level src namespace, which is already imported; "
            "run this experiment in a fresh process"
        )
    prepared = prepare_grid_source(source)
    sys.path.insert(0, str(prepared))
    try:
        model_module = importlib.import_module("src.models.modules.semantic_id.tiger_generation_model")
        kmeans_module = importlib.import_module("src.models.modules.clustering.mini_batch_kmeans")
        initializer_module = importlib.import_module("src.components.clustering_initializers")
        distance_module = importlib.import_module("src.components.distance_functions")
    except Exception:
        # A failed import must not poison the next attempt or another library.
        for name in tuple(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        sys.path.remove(str(prepared))
        raise
    _loaded_grid = SimpleNamespace(
        SemanticIDEncoderDecoder=model_module.SemanticIDEncoderDecoder,
        MiniBatchKMeans=kmeans_module.MiniBatchKMeans,
        KMeansPlusPlusInitInitializer=initializer_module.KMeansPlusPlusInitInitializer,
        SquaredEuclideanDistance=distance_module.SquaredEuclideanDistance,
        source_root=prepared,
        revision=GRID_REVISION,
        python_tree_sha256=GRID_PREPARED_TREE_SHA256,
    )
    return _loaded_grid


def _positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def build_grid_generator(
    sids: Any,
    config: Mapping[str, Any],
    seed: int = 42,
    device: str = "cpu",
    source_root: str | Path | None = None,
) -> Any:
    """Construct GRID's encoder-decoder using an item-by-SID integer matrix.

    Defaults follow TIGER's documented layer dimensions, not GRID's extra FF
    hidden layer. The resulting model is trained from scratch. User hashing is
    GRID's integer-remainder mapping and must be recorded as a reproduction
    difference from the original unspecified hashing implementation.
    """
    import numpy as np
    import torch
    from transformers import T5Config, T5EncoderModel
    from transformers.models.t5.modeling_t5 import T5Stack

    grid = load_grid(source_root)
    values = np.asarray(sids)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("sids must be a nonempty item-by-hierarchy matrix")
    if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
        raise ValueError("sids must contain nonnegative integers")
    width = _positive_int(config, "codebook_size", 256)
    if int(values.max()) >= width:
        raise ValueError("SID code or collision suffix exceeds the configured codebook_size")
    if len(np.unique(values, axis=0)) != len(values):
        raise ValueError("Complete SIDs must uniquely identify every catalogue item")
    beam_width = _positive_int(config, "beam_width", 10)
    if beam_width > width:
        raise ValueError("beam_width must not exceed codebook_size")
    if config.get("should_check_prefix", False):
        raise ValueError("Pinned GRID constrained beam search has an upstream mask bug; use free-form search")
    d_model = _positive_int(config, "d_model", 128)
    encoder_layers = _positive_int(config, "encoder_layers", 4)
    decoder_layers = _positive_int(config, "decoder_layers", 4)
    num_heads = _positive_int(config, "num_heads", 6)
    d_kv = _positive_int(config, "d_kv", 64)
    d_ff = _positive_int(config, "d_ff", 1024)
    dropout = float(config.get("dropout", 0.1))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    user_bins = config.get("num_user_bins", 2000)
    if user_bins is not None and (isinstance(user_bins, bool) or not isinstance(user_bins, int) or user_bins < 0):
        raise ValueError("num_user_bins must be a nonnegative integer or null")
    mlp_layers = config.get("mlp_layers")
    if mlp_layers is not None and (isinstance(mlp_layers, bool) or not isinstance(mlp_layers, int) or mlp_layers < 1):
        raise ValueError("mlp_layers must be a positive integer or null")
    torch.manual_seed(seed)
    encoder_config = T5Config(
        vocab_size=width, d_model=d_model, d_ff=d_ff, d_kv=d_kv,
        num_heads=num_heads, num_layers=encoder_layers, dropout_rate=dropout,
        feed_forward_proj="relu",
    )
    decoder_config = T5Config(
        vocab_size=width, d_model=d_model, d_ff=d_ff, d_kv=d_kv,
        num_heads=num_heads, num_layers=decoder_layers, dropout_rate=dropout,
        feed_forward_proj="relu", is_decoder=True, is_encoder_decoder=False,
    )
    model = grid.SemanticIDEncoderDecoder(
        huggingface_model=T5EncoderModel(encoder_config),
        decoder=T5Stack(decoder_config, embed_tokens=torch.nn.Embedding(width, d_model)),
        codebooks=torch.as_tensor(values.T.copy(), dtype=torch.long),
        num_hierarchies=values.shape[1], num_embeddings_per_hierarchy=width,
        embedding_dim=d_model, num_user_bins=user_bins,
        mlp_layers=mlp_layers, should_add_sep_token=bool(config.get("separator_token", False)),
        should_check_prefix=False, top_k_for_generation=beam_width,
        feature_to_model_input_map={"sequence_data": "input_ids", "user_id": "user_id"},
        postprocessor=None, aggregator=None, loss_function=torch.nn.CrossEntropyLoss(),
        optimizer=None, scheduler=None, evaluator=None, weight_tying=True, compile=False,
    ).to(device)
    model.rhpr_grid_provenance = {
        "revision": GRID_REVISION, "python_tree_sha256": GRID_PREPARED_TREE_SHA256,
        "class": "src.models.modules.semantic_id.tiger_generation_model.SemanticIDEncoderDecoder",
        "constrained_beam_search": False,
        "mlp_layers": mlp_layers, "user_hash": "integer_remainder",
    }
    return model


def fit_grid_rkmeans(
    embeddings: Any,
    config: Mapping[str, Any],
    seed: int = 42,
    device: str = "cpu",
    source_root: str | Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Fit GRID MiniBatchKMeans sequentially on normalized residuals.

    ``passes_per_level`` counts complete shuffled catalogue passes, unlike
    upstream's unbounded distributed step schedule. K-means++ initialization is
    a separate, explicitly counted operation. No generator gradients enter this
    fixed tokenizer. Returned codes have no collision suffix yet.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional

    grid = load_grid(source_root)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError("embeddings must be a finite, nonempty item-by-dimension matrix")
    width = _positive_int(config, "codebook_size", 256)
    levels = _positive_int(config, "num_hierarchies", 3)
    passes = _positive_int(config, "passes_per_level", 10)
    batch_size = _positive_int(config, "batch_size", 2048)
    init_limit = _positive_int(config, "init_buffer_size", 3072)
    if len(array) < width:
        raise ValueError("Catalogue must contain at least codebook_size items")
    if init_limit < width:
        raise ValueError("init_buffer_size must be at least codebook_size")
    if config.get("normalize_residuals", True) is not True:
        raise ValueError("This GRID RK-Means protocol requires normalize_residuals=true")
    tensor_device = torch.device(device)
    gpu_devices = []
    if tensor_device.type == "cuda":
        gpu_devices = [tensor_device.index if tensor_device.index is not None else torch.cuda.current_device()]

    def synchronize() -> None:
        if tensor_device.type == "cuda":
            torch.cuda.synchronize(tensor_device)

    synchronize()
    started = time.perf_counter()
    stage_reports = []
    all_codes = []
    all_centroids = []
    with torch.random.fork_rng(devices=gpu_devices), torch.no_grad():
        torch.manual_seed(seed)
        residual = torch.as_tensor(array, device=tensor_device).clone()
        for level in range(levels):
            synchronize()
            stage_started = time.perf_counter()
            residual = functional.normalize(residual, dim=-1)
            distance = grid.SquaredEuclideanDistance()
            initializer = grid.KMeansPlusPlusInitInitializer(
                n_clusters=width, distance_function=distance, initialize_on_cpu=False,
            )
            buffer_size = min(init_limit, len(array))
            model = grid.MiniBatchKMeans(
                n_clusters=width, n_features=array.shape[1], distance_function=distance,
                initializer=initializer, init_buffer_size=buffer_size,
                update_manually=True,
            ).to(tensor_device)
            # These upstream tensors are plain attributes, not registered buffers.
            model.init_buffer = torch.empty(0, device=tensor_device)
            model.cluster_counts = torch.zeros(width, device=tensor_device)
            initialization_indices = torch.randperm(len(array), device=tensor_device)[:buffer_size]
            model.model_step(residual[initialization_indices])
            if not model.is_initial_step:
                raise RuntimeError("GRID failed to complete its k-means++ initialization buffer")
            synchronize()
            initialized_at = time.perf_counter()
            updates = 0
            for _ in range(passes):
                order = torch.randperm(len(array), device=tensor_device)
                for start in range(0, len(array), batch_size):
                    model.model_step(residual[order[start:start + batch_size]])
                    updates += 1
            if not model.is_initialized or model.is_initial_step:
                raise RuntimeError("GRID initialization flags did not advance on the first training batch")
            counted_points = int(model.cluster_counts.sum().item())
            if counted_points != len(array) * passes:
                raise RuntimeError("GRID centroid counts disagree with completed catalogue passes")
            synchronize()
            trained_at = time.perf_counter()
            assignment_parts = []
            for start in range(0, len(array), batch_size):
                assignment_parts.append(model.predict_step(residual[start:start + batch_size], return_embeddings=False))
            assignments = torch.cat(assignment_parts)
            centroids = model.get_centroids().detach()
            next_residual = residual - centroids[assignments]
            if not torch.isfinite(centroids).all() or not torch.isfinite(next_residual).all():
                raise FloatingPointError("GRID RK-Means produced nonfinite centroids or residuals")
            all_codes.append(assignments.cpu().numpy().copy())
            all_centroids.append(centroids.cpu().numpy().copy())
            normalized_residual_mse = float(next_residual.square().mean().item())
            synchronize()
            stage_reports.append({
                "level": level, "initialization_points": buffer_size,
                "training_points": counted_points, "updates": updates,
                "used_codes": int(torch.unique(assignments).numel()),
                "normalized_residual_mse": normalized_residual_mse,
                "initialization_seconds": initialized_at - stage_started,
                "training_seconds": trained_at - initialized_at,
                "assignment_seconds": time.perf_counter() - trained_at,
            })
            residual = next_residual
    synchronize()
    report = {
        "algorithm": "GRID MiniBatchKMeans with manual centroid updates",
        "revision": GRID_REVISION, "python_tree_sha256": GRID_PREPARED_TREE_SHA256,
        "seed": seed, "device": str(tensor_device), "num_items": len(array),
        "embedding_dim": array.shape[1], "num_hierarchies": levels,
        "codebook_size": width, "passes_per_level": passes, "batch_size": batch_size,
        "normalize_residuals": True,
        "normalization_note": "Each level normalizes its incoming residual; MSE is in that normalized space",
        "initialization": "upstream KMeansPlusPlusInitInitializer",
        "elapsed_seconds": time.perf_counter() - started, "levels": stage_reports,
    }
    return np.stack(all_codes, axis=1), np.stack(all_centroids, axis=0), report
