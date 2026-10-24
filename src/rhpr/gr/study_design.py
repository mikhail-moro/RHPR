"""Explicit quantizer x generator designs and immutable index imports."""
from pathlib import Path
import json
import shutil

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.sid import SidIndex


def study_jobs(config):
    seeds = config.get("model_seeds")
    if seeds is None:
        return [{"quantizer_seed": q, "model_seed": config.get("model_seed", 42), "run_id": f"rq-{q}"}
                for q in config["quantizer_seeds"]]
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
            or config["model_seed"] != seeds[0]):
        raise ValueError("model_seeds must be distinct nonnegative integers, starting with model_seed")
    # Alternate index order to distribute each index over both physical cards.
    return [{"quantizer_seed": q, "model_seed": m, "run_id": f"rq-{q}-ms-{m}"}
            for i, m in enumerate(seeds)
            for q in (config["quantizer_seeds"] if i % 2 == 0 else config["quantizer_seeds"][::-1])]


def initial_hash_for(reference, model_seed):
    return reference[str(model_seed)] if isinstance(reference, dict) else reference


def load_frozen_index(values, spec, source, output):
    """Import exact original codes/centroids; never refit or silently fall back."""
    from rhpr.offline_runtime import resolve_dataset_directory
    output = Path(output)
    parent = None
    if source.get("parent_artifacts"):
        from rhpr.gr.resources import verify_parent_artifacts
        parent = verify_parent_artifacts(source["parent_artifacts"])
    resolved = {}
    for name in ("semantic-ids.npy", "quantizer-centroids.npy", "quantizer.json"):
        entry = source["files"][name]
        path = parent / f"rq-{source['quantizer_seed']}" / name if parent is not None else Path(entry["path"])
        if not path.is_file():
            path = resolve_dataset_directory(path.parent) / path.name
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen index digest mismatch: {name}")
        resolved[name] = path
    saved = np.load(resolved["semantic-ids.npy"], allow_pickle=False)
    if saved.shape != (len(values), spec["levels"] + 1) or saved.dtype.kind not in "iu":
        raise ValueError("Frozen SID shape differs from catalogue/levels")
    index = SidIndex(saved[:, :-1], width=spec["codebook_size"])
    np.testing.assert_array_equal(index.sids, saved)
    centers = np.load(resolved["quantizer-centroids.npy"], allow_pickle=False)
    if centers.shape != (spec["levels"], spec["codebook_size"], values.shape[1]) or not np.isfinite(centers).all():
        raise ValueError("Frozen centroid shape/values differ")
    original = json.loads(resolved["quantizer.json"].read_text())
    for key in ("levels", "codebook_size"):
        if original[key] != spec[key]:
            raise ValueError(f"Frozen quantizer {key} differs")
    if original["seed"] != source["quantizer_seed"]:
        raise ValueError("Frozen quantizer seed differs")
    for name, path in resolved.items():
        shutil.copyfile(path, output / name)
    (output / "frozen-index.json").write_text(json.dumps(source, indent=2) + "\n")
    return index, centers
