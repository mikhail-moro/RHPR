"""Sequential full-catalogue Lloyd codebooks on immutable continuous AE latents."""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write


def fit_residual_kmeans(latents, spec, seed):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    for key in ("num_hierarchies", "codebook_size", "max_iter", "n_init", "cpu_threads"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(f"RQ-KMeans {key} must be positive integer")
    if spec.get("normalize_residuals") is not False or spec.get("normalize_latents") is not False:
        raise ValueError("AE residual KMeans preserves raw latent and residual scales")
    if type(spec.get("tol")) not in (int, float) or not np.isfinite(spec["tol"]) or spec["tol"] <= 0:
        raise ValueError("RQ-KMeans tol must be positive finite")
    z = np.asarray(latents)
    if z.ndim != 2 or len(z) < spec["codebook_size"] or not z.shape[1] or z.dtype.kind != "f" or not np.isfinite(z).all():
        raise ValueError("RQ-KMeans requires finite floating catalogue latents with enough rows")
    residual = z.astype(np.float32, copy=True)
    if not np.isfinite(residual).all():
        raise ValueError("AE latents overflow float32")
    books, codes, levels = [], [], []
    started = time.monotonic()
    with threadpool_limits(limits=spec["cpu_threads"]):
        for level in range(spec["num_hierarchies"]):
            before = time.monotonic()
            model = KMeans(n_clusters=spec["codebook_size"], init="k-means++", n_init=spec["n_init"],
                max_iter=spec["max_iter"], tol=spec["tol"], random_state=seed+level, algorithm="lloyd")
            model.fit(residual)
            assigned = model.predict(residual)
            books.append(model.cluster_centers_.copy());codes.append(assigned)
            residual = residual - model.cluster_centers_[assigned]
            row = {"level": level+1, "iterations": int(model.n_iter_),
                "hit_iteration_cap": model.n_iter_ == spec["max_iter"],
                "inertia": float(model.inertia_), "seconds": time.monotonic()-before}
            levels.append(row);_event("ae_rqkmeans_level_complete", seed=seed, **row)
    return np.stack(codes, 1), np.stack(books), {"kind": "ae_residual_kmeans", "seed": seed,
        "spec": spec, "levels": levels, "fit_seconds": time.monotonic()-started,
        "sklearn": importlib.metadata.version("scikit-learn"), "numpy": np.__version__,
        "scope": "full catalogue raw AE latents; sequential residual Lloyd; no neural updates"}


def prepare_ae_kmeans_index(config, trial, parent, source_ids, directory):
    from rhpr.gr.generator_suite import verify_file
    from rhpr.gr.codebook_diagnostics import diagnose_codebooks

    latent_path = verify_file(parent, trial["ae_latents"])
    report_path = verify_file(parent, trial["ae_report"])
    ae = json.loads(report_path.read_text())
    if (ae["seed"] != trial["quantizer_seed"] or ae.get("quantization_enabled") is not False
            or ae["steps"] != trial["ae_steps"] or ae["latent_dim"] != config["quantizer"]["latent_dim"]
            or ae["checkpoint_sha256"] != trial["ae_checkpoint_sha256"]):
        raise ValueError("AE latent source/report differs from registered checkpoint")
    latents = np.load(latent_path, allow_pickle=False)
    if latents.shape != (len(source_ids), config["quantizer"]["latent_dim"]):
        raise ValueError("AE latent matrix does not match catalogue/dimension")
    codes, books, report = fit_residual_kmeans(latents, config["quantizer"], trial["quantizer_seed"])
    diagnostics = diagnose_codebooks(codes, latents, books)
    report.update(autoencoder_retrained=False, quantizer_fitted=True, ae_steps=ae["steps"],
        ae_checkpoint_sha256=ae["checkpoint_sha256"], ae_latents=trial["ae_latents"], ae_report=trial["ae_report"],
        diagnostics=diagnostics)
    np.save(directory / "codebooks.npy", books, allow_pickle=False)
    _write(directory / "kmeans.json", report)
    return codes, {"ae_latents": trial["ae_latents"], "ae_report": trial["ae_report"],
        "ae_checkpoint_sha256": ae["checkpoint_sha256"], "autoencoder_retrained": False,
        "quantizer_retrained": True, "fit_method": "full_catalogue_residual_lloyd",
        "codebooks_sha256": file_sha256(directory / "codebooks.npy"),
        "fit_report_sha256": file_sha256(directory / "kmeans.json")}
