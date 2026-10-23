"""Reuse a byte-pinned completed tokenizer while retraining the generator."""
from __future__ import annotations

import json
import shutil

from rhpr.gr.experiment import _write


def reuse_tokenizer(config, users_count, original_ids, output):
    import numpy as np
    from rhpr.gr.resources import verify_parent_artifacts
    from rhpr.gr.sid import SidIndex

    parent = verify_parent_artifacts(config["parent"])
    previous = json.loads((parent / "run-manifest.json").read_text())["config"]
    if previous["data"]["sha256"] != config["data"]["sha256"]:
        raise ValueError("cached tokenizer belongs to a different dataset")
    for key in ("kind", "num_hierarchies", "codebook_size", "steps_per_level", "logical_ranks",
                "batch_size_per_rank", "init_buffer_size", "normalize_residuals", "distance_backend"):
        if previous["quantizer"].get(key) != config["quantizer"].get(key):
            raise ValueError(f"cached quantizer setting differs: {key}")
    for key in ("kind", "model", "revision", "max_length", "pooling"):
        if previous["representation"].get(key) != config["representation"].get(key):
            raise ValueError(f"cached representation setting differs: {key}")
    catalog = json.loads((parent / "sid-catalogue.json").read_text())
    if catalog["original_item_ids"] != original_ids:
        raise ValueError("cached tokenizer catalogue order differs")
    values = np.load(parent / "semantic-ids.npy", allow_pickle=False)
    quantizer = config["quantizer"]
    if values.shape != (len(original_ids), quantizer["num_hierarchies"] + 1):
        raise ValueError("cached SID shape differs")
    index = SidIndex(values[:, :-1], width=quantizer["codebook_size"])
    if not np.array_equal(index.sids, values):
        raise ValueError("cached collision suffix differs from the registered SID policy")
    incoming = json.loads((parent / "input-provenance.json").read_text())
    if incoming["counts"]["users"] != users_count:
        raise ValueError("cached tokenizer source has a different user population")
    for name in ("semantic-ids.npy", "quantizer-centroids.npy", "sid-catalogue.json"):
        shutil.copyfile(parent / name, output / name)
    representation = json.loads((parent / "representation.json").read_text())
    report = json.loads((parent / "quantizer.json").read_text())
    representation.update(source_encoding_seconds=representation["seconds"], seconds=0., reused=True)
    report.update(source_fitting_seconds=report["elapsed_seconds"], elapsed_seconds=0., reused=True)
    _write(output / "representation.json", representation)
    _write(output / "quantizer.json", report)
    _write(output / "sid-diagnostics.json", index.diagnostics())
    _write(output / "tokenizer-reuse.json", {"parent": config["parent"],
        "reused_semantic_ids": True, "encoder_and_quantizer_not_rerun": True,
        "generator_weights_reused": False})
    return index, representation, report
