"""Build the immutable private Kaggle input bundle from audited local inputs.

No uploads, GPU use, or implicit downloads. Keep the output outside the checkout.
The download sources and dependency locks are documented in docs/gr-baseline.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rhpr.gr.data import file_sha256
from rhpr.gr.grid import GRID_REVISION, GRID_SOURCE_TREE_SHA256, _verify_source


def build(grid: Path, data: Path, sentence_model: Path, wheels: Path, output: Path) -> dict:
    output = output.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError("downloaded experiment resources must stay outside the checkout")
    if output.exists() and any(output.iterdir()):
        raise ValueError("resource output must be empty")
    _verify_source(grid)
    output.mkdir(parents=True, exist_ok=True)
    for source, name in ((data, "data"), (sentence_model, "sentence-t5-base"), (wheels, "wheels")):
        shutil.copytree(source, output / name, ignore=shutil.ignore_patterns("__pycache__", ".cache"))
    # Deterministic archive bytes, including upstream redistribution notices.
    paths = [path for base in ("src", "configs") for path in (grid / base).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts]
    paths += [grid / name for name in ("LICENSE", "notices.txt", "README.md")]
    # Keep an opaque extension so Kaggle does not unpack this *inner* archive
    # during dataset ingestion; its exact bytes are checked by the manifest.
    with zipfile.ZipFile(output / "grid-source.bin", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo(path.relative_to(grid).as_posix(), (2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    files = [{"path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size,
              "sha256": file_sha256(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"schema": "rhpr-gr-resources-v1", "files": files,
                "grid_repository": "https://github.com/snap-research/GRID", "grid_revision": GRID_REVISION,
                "grid_original_python_tree_sha256": GRID_SOURCE_TREE_SHA256,
                "sentence_model": "sentence-transformers/sentence-t5-base",
                "sentence_model_revision": "fc5d4628481afbbaaacd7af6bb07cf9d3865f781",
                "purpose": "private noncommercial research; retain upstream licenses and notices"}
    (output / "resources-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest_sha256": file_sha256(output / "resources-manifest.json"),
            "files": len(files), "total_size_bytes": sum(row["size_bytes"] for row in files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "data", "sentence-model", "wheels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.grid, args.data, args.sentence_model, args.wheels, args.output), indent=2))
