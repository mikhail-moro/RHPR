"""Build a private GenPAS CPU/TPU input bundle without TensorFlow dependencies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhpr.gr.data import file_sha256, load_beauty
from rhpr.gr.genpas import GENPAS_REVISION, prepare_source


def build(source, data, wheels, output):
    prepared = prepare_source(source)
    shutil.rmtree(prepared)  # Verification only; package the unmodified tree.
    _, _, _, counts = load_beauty(data)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output must be empty")
    for original, name in ((data, "data"), (wheels, "wheels")):
        shutil.copytree(original, output / name, copy_function=os.link)
    with zipfile.ZipFile(output / "genpas-source.bin", "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [p for folder in ("src", "configs") for p in (source / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
        paths += [source / "License+Attribution.txt", source / "README.md"]
        for path in sorted(paths):
            info = zipfile.ZipInfo(path.relative_to(source).as_posix(), (2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    manifest = {"schema": "rhpr-genpas-resources-v1", "source_revision": GENPAS_REVISION,
        "data_counts": counts, "data": "lossless author Beauty TFRecord conversion; original splits",
        "files": [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size,
                   "sha256": file_sha256(p)} for p in sorted(output.rglob("*")) if p.is_file()]}
    (output / "resources-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest_sha256": file_sha256(output / "resources-manifest.json"), "counts": counts,
        "wheels_manifest_sha256": file_sha256(wheels / "runtime-manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "wheels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.data, args.wheels, args.output), indent=2))
