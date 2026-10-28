"""Verify the immutable input bundle before running any learned components."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from rhpr.gr.data import file_sha256


def verify_resources(directory: Path, manifest_sha256: str) -> dict:
    directory = directory.resolve()
    path = directory / "resources-manifest.json"
    if file_sha256(path) != manifest_sha256:
        raise ValueError("GR input manifest differs from the registered experiment")
    manifest = json.loads(path.read_text())
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                or "\\" in entry["path"] or relative in seen):
            raise ValueError("unsafe resource path")
        seen.add(relative)
        source = directory / relative
        if not source.resolve().is_relative_to(directory) or source.is_symlink():
            raise ValueError("unsafe resource path")
        if (not source.is_file() or source.stat().st_size != entry["size_bytes"]
                or file_sha256(source) != entry["sha256"]):
            raise ValueError(f"GR input digest mismatch: {relative}")
    return manifest


def unpack_grid(source: Path, destination: Path) -> Path:
    """Extract only regular pinned source files, rejecting archive path escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        seen = set()
        for item in archive.infolist():
            path = Path(item.filename)
            mode = (item.external_attr >> 16) & 0o170000
            target = root / path
            if (not path.parts or path.is_absolute() or ".." in path.parts
                    or "\\" in item.filename or path in seen
                    or mode not in {0, 0o100000, 0o040000}
                    or not target.resolve().is_relative_to(root)
                    or target.is_symlink()):
                raise ValueError("unsafe GRID archive entry")
            seen.add(path)
        archive.extractall(destination)
    return destination


def verify_parent_artifacts(parent: dict, input_root: Path = Path("/kaggle/input")) -> Path:
    """Resolve a private parent mount and verify bytes plus source revision.

    Hashes, rather than a mutable notebook slug, determine which completed run
    is being reused. The optional path also supports local CPU fixtures.
    """
    kernel = parent["kernel"]
    parts = kernel.split("/")
    if len(parts) != 2 or any(not value or value in {".", ".."} for value in parts):
        raise ValueError("parent kernel must have owner/slug form")
    owner, slug = parts
    candidates = []
    if parent.get("path"):
        candidates.append(Path(parent["path"]))
    candidates.extend([input_root / slug, input_root / owner / slug,
                       input_root / "notebooks" / owner / slug,
                       input_root / "kernels" / owner / slug])
    def locations():
        # Check known mounts before any recursive scan of remote input storage.
        yield from candidates
        # New Kaggle mount layouts can append a version or add a namespace.
        if input_root.exists():
            yield from (p.parent for p in input_root.rglob("run-manifest.json")
                        if slug in p.parts)
    files = parent["files"]
    if "run-manifest.json" not in files:
        raise ValueError("parent provenance manifest must be pinned")
    for name in files:
        if not name or Path(name).name != name or "\\" in name:
            raise ValueError("parent artifact names must be safe basenames")
    for directory in locations():
        if not all((directory / name).is_file() for name in files):
            continue
        for name, expected in files.items():
            path = directory / name
            if (path.stat().st_size != expected["size_bytes"]
                    or file_sha256(path) != expected["sha256"]):
                raise ValueError(f"parent artifact differs from the registered run: {name}")
        manifest = json.loads((directory / "run-manifest.json").read_text())
        if manifest.get("git_sha") != parent["source_code_sha"]:
            raise ValueError("parent source revision differs from the registered run")
        return directory
    raise FileNotFoundError("the registered private parent artifacts are not mounted")
