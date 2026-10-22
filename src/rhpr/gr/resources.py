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
