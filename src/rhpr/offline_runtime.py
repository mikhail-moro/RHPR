"""Install a hash-locked private wheel bundle before importing GPU libraries."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def resolve_dataset_directory(requested: str | Path) -> Path:
    path = Path(requested)
    candidates = [path]
    prefix = Path("/kaggle/input/datasets")
    try:
        relative = path.relative_to(prefix)
        if len(relative.parts) >= 2:
            candidates.append(Path("/kaggle/input") / Path(*relative.parts[1:]))
    except ValueError:
        pass
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"dataset directory not found: {', '.join(map(str, candidates))}")


def prepare_wheel_links(source: Path, destination: Path, manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Restore Kaggle-sanitized wheel names through links; preserve all bytes."""
    destination.mkdir(parents=True, exist_ok=True)
    repairs = []
    names = set()
    for row in manifest["wheels"]:
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".whl") or name in names:
            raise ValueError("manifest wheel names must be unique safe basenames")
        names.add(name)
        actual = source / name
        if not actual.is_file():
            actual = source / name.replace("+", "")
        if not actual.is_file() or actual.stat().st_size != row["size_bytes"]:
            raise ValueError(f"mounted wheel is missing or has the wrong size: {name}")
        link = destination / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != actual.resolve():
                raise ValueError("existing runtime wheel link points elsewhere")
        else:
            link.symlink_to(actual)
        if actual.name != name:
            repairs.append({"mounted_name": actual.name, "restored_name": name})
    return repairs


def install_offline_runtime(runtime: dict[str, Any] | None, working: Path) -> dict[str, Any] | None:
    if not runtime:
        return None
    if "torch" in sys.modules:
        raise RuntimeError("offline runtime must be installed before importing torch")
    expected_minor = runtime["target_python_minor"]
    actual_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_minor != expected_minor:
        raise RuntimeError(f"locked native wheels require Python {expected_minor}, received {actual_minor}")
    source = resolve_dataset_directory(runtime["wheel_path"])
    requirement_name = runtime["requirements_file"]
    if Path(requirement_name).name != requirement_name:
        raise ValueError("requirements_file must be a basename in the wheel dataset")
    requirements = source / requirement_name
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    if digest != runtime["requirements_sha256"]:
        raise ValueError("offline requirements lock differs from the registered experiment")
    manifest_bytes = (source / "runtime-manifest.json").read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != runtime["manifest_sha256"]:
        raise ValueError("offline wheel manifest differs from the registered experiment")
    manifest = json.loads(manifest_bytes)
    target = Path(runtime["target_site"])
    # Keep the ~8GB installed runtime outside Kaggle's collected output tree.
    temporary = target.parent / "runtime-install-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    wheel_links = target.parent / "runtime-wheel-links"
    filename_repairs = prepare_wheel_links(source, wheel_links, manifest)
    env = dict(os.environ)
    env.update(TMPDIR=str(temporary), PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-cache-dir",
               "--find-links", str(wheel_links), "--require-hashes", "--only-binary=:all:", "--ignore-installed",
               "--no-compile", "--target", str(target), "-r", str(requirements)]
    started = time.monotonic()
    with (working / "runtime-install.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(target))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import selective_scan_cuda
    import causal_conv1d_cuda
    from mamba_ssm.modules.mamba_simple import Mamba
    if torch.__version__ != runtime["target_torch_version"]:
        raise RuntimeError("offline torch version differs from the registered runtime")
    if not torch._C._GLIBCXX_USE_CXX11_ABI:
        raise RuntimeError("the selected CUDA wheels require PyTorch's CXX11 ABI")
    if not callable(getattr(selective_scan_cuda, "fwd", None)) or not callable(getattr(selective_scan_cuda, "bwd", None)):
        raise RuntimeError("native Mamba selective scan extension is incomplete")
    report = {"installed_offline": True, "requirements_sha256": digest,
              "manifest_sha256": manifest_digest, "wheel_filename_repairs": filename_repairs,
              "installation_seconds": time.monotonic() - started,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "python_minor": actual_minor, "cxx11_abi": True,
              "native_mamba1_import": bool(Mamba), "causal_conv1d_cuda_import": bool(causal_conv1d_cuda),
              "target_site": str(target), "source_dataset": runtime["wheel_path"]}
    (working / "runtime-bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "offline_runtime_ready", **report}), flush=True)
    return report
