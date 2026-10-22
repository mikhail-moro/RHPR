"""Small offline GR dependency bootstrap; reuse the platform's CUDA PyTorch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def install_gr_runtime(runtime: dict, working: Path) -> dict:
    from rhpr.offline_runtime import resolve_dataset_directory

    expected_minor = runtime["target_python_minor"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected_minor:
        raise RuntimeError(f"GR dependency wheels require Python {expected_minor}")
    if "torch" in sys.modules:
        raise RuntimeError("install the GR runtime before importing torch")
    source = resolve_dataset_directory(runtime["wheel_path"])
    for name, expected in (("requirements-lock.txt", runtime["requirements_sha256"]),
                           ("runtime-manifest.json", runtime["manifest_sha256"])):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"GR runtime digest mismatch: {name}")
    manifest = json.loads((source / "runtime-manifest.json").read_text())
    if manifest["target_python_minor"] != expected_minor:
        raise ValueError("GR runtime manifest targets a different Python version")
    for wheel in manifest["wheels"]:
        package = wheel["name"].lower().replace("_", "-")
        if package in {"torch", "torchvision", "torchaudio", "triton"} or package.startswith("nvidia-"):
            raise ValueError("GR dependency bundle must preserve the platform PyTorch stack")
    # Kaggle sanitizes '+' in wheel basenames. Restore through verified links.
    from rhpr.offline_runtime import prepare_wheel_links
    target = Path(runtime["target_site"])
    links = target.parent / "gr-wheel-links"
    prepare_wheel_links(source, links, manifest)
    started = time.monotonic()
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-cache-dir",
               "--no-deps", "--ignore-installed", "--no-compile", "--require-hashes",
               "--only-binary=:all:",
               "--find-links", str(links), "--target", str(target),
               "-r", str(source / "requirements-lock.txt")]
    with (working / "runtime-install.log").open("w") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(target))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HOME=str(target.parent / "gr-huggingface"), TOKENIZERS_PARALLELISM="false",
                      USE_TF="0", USE_FLAX="0")
    import torch
    if torch.__version__.split("+", 1)[0] != manifest["expected_torch_version"].split("+", 1)[0]:
        raise RuntimeError("platform PyTorch version differs from the registered GR runtime")
    import transformers
    import sentence_transformers
    import lightning
    import torchmetrics
    import hydra
    report = {"kind": "gr", "installed_offline": True,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": transformers.__version__,
              "sentence_transformers": sentence_transformers.__version__,
              "lightning": lightning.__version__, "torchmetrics": torchmetrics.__version__,
              "hydra": hydra.__version__,
              "installation_seconds": time.monotonic() - started,
              "requirements_sha256": runtime["requirements_sha256"],
              "manifest_sha256": runtime["manifest_sha256"],
              "platform_torch_reused": True}
    (working / "runtime-bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "gr_runtime_ready", **report}), flush=True)
    return report
