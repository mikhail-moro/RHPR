from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _embedded_package() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((ROOT / "src" / "rhpr").glob("*.py")):
            archive.writestr(f"rhpr/{source.name}", source.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_runner(config_text: str, git_sha: str) -> str:
    package = _embedded_package()
    return f'''from __future__ import annotations

import base64
import os
import sys
from pathlib import Path


EMBEDDED_PACKAGE = {package!r}
EXPERIMENT_CONFIG = {config_text!r}
GIT_SHA = {git_sha!r}
WORKING = Path("/kaggle/working")
PACKAGE_ARCHIVE = WORKING / "rhpr-package.zip"
CONFIG_PATH = WORKING / "experiment.json"

WORKING.mkdir(parents=True, exist_ok=True)
PACKAGE_ARCHIVE.write_bytes(base64.b64decode(EMBEDDED_PACKAGE))
CONFIG_PATH.write_text(EXPERIMENT_CONFIG, encoding="utf-8")
os.environ["GITHUB_SHA"] = GIT_SHA
sys.path.insert(0, str(PACKAGE_ARCHIVE))

from rhpr.runner import execute  # noqa: E402


result = execute(CONFIG_PATH, WORKING)
print(result["metrics"])
'''


def build(experiment: Path, kernel_id: str, accelerator: str, output: Path) -> Path:
    experiment = experiment.resolve()
    experiment.relative_to((ROOT / "experiments").resolve())
    if "/" not in kernel_id or kernel_id.startswith("/") or kernel_id.endswith("/"):
        raise ValueError("kernel id must have the form username/slug")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")

    config_text = experiment.read_text(encoding="utf-8")
    config = json.loads(config_text)
    kaggle_config = config.get("kaggle", {})
    if not isinstance(kaggle_config, dict):
        raise ValueError("config.kaggle must be an object")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    git_sha = os.environ.get("GITHUB_SHA", "local")
    (output / "run.py").write_text(_render_runner(config_text, git_sha), encoding="utf-8")

    kernel_slug = kernel_id.split("/", 1)[1]

    metadata = {
        "id": kernel_id,
        "title": kernel_slug.replace("-", " ").replace("_", " ").title(),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": accelerator == "gpu",
        "enable_internet": False,
        "dataset_sources": kaggle_config.get("dataset_sources", []),
        "competition_sources": kaggle_config.get("competition_sources", []),
        "kernel_sources": kaggle_config.get("kernel_sources", []),
        "model_sources": kaggle_config.get("model_sources", []),
    }
    metadata_path = output / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained Kaggle kernel")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "kaggle")
    args = parser.parse_args()
    print(build(args.experiment, args.kernel_id, args.accelerator, args.output))


if __name__ == "__main__":
    main()
