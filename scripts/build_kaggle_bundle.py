from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build(experiment: Path, kernel_id: str, accelerator: str, output: Path) -> Path:
    experiment = experiment.resolve()
    experiment.relative_to((ROOT / "experiments").resolve())
    if "/" not in kernel_id or kernel_id.startswith("/") or kernel_id.endswith("/"):
        raise ValueError("kernel id must have the form username/slug")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")

    config = json.loads(experiment.read_text(encoding="utf-8"))
    kaggle_config = config.get("kaggle", {})
    if not isinstance(kaggle_config, dict):
        raise ValueError("config.kaggle must be an object")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "rhpr", output / "rhpr")
    shutil.copy2(ROOT / "kaggle" / "run.py", output / "run.py")
    shutil.copy2(experiment, output / "experiment.json")

    metadata = {
        "id": kernel_id,
        "title": f"RHPR: {config.get('name', experiment.stem)}",
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
