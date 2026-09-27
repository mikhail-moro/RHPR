from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rhpr.experiment import run_experiment
from rhpr.metrics import validate_metrics


def execute(config_path: Path, output_dir: Path) -> dict[str, object]:
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    started_at = datetime.now(UTC)
    metrics = validate_metrics(run_experiment(config))
    finished_at = datetime.now(UTC)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_lines = ["# Experiment result", ""]
    summary_lines.extend(f"- **{key}**: {value}" for key, value in metrics.items())
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "python": platform.python_version(),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "config": config,
    }
    (output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metrics": metrics, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RHPR experiment")
    parser.add_argument("--config", type=Path, default=Path("experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local"))
    args = parser.parse_args()
    result = execute(args.config, args.output)
    print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
