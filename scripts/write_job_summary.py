from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    lines = ["# Kaggle experiment", "", f"Configuration: `{args.experiment}`", ""]
    if args.metrics.is_file():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        lines.extend(["| Metric | Value |", "| --- | --- |"])
        lines.extend(f"| `{key}` | {value} |" for key, value in sorted(metrics.items()))
    else:
        lines.append("No `metrics.json` was produced. Inspect the workflow logs.")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
