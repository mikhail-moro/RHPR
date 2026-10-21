from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--submission", type=Path)
    args = parser.parse_args()

    lines = ["# Kaggle experiment", "", f"Configuration: `{args.experiment}`", ""]
    submission = (json.loads(args.submission.read_text(encoding="utf-8"))
                  if args.submission is not None and args.submission.is_file() else None)
    if submission is not None and submission.get("status") == "submitted" and submission.get("quality_complete") is False:
        kernel_id = submission.get("kernel_id")
        lines.extend([
            "**Submitted to Kaggle; training and quality evaluation are pending.**",
            "This workflow only submitted the kernel. It did not wait for completion or collect results.",
            "",
            f"Kernel: `{kernel_id}`",
            f"Submitted version: `{submission.get('kernel_version')}`",
            f"Source revision: `{submission.get('source_code_sha')}`",
            "",
            "Inspect the submitted Kaggle version and collect its outputs after completion.",
        ])
    elif args.metrics.is_file():
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
