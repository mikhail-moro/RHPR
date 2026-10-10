from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rhpr.runner import execute  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "local")
    args = parser.parse_args()
    result = execute(args.experiment, args.output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
