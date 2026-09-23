from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rhpr.runner import execute  # noqa: E402


if __name__ == "__main__":
    result = execute(HERE / "experiment.json", Path("/kaggle/working"))
    print(result["metrics"])
