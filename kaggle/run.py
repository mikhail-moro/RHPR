from __future__ import annotations

import sys
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rhpr.runner import execute  # noqa: E402


if __name__ == "__main__":
    from rhpr.offline_runtime import install_offline_runtime
    install_offline_runtime(json.loads((HERE / "experiment.json").read_text()).get("runtime"), Path("/kaggle/working"))
    result = execute(HERE / "experiment.json", Path("/kaggle/working"))
    print(result["metrics"])
