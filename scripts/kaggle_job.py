from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


STATUS_PATTERN = re.compile(r"status\s+[\"']?([a-z0-9_.-]+)", re.IGNORECASE)
SUCCESS_STATES = {"complete", "completed"}
FAILURE_STATES = {"error", "failed", "cancelled", "canceled"}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True, capture_output=True)


def parse_kernel_status(output: str) -> str:
    match = STATUS_PATTERN.search(output)
    if not match:
        raise RuntimeError(f"could not parse Kaggle status: {output[-1000:]}")
    return match.group(1).rsplit(".", 1)[-1].lower()


def kernel_status(kernel_id: str) -> tuple[str, str]:
    result = run(["kaggle", "kernels", "status", kernel_id], check=False)
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        raise RuntimeError(f"could not read Kaggle status: {output[-1000:]}")
    return parse_kernel_status(output), output


def copy_result_file(search_root: Path, name: str, destination: Path) -> None:
    matches = list(search_root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {len(matches)}")
    shutil.copy2(matches[0], destination / name)


def execute(bundle: Path, kernel_id: str, output: Path, timeout_minutes: int) -> None:
    if timeout_minutes < 1 or timeout_minutes > 720:
        raise ValueError("timeout must be between 1 and 720 minutes")
    output.mkdir(parents=True, exist_ok=True)

    pushed = run(["kaggle", "kernels", "push", "-p", str(bundle)])
    print(pushed.stdout, end="")

    # A fixed kernel slug may briefly report the previous version as complete.
    time.sleep(15)
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        status, raw = kernel_status(kernel_id)
        print(raw, flush=True)
        if status in SUCCESS_STATES:
            break
        if status in FAILURE_STATES:
            raise RuntimeError(f"Kaggle kernel finished with status {status}")
        if time.monotonic() >= deadline:
            run(["kaggle", "kernels", "cancel", kernel_id], check=False)
            raise TimeoutError(f"Kaggle kernel exceeded {timeout_minutes} minutes")
        time.sleep(30)

    downloaded = output / "downloaded"
    downloaded.mkdir(exist_ok=True)
    result = run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(downloaded), "--force"]
    )
    print(result.stdout, end="")
    copy_result_file(downloaded, "metrics.json", output)
    copy_result_file(downloaded, "summary.md", output)
    copy_result_file(downloaded, "run-manifest.json", output)
    json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and wait for a Kaggle kernel")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=int, default=360)
    args = parser.parse_args()
    execute(args.bundle, args.kernel_id, args.output, args.timeout_minutes)


if __name__ == "__main__":
    main()
