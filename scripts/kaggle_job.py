from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STATUS_PATTERN = re.compile(r"status\s+[\"']?([a-z0-9_.-]+)", re.IGNORECASE)
PUSH_VERSION_PATTERN = re.compile(r"Kernel version\s+(\d+)\s+successfully pushed", re.IGNORECASE)
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
    # Suites contain per-variant metrics; the root aggregate is authoritative.
    root_file = search_root / name
    if root_file.is_file():
        shutil.copy2(root_file, destination / name)
        return
    matches = list(search_root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {len(matches)}")
    shutil.copy2(matches[0], destination / name)


def write_collection_manifest(
    downloaded: Path,
    output: Path,
    kernel_id: str,
    output_file_pattern: str | None,
    *,
    kernel_version: int | None = None,
    strict_checkpoint_manifest: bool = True,
) -> None:
    files = [
        {"path": path.relative_to(downloaded).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(downloaded.rglob("*"))
        if path.is_file()
    ]
    downloaded_sizes = {entry["path"]: entry["size_bytes"] for entry in files}
    checkpoint_manifest_path = downloaded / "checkpoint-manifest.json"
    checkpoints = []
    checkpoint_manifest_error_type = None
    if checkpoint_manifest_path.is_file():
        try:
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
            for checkpoint in checkpoint_manifest.get("checkpoints", []):
                relative_path = checkpoint["path"]
                path = Path(relative_path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"checkpoint path must stay within kernel outputs: {relative_path}")
                size = downloaded_sizes.get(path.as_posix())
                checkpoints.append(
                    {
                        "path": path.as_posix(),
                        "sha256": checkpoint.get("sha256"),
                        "kaggle_size_bytes": checkpoint.get("size_bytes"),
                        "downloaded": size is not None,
                        "downloaded_size_bytes": size,
                        "storage": "downloaded" if size is not None else "retained_on_private_kaggle",
                    }
                )
        except Exception as error:
            if strict_checkpoint_manifest:
                raise
            checkpoints = []
            checkpoint_manifest_error_type = type(error).__name__
    manifest = {
        "kernel_id": kernel_id,
        "kernel_version": kernel_version,
        "versioned_kernel_id": f"{kernel_id}/{kernel_version}" if kernel_version is not None else None,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file_pattern": output_file_pattern,
        "filtered_download": output_file_pattern is not None,
        "downloaded_files": files,
        "downloaded_file_count": len(files),
        "downloaded_size_bytes": sum(entry["size_bytes"] for entry in files),
        "checkpoint_manifest_downloaded": checkpoint_manifest_path.is_file(),
        "checkpoints": checkpoints,
    }
    if checkpoint_manifest_error_type is not None:
        manifest["checkpoint_manifest_error_type"] = checkpoint_manifest_error_type
    (output / "output-collection-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def download_outputs(kernel_id: str, downloaded: Path, output_file_pattern: str | None) -> None:
    downloaded.mkdir(exist_ok=True)
    command = ["kaggle", "kernels", "output", kernel_id, "-p", str(downloaded), "--force"]
    if output_file_pattern is not None:
        command.extend(["--file-pattern", output_file_pattern])
    result = run(command)
    print(result.stdout, end="")


def collect_failed_outputs(
    output: Path,
    kernel_id: str,
    output_file_pattern: str | None,
    *,
    kernel_version: int | None,
    terminal_status: str,
    last_observed_status: str | None,
    primary_error: BaseException,
    collection_errors: list[dict],
) -> None:
    """Retain diagnostics without treating incomplete output as a successful run."""
    downloaded = output / "downloaded"
    downloaded.mkdir(exist_ok=True)
    try:
        download_outputs(kernel_id, downloaded, output_file_pattern)
    except Exception as error:
        collection_errors.append({"stage": "download_outputs", "error_type": type(error).__name__})

    # The CLI writes execution logs as <kernel-slug>.log, independently of the
    # file filter. If a partial download failed before that step, retry logs only.
    kernel_slug = kernel_id.split("/")[1]
    cli_log = downloaded / f"{kernel_slug}.log"
    kernel_log = downloaded / "kernel.log"
    if not kernel_log.is_file() and not cli_log.is_file():
        try:
            download_outputs(kernel_id, downloaded, r"(?!)")
        except Exception as error:
            collection_errors.append({"stage": "download_kernel_log", "error_type": type(error).__name__})
    if not kernel_log.is_file() and cli_log.is_file():
        try:
            shutil.copy2(cli_log, kernel_log)
        except Exception as error:
            collection_errors.append({"stage": "copy_kernel_log", "error_type": type(error).__name__})

    copied = []
    missing = []
    for name in ("metrics.json", "summary.md", "run-manifest.json"):
        # Do not promote a completed trial's metrics to the aggregate suite result.
        if not (downloaded / name).is_file():
            missing.append(name)
            continue
        try:
            shutil.copy2(downloaded / name, output / name)
            copied.append(name)
        except Exception as error:
            collection_errors.append({"stage": f"copy_{name}", "error_type": type(error).__name__})
    try:
        write_collection_manifest(
            downloaded, output, kernel_id, output_file_pattern,
            kernel_version=kernel_version, strict_checkpoint_manifest=False,
        )
    except Exception as error:
        collection_errors.append({"stage": "write_collection_manifest", "error_type": type(error).__name__})

    # Exception messages and subprocess stdout/stderr can contain credentials or
    # signed download URLs. Record only exception classes and controlled metadata.
    report = {
        "kernel_id": kernel_id,
        "kernel_version": kernel_version,
        "terminal_status": terminal_status,
        "last_observed_kernel_status": last_observed_status,
        "completed_successfully": False,
        "primary_error_type": type(primary_error).__name__,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "copied_root_files": copied,
        "missing_root_files": missing,
        "kernel_log_downloaded": kernel_log.is_file(),
        "collection_errors": collection_errors,
    }
    (output / "terminal-status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def execute(
    bundle: Path,
    kernel_id: str,
    output: Path,
    timeout_minutes: int,
    *,
    output_file_pattern: str | None = None,
) -> None:
    if timeout_minutes < 1 or timeout_minutes > 720:
        raise ValueError("timeout must be between 1 and 720 minutes")
    output_file_pattern = output_file_pattern or None
    if output_file_pattern is not None:
        re.compile(output_file_pattern)
    output.mkdir(parents=True, exist_ok=True)

    pushed = run(["kaggle", "kernels", "push", "-p", str(bundle)])
    print(pushed.stdout, end="")
    version_match = PUSH_VERSION_PATTERN.search(pushed.stdout)
    kernel_version = int(version_match.group(1)) if version_match else None

    terminal_status = "status_poll_error"
    last_observed_status = None
    collection_errors: list[dict] = []
    try:
        # A fixed kernel slug may briefly report the previous version as complete.
        time.sleep(15)
        deadline = time.monotonic() + timeout_minutes * 60
        while True:
            status, raw = kernel_status(kernel_id)
            last_observed_status = status
            print(raw, flush=True)
            if status in SUCCESS_STATES:
                break
            if status in FAILURE_STATES:
                terminal_status = status
                raise RuntimeError(f"Kaggle kernel finished with status {status}")
            if time.monotonic() >= deadline:
                terminal_status = "timeout"
                try:
                    cancelled = run(["kaggle", "kernels", "cancel", kernel_id], check=False)
                    if cancelled.returncode:
                        collection_errors.append({"stage": "cancel", "returncode": cancelled.returncode})
                except Exception as error:
                    collection_errors.append({"stage": "cancel", "error_type": type(error).__name__})
                raise TimeoutError(f"Kaggle kernel exceeded {timeout_minutes} minutes")
            time.sleep(30)
    except BaseException as error:
        try:
            collect_failed_outputs(
                output, kernel_id, output_file_pattern, kernel_version=kernel_version,
                terminal_status=terminal_status, last_observed_status=last_observed_status,
                primary_error=error, collection_errors=collection_errors,
            )
        except BaseException:
            # Cleanup must never replace the failure that caused collection.
            pass
        raise

    downloaded = output / "downloaded"
    download_outputs(kernel_id, downloaded, output_file_pattern)
    copy_result_file(downloaded, "metrics.json", output)
    copy_result_file(downloaded, "summary.md", output)
    copy_result_file(downloaded, "run-manifest.json", output)
    json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    write_collection_manifest(
        downloaded, output, kernel_id, output_file_pattern, kernel_version=kernel_version
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and wait for a Kaggle kernel")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-minutes", type=int, default=360)
    parser.add_argument(
        "--output-file-pattern",
        help="Download only kernel output filenames matching this regex; unselected files stay on Kaggle",
    )
    args = parser.parse_args()
    execute(
        args.bundle,
        args.kernel_id,
        args.output,
        args.timeout_minutes,
        output_file_pattern=args.output_file_pattern,
    )


if __name__ == "__main__":
    main()
