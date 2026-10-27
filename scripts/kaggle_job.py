from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


STATUS_PATTERN = re.compile(r"status\s+[\"']?([a-z0-9_.-]+)", re.IGNORECASE)
PUSH_VERSION_PATTERN = re.compile(r"Kernel version\s+(\d+)\s+successfully pushed", re.IGNORECASE)
SUCCESS_STATES = {"complete", "completed"}
FAILURE_STATES = {"error", "failed", "cancelled", "canceled"}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    try:
        return subprocess.run(command, check=check, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        # The CLI's server response used to disappear behind CalledProcessError.
        # Keep diagnostics useful without copying auth values into job logs.
        diagnostic = "\n".join(x for x in (error.stdout, error.stderr) if x)
        for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "TELEGRAM_BOT_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            value = os.environ.get(name)
            if value:
                diagnostic = diagnostic.replace(value, "[redacted]")
        diagnostic = re.sub(r"(?i)(Bearer\s+)[^\s'\"]+", r"\1[redacted]", diagnostic)
        diagnostic = re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\"]+", r"\1?[redacted]", diagnostic)
        print(diagnostic[-4000:], flush=True)
        raise


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


def _external_checkpoint_reference(checkpoint: dict, config: dict) -> dict:
    """Describe a pinned input checkpoint without resolving it on the collector host."""
    def reject() -> None:
        raise ValueError("external checkpoint must match pinned reused_controls metadata and input mount")

    controls = config.get("reused_controls", {})
    kernel = controls.get("kernel")
    source_sha = controls.get("source_code_sha")
    if (checkpoint.get("reused") is not True or not isinstance(kernel, str)
            or re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel) is None
            or not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None
            or checkpoint.get("external_kernel") != kernel
            or checkpoint.get("source_code_sha") != source_sha):
        reject()
    family = checkpoint.get("training_model")
    spec = controls.get("models", {}).get(family, {})
    expected = spec.get("checkpoint", {})
    original = checkpoint.get("original_path")
    digest, size = expected.get("sha256"), expected.get("size_bytes")
    original_trial = checkpoint.get("original_trial")
    registered_trials = config.get("architecture_trials", {}).get(family, [])
    if (not isinstance(original, str) or not original or "\\" in original
            or any(part in {"", ".", ".."} for part in original.split("/"))
            or PurePosixPath(original).is_absolute() or original != expected.get("file")
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or size <= 0
            or type(checkpoint.get("size_bytes")) is not int
            or checkpoint.get("sha256") != digest or checkpoint.get("size_bytes") != size
            or not isinstance(original_trial, dict)
            or original_trial.get("name") != spec.get("original_trial")
            or not isinstance(spec.get("original_trial"), str)
            or not registered_trials or checkpoint.get("trial") != registered_trials[0]
            or ("best_epoch" in expected and checkpoint.get("best_epoch") != expected["best_epoch"])):
        reject()

    # Kaggle varies the input mount layout. Only known roots or the explicitly
    # configured, bounded input root are accepted; no filesystem lookup is needed.
    owner, slug = kernel.split("/")
    base = PurePosixPath("/kaggle/input")
    roots = {base / slug, base / owner / slug,
             base / "notebooks" / owner / slug, base / "kernels" / owner / slug}
    configured = controls.get("checkpoint_root")
    if not isinstance(configured, str) or not configured or "\\" in configured:
        reject()
    configured_path = PurePosixPath(configured)
    if ".." in configured_path.parts or "." in configured.split("/"):
        reject()
    if not configured_path.is_absolute():
        configured_path = base / configured_path
    try:
        configured_parts = configured_path.relative_to(base).parts
    except ValueError:
        reject()
    if not 1 <= len(configured_parts) <= 4 or configured_parts[-1] != slug:
        reject()
    roots.add(configured_path)
    mounted = checkpoint.get("path")
    if (not isinstance(mounted, str) or checkpoint.get("verified_mount_path") != mounted
            or mounted not in {str(root / original) for root in roots}):
        reject()
    return {
        "path": mounted,
        "original_path": original,
        "verified_mount_path": mounted,
        "external_kernel": kernel,
        "source_code_sha": source_sha,
        "training_model": family,
        "original_trial": original_trial,
        "reused": True,
        "sha256": digest,
        "kaggle_size_bytes": size,
        "downloaded": False,
        "downloaded_size_bytes": None,
        "storage": "retained_on_private_parent_kaggle",
    }


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
                if checkpoint.get("reused") is True:
                    checkpoints.append(_external_checkpoint_reference(checkpoint, checkpoint_manifest.get("config", {})))
                    continue
                if any(key in checkpoint for key in ("external_kernel", "verified_mount_path", "original_path")):
                    raise ValueError("external checkpoint requires explicit reused=true metadata")
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
    submit_only: bool = False,
) -> None:
    if timeout_minutes < 1 or timeout_minutes > 720:
        raise ValueError("timeout must be between 1 and 720 minutes")
    output_file_pattern = output_file_pattern or None
    if output_file_pattern is not None:
        re.compile(output_file_pattern)
    output.mkdir(parents=True, exist_ok=True)

    submission_metadata = None
    if submit_only:
        # Pin the submitted bundle before the external mutation. Do not copy
        # raw CLI output, environment variables, or execution results here.
        metadata_bytes = (bundle / "kernel-metadata.json").read_bytes()
        metadata = json.loads(metadata_bytes)
        if metadata.get("id") != kernel_id:
            raise ValueError("bundle kernel id does not match the requested kernel")
        source_sha = os.environ.get("GITHUB_SHA", "")
        submission_metadata = {
            "source_code_sha": source_sha if re.fullmatch(r"[0-9a-f]{40}", source_sha) else None,
            "kernel_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "runner_sha256": hashlib.sha256((bundle / "run.py").read_bytes()).hexdigest(),
        }

    # A submit-only workflow has no polling process left to enforce its limit.
    # Ask Kaggle to enforce the same runtime ceiling on the worker itself.
    pushed = run(["kaggle", "kernels", "push", "-p", str(bundle),
                  "--timeout", str(timeout_minutes * 60)])
    print(pushed.stdout, end="")
    version_match = PUSH_VERSION_PATTERN.search(pushed.stdout)
    kernel_version = int(version_match.group(1)) if version_match else None

    if submit_only:
        submission = {
            **submission_metadata,
            "kernel_id": kernel_id,
            "kernel_version": kernel_version,
            "versioned_kernel_id": f"{kernel_id}/{kernel_version}" if kernel_version is not None else None,
            "status": "submitted",
            "quality_complete": False,
            "worker_timeout_seconds": timeout_minutes * 60,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
        return

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
        "--submit-only", action="store_true",
        help="Return after submission without polling, cancelling, or collecting training results",
    )
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
        submit_only=args.submit_only,
    )


if __name__ == "__main__":
    main()
