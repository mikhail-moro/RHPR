"""Stage an RHPR experiment, run it with the official Colab CLI and retrieve results."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
GPU_TYPES = ("T4", "L4", "A100", "H100")
GPU_EXPERIMENTS = {"gr_ml1m_last8", "gr_ml1m_last8_pairs", "gr_ml1m_mixed64_input_ablations"}
GPU_ALLOCATION_TIMEOUT_SECONDS = 45 * 60
REMOTE = "/content/rhpr-job"
SLUG = r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+"


def checked_path(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("expected a relative input path")
    return path


def colab_config(config, gpu, trial=None):
    config = copy.deepcopy(config)
    if gpu == "cpu":
        if config.get("kind", "synthetic") != "synthetic":
            raise ValueError("the CPU smoke run expects a synthetic experiment")
    else:
        if gpu not in GPU_TYPES or config.get("kind") not in GPU_EXPERIMENTS:
            raise ValueError("Colab GPU support is available for the ML1M Last-8 experiment family")
        config["execution"].update(colab_gpu=gpu, worker_devices=["cuda:0"])
    if trial is not None:
        if gpu != "T4" or not config.get("depth_pilot"):
            raise ValueError("single-trial selection is available for the depth pilot on Colab T4")
        selected = [spec for spec in config["trials"] if spec["id"] == trial]
        if len(selected) != 1:
            raise ValueError("expected one registered depth pilot trial ID")
        config["trials"] = selected
    return config


def input_plan(config):
    kaggle = config.get("kaggle", {})
    if kaggle.get("competition_sources") or kaggle.get("model_sources"):
        raise ValueError("Colab staging supports Kaggle datasets and notebook outputs")
    datasets = list(dict.fromkeys(kaggle.get("dataset_sources", [])))
    parents = {kernel: {} for kernel in kaggle.get("kernel_sources", [])}
    for slug in (*datasets, *parents):
        if not re.fullmatch(SLUG, slug):
            raise ValueError("Kaggle sources must use owner/slug")
    for entry in config.values():
        if isinstance(entry, dict) and entry.get("kernel") in parents and "files" in entry:
            files = parents[entry["kernel"]]
            for path, expected in entry["files"].items():
                checked_path(path)
                if path in files and files[path] != expected:
                    raise ValueError("conflicting hashes for one parent artifact")
                files[path] = expected
    for files in parents.values():
        if not files or "run-manifest.json" not in files:
            raise ValueError("Colab requires pinned output files for every parent notebook")
    return datasets, parents


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_location(path):
    relative = PurePosixPath(path).relative_to("/kaggle/input/datasets")
    owner, slug, *subdirectory = checked_path(relative.as_posix()).parts
    return f"{owner}/{slug}", PurePosixPath(*subdirectory)


def download_kaggle_input(args, kind, log):
    for delay in (5, 10, 20, 40, 60, 0):
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
        log.write(result.stdout)
        log.flush()
        if result.returncode == 0:
            return
        match = re.search(rb"\bHTTP(?: status| Error)?[ :]+([45][0-9]{2})\b|\b([45][0-9]{2}) (?:Client|Server) Error", result.stdout)
        status = int(match.group(1) or match.group(2)) if match else None
        if not delay or status not in {429, 500, 502, 503, 504}:
            reason = f"HTTP {status}" if status is not None else "undetermined"
            print(f"Kaggle {kind} download failed: exit {result.returncode}, {reason}", flush=True)
            result.check_returncode()
        time.sleep(delay)


def download_dataset_file(root, slug, name, expected, log):
    directory = root / "datasets" / slug
    destination = directory / name
    if not destination.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        args = ["kaggle", "datasets", "download", slug, "-f", str(name),
                "--path", str(directory), "--unzip", "--quiet"]
        download_kaggle_input(args, "dataset", log)
    if not destination.is_file() or sha256(destination) != expected:
        raise ValueError(f"Kaggle dataset file differs from its pin: {slug}/{name}")
    return destination


def stage_datasets(root, config, datasets, log):
    # Some private Kaggle datasets have no downloadable whole-dataset ZIP.
    # Fetch the same individual files named by the experiment's manifests.
    task_files = {}
    for key in ("resources", "modern_runtime"):
        if key not in config:
            continue
        spec = config[key]
        slug, subdirectory = dataset_location(spec["path"])
        if slug not in datasets:
            raise ValueError(f"missing dataset source for {key}")
        manifest = download_dataset_file(root, slug, subdirectory / "resources-manifest.json",
                                         spec["manifest_sha256"], log)
        prefixes = tuple(spec.get("include_prefixes", ()))
        task_files[slug] = {subdirectory / checked_path(item["path"]): item["sha256"]
                            for item in json.loads(manifest.read_text())["files"]
                            if not prefixes or item["path"].startswith(prefixes)}
    movie = config.get("movie_input")
    if movie:
        slug, subdirectory = dataset_location(movie["path"])
        if slug in datasets:
            task_files[slug] = {subdirectory / "ml-1m-source.bin": movie["source_zip_sha256"]}
    if set(task_files) != set(datasets):
        raise ValueError("Colab requires pinned files for every Kaggle dataset source")
    for slug, files in task_files.items():
        for name, sha in files.items():
            download_dataset_file(root, slug, name, sha, log)
    runtime = config.get("runtime")
    if runtime:
        slug, subdirectory = dataset_location(runtime["wheel_path"])
        for name, key in (("runtime-manifest.json", "manifest_sha256"),
                          ("requirements-lock.txt", "requirements_sha256")):
            path = root / "datasets" / slug / subdirectory / name
            if not path.is_file() or sha256(path) != runtime[key]:
                raise ValueError(f"Kaggle runtime file differs from its pin: {name}")


def stage_inputs(root, config, log):
    datasets, parents = input_plan(config)
    stage_datasets(root, config, datasets, log)
    for slug, files in parents.items():
        target = root / "notebooks" / slug
        target.mkdir(parents=True)
        pattern = r"^(?:" + "|".join(map(re.escape, sorted(files))) + r")$"
        download_kaggle_input(["kaggle", "kernels", "output", slug, "--path", str(target),
                              "--file-pattern", pattern, "--page-size", "200", "--quiet"], "notebook", log)
        for name, expected in files.items():
            path = target / name
            if (not path.is_file() or not path.resolve().is_relative_to(target)
                    or path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]):
                raise ValueError(f"Kaggle parent differs from its pin: {slug}/{name}")
        # The Kaggle API may attach stdout separately from the requested files.
        for path in target.rglob("*"):
            if path.is_file() and path.relative_to(target).as_posix() not in files:
                path.unlink()


def run_colab_phase(args, phase, log, *, timeout, stdin=None, stdout=None, check=True):
    """Expose the failed phase and allowlisted diagnostics in public CI."""
    try:
        result = subprocess.run(args, stdin=stdin, stdout=log if stdout is None else stdout,
                                stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        log.write(error.stderr or b"")
        print(f"Colab phase {phase}: timeout", flush=True)
        raise
    detail = result.stderr or b""
    log.write(detail)
    log.flush()
    if phase == "allocation":
        retries = detail.count(b"Colab allocation returned ")
        print(f"Colab allocation finished (retry count {retries}, exit {result.returncode})", flush=True)
    if result.returncode:
        if phase == "allocation":
            status = re.search(rb"^Colab backend HTTP ([45][0-9]{2})$", detail, re.MULTILINE)
        else:
            status = re.search(rb"\bHTTP(?: status)? ([45][0-9]{2})\b", detail)
        reason = next((label for needle, label in (
            (b"Allocation refused", "allocation-refused"),
            (b"Backend rejected accelerator", "accelerator-rejected"),
            (b"Colab allocation exception: ValidationError", "invalid-backend-response"),
            (b"Colab allocation exception: JSONDecodeError", "invalid-backend-json"),
            (b"Colab allocation exception: ReadTimeout", "backend-read-timeout"),
            (b"Colab allocation exception: ConnectTimeout", "backend-connect-timeout"),
            (b"Colab allocation exception: ConnectionError", "backend-connection-error"),
            (b"SSH session limit reached", "ssh-session-limit"),
            (b"SSH requires", "ssh-unavailable"),
            (b"Permission denied (publickey", "ssh-public-key-rejected"),
            (b"Host key verification failed", "ssh-host-key-rejected"),
            (b"File exists", "remote-path-already-exists"),
            (b"Connection closed", "connection-closed"),
        ) if needle in detail), "undetermined")
        http = f", HTTP {status.group(1).decode('ascii')}" if status else ""
        print(f"Colab phase {phase}: exit {result.returncode}, {reason}{http}", flush=True)
        if check:
            result.check_returncode()
    return result


def run_checkpointed_training(remote_run, working, minutes, output, log):
    status_file = f"{working}/training-exit"
    epoch_file = f"{working}/checkpoint-epoch"
    script = (f"timeout -k 30s {minutes * 60}s python3 -u {REMOTE}/job/colab_runtime.py; "
              f"printf '%s\\n' \"$?\" > {status_file}.partial && mv {status_file}.partial {status_file}")
    run_colab_phase(remote_run + [
        f"nohup bash -c {shlex.quote(script)} > {working}/colab-run.log 2>&1 < /dev/null &",
    ], "execution-start", log, timeout=60)

    probe = remote_run + [
        f"for path in {status_file} {epoch_file}; do if test -f \"$path\"; "
        'then cat "$path"; else echo -1; fi; done',
    ]
    last_epoch = 0
    failures = 0
    deadline = time.monotonic() + minutes * 60 + 90
    while time.monotonic() < deadline:
        time.sleep(min(60, max(0, deadline - time.monotonic())))
        try:
            result = run_colab_phase(probe, "execution-status", log, timeout=45, stdout=subprocess.PIPE, check=False)
        except subprocess.TimeoutExpired:
            result = None
        status = re.fullmatch(rb"(-1|[0-9]{1,3})\n(-1|[0-9]+)\n", result.stdout) if result and not result.returncode else None
        if not status:
            failures += 1
            if failures == 5:
                raise RuntimeError("Lost contact with the Colab runtime during training")
            continue
        failures = 0
        exit_code, epoch = map(int, status.groups())
        if exit_code != 0 and epoch > last_epoch:
            pending = output / "checkpoint.partial"
            try:
                with pending.open("wb") as stream:
                    saved = run_colab_phase(remote_run + [f"cat {working}/checkpoint.tar"],
                                            "checkpoint-download", log, timeout=90, stdout=stream, check=False)
                if not saved.returncode and tarfile.is_tarfile(pending):
                    pending.rename(output / "checkpoint.tar")
                    last_epoch = epoch
                    print(f"Colab checkpoint saved after epoch {epoch}", flush=True)
            except subprocess.TimeoutExpired:
                pass
            finally:
                pending.unlink(missing_ok=True)
        if exit_code >= 0:
            return exit_code
    raise subprocess.TimeoutExpired(probe, minutes * 60 + 90)


def run_job(experiment, gpu, minutes, output, state, session):
    filename, separator, trial = experiment.partition("#")
    experiment = (ROOT / checked_path(filename)).resolve()
    experiment.relative_to((ROOT / "experiments").resolve())
    config = colab_config(json.loads(experiment.read_text()), gpu, trial if separator else None)
    output.mkdir(parents=True, exist_ok=False)
    cli = ["colab", "--auth", "adc", "--config", str(state)]

    with (output / "deployment.log").open("wb") as log, tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        remote = stage / "job"
        remote.mkdir()
        (remote / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy(ROOT / "scripts/colab_runtime.py", remote)
        shutil.copytree(ROOT / "src/rhpr", remote / "src/rhpr", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (remote / "deployment.json").write_text(json.dumps(dict(
            provider="colab", gpu_requested=gpu, git_sha=revision, session=session,
            experiment=str(experiment.relative_to(ROOT)), experiment_sha256=sha256(experiment),
            runtime_config_sha256=sha256(remote / "experiment.json")),
            indent=2) + "\n")
        kaggle = stage / "kaggle"
        (kaggle / "input").mkdir(parents=True)
        (kaggle / "working").mkdir()
        (kaggle / "temp").mkdir()
        print("Preparing experiment inputs", flush=True)
        stage_inputs(kaggle / "input", config, log)
        print(f"Opening Colab session {session} on {gpu}", flush=True)
        state.parent.mkdir(parents=True, exist_ok=True)
        identity = stage / "ssh"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(identity)], check=True)
        remote_run = cli + ["ssh", "-s", session, "-i", str(identity), "--"]
        transfer = stage / "input.tar"
        with tarfile.open(transfer, "w") as archive:
            archive.add(remote, "job")
            archive.add(kaggle, "kaggle")

        new_session = [sys.executable, str(ROOT / "scripts/colab_allocate.py")]
        if gpu != "cpu":
            new_session.append("--retry-transient")
        new_session.extend([*cli[1:], "new", "-s", session])
        if gpu != "cpu":
            new_session += ["--gpu", gpu]
        try:
            run_colab_phase(new_session, "allocation", log,
                            timeout=GPU_ALLOCATION_TIMEOUT_SECONDS if gpu != "cpu" else 300)
            with transfer.open("rb") as stream:
                run_colab_phase(remote_run + [
                    f"mkdir -p {REMOTE} && tar -xf - -C {REMOTE}",
                ], "upload", log, stdin=stream, timeout=1200)
            print("Running experiment on Colab", flush=True)
            # The remote deadline also applies if the GitHub runner disconnects.
            working = f"{REMOTE}/kaggle/working"
            if separator and gpu == "T4":
                returncode = run_checkpointed_training(remote_run, working, minutes, output, log)
            else:
                training = run_colab_phase(remote_run + [
                    f"timeout -k 30s {minutes * 60}s python3 -u {REMOTE}/job/colab_runtime.py "
                    f"> {working}/colab-run.log 2>&1",
                ], "execution", log, timeout=minutes * 60 + 90, check=False)
                returncode = training.returncode
            print("Retrieving Colab results", flush=True)
            pending = output / "results.partial"
            with pending.open("wb") as stream:
                run_colab_phase(remote_run + [f"tar -C {working} --exclude=./checkpoint.tar -cf - ."], "result-download", log,
                                stdout=stream, timeout=1200)
            pending.rename(output / "results.tar")
            if returncode:
                raise subprocess.CalledProcessError(returncode, "Colab training")
        finally:
            if state.is_file():
                print(f"Releasing Colab session {session}", flush=True)
                stopped = subprocess.run(cli + ["stop", "-s", session], stdout=log, stderr=log, timeout=60)
                if stopped.returncode:
                    print("Colab did not confirm session release. See Manage sessions.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="experiments/baseline.json")
    parser.add_argument("--gpu", choices=("cpu", *GPU_TYPES), default="cpu")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--session", default=f"rhpr-{uuid.uuid4().hex[:10]}")
    args = parser.parse_args()
    if not 1 <= args.minutes <= 300 or not re.fullmatch(r"rhpr-[a-z0-9-]{1,50}", args.session):
        parser.error("expected 1–300 minutes and a safe rhpr- session name")
    try:
        run_job(args.experiment, args.gpu, args.minutes, args.output.resolve(), args.state.resolve(), args.session)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        # Public CI prints phase markers only. Detailed service errors can
        # include private URLs and are recorded in deployment.log.
        if isinstance(error, ValueError):
            parser.exit(1, f"Colab deployment failed: {error}\n")
        parser.exit(1, f"Colab deployment failed ({type(error).__name__}); see deployment.log.\n")


if __name__ == "__main__":
    main()
