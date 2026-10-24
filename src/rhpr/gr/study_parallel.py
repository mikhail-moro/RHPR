"""Independent seed workers, one visible GPU per fresh Python process.

Parent prepares data/indexes once. Workers own disjoint table shards and seed
directories; only the parent creates DuckDB views, inventory and global status.
An explicit CPU worker mode tests the same orchestration without claiming CUDA
validation. A GPU request never falls back to CPU or fewer devices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def seed_assignments(config):
    execution = config["execution"]
    devices = execution.get("worker_devices")
    from rhpr.gr.study_design import study_jobs
    jobs = study_jobs(config)
    seeds = config["quantizer_seeds"]
    if not isinstance(devices, list) or not devices or len(devices) > len(jobs):
        raise ValueError("Worker devices must be a nonempty list, at most one worker per seed")
    if type(execution.get("cpu_threads_per_worker")) is not int or execution["cpu_threads_per_worker"] < 1:
        raise ValueError("cpu_threads_per_worker must be positive")
    if config["training"]["device"] == "cuda":
        if any(not isinstance(d, str) or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) is None for d in devices):
            raise ValueError("CUDA workers require explicit cuda:N devices")
        if len(set(devices)) != len(devices):
            raise ValueError("Each CUDA worker must own a different GPU")
    elif any(d != "cpu" for d in devices):
        raise ValueError("CPU verification requires explicit CPU workers")
    if config.get("model_seeds") is None:
        return [{"slot": i, "device": device, "seeds": seeds[i::len(devices)]}
                for i, device in enumerate(devices)]
    return [{"slot": i, "device": device, "jobs": jobs[i::len(devices)],
             "seeds": [j["quantizer_seed"] for j in jobs[i::len(devices)]]}
            for i, device in enumerate(devices)]


def check_devices(config):
    import torch
    assignments = seed_assignments(config)
    if config["training"]["device"] == "cuda":
        count = torch.cuda.device_count()
        if any(int(a["device"].split(":")[1]) >= count for a in assignments):
            raise RuntimeError(f"Requested GPUs unavailable: need {config['execution']['worker_devices']}, found {count}; no fallback")
    return assignments


def worker_environment(config, assignment, *, environment=None, python_paths=None):
    env = dict(os.environ if environment is None else environment)
    if assignment["device"] != "cpu":
        logical = int(assignment["device"].split(":")[1])
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            selected = str(logical)
        else:
            devices = visible.split(",")
            if logical >= len(devices) or not devices[logical].strip() or devices[logical].strip() == "-1":
                raise ValueError("Requested logical GPU is absent from inherited CUDA_VISIBLE_DEVICES")
            selected = devices[logical].strip()
        env["CUDA_VISIBLE_DEVICES"] = selected
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    # Inherit the already installed offline wheel site and embedded RHPR zip.
    # Do not dump the inherited environment into outputs/logs.
    paths = sys.path if python_paths is None else python_paths
    env["PYTHONPATH"] = os.pathsep.join(str(Path.cwd()) if not p else str(p) for p in paths)
    threads = str(config["execution"]["cpu_threads_per_worker"])
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env


def collect_results(output, assignments):
    result, failures = {"runs": sum(len(a["seeds"]) for a in assignments), "parallel_workers": len(assignments)}, []
    completed = 0
    for assignment in assignments:
        status_path = output / "workers" / f"worker-{assignment['slot']}.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        failures.extend(status.get("failures", []))
        for seed, metrics in status.get("results", {}).items():
            completed += 1
            prefix = str(seed).replace("-", "_") if str(seed).startswith("rq-") else f"rq_{seed}"
            result.update({prefix + "_" + key: value for key, value in metrics.items()})
    result["completed_runs"] = completed
    return result, failures


def forward_events(reader, slot):
    """Surface live training/eval progress while retaining full worker logs."""
    from rhpr.gr.experiment import _event
    while True:
        start = reader.tell()
        line = reader.readline()
        if not line:
            return
        if not line.endswith("\n"):
            reader.seek(start)
            return
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and str(event.get("phase", "")).startswith("sid_study_"):
            _event("sid_study_worker_event", worker_slot=slot, worker_event=event)


def launch_workers(config, output, source, windows, initial_hash):
    from rhpr.gr.experiment import _event
    output = Path(output).resolve()
    assignments = check_devices(config)
    directory = output / "workers"
    directory.mkdir()
    plan = {"config": config, "output": str(output), "source": str(Path(source).resolve()),
            "windows": {str(epoch): str(path.relative_to(output)) for epoch, path in windows.items()},
            "initial_state_sha256": initial_hash, "assignments": assignments}
    plan_path = directory / "plan.json"
    atomic_json(plan_path, plan)
    processes, streams, readers, last_state = [], [], [], None
    try:
        for assignment in assignments:
            slot = assignment["slot"]
            stream = (directory / f"worker-{slot}.log").open("w")
            streams.append(stream)
            process = subprocess.Popen([sys.executable, "-m", "rhpr.gr.study_parallel", "--plan", str(plan_path),
                "--slot", str(slot)], env=worker_environment(config, assignment), stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            readers.append((directory / f"worker-{slot}.log").open(errors="replace"))
            _event("sid_study_worker_started", **assignment, pid=process.pid)
        while True:
            for slot, reader in enumerate(readers):
                forward_events(reader, slot)
            result, failures = collect_results(output, assignments)
            state = (result["completed_runs"], len(failures))
            if state != last_state:
                atomic_json(output / "partial-metrics.json", result)
                _event("sid_study_parallel_progress", completed_runs=state[0], failed_runs=state[1], total_runs=result["runs"])
                last_state = state
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(.5)
        # Collect again AFTER exit: a process may commit its final result
        # between the last status read and poll() observing its exit.
        result, failures = collect_results(output, assignments)
        for slot, reader in enumerate(readers):
            forward_events(reader, slot)
        atomic_json(output / "partial-metrics.json", result)
        codes = [process.returncode for process in processes]
        if any(code != 0 for code in codes) or failures or result["completed_runs"] != result["runs"]:
            raise RuntimeError(f"Incomplete SID study: {result['completed_runs']}/{result['runs']} seeds, worker exits {codes}; see workers/*.json and *.log")
        return result
    finally:
        # Cancellation or launcher failure must not leave detached GPU jobs.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for stream in streams:
            stream.close()
        for reader in readers:
            reader.close()


def run_worker(plan_path, slot):
    plan = json.loads(Path(plan_path).read_text())
    assignment = plan["assignments"][slot]
    if assignment["slot"] != slot:
        raise ValueError("Worker assignment mismatch")
    config, output = plan["config"], Path(plan["output"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from rhpr.gr.data import load_beauty
    from rhpr.gr.genpas import activate_source
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import train_one_seed
    from rhpr.gr.study_store import StudyStore
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if assignment["device"] != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA worker must see exactly its assigned GPU")
        torch.cuda.set_device(0)
    _, users, _, _ = load_beauty(output / "inputs", config["data"])
    source = activate_source(plan["source"])
    store = StudyStore(output, shard=f"worker-{slot}")
    cases = store.read("cases")
    windows = {int(epoch): output / path for epoch, path in plan["windows"].items()}
    status = {**assignment, "pid": os.getpid(), "status": "running", "results": {}, "failures": [],
              "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "hardware": torch.cuda.get_device_name(0) if assignment["device"] != "cpu" else "cpu"}
    status_path = output / "workers" / f"worker-{slot}.json"
    atomic_json(status_path, status)
    from rhpr.gr.study_design import initial_hash_for
    jobs = assignment.get("jobs") or [{"quantizer_seed": q, "model_seed": config["model_seed"], "run_id": f"rq-{q}"}
                                       for q in assignment["seeds"]]
    for job in jobs:
        seed, run_id = job["quantizer_seed"], job["run_id"]
        local_config = {**config, "model_seed": job["model_seed"]}
        status["current_run"] = run_id
        status["current_seed"] = seed
        atomic_json(status_path, status)
        try:
            sids = np.load(output / f"rq-{seed}" / "semantic-ids.npy", allow_pickle=False)
            index = SidIndex(sids[:, :-1], width=config["quantizer"]["codebook_size"])
            np.testing.assert_array_equal(index.sids, sids)
            _, result = train_one_seed(seed, index, users, cases, windows, local_config, store, source, output,
                initial_hash_for(plan["initial_state_sha256"], job["model_seed"]), run_id=run_id)
            status["results"][run_id if config.get("model_seeds") is not None else str(seed)] = result
        except Exception as error:
            # Preserve partial evidence, attempt other assigned seeds, never
            # retry with altered LR/precision/budget or label the run complete.
            store.flush()
            failure = {"seed": seed, "run_id": run_id, "model_seed": job["model_seed"], "type": type(error).__name__, "message": str(error)}
            status["failures"].append(failure)
            (output / run_id).mkdir(exist_ok=True)
            atomic_json(output / run_id / "failure.json", failure)
            traceback.print_exc()
        atomic_json(status_path, status)
    status["status"] = "failed" if status["failures"] else "complete"
    status["current_seed"] = None
    atomic_json(status_path, status)
    return 1 if status["failures"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(run_worker(args.plan, args.slot))


if __name__ == "__main__":
    main()
