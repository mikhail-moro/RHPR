"""Independent AE hypothesis workers; one isolated process per GPU."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from rhpr.gr.experiment import _event, _write
from rhpr.gr.study_parallel import atomic_json, worker_environment


def assignments(config, available_gpus):
    devices = config["execution"]["worker_devices"]
    trials = config["trials"]
    ids = [t["id"] for t in trials]
    if len(devices) != 2 or len(trials) < 2:
        raise ValueError("AE suite requires two distinct workers and at least two trials")
    if type(config["execution"].get("cpu_threads_per_worker")) is not int or config["execution"]["cpu_threads_per_worker"] < 1:
        raise ValueError("cpu_threads_per_worker must be positive")
    if any(not name or not name.replace("_", "").isalnum() for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("trial IDs must be unique safe names")
    if config["training"]["device"] == "cuda":
        if devices != ["cuda:0", "cuda:1"] or available_gpus < 2:
            raise RuntimeError("AE suite requires two CUDA GPUs; no hardware fallback")
    elif devices != ["cpu", "cpu"]:
        raise ValueError("CPU verification needs two explicit CPU workers")
    return [{"slot": i, "device": d, "trials": trials[i::2]} for i,d in enumerate(devices)]


def run_worker(plan_path, slot):
    import numpy as np
    import torch
    from rhpr.gr.data import load_beauty
    from rhpr.gr.field_autoencoder import fit_field_autoencoder

    plan = json.loads(Path(plan_path).read_text())
    config, assignment = plan["config"], plan["assignments"][slot]
    torch.set_num_threads(config["execution"]["cpu_threads_per_worker"])
    device = "cpu"
    if assignment["device"] != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("AE worker must see exactly its assigned GPU")
        device = "cuda:0";torch.cuda.set_device(0)
    texts, _, _, _ = load_beauty(Path(plan["resources"]) / "data", config["data"])
    with np.load(plan["parent_fields"], allow_pickle=False) as source:
        titles, observed = source["title"], source["title_observed"]
    root = Path(plan["output"])
    status_path = root / "workers" / f"worker-{slot}.json"
    status = {"slot": slot, "device": assignment["device"], "status": "running", "results": {}, "failures": [],
        "hardware": torch.cuda.get_device_name(0) if device != "cpu" else "cpu"}
    atomic_json(status_path, status)
    for trial in assignment["trials"]:
        name = trial["id"];status["current_trial"] = name;atomic_json(status_path, status)
        _event("ae_suite_trial_start", trial=name, worker=slot, loss_policy=trial["loss_policy"])
        directory = root / "trials" / name;directory.mkdir(parents=True, exist_ok=True)
        spec = {**config["quantizer"], "loss_policy": trial["loss_policy"]}
        try:
            _, report, _ = fit_field_autoencoder(texts, config["representation"], spec,
                config["seed"], device, plan["source"], plan["recipe"], directory,
                title_embeddings=titles, title_observed=observed)
            status["results"][name] = report
            atomic_json(directory / "trial-manifest.json", {"status": "complete", "trial": trial,
                "seed": config["seed"], "device": assignment["device"], "code_sha": os.environ.get("GITHUB_SHA", "local"),
                "initial_state_sha256": report["initial_state_sha256"]})
            _event("ae_suite_trial_complete", trial=name, worker=slot, fields=report["field_reconstruction"])
        except Exception as error:
            failure = {"trial": name, "exception_type": type(error).__name__, "message": str(error)}
            (directory / "failure.log").write_text(traceback.format_exc())
            atomic_json(directory / "trial-manifest.json", {"status": "failed", **failure})
            status["failures"].append(failure);_event("ae_suite_trial_failed", **failure)
        atomic_json(status_path, status)
        gc.collect()
        if device != "cpu":torch.cuda.empty_cache()
    status["status"] = "complete";atomic_json(status_path, status)


def launch_workers(plan_path):
    plan_path = Path(plan_path).resolve();plan = json.loads(plan_path.read_text())
    config, output = plan["config"], Path(plan["output"])
    processes, streams, readers, current = [], [], [], {}
    try:
        for a in plan["assignments"]:
            path = output / "workers" / f"worker-{a['slot']}.log"
            stream = path.open("w");streams.append(stream)
            p = subprocess.Popen([sys.executable, "-u", "-m", "rhpr.gr.ae_suite", "--plan", str(plan_path),
                "--slot", str(a["slot"])], stdout=stream, stderr=subprocess.STDOUT,
                env=worker_environment(config, a))
            processes.append(p);readers.append(path.open())
        while True:
            alive = any(p.poll() is None for p in processes)
            for slot, reader in enumerate(readers):
                while True:
                    position = reader.tell();line = reader.readline()
                    if not line:break
                    if not line.endswith("\n"):reader.seek(position);break
                    try:event = json.loads(line)
                    except ValueError:continue
                    if not isinstance(event, dict):continue
                    if event.get("phase") == "ae_suite_trial_start":current[slot] = event["trial"]
                    _event("ae_suite_worker_event", worker=slot, trial=current.get(slot), event=event)
            if not alive:break
            time.sleep(.5)
        results, failures = {}, []
        for a,p in zip(plan["assignments"], processes):
            status_path = output / "workers" / f"worker-{a['slot']}.json"
            if status_path.exists():
                status = json.loads(status_path.read_text());results.update(status["results"]);failures.extend(status["failures"])
            if p.returncode != 0:failures.append({"worker": a["slot"], "exit_code": p.returncode})
        hashes = {r["initial_state_sha256"] for r in results.values()}
        if len(hashes) > 1:failures.append({"message": "initial neural weights differ between trials"})
        summary = {"results": results, "failures": failures, "parallel_workers": len(processes)}
        atomic_json(output / "suite-results.json", summary)
        if failures or len(results) != len(config["trials"]):
            raise RuntimeError("AE suite incomplete; see suite-results.json and worker logs")
        return summary
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for stream in streams + readers:stream.close()


def run_autoencoder_suite(config, output):
    import numpy as np
    import torch
    from rhpr.gr.data import load_beauty
    from rhpr.gr.resources import verify_resources, verify_parent_artifacts, unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.offline_runtime import resolve_dataset_directory

    output = Path(output).resolve();started = time.monotonic()
    plan_assignments = assignments(config, torch.cuda.device_count())
    resources = resolve_dataset_directory(config["resources"]["path"])
    manifest = verify_resources(resources, config["resources"]["manifest_sha256"])
    parent = verify_parent_artifacts(config["parent"])
    texts, _, ids, counts = load_beauty(resources / "data", config["data"])
    with np.load(parent / "catalogue-fields.npz", allow_pickle=False) as source:
        if source["title"].shape != (len(texts), 768) or not np.isfinite(source["title"]).all():
            raise ValueError("parent title targets do not match catalogue")
    scratch = Path(config["resources"]["scratch_path"])
    original = unpack_grid(resources / "grid-source.bin", scratch / "original")
    source = prepare_grid_source(original, scratch / "runtime")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    (output / "workers").mkdir()
    _write(output / "input-provenance.json", {"manifest": manifest, "counts": counts, "parent": config["parent"],
        "original_item_ids": ids, "initialization": f"same seed{config['seed']} and source constructor per independent trial"})
    plan = {"config": config, "assignments": plan_assignments, "output": str(output), "resources": str(resources),
        "source": str(source), "recipe": str(original / "configs/experiment/rqvae_train_flat.yaml"),
        "parent_fields": str(parent / "catalogue-fields.npz")}
    plan_path = output / "workers/plan.json";atomic_json(plan_path, plan)
    _event("ae_suite_start", assignments=plan_assignments)
    summary = launch_workers(plan_path)
    metrics = {"trials": len(summary["results"]), "parallel_workers": 2, "elapsed_seconds": time.monotonic()-started,
        "quantization_enabled": False, "generator_trained": False, "test_evaluations": 0}
    for name,r in summary["results"].items():
        metrics[name + "_steps"] = r["steps"]
        for field, values in r["field_reconstruction"].items():
            metrics[name + "_" + field + "_loss"] = values["loss"]
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--plan", required=True);parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args();run_worker(args.plan, args.slot)
