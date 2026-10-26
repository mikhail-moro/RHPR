"""Independent field AE/RQ hypothesis workers; one isolated process per GPU."""
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
    jobs = [{"slot": i, "device": d, "trials": trials[i::2]} for i,d in enumerate(devices)]
    if config.get("warmup"):
        seeds = []
        for job in jobs:
            values = {t.get("seed") for t in job["trials"]}
            if len(values) != 1 or any(type(s) is not int or s < 0 for s in values):
                raise ValueError("staged worker requires one explicit pipeline seed")
            seeds.append(next(iter(values)))
            if sorted(t.get("freeze_policy", "") for t in job["trials"]) != ["ae", "embeddings", "joint"]:
                raise ValueError("staged seed needs joint, ae and embeddings freeze branches")
            if any(t["loss_policy"] != config["warmup"]["loss_policy"] for t in job["trials"]):
                raise ValueError("warm-up and RQ objectives must match")
        if len(set(seeds)) != 2 or config["quantizer"]["kind"] != "grid_field_rqvae":
            raise ValueError("staged suite requires two pipeline seeds and native RQ")
    return jobs


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
    warmup_checkpoint = None
    if config.get("warmup"):
        seed = assignment["trials"][0]["seed"]
        directory = root / "warmup" / f"seed{seed}"
        try:
            _event("ae_suite_warmup_start", worker=slot, seed=seed)
            warm_spec = {**config["quantizer"], **config["warmup"], "kind": "continuous_field_autoencoder"}
            _, report, _ = fit_field_autoencoder(texts, config["representation"], warm_spec,
                seed, device, plan["source"], plan["recipe"], directory,
                title_embeddings=titles, title_observed=observed)
            warmup_checkpoint = directory / "autoencoder.pt"
            status["warmup"] = report
            atomic_json(status_path, status)
            _event("ae_suite_warmup_complete", worker=slot, seed=seed, steps=report["steps"], stopping=report["stopping"])
            gc.collect()
            if device != "cpu":torch.cuda.empty_cache()
        except Exception as error:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "failure.log").write_text(traceback.format_exc())
            status["failures"].append({"stage": "warmup", "seed": seed,
                "exception_type": type(error).__name__, "message": str(error),
                "dependent_trials": [t["id"] for t in assignment["trials"]]})
            status["status"] = "failed";atomic_json(status_path, status)
            _event("ae_suite_warmup_failed", **status["failures"][-1])
            return
    for trial in assignment["trials"]:
        name = trial["id"];status["current_trial"] = name;atomic_json(status_path, status)
        _event("ae_suite_trial_start", trial=name, worker=slot, loss_policy=trial["loss_policy"])
        directory = root / "trials" / name;directory.mkdir(parents=True, exist_ok=True)
        spec = {**config["quantizer"], "loss_policy": trial["loss_policy"]}
        if warmup_checkpoint is not None:
            spec["freeze_policy"] = trial["freeze_policy"]
        if config.get("expected_initial_state_sha256"):
            spec["expected_initial_state_sha256"] = config["expected_initial_state_sha256"]
        try:
            fit = fit_field_autoencoder
            if spec["kind"] == "grid_field_rqvae":
                from rhpr.gr.field_rqvae import fit_field_rqvae
                fit = fit_field_rqvae
            kwargs = {"title_embeddings": titles, "title_observed": observed}
            if warmup_checkpoint is not None:kwargs["warmup_checkpoint"] = warmup_checkpoint
            seed = trial.get("seed", config["seed"])
            _, report, _ = fit(texts, config["representation"], spec,
                seed, device, plan["source"], plan["recipe"], directory, **kwargs)
            status["results"][name] = report
            atomic_json(directory / "trial-manifest.json", {"status": "complete", "trial": trial,
                "seed": seed, "device": assignment["device"], "code_sha": os.environ.get("GITHUB_SHA", "local"),
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


def launch_workers(plan_path, *, worker_module="rhpr.gr.ae_suite", event_prefix="ae_suite"):
    plan_path = Path(plan_path).resolve();plan = json.loads(plan_path.read_text())
    config, output = plan["config"], Path(plan["output"])
    processes, streams, readers, current = [], [], [], {}
    try:
        for a in plan["assignments"]:
            path = output / "workers" / f"worker-{a['slot']}.log"
            stream = path.open("w");streams.append(stream)
            p = subprocess.Popen([sys.executable, "-u", "-m", worker_module, "--plan", str(plan_path),
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
                    if event.get("phase") == event_prefix + "_trial_start":current[slot] = event["trial"]
                    _event(event_prefix + "_worker_event", worker=slot, trial=current.get(slot), event=event)
            if not alive:break
            time.sleep(.5)
        results, failures, warmups = {}, [], {}
        for a,p in zip(plan["assignments"], processes):
            status_path = output / "workers" / f"worker-{a['slot']}.json"
            if status_path.exists():
                status = json.loads(status_path.read_text());results.update(status["results"]);failures.extend(status["failures"])
                if status.get("warmup"):warmups[str(status["warmup"]["seed"])] = status["warmup"]
            if p.returncode != 0:failures.append({"worker": a["slot"], "exit_code": p.returncode})
        by_seed = config.get("warmup") or config.get("compare_initialization_by_seed")
        def initialization_group(row):
            if config.get("compare_initialization_by_shape"):
                return row["initialization_group"]
            return str(row["seed"]) if by_seed else "all"
        groups = {initialization_group(r) for r in results.values()}
        for group in groups:
            rows = [r for r in results.values() if initialization_group(r) == group]
            for key in ("initial_state_sha256", "initial_full_state_sha256"):
                if len({r[key] for r in rows if key in r}) > 1:
                    failures.append({"message": f"{key} differs between trials in group {group}"})
            if config.get("warmup") and len({r["warmup"]["checkpoint_sha256"] for r in rows}) > 1:
                failures.append({"message": f"warm-up checkpoints differ in group {group}"})
        summary = {"results": results, "failures": failures, "parallel_workers": len(processes), "warmups": warmups}
        atomic_json(output / "suite-results.json", summary)
        if failures or len(results) != len(config["trials"]):
            raise RuntimeError(f"{event_prefix} incomplete; see suite-results.json and worker logs")
        return summary
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for stream in streams + readers:stream.close()


def run_autoencoder_suite(config, output):
    """Shared field suite entry point; historical name also supports RQ."""
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
        "original_item_ids": ids, "initialization": "one shared continuous AE per pipeline seed; identical RQ initialization within seed"
            if config.get("warmup") else f"same seed{config['seed']} and source constructor per independent trial"})
    plan = {"config": config, "assignments": plan_assignments, "output": str(output), "resources": str(resources),
        "source": str(source), "recipe": str(original / "configs/experiment/rqvae_train_flat.yaml"),
        "parent_fields": str(parent / "catalogue-fields.npz")}
    plan_path = output / "workers/plan.json";atomic_json(plan_path, plan)
    if config.get("ae_controls"):
        _write(output / "ae-controls.json", config["ae_controls"])
    if config.get("rq64_controls"):
        _write(output / "rq64-controls.json", config["rq64_controls"])
    _event("ae_suite_start", assignments=plan_assignments)
    summary = launch_workers(plan_path)
    quantized = config["quantizer"]["kind"] == "grid_field_rqvae"
    expected_initial = config.get("expected_initial_state_sha256")
    if expected_initial and any(r["initial_state_sha256"] != expected_initial for r in summary["results"].values()):
        raise ValueError("initial neural weights differ from the completed AE controls")
    metrics = {"trials": len(summary["results"]), "parallel_workers": 2, "elapsed_seconds": time.monotonic()-started,
        "catalogue_items": len(texts),
        "quantization_enabled": quantized, "generator_trained": False, "test_evaluations": 0}
    for name,r in summary["results"].items():
        metrics[name + "_steps"] = r["steps"]
        for field, values in r["field_reconstruction"].items():
            metrics[name + "_" + field + "_loss"] = values["loss"]
        if quantized:
            metrics[name + "_fit_seconds"] = r["fit_seconds"]
            metrics[name + "_reconstruction_updates"] = r["reconstruction_updates"]
            metrics[name + "_initialization_only_batches"] = r["initialization_only_batches"]
            for key in ("semantic_unique_ids", "max_collision_group", "suffix_representable"):
                metrics[name + "_" + key] = r["semantic_code_statistics"][key]
            metrics[name + "_collided_items"] = r["semantic_code_statistics"]["prefixes"][-1]["collided_items"]
            for level, used in enumerate(r["used_codes_per_level"], 1):
                metrics[name + "_used_codes_level_" + str(level)] = used
    for seed,r in summary.get("warmups", {}).items():
        metrics["warmup_seed" + seed + "_steps"] = r["steps"]
        metrics["warmup_seed" + seed + "_seconds"] = r["elapsed_seconds"]
        metrics["warmup_seed" + seed + "_spectral_plateau"] = r["stopping"]["reason"] == "spectral_plateau"
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--plan", required=True);parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args();run_worker(args.plan, args.slot)
