"""Matched TIGER fits on byte-pinned, completed field-RQ semantic indices."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.experiment import _evaluate, _event, _write
from rhpr.gr.study_parallel import atomic_json


def assignments(config, gpu_count):
    from rhpr.gr.ae_suite import assignments as base_assignments
    jobs = base_assignments(config, gpu_count)
    if config["training"].get("sampling") != "all_train_prefixes":
        raise ValueError("frozen SID suite requires the all_train_prefixes control")
    if config["generator"]["codebook_size"] != config["quantizer"]["codebook_size"]:
        raise ValueError("generator and index widths differ")
    return jobs


def verify_file(root, record):
    """Permit pinned nested trial outputs, rejecting traversal and symlinks."""
    name = record["path"]
    relative = Path(name)
    if (not name or not relative.parts or relative.is_absolute() or ".." in relative.parts or "\\" in name):
        raise ValueError("unsafe frozen-index artifact path")
    root = Path(root).resolve();path = root / relative
    if (not path.resolve().is_relative_to(root)
            or any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1))):
        raise ValueError("unsafe frozen-index artifact path")
    if (not path.is_file() or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]):
        raise ValueError(f"frozen-index artifact digest mismatch: {name}")
    return path


def prepare_indices(config, parent, source_ids, output):
    import numpy as np
    from rhpr.gr.sid import SidIndex

    manifest = json.loads((parent / "run-manifest.json").read_text())
    provenance = json.loads((parent / "input-provenance.json").read_text())
    if (manifest["config"]["data"]["sha256"] != config["data"]["sha256"]
            or provenance["original_item_ids"] != source_ids):
        raise ValueError("parent index catalogue/order differs from generator data")
    records = {}
    for trial in config["trials"]:
        directory = output / "indices" / trial["id"];directory.mkdir(parents=True)
        if config["quantizer"]["kind"] == "ae_residual_kmeans":
            from rhpr.gr.ae_rqkmeans import prepare_ae_kmeans_index
            codes, origin = prepare_ae_kmeans_index(config, trial, parent, source_ids, directory)
            report = None
        else:
            codes_path = verify_file(parent, trial["semantic_codes"])
            report_path = verify_file(parent, trial["quantizer_report"])
            report = json.loads(report_path.read_text())
            if (report["seed"] != trial["quantizer_seed"] or report["warmup"]["freeze_policy"] != trial["freeze_policy"]
                    or report["config"]["num_hierarchies"] != config["quantizer"]["num_hierarchies"]
                    or report["config"]["codebook_size"] != config["quantizer"]["codebook_size"]
                    or not report["warmup"]["frozen_state_verified"]):
                raise ValueError("parent quantizer report differs from declared trial")
            codes = np.load(codes_path, allow_pickle=False)
            origin = {"semantic_codes": trial["semantic_codes"], "quantizer_report": trial["quantizer_report"],
                "quantizer_retrained": False}
        if codes.shape != (len(source_ids), config["quantizer"]["num_hierarchies"]):
            raise ValueError("parent code matrix does not match catalogue/levels")
        index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
        diagnostics = index.diagnostics()
        if report is not None and any(diagnostics[k] != report["semantic_code_statistics"][k]
               for k in ("semantic_unique_ids", "max_collision_group")):
            raise ValueError("parent semantic statistics disagree with saved codes")
        np.save(directory / "semantic-codes.npy", codes, allow_pickle=False)
        np.save(directory / "semantic-ids.npy", index.sids, allow_pickle=False)
        _write(directory / "sid-diagnostics.json", diagnostics)
        record = {"trial": trial["id"], "parent": config["parent"]["kernel"],
            "source_code_sha": manifest["git_sha"], "quantizer_seed": trial["quantizer_seed"],
            "freeze_policy": trial["freeze_policy"], **origin,
            "semantic_ids_sha256": file_sha256(directory / "semantic-ids.npy"), "diagnostics": diagnostics}
        _write(directory / "index-provenance.json", record);records[trial["id"]] = record
    _write(output / "sid-catalogue.json", {"original_item_ids": source_ids,
        "row_order": "internal_item_id_0_based", "indices": records})
    return records


def training_manifest(users, config):
    """No extra target sampling: record the shared original trainer's case order."""
    import numpy as np
    train = build_examples(users, config["data"]["history_items"], "train")
    digest = hashlib.sha256()
    for array in train:
        digest.update(str(array.shape).encode());digest.update(array.dtype.str.encode());digest.update(array.tobytes())
    epochs = config["training"]["max_epochs"]
    return {"sampling": "all_train_prefixes", "examples_per_epoch": len(train[2]),
        "seed": config["seed"], "train_cases_sha256": digest.hexdigest(),
        "epoch_order_sha256": {str(e): hashlib.sha256(np.random.default_rng(config["seed"]+e)
            .permutation(len(train[2])).tobytes()).hexdigest() for e in range(1, epochs+1)}}


def run_trial(trial, config, users, output, source, device, expected_initial=None):
    import numpy as np
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.rqvae_experiment import _train_rqvae_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    directory = output / "trials" / trial["id"];directory.mkdir(parents=True, exist_ok=True)
    codes = np.load(output / "indices" / trial["id"] / "semantic-codes.npy", allow_pickle=False)
    index = SidIndex(codes, width=config["quantizer"]["codebook_size"])
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    model = build_grid_generator(index.sids, config["generator"], config["seed"], device, source)
    # SID lookup buffers intentionally differ; hash the actual tied neural
    # parameters, not the immutable item-to-code lookup supplied to the model.
    initial = state_digest(dict(model.named_parameters()))
    parameters = sum(p.numel() for p in model.parameters())
    _write(directory / "generator.json", {"parameters": parameters, "config": config["generator"],
        "initial_parameter_sha256": initial, "provenance": model.rhpr_grid_provenance})
    _write(directory / "initialization.json", {"initial_state_sha256": initial,
        "scope": "named neural parameters; SID lookup data differ by design", "seed": config["seed"]})
    if expected_initial is not None and initial != expected_initial:
        raise ValueError("generator neural initialization differs from the shared control")
    selection = _train_rqvae_generator(model, users, index, config, directory, device)
    if selection["epochs"] != config["training"]["max_epochs"]:
        raise RuntimeError("generator stopped before its declared complete epoch budget")
    test = build_examples(users, config["data"]["history_items"], "test")
    scores, ids, probabilities = _evaluate(model, test, index,
        config["training"]["evaluation_batch_size"], device, precision=config["training"]["evaluation_precision"])
    _write(directory / "test-metrics.json", scores)
    np.savez_compressed(directory / "test-predictions.npz", semantic_ids=ids, scores=probabilities,
        user_ids=test[1], target_item_ids=test[2])
    result = {"trial": trial["id"], "seed": config["seed"], "quantizer_seed": trial["quantizer_seed"],
        "freeze_policy": trial["freeze_policy"], "initial_state_sha256": initial, "parameters": parameters,
        "selection": selection, "test": scores,
        "quantizer_retrained": config["quantizer"]["kind"] == "ae_residual_kmeans", "test_evaluations": 1}
    _write(directory / "result.json", result)
    return result


def run_worker(plan_path, slot):
    import torch
    plan = json.loads(Path(plan_path).read_text());config = plan["config"]
    assignment = plan["assignments"][slot];output = Path(plan["output"])
    torch.set_num_threads(config["execution"]["cpu_threads_per_worker"])
    device = "cpu"
    if assignment["device"] != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("generator worker must see exactly its assigned GPU")
        torch.cuda.set_device(0);device = "cuda:0"
    _, users, _, _ = load_beauty(Path(plan["resources"]) / "data", config["data"])
    status = {"slot": slot, "device": assignment["device"], "status": "running", "results": {}, "failures": [],
        "hardware": torch.cuda.get_device_name(0) if device != "cpu" else "cpu"}
    status_path = output / "workers" / f"worker-{slot}.json";atomic_json(status_path, status)
    for trial in assignment["trials"]:
        name = trial["id"];status["current_trial"] = name;atomic_json(status_path, status)
        _event("generator_suite_trial_start", trial=name, stage="frozen_sid_generator", worker=slot)
        try:
            result = run_trial(trial, config, users, output, plan["source"], device, plan.get("initial_state_sha256"))
            status["results"][name] = result
            _event("frozen_sid_generator_complete", trial=name, test=result["test"])
        except Exception as error:
            directory = output / "trials" / name;directory.mkdir(parents=True, exist_ok=True)
            (directory / "failure.log").write_text(traceback.format_exc())
            failure = {"trial": name, "exception_type": type(error).__name__, "message": str(error)}
            status["failures"].append(failure);_event("frozen_sid_generator_failed", **failure)
        atomic_json(status_path, status);gc.collect()
        if device != "cpu":torch.cuda.empty_cache()
    status["status"] = "complete";atomic_json(status_path, status)


def run_generator_suite(config, output):
    import torch
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.resources import verify_resources, verify_parent_artifacts, unpack_grid
    from rhpr.gr.grid import prepare_grid_source, build_grid_generator
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.sid import SidIndex
    import numpy as np
    from rhpr.offline_runtime import resolve_dataset_directory

    output = Path(output).resolve();started = time.monotonic()
    jobs = assignments(config, torch.cuda.device_count())
    resources = resolve_dataset_directory(config["resources"]["path"])
    resource_manifest = verify_resources(resources, config["resources"]["manifest_sha256"])
    parent = verify_parent_artifacts(config["parent"])
    _, users, source_ids, counts = load_beauty(resources / "data", config["data"])
    indices = prepare_indices(config, parent, source_ids, output)
    stream = training_manifest(users, config)
    if stream["examples_per_epoch"] != counts["train_examples"]:
        raise ValueError("training manifest disagrees with original data counts")
    if config.get("expected_train_cases_sha256", stream["train_cases_sha256"]) != stream["train_cases_sha256"]:
        raise ValueError("training cases differ from completed TIGER control")
    _write(output / "shared-training-stream.json", stream)
    scratch = Path(config["resources"]["scratch_path"])
    original = unpack_grid(resources / "grid-source.bin", scratch / "original")
    source = prepare_grid_source(original, scratch / "runtime")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    first = config["trials"][0]["id"]
    index = SidIndex(np.load(output / "indices" / first / "semantic-codes.npy", allow_pickle=False),
        width=config["quantizer"]["codebook_size"])
    control = build_grid_generator(index.sids, config["generator"], config["seed"], "cpu", source)
    initial = state_digest(dict(control.named_parameters()))
    if config.get("expected_initial_state_sha256", initial) != initial:
        raise ValueError("generator neural parameters differ from completed TIGER control")
    _write(output / "initialization.json", {"initial_state_sha256": initial,
        "scope": "shared CPU neural-parameter control; fixed SID lookup data excluded", "seed": config["seed"]})
    del control;gc.collect()
    (output / "workers").mkdir()
    _write(output / "input-provenance.json", {"manifest": resource_manifest, "counts": counts,
        "parent": config["parent"], "indices": indices, "telemetry_enabled": False,
        "training_cases": stream, "generator_seed": config["seed"]})
    if config.get("controls"):
        _write(output / "controls.json", config["controls"])
    plan = {"config": config, "assignments": jobs, "output": str(output), "resources": str(resources), "source": str(source),
        "initial_state_sha256": initial}
    plan_path = output / "workers/plan.json";atomic_json(plan_path, plan)
    summary = launch_workers(plan_path, worker_module="rhpr.gr.generator_suite", event_prefix="generator_suite")
    metrics = {"runs": len(summary["results"]), "parallel_workers": 2, "generator_seed": config["seed"],
        "quantizer_retrained": config["quantizer"]["kind"] == "ae_residual_kmeans",
        "autoencoder_retrained": False, "telemetry_enabled": False, "catalogue_items": len(source_ids),
        "elapsed_seconds": time.monotonic()-started}
    for name, result in summary["results"].items():
        metrics[name + "_epochs"] = result["selection"]["epochs"]
        metrics[name + "_selected_epoch"] = result["selection"]["selected_epoch"]
        metrics[name + "_training_seconds"] = result["selection"]["training_seconds"]
        metrics[name + "_validation_seconds"] = result["selection"]["validation_seconds"]
        metrics[name + "_test_evaluations"] = result["test_evaluations"]
        for key, value in result["test"].items():metrics[name + "_test_" + key.replace("@", "")] = value
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser();parser.add_argument("--plan", required=True);parser.add_argument("--slot", type=int, required=True)
    args = parser.parse_args();run_worker(args.plan, args.slot)
