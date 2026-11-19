"""Execution topology for the registered ML1M Last-8 experiments."""
from __future__ import annotations

import torch


def colab_gpu_matches(requested, actual):
    return requested in {"T4", "L4", "A100", "H100"} and requested in actual.upper().split("-")[0].split()


def devices_for(config):
    execution = config["execution"]
    requested = execution.get("colab_gpu")
    devices = execution["worker_devices"]
    if requested:
        if (devices != ["cuda:0"] or torch.cuda.device_count() != 1
                or not colab_gpu_matches(requested, torch.cuda.get_device_name(0))):
            raise RuntimeError(f"Colab requires exactly one {requested}; no fallback")
    elif (devices != ["cuda:0", "cuda:1"] or torch.cuda.device_count() != 2
          or any(torch.cuda.get_device_capability(i) != (7, 5) for i in range(2))):
        raise RuntimeError("requires two T4s; no fallback")
    return devices


def check_worker(config, transformers_version):
    requested = config["execution"].get("colab_gpu")
    if torch.cuda.device_count() != 1 or transformers_version != "5.22.1":
        raise RuntimeError("requires one isolated GPU and Transformers 5.22.1")
    if requested:
        devices_for(config)
    elif torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("requires an isolated T4")


def assignments_for(config, devices):
    if config["execution"].get("colab_gpu"):
        return [{"slot": 0, "device": devices[0], "trials": config["trials"]}]
    from rhpr.gr.ae_suite import assignments
    return assignments(config, len(devices))
