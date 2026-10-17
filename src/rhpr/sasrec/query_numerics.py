"""Failure-only forward replay; diagnostics never change training or selection.

Hooks localize the first observed nonfinite module boundary, not the precise
functional/CUDA operation inside it. No raw events, labels or sampled IDs are
serialized. The FP32 replay is evidence, never a fallback training step.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from rhpr.sasrec.long_training_profile import _loss
from rhpr.sasrec.query_lr_restart import _preserve_rng


def capture_training_rng(device: torch.device) -> dict[str, torch.Tensor | None]:
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _tensors(value: Any, path: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _tensors(child, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensors(child, f"{path}.{key}")


def _stats(value: torch.Tensor) -> dict[str, Any]:
    """Bound temporary storage even for the large tied item embedding table."""
    x = value.detach().reshape(-1)
    result: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype),
        "numel": x.numel(), "nan": 0, "posinf": 0, "neginf": 0,
        "finite_min": None, "finite_max": None}
    for chunk in x.split(1_048_576):
        finite = torch.isfinite(chunk)
        if chunk.is_floating_point():
            result["nan"] += int(torch.isnan(chunk).sum())
            result["posinf"] += int(torch.isposinf(chunk).sum())
            result["neginf"] += int(torch.isneginf(chunk).sum())
        selected = chunk[finite]
        if selected.numel():
            lo, hi = float(selected.min()), float(selected.max())
            result["finite_min"] = lo if result["finite_min"] is None else min(lo, result["finite_min"])
            result["finite_max"] = hi if result["finite_max"] is None else max(hi, result["finite_max"])
    result["all_finite"] = result["nan"] + result["posinf"] + result["neginf"] == 0
    return result


def _replay(model: torch.nn.Module, batch: dict[str, torch.Tensor], *,
            device: torch.device, amp: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"autocast_enabled": amp and device.type == "cuda",
        "autocast_dtype": "torch.float16", "first_nonfinite": None,
        "first_nonfinite_input": None, "first_nonfinite_output": None}
    handles = []

    def inspect(name: str, boundary: str, value: Any) -> None:
        key = f"first_nonfinite_{boundary}"
        if result[key] is not None:
            return
        for path, tensor in _tensors(value):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                record = {"module": name, "boundary": boundary, "tensor_path": path, **_stats(tensor)}
                result[key] = record
                if result["first_nonfinite"] is None:
                    result["first_nonfinite"] = record
                return

    try:
        for name, module in model.named_modules():
            handles.append(module.register_forward_pre_hook(
                lambda _module, inputs, name=name: inspect(name or "<root>", "input", inputs)))
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, output, name=name: inspect(name or "<root>", "output", output)))
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=result["autocast_enabled"]):
                encoded = model.encode(batch["item_ids"], batch["valid_mask"])
            result["encoded"] = [{"tensor_path": path, **_stats(tensor)}
                                 for path, tensor in _tensors(encoded) if tensor.is_floating_point()]
            loss = _loss(model, encoded, batch, model.compressor.tail_size, batch["negative_ids"].shape[-1])
            result["loss_finite"] = bool(torch.isfinite(loss))
            result["loss"] = float(loss) if result["loss_finite"] else None
    except Exception as error:
        # Exception text can contain tensor values/IDs; retain only its class.
        result["error_type"] = type(error).__name__
    finally:
        for handle in handles:
            handle.remove()
    return result


def diagnose_nonfinite_batch(model: torch.nn.Module, batch_arrays: dict[str, np.ndarray],
                             device: torch.device, amp: bool,
                             rng_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay the failed batch with unchanged weights, mode, gradients and RNG.

Call immediately after a forward failure and before any optimizer update.
FP32 means disabled autocast with the original FP32 master parameters; custom
operators retain their own explicitly chosen precision.
"""
    report: dict[str, Any] = {"diagnostic_only": True, "model_training": model.training,
        "rng_reproduces_failed_forward": rng_state is not None,
        "input_shapes": {key: list(value.shape) for key, value in batch_arrays.items()}}
    modes = [(module, module.training) for module in model.modules()]
    buffers = []
    try:
        buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
        with _preserve_rng(device):
            state = capture_training_rng(device) if rng_state is None else rng_state
            batch = {key: torch.from_numpy(value).to(device) for key, value in batch_arrays.items()}
            report["parameters"] = {name: _stats(parameter) for name, parameter in model.named_parameters()}
            report["parameters_finite"] = all(row["all_finite"] for row in report["parameters"].values())
            for name, enabled in (("training_precision", amp), ("fp32", False)):
                _restore_rng(state, device)
                with torch.no_grad():
                    for buffer, original in buffers:
                        buffer.copy_(original)
                report[name] = _replay(model, batch, device=device, amp=enabled)
    except Exception as error:
        report["error_type"] = type(error).__name__
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for module, training in modes:
            module.training = training
    return report
