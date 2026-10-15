"""Cost preflight and strict native-scan checks for the Query SSM pilot.

Measurements use disposable models and real successful optimizer updates. Batch
selection uses throughput alone; validation recommendation quality is discarded.
"""
from __future__ import annotations

import math
import copy
import importlib
import importlib.machinery
import importlib.metadata
import time
from typing import Any

import torch

from rhpr.sasrec.daily_training_profile import _measure_daily_rate
from rhpr.sasrec.long_quality import _cleanup, _optimizer, _scaler, _sync
from rhpr.sasrec.long_training_profile import _train_shape


def _comparison(reference: torch.Tensor, candidate: torch.Tensor, *,
                name: str, atol: float, rtol: float) -> dict[str, Any]:
    """Materialize only tiny audit tensors on CPU and fail on any mismatch."""
    expected, observed = reference.detach().float().cpu(), candidate.detach().float().cpu()
    if expected.shape != observed.shape or not torch.isfinite(expected).all() or not torch.isfinite(observed).all():
        raise RuntimeError(f"native Query SSM audit: {name} has incompatible shapes or nonfinite values")
    difference = (expected - observed).abs()
    maximum = float(difference.max()) if difference.numel() else 0.0
    if not torch.allclose(expected, observed, atol=atol, rtol=rtol):
        raise RuntimeError(f"native Query SSM audit: {name} differs from CPU reference (max abs error {maximum:.6g})")
    return {"name": name, "passed": True, "max_absolute_error": maximum,
            "reference_norm": float(expected.norm()), "native_norm": float(observed.norm()),
            "atol": atol, "rtol": rtol}


def _critical_gradient_norms(model: torch.nn.Module, inputs: torch.Tensor,
                             old_lengths: tuple[int, ...], query_size: int) -> dict[str, float]:
    parameters = dict(model.named_parameters())
    names = ("A_log", "write_projection.weight", "delta_projection.weight",
             "delta_projection.bias", "query_gate.weight", "query_generator.queries",
             "read_projection.weight")
    norms = {}
    for name in names:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"native Query SSM audit: missing or nonfinite {name} gradient")
        norms[name] = float(gradient.detach().float().norm())
    if inputs.grad is None or not torch.isfinite(inputs.grad).all():
        raise RuntimeError("native Query SSM audit: missing or nonfinite input gradient")
    norms["old_input"] = float(torch.cat([
        inputs.grad[index, :old].flatten() for index, old in enumerate(old_lengths)]).float().norm())
    norms["query_context_input"] = float(torch.cat([
        inputs.grad[index, old:old + query_size].flatten()
        for index, old in enumerate(old_lengths)]).float().norm())
    if any(not math.isfinite(norm) or norm <= 0 for norm in norms.values()):
        raise RuntimeError("native Query SSM audit: a critical recurrence/query/input gradient is zero or nonfinite")
    return norms


def audit_native_query_scan(device: torch.device) -> dict[str, Any]:
    """Check actual CUDA output/backward against CPU recurrence before data load.

    Both write-conditioning variants are audited end to end. The loss touches
    only summary outputs, so nonzero old-history gradients cannot come from the
    untouched raw tail. Native failure aborts; GPU reference fallback is forbidden.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native Query SSM audit requires an actual CUDA device")
    try:
        extension = importlib.import_module("selective_scan_cuda")
        interface = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("native Query SSM audit cannot load the Mamba CUDA extension; no GPU fallback is allowed") from error
    native_path = str(getattr(extension, "__file__", ""))
    if (not any(native_path.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
            or not callable(getattr(extension, "fwd", None))
            or not callable(getattr(extension, "bwd", None))
            or not callable(getattr(interface, "selective_scan_fn", None))):
        raise RuntimeError("native Query SSM audit requires a compiled selective_scan_cuda extension with fwd/bwd")
    from rhpr.sasrec.query_ssm import QuerySSMCompressor
    try:
        version = importlib.metadata.version("mamba-ssm")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"

    rows = []
    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    # Preserve the run's model/window RNG. All test data originate from this
    # forked CPU generator; CUDA models are exact copies of CPU parameters.
    with torch.random.fork_rng(devices=[cuda_index]):
        torch.random.default_generator.manual_seed(918273)
        for conditioning in ("recent", "slot_only"):
            reference = QuerySSMCompressor(d_model=8, summary_tokens=2, tail_size=6,
                query_context_size=4, n_heads=2, d_state=4, d_inner=8, gate_rank=4,
                write_conditioning=conditioning, dropout=0.0)
            native = copy.deepcopy(reference).to(device)
            values = torch.randn(2, 13, 8)
            mask = torch.arange(13)[None, :] < torch.tensor([13, 11])[:, None]
            values = values.masked_fill(~mask[:, :, None], 0)
            cpu_input = values.clone().requires_grad_(True)
            gpu_input = values.to(device).requires_grad_(True)
            weight = torch.randn(2, 2, 8)
            expected = reference(cpu_input, mask)
            observed = native(gpu_input, mask.to(device))
            if reference.last_scan_backend != "cpu_reference" or native.last_scan_backend != "mamba_native_cuda":
                raise RuntimeError("native Query SSM audit executed an unexpected scan backend")
            if not torch.equal(expected[1], observed[1].cpu()) or not torch.equal(expected[2], observed[2].cpu()):
                raise RuntimeError("native Query SSM audit: packed masks/source positions differ")
            checks = [_comparison(expected[0], observed[0], name="packed_output", atol=1e-4, rtol=2e-3)]
            ((expected[0][:, :2].float() * weight).sum()).backward()
            ((observed[0][:, :2].float() * weight.to(device)).sum()).backward()
            checks.append(_comparison(cpu_input.grad, gpu_input.grad, name="input_gradient", atol=2e-4, rtol=3e-3))
            native_parameters = dict(native.named_parameters())
            for name, parameter in reference.named_parameters():
                gradient = native_parameters[name].grad
                if parameter.grad is None or gradient is None:
                    raise RuntimeError(f"native Query SSM audit: disconnected parameter {name}")
                checks.append(_comparison(parameter.grad, gradient,
                    name=f"parameter_gradient:{name}", atol=2e-4, rtol=3e-3))
            critical = _critical_gradient_norms(native, gpu_input, (7, 5), 4)

            # The real run uses autocast FP16. Check its actual native forward
            # and backward as well, against the same FP32 native model/output.
            half_model = copy.deepcopy(reference).to(device)
            half_model.zero_grad(set_to_none=True)
            half_input = values.to(device).requires_grad_(True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                half_output = half_model(half_input, mask.to(device))[0]
            ((half_output[:, :2].float() * weight.to(device)).sum()).backward()
            if half_model.last_scan_backend != "mamba_native_cuda":
                raise RuntimeError("native Query SSM audit: FP16 executed an unexpected backend")
            half_check = _comparison(observed[0], half_output, name="autocast_fp16_output", atol=2e-2, rtol=3e-2)
            for name, parameter in half_model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"native Query SSM audit: missing/nonfinite FP16 gradient {name}")
            half_critical = _critical_gradient_norms(half_model, half_input, (7, 5), 4)
            rows.append({"write_conditioning": conditioning, "passed": True,
                         "native_backend": native.last_scan_backend,
                         "reference_backend": reference.last_scan_backend,
                         "checks": checks, "critical_native_gradient_norms": critical,
                         "autocast_fp16": {"output_check": half_check,
                             "all_parameter_gradients_finite": True,
                             "critical_gradient_norms": half_critical}})
            del reference, native, half_model, cpu_input, gpu_input, half_input
    _sync(device)
    _cleanup()
    return {"passed": True, "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda),
            "mamba_ssm_version": version, "native_extension_path": native_path,
            "native_extension_fwd_bwd_verified": True, "python_gpu_fallback": False,
            "summary_uses_differentiable_scan_output": True,
            "tiny_shape": {"batch_size": 2, "d_model": 8, "summary_tokens": 2,
                "old_lengths": [7, 5], "raw_tail_size": 6, "query_context_size": 4,
                "d_state": 4, "d_inner": 8, "gate_rank": 4}, "variants": rows}


def _build_query_model(config: dict[str, Any], variant: dict[str, Any], num_items: int):
    # The runner imports this module, so defer the inverse dependency until use.
    from rhpr.sasrec.query_pilot import build_query_model
    return build_query_model(config, variant, num_items)


def _positive_rate(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _choose_training_batch(rows: list[dict[str, Any]]) -> tuple[int, float]:
    usable = [row for row in rows if row["status"] == "ok"
              and row.get("measured_successful_updates", 0) > 0
              and row.get("amp_skipped_updates", 0) == 0
              and _positive_rate(row.get("train_windows_per_second_with_prepare_h2d_sync"))]
    if not usable:
        raise RuntimeError("no physical batch completed a finite, successful optimizer update")
    best = max(usable, key=lambda row: row["train_windows_per_second_with_prepare_h2d_sync"])
    return best["physical_batch_size"], float(best["train_windows_per_second_with_prepare_h2d_sync"])


def profile_query_variant(config: dict[str, Any], variant: dict[str, Any], cohort: Any,
                          data: Any, schedule: tuple[Any, ...], device: torch.device) -> dict[str, Any]:
    """Measure prepare/transfer/train and lazy full-catalogue daily evaluation."""
    train, bench = config["training"], config["benchmark"]
    training_tail = train["raw_tail_size"] + train["supervised_targets"] - 1
    recent = variant["kind"] == "recent"
    training_context = training_tail if recent else train["raw_context_length"]
    serving_context = train["raw_tail_size"] if recent else train["serving_context_length"]
    if len(schedule) < train["effective_batch_size"]:
        raise ValueError("cost preflight requires one complete effective batch of registered windows")
    counts = {split: data.summary[f"{split}_anchors"] for split in ("validation", "test")}
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("daily anchor counts must be positive integers")
    profile = {**train, "tail_size": train["supervised_targets"],
               "update_repetitions": 1, "warmup_microbatches": 3}
    started = time.monotonic()
    rows = []
    for physical in train["physical_batch_sizes"]:
        model = _build_query_model(config, variant, cohort.num_items).to(device)
        model.compressor.tail_size = training_tail
        optimizer, scaler = _optimizer(model, train), _scaler(device, train)
        try:
            row = _train_shape(model, cohort, schedule, profile, training_context,
                               physical, device, optimizer, scaler)
        except torch.OutOfMemoryError:
            row = {"status": "oom", "physical_batch_size": physical}
        except FloatingPointError as error:
            row = {"status": "nonfinite", "physical_batch_size": physical, "reason": str(error)}
        finally:
            del optimizer, scaler, model
            _cleanup()
        rows.append(row)
    training_batch, training_rate = _choose_training_batch(rows)

    candidates = [batch for batch in bench["batch_sizes"]
                  if batch <= bench["evaluation_batch_size"] and batch > 1]
    if not candidates:
        raise ValueError("query cost preflight requires at least one parallel evaluation batch")
    evaluations = []
    # This model has never seen a training update, including a timing warmup.
    model = _build_query_model(config, variant, cohort.num_items).to(device)
    model.compressor.tail_size = train["raw_tail_size"]
    try:
        for batch_size in candidates:
            try:
                if device.type == "cuda":
                    _sync(device)
                    torch.cuda.reset_peak_memory_stats(device)
                row = _measure_daily_rate(model, cohort, data, context_length=serving_context,
                    batch_size=batch_size, device=device, amp=train["amp"],
                    item_chunk_size=bench["item_chunk_size"], sample_count=128)
                if device.type == "cuda":
                    _sync(device)
                    row.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
            except torch.OutOfMemoryError:
                row = {"status": "oom", "batch_size": batch_size}
                _cleanup()
            evaluations.append(row)
    finally:
        del model
        _cleanup()
    usable = [row for row in evaluations if row["status"] == "ok"
              and _positive_rate(row.get("anchors_per_second"))]
    if not usable:
        raise RuntimeError(f"{variant['name']}: no parallel daily evaluation batch succeeded")
    best = max(usable, key=lambda row: row["anchors_per_second"])
    return {**variant, "training_batch_size": training_batch,
            "evaluation_batch_size": best["batch_size"],
            "estimated_train_pass_seconds": len(cohort.users) / training_rate,
            "estimated_validation_seconds": counts["validation"] / best["anchors_per_second"],
            "estimated_test_seconds": counts["test"] / best["anchors_per_second"],
            "training": rows, "evaluation": evaluations,
            "training_context_length": training_context, "serving_context_length": serving_context,
            "training_raw_tail_size": training_tail, "serving_raw_tail_size": train["raw_tail_size"],
            "training_selection": "fastest successful finite effective update, including CPU prepare, H2D and synchronization",
            "evaluation_selection": "fastest lazy daily exact top100 throughput; no quality selection",
            "validation_anchors": counts["validation"], "test_anchors": counts["test"],
            "profile_seconds": time.monotonic() - started,
            "profile_weights_reused_for_training": False}
