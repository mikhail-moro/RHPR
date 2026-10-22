"""Bounded profiling of the existing GR checkpoint, without quality training.

Training measurements restore the same model, optimizer and RNG state between
cases. Larger *effective* batches are labelled explicitly. Validation-only
precision comparisons never select a model or inspect held-out test targets.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import json
import math
from pathlib import Path
import time

from rhpr.gr.data import UserSequence, build_examples, file_sha256


class ProfileBudgetExhausted(RuntimeError):
    pass


class ProfileDeadline:
    def __init__(self, seconds: float, clock=time.monotonic):
        if not math.isfinite(seconds) or not 0 < seconds <= 600:
            raise ValueError("profile.max_seconds must be in (0,600]")
        self.clock, self.started, self.seconds = clock, clock(), seconds

    def check(self) -> None:
        if self.clock() - self.started >= self.seconds:
            raise ProfileBudgetExhausted("registered profile time budget exhausted")

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_profile(config: dict) -> dict:
    specification = dict(config["profile"])
    for name, default in (("warmup", 2), ("iterations", 5), ("validation_users", 512)):
        specification[name] = _positive_int(specification.get(name, default), f"profile.{name}")
    for name, default in (("train_batch_sizes", [128, 256, 512, 1024]),
                          ("evaluation_batch_sizes", [32, 64, 128, 256])):
        values = specification.get(name, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"profile.{name} must be a nonempty list of distinct batch sizes")
        values = [_positive_int(value, name) for value in values]
        if len(set(values)) != len(values):
            raise ValueError(f"profile.{name} must be distinct")
        specification[name] = values
    precisions = specification.get("precisions", ["fp32", "fp16"])
    if (not isinstance(precisions, list) or not precisions
            or any(not isinstance(value, str) for value in precisions)
            or len(set(precisions)) != len(precisions)
            or any(value not in {"fp32", "fp16", "bf16"} for value in precisions)):
        raise ValueError("profile.precisions must contain distinct fp32/fp16/bf16 values")
    # Always establish the FP32 validation reference before reduced precision.
    specification["precisions"] = ["fp32", *[x for x in precisions if x != "fp32"]]
    seconds = float(specification.get("max_seconds", 480))
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("profile.max_seconds must be in (0,600]")
    specification["max_seconds"] = seconds
    if config.get("evaluation", {}).get("test_after_selection", False):
        raise ValueError("profiling cannot enable held-out test evaluation")
    if specification.get("architectures"):
        if (len(specification["architectures"]) != 1
                or specification["architectures"][0].get("use_parent_checkpoint") is not True):
            raise ValueError("this profile compares the single existing parent architecture only")
    return specification


def verified_parent(parent: dict) -> Path:
    """Require the trained parent, then use the shared hash-pinned resolver."""
    from rhpr.gr.resources import verify_parent_artifacts

    required = {"generator-best.pt", "semantic-ids.npy", "run-manifest.json"}
    if not required.issubset(parent["files"]):
        raise ValueError("parent must pin checkpoint, SIDs and run manifest")
    return verify_parent_artifacts(parent).resolve()


def _profile_data(directory: Path, expected: dict):
    """Load only train/validation identities; deliberately ignore test fields."""
    for name, digest in expected["sha256"].items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError(f"profile data hash mismatch: {name}")
    catalog = json.loads((directory / "catalog.json").read_text())
    source_ids = [row["item_id"] for row in catalog]
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("catalogue source IDs must be unique")
    mapping = {item: index for index, item in enumerate(source_ids)}
    users = []
    for row in json.loads((directory / "sequences.json").read_text()):
        if len(row["train"]) < 2:
            raise ValueError("profile requires causal train examples")
        # The sentinel deliberately prevents an accidental test-example path.
        users.append(UserSequence(row["user_id"], tuple(mapping[x] for x in row["train"]),
                                  mapping[row["validation"]], -1))
    counts = {"items": len(catalog), "users": len(users),
              "train_events": sum(len(x.train) for x in users),
              "train_examples": sum(len(x.train) - 1 for x in users)}
    for key, actual in counts.items():
        if expected.get(key, actual) != actual:
            raise ValueError(f"profile data count differs: {key}")
    if len({x.user_id for x in users}) != len(users):
        raise ValueError("profile user IDs must be unique")
    return users, counts


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def precision_supported(precision: str, device: str) -> bool:
    if precision == "fp32":
        return True
    if not device.startswith("cuda"):
        return False
    if precision == "bf16":
        import torch
        return torch.cuda.get_device_capability(device)[0] >= 8
    return precision == "fp16"


def _autocast(precision: str, device: str):
    if precision == "fp32":
        return nullcontext()
    import torch
    return torch.autocast(device_type="cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def _memory(device: str) -> dict:
    if not device.startswith("cuda"):
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    import torch
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _clear(device: str, model=None) -> None:
    if model is not None:
        model.zero_grad(set_to_none=True)
    gc.collect()
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def _is_oom(error: BaseException) -> bool:
    import torch
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _step(model, provider, effective: int, physical: int, optimizer, scaler,
          precision: str, device: str, grad_clip: float) -> dict:
    import torch
    from rhpr.gr.experiment import _loss

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, effective, physical):
        end = min(start + physical, effective)
        batch = provider(start, end)
        with _autocast(precision, device):
            loss = _loss(model, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite profiling loss")
        loss_sum += loss.detach().float().item() * (end - start) / effective
        scaler.scale(loss * (end - start) / effective).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    finite = bool(torch.isfinite(norm))
    if not finite and precision != "fp16":
        raise FloatingPointError("nonfinite profiling gradients")
    scaler.step(optimizer)
    scaler.update()
    return {"loss": loss_sum, "optimizer_update_applied": finite,
            "loss_scale": float(scaler.get_scale())}


def benchmark_training(model, examples, sids, indices, physical: int, precision: str,
                       mode: str, training: dict, specification: dict,
                       deadline: ProfileDeadline, device: str) -> dict:
    """Measure real updates, restoring weights, RNG and training mode on exit."""
    import numpy as np
    import torch
    from rhpr.gr.experiment import _model_batch

    if mode not in {"full_pipeline", "gpu_resident"}:
        raise ValueError("unknown training profile mode")
    effective = max(physical, training["batch_size"])
    if len(indices) < effective:
        raise ValueError("insufficient fixed profile indices for the effective batch")
    selected = np.asarray(indices[:effective])
    contexts, users, targets = [array[selected] for array in examples]
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    was_training = model.training
    gpu_devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=gpu_devices):
            torch.manual_seed(specification.get("seed", 42))
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                         weight_decay=training["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024)
            resident = (_model_batch(contexts, users, targets, sids, device)
                        if mode == "gpu_resident" else None)

            def provider(start, end):
                if resident is not None:
                    return {key: value[start:end] for key, value in resident.items()}
                return _model_batch(contexts[start:end], users[start:end], targets[start:end], sids, device)

            warmup = []
            for _ in range(specification["warmup"]):
                deadline.check()
                warmup.append(_step(model, provider, effective, physical, optimizer, scaler,
                                    precision, device, training["grad_clip_norm"]))
            _synchronize(device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            durations, measurements = [], []
            for _ in range(specification["iterations"]):
                deadline.check()
                _synchronize(device)
                started = time.perf_counter()
                measurements.append(_step(model, provider, effective, physical, optimizer, scaler,
                                          precision, device, training["grad_clip_norm"]))
                _synchronize(device)
                durations.append(time.perf_counter() - started)
            seconds = sum(durations) / len(durations)
            applied = sum(row["optimizer_update_applied"] for row in measurements)
            return {"status": "ok" if applied == len(measurements) else "amp_skipped_updates",
                    "mode": mode, "precision": precision, "physical_batch_size": physical,
                    "effective_batch_size": effective,
                    "baseline_effective_batch_preserved": effective == training["batch_size"],
                    "microbatches_per_update": math.ceil(effective / physical),
                    "warmup_updates_applied": sum(row["optimizer_update_applied"] for row in warmup),
                    "measured_updates_applied": applied, "measured_updates_attempted": len(measurements),
                    "step_seconds_mean": seconds, "step_seconds_min": min(durations),
                    "step_seconds_max": max(durations), "step_seconds": durations,
                    "examples_per_second": effective / seconds,
                    "estimated_train_epoch_seconds": len(examples[2]) / effective * seconds,
                    "last_loss": measurements[-1]["loss"],
                    "loss_scale": measurements[-1]["loss_scale"], **_memory(device)}
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        model.train(was_training)
        _clear(device)


def detailed_training_profile(model, examples, sids, indices, physical: int,
                              precision: str, training: dict, device: str,
                              output: Path) -> dict:
    """One instrumented step: synchronizing stage boundaries changes timing."""
    import torch
    from rhpr.gr.experiment import _loss, _model_batch

    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    was_training = model.training
    random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    selected = indices[:physical]
    contexts, users, targets = [array[selected] for array in examples]
    stages = {}

    def timed(name, function):
        _synchronize(device)
        started = time.perf_counter()
        result = function()
        _synchronize(device)
        stages[name] = time.perf_counter() - started
        return result

    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                     weight_decay=training["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024)
        cpu = timed("host_batch_assembly_seconds", lambda: _model_batch(contexts, users, targets, sids, "cpu"))
        batch = timed("host_to_device_seconds", lambda: {key: value.to(device) for key, value in cpu.items()})
        optimizer.zero_grad(set_to_none=True)
        with _autocast(precision, device):
            loss = timed("forward_and_loss_seconds", lambda: _loss(model, batch))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite instrumented profiling loss")
        timed("backward_seconds", lambda: scaler.scale(loss).backward())

        def update():
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"])
            if not bool(torch.isfinite(norm)) and precision != "fp16":
                raise FloatingPointError("nonfinite instrumented profiling gradients")
            scaler.step(optimizer)
            scaler.update()

        timed("unscale_clip_adam_seconds", update)
        model.load_state_dict(initial)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        report = {"synchronized_stage_timing": True, "includes_cold_adam_state_allocation": True,
                  "warning": "Instrumentation adds synchronization; use throughput cases for end-to-end speed",
                  "operator_profile_uses_initialized_adam_state": True,
                  "physical_batch_size": physical, "precision": precision, **stages}
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True, with_flops=True) as profile:
            _step(model, lambda start, end: {key: value[start:end] for key, value in batch.items()},
                  physical, physical, optimizer, scaler, precision, device, training["grad_clip_norm"])
            _synchronize(device)
        events = []
        for event in profile.key_averages():
            events.append({"operator": event.key, "calls": event.count,
                           "self_cpu_time_us": event.self_cpu_time_total,
                           "cpu_time_total_us": event.cpu_time_total,
                           "device_time_total_us": getattr(event, "device_time_total", 0),
                           "self_device_time_us": getattr(event, "self_device_time_total", 0),
                           "flops": event.flops})
        report["operators_by_device_time"] = sorted(events, key=lambda row: row["device_time_total_us"], reverse=True)[:30]
        report["operators_by_self_cpu_time"] = sorted(events, key=lambda row: row["self_cpu_time_us"], reverse=True)[:30]
        report["recognized_operator_flops_per_physical_batch"] = sum(row["flops"] or 0 for row in events)
        report["flops_note"] = "torch.profiler estimates recognized operators in one forward/backward/update; unsupported operators are omitted"
        (output / "training-profile-operators.txt").write_text(profile.key_averages().table(
            sort_by="self_device_time_total" if device.startswith("cuda") else "self_cpu_time_total", row_limit=40))
        return report
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(initial)
        model.train(was_training)
        torch.set_rng_state(random_state)
        if cuda_random_state is not None:
            torch.cuda.set_rng_state_all(cuda_random_state)
        _clear(device)


def prediction_agreement(reference, candidate) -> dict:
    import numpy as np
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("prediction arrays must share [users,beams,SID] shape")
    if not len(reference) or not reference.shape[1]:
        raise ValueError("agreement requires predictions")
    same = np.all(reference == candidate, axis=2)
    overlap = []
    for left, right in zip(reference, candidate):
        lset, rset = {tuple(x) for x in left[:10]}, {tuple(x) for x in right[:10]}
        overlap.append(len(lset & rset) / max(1, len(lset)))
    return {"top1_exact_agreement": float(same[:, 0].mean()),
            "ranked_beam_exact_agreement": float(same.mean()),
            "top10_set_overlap": float(np.mean(overlap))}


def benchmark_evaluation(model, examples, sid_index, batch_size: int, precision: str,
                         device: str, deadline: ProfileDeadline, warmup: int = 1):
    import numpy as np
    import torch
    from rhpr.gr.experiment import _model_batch
    from rhpr.gr.sid import evaluate_ranked_sids

    contexts, users, targets = examples
    model.eval()
    predictions, zero_scores, score_count = [], 0, 0
    # Warm actual autoregressive beam generation, rather than just the encoder.
    # This is excluded from throughput and checked against the global budget.
    with torch.inference_mode():
        end = min(batch_size, len(targets))
        batch = _model_batch(contexts[:end], users[:end], targets[:end], sid_index.sids, device)
        for _ in range(warmup):
            deadline.check()
            with _autocast(precision, device):
                model.generate(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask_encoder"],
                               user_id=batch["user_id"])
    _synchronize(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(targets), batch_size):
            deadline.check()
            end = begin + batch_size
            batch = _model_batch(contexts[begin:end], users[begin:end], targets[begin:end], sid_index.sids, device)
            with _autocast(precision, device):
                ids, scores = model.generate(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask_encoder"],
                                             user_id=batch["user_id"])
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("nonfinite profiling beam scores")
            if precision != "fp32" and scores.dtype != torch.float32:
                raise FloatingPointError("mixed-precision generation must retain FP32 beam probabilities")
            zero_scores += int((scores == 0).sum().item())
            score_count += scores.numel()
            predictions.append(ids.cpu().numpy())
    _synchronize(device)
    seconds = time.perf_counter() - started
    ids = np.concatenate(predictions)
    metrics = evaluate_ranked_sids(sid_index, ids, targets, cutoffs=(5, 10))
    return {"status": "ok", "precision": precision, "batch_size": batch_size,
            "examples": len(targets), "warmup_batches": warmup, "seconds": seconds,
            "users_per_second": len(targets) / seconds,
            "zero_beam_score_fraction": zero_scores / score_count,
            "validation_metrics": metrics, **_memory(device)}, ids


def run_gr_profile(config: dict, output: Path) -> dict:
    import numpy as np
    import torch
    from rhpr.gr.experiment import _event, _write
    from rhpr.gr.grid import build_grid_generator, fit_grid_rkmeans, prepare_grid_source
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory

    specification = validate_profile(config)
    deadline = ProfileDeadline(specification["max_seconds"])
    device = specification.get("device", config["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("registered profiling experiment requires CUDA")
    torch.set_num_threads(specification.get("cpu_threads", config["training"]["cpu_threads"]))
    root = verified_parent(config["parent"])
    resources = resolve_dataset_directory(config["resources"]["path"])
    verify_resources(resources, config["resources"]["manifest_sha256"])
    scratch = Path(config["resources"].get("scratch_path", "/kaggle/temp/rhpr-gr-profile"))
    grid = prepare_grid_source(unpack_grid(resources / "grid-source.bin", scratch / "original"), scratch / "runtime")
    users, counts = _profile_data(resources / "data", config["data"])
    codes = np.load(root / "semantic-ids.npy", allow_pickle=False)
    if len(codes) != counts["items"]:
        raise ValueError("parent SID catalogue has a different item count")
    # Reconstruct from semantic coordinates and verify the saved disambiguation.
    index = SidIndex(codes[:, :-1], width=config["generator"]["codebook_size"])
    if not np.array_equal(index.sids, codes):
        raise ValueError("parent SID collision suffix differs from the registered catalogue order")
    checkpoint = torch.load(root / "generator-best.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("config", {}).get("data", {}).get("sha256") != config["data"]["sha256"]:
        raise ValueError("parent checkpoint uses a different data catalogue or split")
    if checkpoint.get("config", {}).get("generator") != config["generator"]:
        raise ValueError("profiling generator must exactly match the loaded parent checkpoint")
    model = build_grid_generator(index.sids, config["generator"], seed=config["seed"], device=device, source_root=grid)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    train = build_examples(users, config["data"]["history_items"], "train")
    validation = build_examples(users, config["data"]["history_items"], "validation")
    selected_users = np.sort(np.random.default_rng(config["seed"]).choice(
        len(users), size=min(specification["validation_users"], len(users)), replace=False))
    validation = tuple(array[selected_users] for array in validation)
    largest = max(max(specification["train_batch_sizes"]), config["training"]["batch_size"])
    if largest > len(train[2]):
        raise ValueError("profiling batch exceeds available train examples")
    indices = np.random.default_rng(config["seed"]).permutation(len(train[2]))[:largest]
    report = {"profile_status": "complete", "config": config, "parent": config["parent"],
              "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "none",
              "torch": torch.__version__, "counts": counts, "test_evaluations": 0,
              "validation_user_ids": validation[1].tolist(), "train_profile_indices": indices.tolist(),
              "train_cases": [], "evaluation_cases": [], "quantizer_cases": [],
              "precision_skips": [], "notes": [
                  "All training cases restore identical parent model weights and begin with fresh Adam/scaler state",
                  "Only physical128/256 preserve the registered effective256; larger batches change optimization",
                  "Training cases are microbenchmarks, not continued training or quality trials",
                  "Validation precision agreement is diagnostic; test targets and test predictions are not used",
                  "Profile deadline includes setup after offline runtime; a running step can extend it slightly"]}

    def save():
        report["elapsed_seconds"] = deadline.elapsed
        _write(output / "gr-profile.json", report)

    def failure(error, **case):
        status = "out_of_memory" if _is_oom(error) else "numerical_failed"
        _clear(device, model)
        return {**case, "status": status, "exception_type": type(error).__name__}

    try:
        # Accuracy references and validation throughput are obtained before
        # train-case updates (which are nevertheless restored after each case).
        reference = None
        reference_metrics = None
        for precision in specification["precisions"]:
            if not precision_supported(precision, device):
                report["precision_skips"].append({"precision": precision, "reason": "not natively supported by this device"})
                continue
            for batch_size in specification["evaluation_batch_sizes"]:
                deadline.check()
                try:
                    case, predictions = benchmark_evaluation(model, validation, index, batch_size,
                                                               precision, device, deadline, specification["warmup"])
                    if reference is None:
                        reference, reference_metrics = predictions, case["validation_metrics"]
                    case["agreement_with_fp32_reference"] = prediction_agreement(reference, predictions)
                    case["validation_ndcg10_delta_from_fp32"] = (
                        case["validation_metrics"]["ndcg@10"] - reference_metrics["ndcg@10"])
                except (RuntimeError, FloatingPointError) as error:
                    if isinstance(error, ProfileBudgetExhausted):
                        raise
                    if not _is_oom(error) and not isinstance(error, FloatingPointError):
                        raise
                    case = failure(error, precision=precision, batch_size=batch_size)
                report["evaluation_cases"].append(case)
                _event("gr_profile_evaluation_case", **case)
                save()
        detailed = False
        for precision in specification["precisions"]:
            if not precision_supported(precision, device):
                continue
            for batch_size in specification["train_batch_sizes"]:
                for mode in ("full_pipeline", "gpu_resident"):
                    deadline.check()
                    try:
                        case = benchmark_training(model, train, index.sids, indices, batch_size,
                                                  precision, mode, config["training"],
                                                  {**specification, "seed": config["seed"]}, deadline, device)
                    except (RuntimeError, FloatingPointError) as error:
                        if isinstance(error, ProfileBudgetExhausted):
                            raise
                        if not _is_oom(error) and not isinstance(error, FloatingPointError):
                            raise
                        case = failure(error, mode=mode, precision=precision, physical_batch_size=batch_size)
                    report["train_cases"].append(case)
                    _event("gr_profile_training_case", **case)
                    save()
                if (not detailed and specification.get("profiler", True) and precision == "fp16"
                        and batch_size == config["training"]["batch_size"]):
                    deadline.check()
                    try:
                        report["detailed_training_profile"] = detailed_training_profile(
                            model, train, index.sids, indices, batch_size, precision,
                            config["training"], device, output)
                    except RuntimeError as error:
                        # CUPTI/kineto may be unavailable in a managed runtime;
                        # do not discard the valid end-to-end measurements.
                        report["detailed_training_profile"] = {
                            "status": "unavailable", "exception_type": type(error).__name__,
                            "out_of_memory": _is_oom(error)}
                        _clear(device, model)
                    detailed = True
                    save()
        for width in specification.get("quantizer_widths", []):
            deadline.check()
            if "item-embeddings.npy" not in config["parent"]["files"]:
                raise ValueError("quantizer profiling requires hash-pinned parent item embeddings")
            embeddings = np.load(root / "item-embeddings.npy", allow_pickle=False)
            quantizer_config = {**config["quantizer"], "codebook_size": _positive_int(width, "quantizer width")}
            semantic, _, timing = fit_grid_rkmeans(embeddings, quantizer_config, config["seed"], device, grid)
            _, collision_counts = np.unique(semantic, axis=0, return_counts=True)
            if int(collision_counts.max()) > width:
                # A successful quantizer fit can still be unusable with the
                # baseline's single fixed-width collision suffix. Report that
                # fact without losing valid generator profiling measurements.
                case = {"status": "collision_suffix_capacity_exceeded",
                        "maximum_collision_group": int(collision_counts.max())}
            else:
                case = {"status": "ok", "sid_diagnostics": SidIndex(semantic, width=width).diagnostics()}
            report["quantizer_cases"].append({**case, "codebook_size": width, "timing": timing,
                                               "generator_training": False})
            save()
    except ProfileBudgetExhausted:
        report["profile_status"] = "time_budget"
    report["recommendation_note"] = "Choose throughput only among finite successful cases; changing effective batch or dtype still requires quality validation"
    save()
    metrics = {"experiment": config["name"], "profile_complete": report["profile_status"] == "complete",
               "profile_seconds": report["elapsed_seconds"], "test_evaluations": 0,
               "training_cases": len(report["train_cases"]), "evaluation_cases": len(report["evaluation_cases"])}
    for case in report["train_cases"]:
        if case["status"] != "ok":
            continue
        key = f"train_{case['mode']}_{case['precision']}_b{case['physical_batch_size']}"
        for name in ("examples_per_second", "step_seconds_mean", "peak_allocated_bytes"):
            metrics[f"{key}_{name}"] = case[name]
    for case in report["evaluation_cases"]:
        if case["status"] == "ok":
            metrics[f"eval_{case['precision']}_b{case['batch_size']}_users_per_second"] = case["users_per_second"]
    return metrics
