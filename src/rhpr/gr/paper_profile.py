"""Runtime measurements for the paper-aligned GRID recipe, without quality claims.

The item encoder is real Flan-T5-XL. Generator timing uses a freshly initialized
GRID model and injective structural placeholder SIDs, never a hybrid checkpoint.
Those placeholders preserve tensor shapes but are not a trained tokenizer.
"""
from __future__ import annotations

import gc
import math
from pathlib import Path
import time

from rhpr.gr.profiling import (
    ProfileBudgetExhausted, _autocast, _clear, _is_oom, _memory,
    _profile_data, _synchronize, precision_supported, prediction_agreement,
)


class PaperProfileBudget:
    def __init__(self, seconds: float, clock=time.monotonic):
        if not math.isfinite(seconds) or not 0 < seconds <= 1200:
            raise ValueError("paper profile budget must be in (0,1200] seconds")
        self.clock, self.started, self.seconds = clock, clock(), seconds

    @property
    def elapsed(self):
        return self.clock() - self.started

    def check(self):
        if self.elapsed >= self.seconds:
            raise ProfileBudgetExhausted("paper runtime profile budget exhausted")


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_paper_profile(config: dict) -> dict:
    result = dict(config["profile"])
    for name, default in (("warmup", 2), ("iterations", 5), ("encoder_items", 256),
                          ("validation_users", 512), ("quantizer_iterations", 20)):
        result[name] = _integer(result.get(name, default), name)
    for name, default in (("encoder_batch_sizes", [8, 16, 32]),
                          ("train_batch_sizes", [128, 256, 64, 32]),
                          ("evaluation_batch_sizes", [32, 64, 128, 256]),
                          ("quantizer_batch_sizes", [2048, 16384])):
        values = result.get(name, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        values = [_integer(x, name) for x in values]
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain distinct sizes")
        result[name] = values
    for name in ("precisions", "encoder_precisions"):
        values = result.get(name, ["fp32", "fp16"])
        if (not isinstance(values, list) or not values
                or any(x not in ("fp32", "fp16", "bf16") for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain distinct supported precision names")
        result[name] = ["fp32", *[x for x in values if x != "fp32"]]
    result["max_seconds"] = float(result.get("max_seconds", 900))
    PaperProfileBudget(result["max_seconds"])
    if config.get("evaluation", {}).get("test_after_selection", False):
        raise ValueError("runtime profiling must not evaluate held-out test targets")
    if config.get("parent"):
        raise ValueError("paper profiling must not reuse a hybrid checkpoint")
    if config["training"]["batch_size"] != 256:
        raise ValueError("paper runtime reference requires effective batch 256")
    return result


def runtime_sid_index(num_items: int, width: int, seed: int):
    """Injective, shape-correct placeholders; no semantic or quality meaning."""
    import numpy as np
    from rhpr.gr.sid import SidIndex

    if type(num_items) is not int or not 0 < num_items <= width ** 3:
        raise ValueError("runtime catalogue exceeds three-code capacity")
    values = np.random.default_rng(seed).choice(width ** 3, size=num_items, replace=False)
    semantic = np.column_stack([(values // width ** power) % width for power in (2, 1, 0)])
    return SidIndex(semantic, width=width)


def masked_mean(last_hidden_state, attention_mask):
    """Official GRID mean pooling: include EOS and exclude right padding."""
    mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)


def embedding_agreement(reference, candidate) -> dict:
    import torch

    left, right = reference.float(), candidate.float()
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("embedding comparisons require equal [items,dimensions] shapes")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("nonfinite encoder profiling embeddings")
    return {"maximum_absolute_difference": float((left - right).abs().max()),
            "mean_relative_l2_difference": float(((left - right).norm(dim=1) /
                left.norm(dim=1).clamp_min(1e-12)).mean()),
            "minimum_cosine_similarity": float(torch.nn.functional.cosine_similarity(left, right).min())}


def _case_failure(error, **identity):
    return {**identity, "status": "out_of_memory" if _is_oom(error) else "numerical_failed",
            "exception_type": type(error).__name__}


def _load_encoder(directory, device):
    """Load the verified encoder without a second full-size CPU allocation."""
    import torch
    from safetensors import safe_open
    from transformers import T5Config, T5EncoderModel

    with safe_open(Path(directory) / "model.safetensors", framework="pt", device="cpu") as source:
        state = {name: source.get_tensor(name) for name in source.keys()}
    with torch.device("meta"):
        model = T5EncoderModel(T5Config.from_pretrained(directory, local_files_only=True))
    if "encoder.embed_tokens.weight" not in state:
        state["encoder.embed_tokens.weight"] = state["shared.weight"]
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("the reference Flan encoder weights must remain FP32")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.eval().to(device)


def profile_encoder(config, texts, device, specification, budget, progress=None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from rhpr.gr.resources import verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    source = config["representation"]
    directory = resolve_dataset_directory(source["path"])
    verify_resources(directory, source["manifest_sha256"])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = _load_encoder(directory, device)
    _synchronize(device)
    load_seconds = time.perf_counter() - started
    chosen = np.sort(np.random.default_rng(config["seed"]).choice(
        len(texts), size=min(len(texts), specification["encoder_items"]), replace=False))
    started = time.perf_counter()
    tokens = tokenizer([texts[i] for i in chosen], max_length=source["max_length"],
                       padding="max_length", truncation=True, add_special_tokens=True,
                       return_tensors="pt")
    tokenize_seconds = time.perf_counter() - started
    result = {"model": source["model"], "revision": source["revision"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "model_load_seconds": load_seconds, "sample_items": len(chosen),
              "catalogue_items": len(texts), "tokenization_seconds": tokenize_seconds,
              "max_length": source["max_length"], "pooling": "masked mean including EOS; no Dense/L2",
              "cases": [], "quality_not_evaluated": True,
              "estimate_note": "Extrapolated dense inference throughput; excludes full catalogue RK fitting and training"}
    try:
        for precision in specification["encoder_precisions"]:
            if not precision_supported(precision, device):
                result["cases"].append({"precision": precision, "status": "unsupported"})
                continue
            for physical in specification["encoder_batch_sizes"]:
                budget.check()
                size = min(physical, len(chosen))
                cpu = {key: value[:size] for key, value in tokens.items()}

                def forward():
                    batch = {key: value.to(device) for key, value in cpu.items()}
                    with torch.inference_mode(), _autocast(precision, device):
                        hidden = model(**batch).last_hidden_state
                        return masked_mean(hidden.float(), batch["attention_mask"]).cpu()

                try:
                    for _ in range(specification["warmup"]):
                        budget.check()
                        forward()
                    _synchronize(device)
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    durations, output = [], None
                    for _ in range(specification["iterations"]):
                        budget.check()
                        _synchronize(device)
                        started = time.perf_counter()
                        output = forward()
                        _synchronize(device)
                        durations.append(time.perf_counter() - started)
                    if not bool(torch.isfinite(output).all()):
                        raise FloatingPointError("nonfinite encoder profile")
                    # Obtain a same-shape FP32 reference outside throughput.
                    if precision == "fp32":
                        agreement = embedding_agreement(output, output)
                    else:
                        batch = {key: value.to(device) for key, value in cpu.items()}
                        with torch.inference_mode():
                            baseline = masked_mean(model(**batch).last_hidden_state,
                                                   batch["attention_mask"]).cpu()
                        agreement = embedding_agreement(baseline, output)
                    seconds = sum(durations) / len(durations)
                    case = {"status": "ok", "precision": precision, "batch_size": size,
                            "seconds_per_batch": seconds, "batch_seconds": durations,
                            "items_per_second": size / seconds,
                            "estimated_catalogue_encoding_seconds": len(texts) / size * seconds
                                + tokenize_seconds * len(texts) / len(chosen),
                            "agreement_with_fp32": agreement, **_memory(device)}
                except (RuntimeError, FloatingPointError) as error:
                    if isinstance(error, ProfileBudgetExhausted):
                        raise
                    if not _is_oom(error) and not isinstance(error, FloatingPointError):
                        raise
                    case = _case_failure(error, precision=precision, batch_size=size)
                    _clear(device)
                result["cases"].append(case)
                if progress is not None:
                    progress(result)
        return result
    finally:
        del model, tokenizer
        gc.collect()
        _clear(device)


def _gradient_norm(model):
    import torch
    gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("profile update produced no gradients")
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))


def _update(model, batch, physical, optimizer, scaler, precision, device, clip):
    import torch
    from rhpr.gr.experiment import _loss

    effective = len(batch["future_ids"])
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, effective, physical):
        end = min(start + physical, effective)
        part = {key: value[start:end].to(device) for key, value in batch.items()}
        with _autocast(precision, device):
            loss = _loss(model, part)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite paper profile training loss")
        loss_sum += float(loss.detach()) * (end - start) / effective
        scaler.scale(loss * (end - start) / effective).backward()
    scaler.unscale_(optimizer)
    if clip is None:
        norm = _gradient_norm(model)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    finite = bool(torch.isfinite(norm))
    if not finite and precision != "fp16":
        raise FloatingPointError("nonfinite paper profile training gradients")
    scaler.step(optimizer)
    scaler.update()
    return {"loss": loss_sum, "optimizer_update_applied": finite,
            "loss_scale": float(scaler.get_scale())}


def benchmark_paper_training(model, stream_factory, physical, precision, mode,
                             training, specification, budget, device):
    """Benchmark actual upstream augmentation or a resident copy of its batch."""
    import torch

    if mode not in ("full_pipeline", "gpu_resident"):
        raise ValueError("unknown paper training benchmark mode")
    effective = max(physical, training["batch_size"])
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    previous_mode = model.training
    devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(specification.get("seed", 42))
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                         weight_decay=training["weight_decay"])
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1024)
            stream = stream_factory(effective)
            resident = ({key: value.to(device) for key, value in stream.next_batch().items()}
                        if mode == "gpu_resident" else None)

            def one_step():
                batch = resident if resident is not None else stream.next_batch()
                return _update(model, batch, physical, optimizer, scaler, precision, device,
                               training.get("grad_clip_norm"))

            warmup = []
            for _ in range(specification["warmup"]):
                budget.check()
                warmup.append(one_step())
            _synchronize(device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            durations, results = [], []
            for _ in range(specification["iterations"]):
                budget.check()
                _synchronize(device)
                started = time.perf_counter()
                results.append(one_step())
                _synchronize(device)
                durations.append(time.perf_counter() - started)
            seconds = sum(durations) / len(durations)
            applied = sum(x["optimizer_update_applied"] for x in results)
            return {"status": "ok" if applied == len(results) else "amp_skipped_updates",
                    "physical_batch_size": physical, "effective_batch_size": effective,
                    "paper_effective_batch_preserved": effective == 256,
                    "mode": mode, "precision": precision,
                    "microbatches_per_update": math.ceil(effective / physical),
                    "seconds_per_update": seconds, "update_seconds": durations,
                    "examples_per_second": effective / seconds,
                    "warmup_updates_applied": sum(x["optimizer_update_applied"] for x in warmup),
                    "measured_updates_applied": applied,
                    "measured_updates_attempted": len(results),
                    "finite_gradient_guard": "foreach gradient norm; included in update time",
                    "gradient_clipping": training.get("grad_clip_norm"),
                    "loss_scale": results[-1]["loss_scale"],
                    "sampling": stream.report(), **_memory(device)}
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(state)
        model.train(previous_mode)
        _clear(device)


def benchmark_paper_generation(model, examples, sids, physical, precision, device,
                               specification, budget):
    """Time beam generation; deliberately compute no retrieval quality metric."""
    import numpy as np
    import torch
    from rhpr.gr.experiment import _model_batch

    contexts, user_ids, targets = examples
    model.eval()
    outputs = []
    with torch.inference_mode():
        sample = _model_batch(contexts[:physical], user_ids[:physical], targets[:physical], sids, device)
        for _ in range(specification["warmup"]):
            budget.check()
            with _autocast(precision, device):
                model.generate(input_ids=sample["input_ids"], attention_mask=sample["attention_mask_encoder"],
                               user_id=sample["user_id"])
        _synchronize(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for start in range(0, len(targets), physical):
            budget.check()
            batch = _model_batch(contexts[start:start + physical], user_ids[start:start + physical],
                                 targets[start:start + physical], sids, device)
            with _autocast(precision, device):
                ids, scores = model.generate(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask_encoder"], user_id=batch["user_id"])
            if not bool(torch.isfinite(scores).all()) or scores.dtype != torch.float32:
                raise FloatingPointError("generation requires finite FP32 beam probabilities")
            outputs.append(ids.cpu().numpy())
        _synchronize(device)
    seconds = time.perf_counter() - started
    return {"status": "ok", "precision": precision, "batch_size": physical,
            "users": len(targets), "seconds": seconds,
            "users_per_second": len(targets) / seconds, **_memory(device)}, np.concatenate(outputs)


def profile_paper_operators(model, stream_factory, training, specification, budget, device, output):
    """One instrumented source-FP32 update, separate from throughput samples."""
    import torch
    from rhpr.gr.experiment import _loss

    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    was_training, cpu_rng = model.training, torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    stages = {}

    def timed(name, operation):
        budget.check()
        _synchronize(device)
        started = time.perf_counter()
        value = operation()
        _synchronize(device)
        stages[name] = time.perf_counter() - started
        return value

    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                     weight_decay=training["weight_decay"])
        stream = stream_factory(training["batch_size"])
        cpu = timed("cpu_author_augmentation_seconds", stream.next_batch)
        batch = timed("host_to_device_seconds", lambda: {key: value.to(device) for key, value in cpu.items()})
        optimizer.zero_grad(set_to_none=True)
        loss = timed("forward_loss_seconds", lambda: _loss(model, batch))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite detailed paper profile loss")
        timed("backward_seconds", loss.backward)

        def gradient_check():
            norm = (_gradient_norm(model) if training.get("grad_clip_norm") is None
                    else torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip_norm"]))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("nonfinite detailed paper profile gradients")

        timed("finite_gradient_guard_seconds", gradient_check)
        timed("adam_seconds", optimizer.step)
        model.load_state_dict(state)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    with_flops=True, profile_memory=True) as profile:
            with torch.profiler.record_function("rhpr_author_augmentation"):
                sampled = stream.next_batch()
            with torch.profiler.record_function("rhpr_training_update"):
                _update(model, sampled, training["batch_size"], optimizer, scaler, "fp32", device,
                        training.get("grad_clip_norm"))
            _synchronize(device)
        operators = [{"operator": event.key, "calls": event.count,
                      "self_cpu_time_us": event.self_cpu_time_total,
                      "device_time_total_us": getattr(event, "device_time_total", 0),
                      "self_device_time_us": getattr(event, "self_device_time_total", 0),
                      "recognized_flops": event.flops}
                     for event in profile.key_averages()]
        (output / "grid-paper-operators.txt").write_text(profile.key_averages().table(
            sort_by="self_device_time_total" if device.startswith("cuda") else "self_cpu_time_total", row_limit=40))
        return {"status": "ok", "precision": "fp32", "effective_batch_size": training["batch_size"],
                "synchronized_stage_timing": stages,
                "top_device_operators": sorted(operators, key=lambda x: x["device_time_total_us"], reverse=True)[:25],
                "top_self_cpu_operators": sorted(operators, key=lambda x: x["self_cpu_time_us"], reverse=True)[:25],
                "recognized_flops_per_update": sum(x["recognized_flops"] or 0 for x in operators),
                "note": "Synchronization/profiler instrumentation changes timing; Adam stage includes cold state allocation; unsupported FLOPs are omitted"}
    finally:
        model.zero_grad(set_to_none=True)
        model.load_state_dict(state)
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state_all(gpu_rng)
        _clear(device)


def profile_quantizer(grid_root, config, specification, budget, device, progress=None):
    import torch
    from rhpr.gr.distances import distance_backend_report, make_grid_distance
    from rhpr.gr.grid import load_grid
    from rhpr.gr.paper_quantizer_profile import benchmark_logical_rank_quantizer

    grid = load_grid(grid_root)
    width = config["quantizer"]["codebook_size"]
    dimension = config["quantizer"].get("embedding_dim", 2048)
    init_size = config["quantizer"].get("init_buffer_size", 3072)
    try:
        cases = [benchmark_logical_rank_quantizer(grid_root, config, specification, budget, device)]
    except RuntimeError as error:
        if isinstance(error, ProfileBudgetExhausted) or not _is_oom(error):
            raise
        cases = [_case_failure(error, mode="logical_rank_operator")]
        _clear(device)
    if progress is not None:
        progress(cases)
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device).index or 0]
                                              if device.startswith("cuda") else []):
        torch.manual_seed(config["seed"])
        for size in specification["quantizer_batch_sizes"]:
            budget.check()
            model = None
            try:
                proxy = torch.nn.functional.normalize(torch.randn(max(size, init_size), dimension,
                                                                  device=device), dim=-1)
                distance = make_grid_distance(grid, config["quantizer"])
                initializer = grid.KMeansPlusPlusInitInitializer(n_clusters=width,
                    distance_function=grid.SquaredEuclideanDistance(), initialize_on_cpu=False)
                model = grid.MiniBatchKMeans(n_clusters=width, n_features=dimension,
                    distance_function=distance, initializer=initializer, init_buffer_size=init_size,
                    update_manually=True).to(device)
                model.init_buffer = torch.empty(0, device=device)
                model.cluster_counts = torch.zeros(width, device=device)
                _synchronize(device)
                started = time.perf_counter()
                model.model_step(proxy[:init_size])
                _synchronize(device)
                initialization_seconds = time.perf_counter() - started
                for _ in range(specification["warmup"]):
                    budget.check()
                    model.model_step(proxy[:size])
                _synchronize(device)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                durations = []
                for _ in range(specification["quantizer_iterations"]):
                    budget.check()
                    _synchronize(device)
                    started = time.perf_counter()
                    model.model_step(proxy[:size])
                    _synchronize(device)
                    durations.append(time.perf_counter() - started)
                seconds = sum(durations) / len(durations)
                case = {"status": "ok", "batch_size": size, "embedding_dim": dimension,
                        **distance_backend_report(config["quantizer"]),
                        "seconds_per_level_update": seconds, "initialization_seconds": initialization_seconds,
                        "estimated_3_levels_1000_updates_seconds": 3 * (1000 * seconds + initialization_seconds),
                        "synthetic_runtime_embeddings": True, "equivalent_to_8_gpu_training": False,
                        "note": "Single-device dense-shape throughput only; not fitted Flan codes or distributed update equivalence",
                        **_memory(device)}
            except RuntimeError as error:
                if isinstance(error, ProfileBudgetExhausted):
                    raise
                if not _is_oom(error):
                    raise
                case = _case_failure(error, batch_size=size)
            finally:
                if model is not None:
                    del model
                if "proxy" in locals():
                    del proxy
                _clear(device)
            cases.append(case)
            if progress is not None:
                progress(cases)
    return cases


def runtime_estimates(report, config):
    """Expose update-based costs; the author stream has no finite epoch unit."""
    validation = [x for x in report["generation_cases"] if x["status"] == "ok" and x["precision"] == "fp32"]
    if not validation:
        return []
    full_validation = min(x["seconds"] / x["users"] for x in validation) * report["counts"]["users"]
    cadence = config["training"].get("validation_interval_updates", 100)
    estimates = []
    for case in report["training_cases"]:
        if case["status"] != "ok" or case["mode"] != "full_pipeline" or not case["paper_effective_batch_preserved"]:
            continue
        for updates in (100, 1000, 10000):
            validations = updates // cadence
            estimates.append({"precision": case["precision"], "physical_batch_size": case["physical_batch_size"],
                "optimizer_updates": updates, "training_examples": updates * 256,
                "training_seconds": updates * case["seconds_per_update"],
                "validation_checks": validations, "estimated_full_validation_seconds": full_validation,
                "estimated_train_plus_validation_seconds": updates * case["seconds_per_update"] + validations * full_validation,
                "estimate_note": "Measured short fresh-model throughput; excludes setup/encoding/RK/test and does not predict convergence"})
    return estimates


def run_grid_paper_profile(config: dict, output: Path) -> dict:
    import json
    import numpy as np
    import torch
    from rhpr.gr.data import build_examples
    from rhpr.gr.experiment import _event, _write
    from rhpr.gr.grid import build_grid_generator, prepare_grid_source
    from rhpr.gr.paper_protocol import PaperBatchStream
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    specification = validate_paper_profile(config)
    budget = PaperProfileBudget(specification["max_seconds"])
    device = specification.get("device", config["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("paper GPU profile requires CUDA")
    torch.set_num_threads(config["training"].get("cpu_threads", 2))
    resources = resolve_dataset_directory(config["resources"]["path"])
    verify_resources(resources, config["resources"]["manifest_sha256"])
    scratch = Path(config["resources"].get("scratch_path", "/kaggle/temp/rhpr-grid-paper-profile"))
    grid = prepare_grid_source(unpack_grid(resources / "grid-source.bin", scratch / "original"), scratch / "runtime")
    users, counts = _profile_data(resources / "data", config["data"])
    texts = [row["text"] for row in json.loads((resources / "data/catalog.json").read_text())]
    report = {"profile_status": "complete", "quality_not_evaluated": True,
              "test_evaluations": 0, "synthetic_runtime_sids": True,
              "config": config, "counts": counts, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else "none",
              "training_cases": [], "generation_cases": [], "quantizer_cases": [],
              "notes": ["Fresh generator; structural placeholder SIDs only, never a hybrid checkpoint",
                        "Actual upstream contiguous-subsequence augmentation and full author training rows",
                        "No recommendation quality is evaluated; fresh-model generation measures serving cost only",
                        "FP32 source reference; FP16 is a numerical/performance diagnostic",
                        "Cooperative profile deadline plus independently enforced Kaggle worker timeout"]}

    def save():
        report["elapsed_seconds"] = budget.elapsed
        report["runtime_estimates"] = runtime_estimates(report, config)
        _write(output / "grid-paper-profile.json", report)

    try:
        def encoder_progress(partial):
            report["encoder"] = partial
            save()

        report["encoder"] = profile_encoder(config, texts, device, specification, budget,
                                            progress=encoder_progress)
        _event("grid_paper_encoder_profile_complete", **report["encoder"])
        save()
        budget.check()
        index = runtime_sid_index(counts["items"], config["generator"]["codebook_size"], config["seed"])
        np.save(output / "runtime-semantic-ids.npy", index.sids, allow_pickle=False)
        model = build_grid_generator(index.sids, config["generator"], seed=config["seed"],
                                     device=device, source_root=grid)
        report["generator_parameters"] = sum(p.numel() for p in model.parameters())
        report["generator_provenance"] = model.rhpr_grid_provenance
        positions = config["training"].get("sequence_length", 120)
        separators = positions // index.sids.shape[1] if config["generator"].get("separator_token") else 0
        report["training_input_shape"] = {
            "sid_positions_including_masked_target": positions,
            "separator_positions": separators,
            "encoder_positions": positions + separators,
            "max_observed_history_items": positions // index.sids.shape[1] - 1,
        }

        def stream_factory(effective):
            training = config["training"]
            return PaperBatchStream(users, index.sids, seed=config["seed"],
                source_batch_size=training.get("source_user_batch_size", 32), effective_batch_size=effective,
                sequence_length=training.get("sequence_length", 120),
                max_augmented_batch=training.get("max_augmented_batch", 512))

        detailed = False
        for precision in specification["precisions"]:
            if not precision_supported(precision, device):
                continue
            for physical in specification["train_batch_sizes"]:
                for mode in ("full_pipeline", "gpu_resident"):
                    budget.check()
                    try:
                        case = benchmark_paper_training(model, stream_factory, physical, precision, mode,
                            config["training"], {**specification, "seed": config["seed"]}, budget, device)
                    except (RuntimeError, FloatingPointError) as error:
                        if isinstance(error, ProfileBudgetExhausted):
                            raise
                        if not _is_oom(error) and not isinstance(error, FloatingPointError):
                            raise
                        case = _case_failure(error, physical_batch_size=physical, precision=precision, mode=mode)
                        _clear(device, model)
                    report["training_cases"].append(case)
                    _event("grid_paper_training_case", **case)
                    save()
                if (not detailed and specification.get("profiler", True) and precision == "fp32"
                        and physical == config["training"]["batch_size"]):
                    budget.check()
                    try:
                        report["operator_profile"] = profile_paper_operators(
                            model, stream_factory, config["training"], specification, budget, device, output)
                    except RuntimeError as error:
                        if isinstance(error, ProfileBudgetExhausted):
                            raise
                        report["operator_profile"] = {"status": "unavailable", "exception_type": type(error).__name__}
                        _clear(device, model)
                    detailed = True
                    save()
        validation = build_examples(users, config["data"]["history_items"], "validation")
        selected = np.sort(np.random.default_rng(config["seed"]).choice(len(users),
            size=min(len(users), specification["validation_users"]), replace=False))
        validation = tuple(array[selected] for array in validation)
        reference = None
        for precision in specification["precisions"]:
            if not precision_supported(precision, device):
                continue
            for physical in specification["evaluation_batch_sizes"]:
                budget.check()
                try:
                    case, ids = benchmark_paper_generation(model, validation, index.sids, physical,
                        precision, device, specification, budget)
                    if reference is None:
                        reference = ids
                    case["beam_agreement_with_fp32"] = prediction_agreement(reference, ids)
                except (RuntimeError, FloatingPointError) as error:
                    if isinstance(error, ProfileBudgetExhausted):
                        raise
                    if not _is_oom(error) and not isinstance(error, FloatingPointError):
                        raise
                    case = _case_failure(error, batch_size=physical, precision=precision)
                    _clear(device, model)
                report["generation_cases"].append(case)
                _event("grid_paper_generation_case", **case)
                save()
        del model
        _clear(device)
        def quantizer_progress(partial):
            report["quantizer_cases"] = partial
            save()

        report["quantizer_cases"] = profile_quantizer(grid, config, specification, budget, device,
                                                      progress=quantizer_progress)
        save()
    except ProfileBudgetExhausted:
        report["profile_status"] = "time_budget"
    save()
    result = {"experiment": config["name"], "profile_complete": report["profile_status"] == "complete",
              "quality_not_evaluated": True, "test_evaluations": 0,
              "synthetic_runtime_sids": True, "profile_seconds": report["elapsed_seconds"],
              "training_cases": len(report["training_cases"]), "generation_cases": len(report["generation_cases"])}
    if "generator_parameters" in report:
        result["generator_parameters"] = report["generator_parameters"]
    for case in report["training_cases"]:
        if case["status"] == "ok":
            result[f"train_{case['mode']}_{case['precision']}_b{case['physical_batch_size']}_seconds_per_update"] = case["seconds_per_update"]
    for case in report["generation_cases"]:
        if case["status"] == "ok":
            result[f"generation_{case['precision']}_b{case['batch_size']}_users_per_second"] = case["users_per_second"]
    return result
