"""GPU SID index study, optionally crossed with generator initializations.

No UI, no implicit Kaggle launch, no automatic precision/LR fallbacks. Raw and
local catalogue scores, exact search and beams always retain their identities.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np

from rhpr.gr.data import file_sha256, load_beauty
from rhpr.gr.experiment import _event, _write
from rhpr.gr.sid import SidIndex, validate_item_embeddings
from rhpr.gr.study_store import StudyStore


def state_digest(state):
    """Tensor-content hash, independent of torch.save archive names/metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


def fit_quantizer(values, spec, seed, output):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    residual, codes, centers, reports = values.copy(), [], [], []
    with threadpool_limits(limits=spec["cpu_threads"]):
        for level in range(spec["levels"]):
            residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
            quantizer = KMeans(n_clusters=spec["codebook_size"], n_init=spec["n_init"],
                max_iter=spec["max_iter"], random_state=seed + level, algorithm="lloyd")
            codes.append(quantizer.fit_predict(residual))
            centers.append(quantizer.cluster_centers_.copy())
            residual -= quantizer.cluster_centers_[codes[-1]]
            reports.append({"level": level, "iterations": quantizer.n_iter_, "inertia": quantizer.inertia_})
    index = SidIndex(np.stack(codes, 1), width=spec["codebook_size"])
    np.save(output / "semantic-ids.npy", index.sids)
    np.save(output / "quantizer-centroids.npy", np.stack(centers))
    _write(output / "quantizer.json", {**spec, "seed": seed, "backend": "sklearn Lloyd",
        "normalization": "L2 normalize each incoming residual", "reports": reports,
        "suffix": "catalogue-row ordinal; includes zero for singleton groups"})
    return index, np.stack(centers)


def build_study_model(index, source, output, spec):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from rhpr.gr.genpas import compose_beauty
    folder = output / "model-input"
    (folder / "sids").mkdir(parents=True)
    # The upstream configuration resolver loads this conventional filename;
    # actual encoder identity/width are recorded separately, never inferred here.
    torch.save(torch.as_tensor(index.sids.T.copy()), folder / "sids/flan-t5-xxl_rkmeans_3_256_seed42.pt")
    cfg = compose_beauty(source, folder, output)
    cfg.model.huggingface_model.config.update({
        "vocab_size": index.width, "d_model": spec["d_model"], "d_ff": spec["d_ff"],
        "d_kv": spec["d_kv"], "num_heads": spec["num_heads"],
        "num_layers": spec["encoder_layers"], "dropout_rate": spec["dropout"]})
    cfg.model.decoder.config.num_layers = spec["decoder_layers"]
    cfg.model.decoder.config.dropout_rate = spec["dropout"]
    cfg.model.num_hierarchies = index.sid_length
    cfg.model.num_user_bins = 0
    cfg.model.mlp_layers = None
    model = instantiate(cfg.model)
    _write(output / "generator.json", {"resolved_config": OmegaConf.to_container(cfg.model, resolve=True),
        "parameters": sum(p.numel() for p in model.parameters()), "study_generator": spec})
    return model


def make_batch(contexts, targets, user_ids, index, input_tokens, device):
    """Canonical RIGHT padding; actual context events never include the label.

    Retains item-aligned input positions for stored histories/interventions.
    No masking placeholder is attended and no partial SID is retained.
    """
    import torch
    if not (len(contexts) == len(targets) == len(user_ids)) or not contexts:
        raise ValueError("Batch alignment error")
    if input_tokens % index.sid_length:
        raise ValueError("Input budget must contain complete SIDs")
    inputs = np.full((len(contexts), input_tokens), -1, dtype=np.int64)
    mask = np.zeros_like(inputs)
    for i, context in enumerate(contexts):
        if not context or len(context) * index.sid_length > input_tokens:
            raise ValueError("Empty or oversized actual context")
        encoded = index.sids[np.asarray(context)].reshape(-1)
        inputs[i, :len(encoded)], mask[i, :len(encoded)] = encoded, 1
    return {"input_ids": torch.as_tensor(inputs, device=device),
        "attention_mask_encoder": torch.as_tensor(mask, device=device),
        "user_id": torch.as_tensor(user_ids, device=device)[:, None],
        "future_ids": torch.as_tensor(index.sids[np.asarray(targets)].copy(), device=device)}


def evaluate_study(model, cases, index, store, run_id, epoch, config, output):
    try:
        return _evaluate_study(model, cases, index, store, run_id, epoch, config, output)
    finally:
        # Preserve completed measurements even if a later catalogue case fails.
        store.flush()


def _evaluate_study(model, cases, index, store, run_id, epoch, config, output):
    import torch
    from rhpr.gr.study_search import ModelScorer, SCORINGS, target_diagnostics, beam_search, exact_catalogue, retrieval_metrics
    spec = config["diagnostics"]
    started = time.monotonic()
    device = next(model.parameters()).device
    model.eval()
    metrics, summaries = [], {name: Counter() for name in SCORINGS}
    panel_scores = output / "catalogue-scores"
    panel_scores.mkdir(exist_ok=True)
    for begin in range(0, len(cases), spec["evaluation_batch_size"]):
        chosen = cases[begin:begin + spec["evaluation_batch_size"]]
        batch = make_batch([c["visible_history"] for c in chosen], [c["target_item"] for c in chosen],
            [c["user_id"] for c in chosen], index, config["training"]["input_tokens"], device)
        scorer = ModelScorer(model, batch, index)
        targets = index.sids[[c["target_item"] for c in chosen]]
        path, raw_target = target_diagnostics(scorer, targets, [c["case_id"] for c in chosen], run_id, epoch)
        store.write("target_path", path)
        exact = {}
        if epoch in spec["exact_epochs"]:
            for owner, case in enumerate(chosen):
                if not case["exact_panel"]:
                    continue
                scores = exact_catalogue(scorer, owner, chunk_size=spec["exact_chunk_size"])
                if not np.isclose(scores["raw"][case["target_item"]], raw_target[owner], rtol=2e-5, atol=2e-5):
                    raise ValueError("Teacher-forced target score disagrees with exact catalogue scoring")
                exact[owner] = scores
                # Vectors retain ALL item scores, not just the exported top100.
                path_file = panel_scores / f"{epoch}-{case['case_id'].replace(':', '-')}.npz"
                np.savez_compressed(path_file, raw=scores["raw"], local=scores["local"],
                                    item_ids=np.arange(index.num_items))
                store.write("artifacts", [{"path": str(path_file.relative_to(store.root)),
                    "sha256": file_sha256(path_file), "role": "exact_catalogue_scores"}])
                for scoring in SCORINGS:
                    ranked = np.lexsort((np.arange(index.num_items), -scores[scoring]))
                    rank = int(np.flatnonzero(ranked == case["target_item"])[0] + 1)
                    mass = float(np.exp(scores["raw"]).sum())
                    if not np.isclose(np.exp(scores["local"]).sum(), 1., rtol=2e-4, atol=2e-6):
                        raise ValueError("Locally normalized catalogue distribution does not sum to one")
                    metrics.append({"run_id": run_id, "epoch": epoch, "case_id": case["case_id"],
                        "scoring": scoring, "search": "exact", "beam_width": None, "target_rank": rank,
                        "raw_target_logprob": float(raw_target[owner]), "valid_catalogue_mass": mass,
                        **retrieval_metrics(rank), "exact_top10_overlap": 1., "target_lost10": False,
                        "target_promoted10": False})
                    store.write("predictions", [{"run_id": run_id, "epoch": epoch, "case_id": case["case_id"],
                        "scoring": scoring, "search": "exact", "beam_width": None, "rank": r,
                        "item_id": int(item), "sid": index.sids[item].tolist(), "log_score": float(scores[scoring][item])}
                        for r, item in enumerate(ranked[:spec["save_top_k"]], 1)])
        for scoring in SCORINGS:
            # Standard beams for every case at every scheduled checkpoint.
            beams, trace = beam_search(scorer, spec["main_beam_width"], scoring, targets=targets)
            requests = [(list(range(len(chosen))), beams, trace, spec["main_beam_width"])]
            # Wide beams only on the common panel and exact-search epochs.
            for width in spec["panel_beam_widths"]:
                if exact and width != spec["main_beam_width"]:
                    owners = sorted(exact)
                    sub_batch = {name: value[owners] for name, value in batch.items()}
                    sub = ModelScorer(model, sub_batch, index)
                    wide, trace_wide = beam_search(sub, width, scoring, targets=targets[owners])
                    requests.append((owners, wide, trace_wide, width))
            for owners, groups, trace, width in requests:
                store.write("beam_trace", [{"run_id": run_id, "epoch": epoch,
                    "case_id": chosen[owners[row["owner"]]]["case_id"], "scoring": scoring,
                    "beam_width": width, **{key: value for key, value in row.items() if key != "owner"}}
                    for row in trace])
                predictions = []
                for local, owner in enumerate(owners):
                    case, group = chosen[owner], groups[local]
                    items = [index.sid_to_item[sid] for sid, _ in group]
                    rank = items.index(case["target_item"]) + 1 if case["target_item"] in items else None
                    overlap = lost = promoted = None
                    if owner in exact:
                        scores = exact[owner][scoring]
                        ranked = np.lexsort((np.arange(index.num_items), -scores))
                        erank = int(np.flatnonzero(ranked == case["target_item"])[0] + 1)
                        overlap = len(set(items[:10]) & set(ranked[:10])) / min(10, index.num_items)
                        lost = erank <= 10 and not (rank is not None and rank <= 10)
                        promoted = erank > 10 and rank is not None and rank <= 10
                    row = {"run_id": run_id, "epoch": epoch, "case_id": case["case_id"],
                        "scoring": scoring, "search": "beam", "beam_width": width, "target_rank": rank,
                        "raw_target_logprob": float(raw_target[owner]), "valid_catalogue_mass": None,
                        **retrieval_metrics(rank), "exact_top10_overlap": overlap, "target_lost10": lost,
                        "target_promoted10": promoted}
                    metrics.append(row)
                    if width == spec["main_beam_width"]:
                        summaries[scoring].update(retrieval_metrics(rank))
                        summaries[scoring]["cases"] += 1
                    predictions.extend({"run_id": run_id, "epoch": epoch, "case_id": case["case_id"],
                        "scoring": scoring, "search": "beam", "beam_width": width, "rank": r,
                        "item_id": item, "sid": list(sid), "log_score": score}
                        for r, ((sid, score), item) in enumerate(zip(group, items, strict=True), 1))
                store.write("predictions", predictions)
        store.write("case_metrics", metrics)
        metrics.clear()
        if begin % (spec["evaluation_batch_size"] * 100) == 0:
            _event("sid_study_evaluation", run=run_id, epoch=epoch, split=chosen[0]["split"],
                   completed=min(begin + len(chosen), len(cases)), total=len(cases))
    summary = {f"{name}_{metric}": float(values[metric] / values["cases"])
            for name, values in summaries.items() for metric in ("recall5", "recall10", "ndcg5", "ndcg10")}
    store.write("evaluations", [{"run_id": run_id, "epoch": epoch, "split": cases[0]["split"],
        "cases": len(cases), "exact_cases": sum(c["exact_panel"] for c in cases) if epoch in spec["exact_epochs"] else 0,
        "seconds": time.monotonic() - started, "metrics_json": json.dumps(summary, sort_keys=True)}])
    return summary


def train_study(model, index, users, cases, windows, config, store, run_id, output):
    import torch
    import torch.nn.functional as F
    spec = config["training"]
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=spec["precision"] == "fp16")
    by_user = {u.user_id: u for u in users}
    completed_examples, effective_updates, results = 0, 0, {}
    # A model RNG stream independent of sklearn, source construction and eval.
    torch.manual_seed(config["model_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["model_seed"])
    probe, total_attempts = None, 0
    if config["diagnostics"].get("training_probe"):
        from rhpr.gr.study_probe import TrainingProbe
        probe = TrainingProbe(config, output, store, run_id)
        probe.sample(model, index, 0, 0, 0)
    for epoch in range(1, spec["max_epochs"] + 1):
        model.train()
        started, loss_total, examples, skipped, attempted = time.monotonic(), 0., 0, 0, 0
        norms, context_counts, target_counts = [], Counter(), Counter()
        item_losses = np.zeros((index.num_items, index.sid_length), dtype=np.float64)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        import pyarrow.parquet as pq
        manifest = pq.read_table(windows[epoch]).to_pylist()
        for begin in range(0, len(manifest), spec["batch_size"]):
            chosen = manifest[begin:begin + spec["batch_size"]]
            contexts = [by_user[r["user_id"]].train[r["visible_start"]:r["end"] - 1] for r in chosen]
            target = [r["target_item"] for r in chosen]
            batch = make_batch(contexts, target, [r["user_id"] for r in chosen], index, spec["input_tokens"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=spec["precision"] == "fp16"):
                hidden = model(**batch)[:, :-1]
                levels = torch.stack([F.cross_entropy(head(hidden[:, level]).float(),
                    batch["future_ids"][:, level], reduction="none")
                    for level, head in enumerate(model.decoder.decoder_mlp)], dim=1)
                loss = levels.sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {run_id} epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip_norm"])
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            was_skipped = scaler.get_scale() < before
            if not was_skipped and not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite unscaled gradient without an AMP-skipped update")
            skipped += int(was_skipped)
            effective_updates += int(not was_skipped)
            attempted += 1
            total_attempts += 1
            if torch.isfinite(norm):
                norms.append(float(norm))
            count = len(chosen)
            loss_total += float(loss.detach()) * count
            examples += count
            for history in contexts:
                context_counts.update(history)
            target_counts.update(target)
            np.add.at(item_losses, target, levels.detach().cpu().numpy())
            if probe and total_attempts % probe.spec["interval_updates"] == 0:
                probe.sample(model, index, epoch, total_attempts, effective_updates)
        if not examples:
            raise ValueError("Empty training epoch")
        completed_examples += examples
        if probe:
            probe.sample(model, index, epoch, total_attempts, effective_updates)
        seconds = time.monotonic() - started
        store.write("exposures", [{"run_id": run_id, "epoch": epoch, "item_id": i,
            "context_count": context_counts[i], "target_count": target_counts[i],
            "target_loss_sum": float(item_losses[i].sum()), "target_level_loss_sums": item_losses[i].tolist()}
            for i in range(index.num_items)])
        evaluation_seconds = 0.
        if epoch % spec["validation_interval_epochs"] == 0:
            checkpoint = output / f"epoch-{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "effective_updates": effective_updates, "examples": completed_examples,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "config": config,
                "run_id": run_id, "sids": index.sids.copy(), "automatic_resume": False}, checkpoint)
            store.write("checkpoints", [{"run_id": run_id, "epoch": epoch, "optimizer_updates": effective_updates,
                "examples": completed_examples, "train_loss": loss_total / examples, "seconds": seconds,
                "path": str(checkpoint.relative_to(store.root)), "sha256": file_sha256(checkpoint)}])
            started_eval = time.monotonic()
            # Evaluation never advances the training/dropout RNG stream.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                validation = evaluate_study(model, [c for c in cases if c["split"] == "validation"],
                    index, store, run_id, epoch, config, output)
                results = {"validation_" + key: value for key, value in validation.items()}
                if epoch == spec["max_epochs"]:
                    test = evaluate_study(model, [c for c in cases if c["split"] == "test"],
                        index, store, run_id, epoch, config, output)
                    results.update({"test_" + key: value for key, value in test.items()})
            evaluation_seconds = time.monotonic() - started_eval
        record = {"run_id": run_id, "epoch": epoch, "examples": examples, "attempted_updates": attempted,
            "effective_updates": attempted - skipped, "skipped_updates": skipped, "mean_loss": loss_total / examples,
            "learning_rate": spec["learning_rate"], "scaler_scale": scaler.get_scale(),
            "grad_norm_mean": float(np.mean(norms)) if norms else None, "training_seconds": seconds,
            "evaluation_seconds": evaluation_seconds,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
        store.write("training_epochs", [record])
        store.flush()
        _event("sid_study_epoch", **record, **results)
    if probe:
        _write(output / "probe-summary.json", {"seconds": probe.seconds, "last_attempted_update": total_attempts,
            "fixed_cases": len(probe.rows), "rng_preserved": True, "optimizer_not_updated_by_probe": True})
    return results


def validate_study_config(config):
    spec, diagnostics = config["training"], config["diagnostics"]
    seeds = config["quantizer_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Quantizer seeds must be distinct nonnegative integers")
    from rhpr.gr.study_design import study_jobs
    study_jobs(config)
    if config.get("frozen_indices") is not None:
        if set(config["frozen_indices"]) != {str(seed) for seed in seeds}:
            raise ValueError("Frozen indices must cover exactly the quantizer seeds")
        for seed in seeds:
            source = config["frozen_indices"][str(seed)]
            if source["quantizer_seed"] != seed:
                raise ValueError("Frozen quantizer seed mismatch")
            for name in ("semantic-ids.npy", "quantizer-centroids.npy", "quantizer.json"):
                entry = source["files"][name]
                if not entry.get("path") or len(entry.get("sha256", "")) != 64:
                    raise ValueError("Frozen index requires paths and SHA256 hashes")
    if diagnostics.get("training_probe"):
        from rhpr.gr.study_probe import validate_probe
        validate_probe(diagnostics["training_probe"])
    for name in ("max_epochs", "validation_interval_epochs", "batch_size", "source_user_batch_size", "max_windows_per_source_batch", "context_items", "input_tokens"):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError(f"Invalid training {name}")
    if spec["max_epochs"] % spec["validation_interval_epochs"]:
        raise ValueError("Final epoch must have scheduled evaluation")
    if spec["precision"] not in {"fp16", "fp32"} or spec["device"] not in {"cuda", "cpu"}:
        raise ValueError("Study supports one GPU FP16/FP32, or CPU FP32 verification")
    if spec["precision"] == "fp16" and spec["device"] != "cuda":
        raise ValueError("FP16 training requires CUDA")
    if config["quantizer"]["levels"] != 3 or config["quantizer"]["codebook_size"] < 2:
        raise ValueError("Current study uses three semantic levels plus a collision suffix")
    if spec["input_tokens"] < 4 * max(spec["context_items"], diagnostics["context_items"]):
        raise ValueError("Visible context exceeds encoder input budget")
    for epoch in diagnostics["exact_epochs"]:
        if epoch < 1 or epoch > spec["max_epochs"] or epoch % spec["validation_interval_epochs"]:
            raise ValueError("Exact-search epochs must be scheduled checkpoints")
    for name in ("context_items", "evaluation_batch_size", "exact_chunk_size", "save_top_k", "main_beam_width", "exact_panel_size"):
        if type(diagnostics[name]) is not int or diagnostics[name] < 1:
            raise ValueError(f"Invalid diagnostics {name}")
    if config.get("execution"):
        from rhpr.gr.study_parallel import seed_assignments
        seed_assignments(config)


def train_one_seed(seed, index, users, cases, windows, config, store, source, output, initial_hash=None, *, run_id=None):
    """Independent model/RNG/optimizer; same implementation in serial/workers."""
    import torch
    run_id = run_id or f"rq-{seed}"
    folder = output / run_id
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(config["training"]["cpu_threads"])
    random.seed(config["model_seed"])
    np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    model = build_study_model(index, source, folder, config["generator"])
    try:
        store.write("artifacts", [{"path": str((folder / "generator.json").relative_to(output)),
            "sha256": file_sha256(folder / "generator.json"), "role": "generator_config"}])
        digest = state_digest(model.state_dict())
        if initial_hash is None:
            initial_hash = digest
            initial_file = f"initial-state-ms-{config['model_seed']}.pt" if config.get("model_seeds") is not None else "initial-state.pt"
            torch.save(model.state_dict(), output / initial_file)
        if digest != initial_hash:
            raise ValueError("Generator initial weights changed between quantizer seeds")
        store.write("runs", [{"run_id": run_id, "quantizer_seed": seed, "model_seed": config["model_seed"],
            "data_seed": config["data_seed"], "initial_state_sha256": digest, "config_json": json.dumps(config, sort_keys=True)}])
        for begin in range(0, len(cases), 1024):
            chosen = cases[begin:begin + 1024]
            batch = make_batch([c["visible_history"] for c in chosen], [c["target_item"] for c in chosen],
                [c["user_id"] for c in chosen], index, config["training"]["input_tokens"], "cpu")
            store.write("model_inputs", [{"run_id": run_id, "case_id": c["case_id"],
                "input_ids": batch["input_ids"][i].tolist(), "attention_mask": batch["attention_mask_encoder"][i].tolist()}
                for i, c in enumerate(chosen)])
        store.flush()
        torch.set_num_threads(config.get("execution", {}).get("cpu_threads_per_worker", config["training"]["cpu_threads"]))
        model.to(config["training"]["device"])
        result = train_study(model, index, users, cases, windows, config, store, run_id, folder)
        _write(folder / "result.json", result)
        store.write("artifacts", [{"path": str((folder / "result.json").relative_to(output)),
            "sha256": file_sha256(folder / "result.json"), "role": "seed_result"}])
        return initial_hash, result
    finally:
        store.flush()
        del model
        gc.collect()
        if config["training"]["device"] == "cuda":
            torch.cuda.empty_cache()


def run_sid_study(config, output):
    output = Path(output).resolve()
    if (output / "schema.json").exists():
        raise ValueError("Study output already exists; refusing duplicate evidence. Use a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "pipeline-status.json", {"status": "running"})
    try:
        result = _run_sid_study(config, output)
    except Exception as error:
        _write(output / "pipeline-status.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise
    _write(output / "pipeline-status.json", {"status": "complete"})
    return result


def _run_sid_study(config, output):
    """Runnable through the stable RHPR/Kaggle entry point; no external writes."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from rhpr.gr.genpas import activate_source
    from rhpr.gr.study_index import prepare_shared, window_manifest, record_index, compare_indices
    validate_study_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = config["training"]
    if spec["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU required; no silent CPU fallback")
    if config.get("execution"):
        from rhpr.gr.study_parallel import check_devices
        check_devices(config)
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from rhpr.offline_runtime import resolve_dataset_directory
    data_path = resolve_dataset_directory(config["data_path"])
    texts, users, original_ids, counts = load_beauty(data_path, config["data"])
    representation = config["representation"]
    source_values = Path(representation["path"])
    if representation.get("parent_artifacts"):
        from rhpr.gr.resources import verify_parent_artifacts
        parent = verify_parent_artifacts(representation["parent_artifacts"])
        source_values = parent / source_values.name
    if file_sha256(source_values) != representation["sha256"]:
        raise ValueError("Frozen item embedding digest mismatch")
    if file_sha256(data_path / "catalog.json") != representation["catalog_sha256"]:
        raise ValueError("Embedding catalogue identity/order mismatch")
    values = validate_item_embeddings(np.load(source_values, allow_pickle=False), len(texts))
    if values.shape[0] < config["quantizer"]["codebook_size"]:
        raise ValueError("More centroids than items")
    np.save(output / "item-embeddings.npy", values)
    _write(output / "study-config.json", config)
    _write(output / "representation.json", {**representation, "shape": list(values.shape),
        "frozen": True, "actual_sha256": file_sha256(output / "item-embeddings.npy")})
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("catalog.json", "sequences.json", "data-manifest.json"):
        if (data_path / name).is_file():
            shutil.copyfile(data_path / name, inputs / name)
    crossed = config.get("model_seeds") is not None
    _write(output / "protocol.json", {"counts": counts,
        "varying": "fixed SID index crossed with generator RNG seed" if crossed else "RQ-KMeans seed only",
        "initialization": "identical tensor hash within each paired model seed" if crossed else "identical state_dict tensor-content hash",
        "selection": "fixed final epoch, no early stopping",
        "sampling": "TIGER/GRID contiguous windows; sampled source-user groups; raw-item manifests shared by all runs",
        "history_padding": "right padding; complete context SIDs only; no attended target placeholder",
        "scorings": ["raw full-vocabulary AR", "valid-child locally renormalized AR"],
        "beam_ties": "lexicographic SID prefix", "exact_ties": "catalogue item ID",
        "candidate_universe": "entire static catalogue; no seen-item filtering",
        "test": "only fixed final epoch; descriptive exploration uses validation",
        "timestamp_status": "not supplied by normalized source; null, never inferred",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_name() if spec["device"] == "cuda" else "cpu"),
        "exact_published_reproduction": False})
    if config.get("execution"):
        protocol = json.loads((output / "protocol.json").read_text())
        protocol["execution"] = config["execution"]
        protocol["parallelism"] = "independent seeds in isolated processes; one visible GPU per worker"
        _write(output / "protocol.json", protocol)
    from importlib.metadata import version
    from rhpr.gr.genpas import GENPAS_REVISION
    _write(output / "input-provenance.json", {"source_revision": GENPAS_REVISION,
        "files": {name: file_sha256(inputs / name) for name in ("catalog.json", "sequences.json")},
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pyarrow", "transformers", "hydra-core")},
        "deterministic_algorithms": True, "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    source_path = Path(config["source_path"])
    if not source_path.exists():
        source_path = resolve_dataset_directory(source_path.parent) / source_path.name
    if source_path.is_file():
        from rhpr.gr.resources import unpack_grid
        source_path = unpack_grid(source_path, output / "verified-source")
    source = activate_source(source_path)
    store = StudyStore(output)
    cases = prepare_shared(store, texts, original_ids, users, context_items=config["diagnostics"]["context_items"],
        panel_size=config["diagnostics"]["exact_panel_size"], panel_seed=config["data_seed"])
    windows = {}
    for epoch in range(1, spec["max_epochs"] + 1):
        manifest = window_manifest(users, seed=config["data_seed"], epoch=epoch,
            source_size=spec["source_user_batch_size"], maximum=spec["max_windows_per_source_batch"],
            context_items=spec["context_items"])
        windows[epoch] = store.write("training_windows", manifest)
    del manifest
    from rhpr.gr.study_design import study_jobs, load_frozen_index
    jobs = study_jobs(config)
    initial_hash, indices, metrics = None, {}, {"runs": len(jobs)}
    # Fit and inspect ALL index realizations before training any generator.
    for seed in config["quantizer_seeds"]:
        run_id = f"rq-{seed}"
        folder = output / run_id
        folder.mkdir()
        if config.get("frozen_indices"):
            index, centers = load_frozen_index(values, config["quantizer"], config["frozen_indices"][str(seed)], folder)
        else:
            index, centers = fit_quantizer(values, config["quantizer"], seed, folder)
        for job in jobs:
            if job["quantizer_seed"] == seed:
                record_index(store, job["run_id"], index, users, cases, values, centers)
        indices[run_id] = index
        for name in ("semantic-ids.npy", "quantizer-centroids.npy", "quantizer.json"):
            store.write("artifacts", [{"path": str((folder / name).relative_to(output)),
                "sha256": file_sha256(folder / name), "role": "quantizer"}])
        if (folder / "frozen-index.json").is_file():
            store.write("artifacts", [{"path": str((folder / "frozen-index.json").relative_to(output)),
                "sha256": file_sha256(folder / "frozen-index.json"), "role": "frozen_index_provenance"}])
    compare_indices(store, {job["run_id"]: indices[f"rq-{job['quantizer_seed']}"] for job in jobs})
    if config["diagnostics"].get("training_probe"):
        from rhpr.gr.study_probe import prepare_probe_cases
        prepare_probe_cases(users, cases, windows, config, output, store)
    store.inventory()
    if config.get("execution"):
        from rhpr.gr.study_parallel import launch_workers
        # Construct the reference on CPU once, independently of worker timing.
        initial_hashes = {}
        for model_seed in config.get("model_seeds", [config["model_seed"]]):
            random.seed(model_seed)
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            reference = build_study_model(indices[f"rq-{config['quantizer_seeds'][0]}"], source,
                output / (f"initial-generator-ms-{model_seed}" if crossed else "initial-generator"), config["generator"])
            initial_hashes[str(model_seed)] = state_digest(reference.state_dict())
            torch.save(reference.state_dict(), output / (f"initial-state-ms-{model_seed}.pt" if crossed else "initial-state.pt"))
            del reference
        initial_hash = initial_hashes if crossed else initial_hashes[str(config["model_seed"])]
        store.flush()
        try:
            metrics = launch_workers(config, output, source_path, windows, initial_hash)
        finally:
            store.inventory()
    else:
        initial_hashes = {}
        for job in jobs:
            local_config = {**config, "model_seed": job["model_seed"]}
            seed = job["quantizer_seed"]
            reference = initial_hashes.get(job["model_seed"])
            digest, result = train_one_seed(seed, indices[f"rq-{seed}"], users, cases, windows,
                local_config, store, source, output, reference, run_id=job["run_id"])
            initial_hashes[job["model_seed"]] = digest
            metrics.update({job["run_id"].replace("-", "_") + "_" + key: value for key, value in result.items()})
            _write(output / "partial-metrics.json", metrics)
            store.inventory()
    initial_files = [f"initial-state-ms-{seed}.pt" for seed in config["model_seeds"]] if crossed else ["initial-state.pt"]
    for name in ("item-embeddings.npy", *initial_files, "study-config.json", "protocol.json", "representation.json", "input-provenance.json",
                 "inputs/catalog.json", "inputs/sequences.json"):
        store.write("artifacts", [{"path": name, "sha256": file_sha256(output / name), "role": name.split(".")[0]}])
    store.inventory()
    # Parquet is sufficient on the training machine; DuckDB is optional there.
    try:
        with store.connect(output / "study.duckdb"):
            pass
    except ImportError:
        pass
    return metrics
