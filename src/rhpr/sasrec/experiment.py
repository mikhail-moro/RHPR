from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rhpr.sasrec.data import (
    SequenceSplit,
    YambdaSequences,
    cold_target_fraction,
    load_yambda_sequences,
    make_training_pair,
    resolve_yambda_parquet_path,
    test_example,
    validation_example,
    training_gap_ids,
    evaluation_gap_ids,
)
from rhpr.sasrec.benchmark import benchmark_inference, score_catalogue
from rhpr.sasrec.diagnostics import context_hash
from rhpr.sasrec.losses import next_item_loss
from rhpr.sasrec.model import SASRecModel


class _TrainingDataset(Dataset[Any]):
    def __init__(self, splits: Sequence[SequenceSplit], max_seq_len: int, *,
                 use_time_features: bool = False, time_unit: float = 1.0,
                 num_gap_buckets: int = 32) -> None:
        self.splits = tuple(split for split in splits if split.train.size >= 2)
        self.max_seq_len = max_seq_len
        self.use_time_features = use_time_features
        self.time_unit = time_unit
        self.num_gap_buckets = num_gap_buckets
        if not self.splits:
            raise ValueError("no user has enough training interactions for shifted targets")

    def __len__(self) -> int:
        return len(self.splits)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs, targets = make_training_pair(self.splits[index].train, self.max_seq_len)
        gaps = (training_gap_ids(self.splits[index], self.max_seq_len,
                                 time_unit=self.time_unit, num_buckets=self.num_gap_buckets)
                if self.use_time_features else np.zeros_like(inputs))
        return inputs, targets, gaps


class _EvaluationDataset(Dataset[Any]):
    def __init__(self, splits: Sequence[SequenceSplit], max_seq_len: int, split_name: str, *,
                 use_time_features: bool = False, time_unit: float = 1.0,
                 num_gap_buckets: int = 32) -> None:
        if split_name == "validation":
            self.example: Callable[[SequenceSplit, int], tuple[np.ndarray, int]] = validation_example
        elif split_name == "test":
            self.example = test_example
        else:
            raise ValueError("split_name must be validation or test")
        self.splits = tuple(splits)
        self.max_seq_len = max_seq_len
        self.split_name = split_name
        self.use_time_features = use_time_features
        self.time_unit = time_unit
        self.num_gap_buckets = num_gap_buckets

    def __len__(self) -> int:
        return len(self.splits)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, np.ndarray, int]:
        context, target = self.example(self.splits[index], self.max_seq_len)
        gaps = (evaluation_gap_ids(self.splits[index], self.max_seq_len, self.split_name,
                                   time_unit=self.time_unit, num_buckets=self.num_gap_buckets)
                if self.use_time_features else np.zeros_like(context))
        return context, target, gaps, self.splits[index].user_id


def _pad_sequences(sequences: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    item_ids = torch.zeros((len(sequences), max_length), dtype=torch.long)
    valid_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        item_ids[row, :length] = torch.from_numpy(sequence)
        valid_mask[row, :length] = True
    return item_ids, valid_mask


def _collate_training(
    batch: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets, gaps = zip(*batch, strict=True)
    item_ids, valid_mask = _pad_sequences(inputs)
    padded_targets, _ = _pad_sequences(targets)
    padded_gaps, _ = _pad_sequences(gaps)
    return item_ids, valid_mask, padded_targets, padded_gaps


def _collate_evaluation(batch: Sequence[Any]) -> tuple[torch.Tensor, ...]:
    contexts, targets, gaps, user_ids = zip(*batch, strict=True)
    item_ids, valid_mask = _pad_sequences(contexts)
    padded_gaps, _ = _pad_sequences(gaps)
    return (item_ids, valid_mask, torch.tensor(targets, dtype=torch.long),
            padded_gaps, torch.tensor(user_ids, dtype=torch.long))


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    collate_fn: Callable[..., Any],
    device: torch.device,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def _full_catalog_ndcg(
    model: SASRecModel,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    k: int,
    item_chunk_size: int,
    amp: bool,
    phase: str,
    max_batches: int | None = None,
    progress_interval_batches: int = 0,
    details: dict[str, Any] | None = None,
    prediction_rows: list[dict[str, Any]] | None = None,
    model_label: str = "full",
    seed: int = 42,
) -> tuple[float, int]:
    model.eval()
    total = 0.0
    total_hits = 0
    observations = 0
    groups = {label: [0.0, 0, 0] for label in ("le128", "129_512", "513_1028")}

    started = time.perf_counter()
    batches = islice(loader, max_batches) if max_batches is not None else loader
    for batch_index, batch in enumerate(batches, start=1):
        item_ids, valid_mask, targets = batch[:3]
        raw_items = item_ids
        gap_ids = batch[3] if len(batch) > 3 else None
        user_ids = batch[4] if len(batch) > 4 else None
        item_ids = item_ids.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if gap_ids is not None and model.use_time_features:
            gap_ids = gap_ids.to(device, non_blocking=True)
        else:
            gap_ids = None
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            states = model.last_state(item_ids, valid_mask, gap_ids)
        _, best_ids = score_catalogue(model, states, k=k, item_chunk_size=item_chunk_size)

        matches = best_ids == targets[:, None]
        found = matches.any(dim=1)
        ranks = matches.to(torch.int64).argmax(dim=1) + 1
        gains = torch.where(found, 1.0 / torch.log2(ranks.float() + 1.0), 0.0)
        total += float(gains.sum().item())
        total_hits += int(found.sum().item())
        lengths = valid_mask.sum(dim=1).cpu().tolist()
        gains_cpu, hits_cpu = gains.cpu().tolist(), found.cpu().tolist()
        for row_index, length in enumerate(lengths):
            label = "le128" if length <= 128 else "129_512" if length <= 512 else "513_1028"
            group = groups[label]
            group[0] += gains_cpu[row_index]
            group[1] += int(hits_cpu[row_index])
            group[2] += 1
        if prediction_rows is not None:
            if user_ids is None:
                raise ValueError("per-user evaluation requires stable user IDs")
            top_ids, rank_values = best_ids.cpu().tolist(), ranks.cpu().tolist()
            target_values = targets.cpu().tolist()
            for row_index, user_id in enumerate(user_ids.tolist()):
                prediction_rows.append({
                    "model": model_label, "time_enabled": model.use_time_features,
                    "seed": seed, "split": phase, "user_id": int(user_id),
                    "context_hash": context_hash(raw_items[row_index, :lengths[row_index]].numpy()),
                    "target": int(target_values[row_index]), "top10": top_ids[row_index],
                    "rank_in_top10": int(rank_values[row_index]) if hits_cpu[row_index] else None,
                    "ndcg_at_10": float(gains_cpu[row_index]),
                    "recall_at_10": int(hits_cpu[row_index]),
                })
        observations += int(targets.shape[0])
        if progress_interval_batches and batch_index % progress_interval_batches == 0:
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "batches_done": batch_index,
                        "batches_total": min(len(loader), max_batches) if max_batches is not None else len(loader),
                        "examples_done": observations,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if observations == 0:
        raise ValueError("evaluation split is empty")
    if details is not None:
        details.update({"recall_at_10": total_hits / observations, "groups": {}})
        for label, (gain_sum, hit_sum, count) in groups.items():
            details["groups"][label] = {"count": count}
            if count:
                details["groups"][label].update({"ndcg_at_10": gain_sum / count,
                                                "recall_at_10": hit_sum / count})
    return total / observations, observations


def _positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _mixed_depth_settings(
    training: dict[str, Any], model: dict[str, Any], evaluation: dict[str, Any],
) -> tuple[torch.Tensor, int] | None:
    settings = training.get("mixed_depth")
    if settings is None:
        return None
    if not isinstance(settings, dict) or model.get("compressor_type") != "recursive_pair_mlp":
        raise ValueError("mixed_depth requires a recursive_pair_mlp model and an object configuration")
    probabilities = settings.get("probabilities")
    if not isinstance(probabilities, list) or not 4 <= len(probabilities) <= 7:
        raise ValueError("mixed_depth probabilities must contain four to seven depth probabilities")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in probabilities):
        raise ValueError("all mixed_depth probabilities must be finite and positive")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("mixed_depth probabilities must sum to one")
    interval = settings.get("validation_interval", 5)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("mixed_depth validation_interval must be a positive integer")
    if evaluation.get("evaluate_test", True):
        raise ValueError("mixed_depth screening must leave test unevaluated")
    return torch.tensor(probabilities, dtype=torch.float64, device="cpu"), interval


def _sample_mixed_depth(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    """Use a private CPU stream, independent of shuffle, dropout, and negatives."""
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _validation_interval(training: dict[str, Any]) -> int:
    interval = training.get("validation_interval", 1)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("training validation_interval must be a positive integer")
    return interval


def _save_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    temporary = output_dir / "history.json.tmp"
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "history.json")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_yambda_experiment(config: dict[str, Any], output_dir: Path, *,
                          dataset: YambdaSequences | None = None) -> dict[str, int | float | str | bool]:
    """Train SASRec and evaluate leave-two-out targets against the full catalogue."""

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    _seed_everything(seed, deterministic)

    data_config = dict(config.get("data", {}))
    model_config = dict(config.get("model", {}))
    training_config = dict(config.get("training", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    loss_config = dict(config.get("loss", {}))
    mixed_depth = _mixed_depth_settings(training_config, model_config, evaluation_config)
    validation_interval = mixed_depth[1] if mixed_depth is not None else _validation_interval(training_config)
    trained_depths = range(len(mixed_depth[0])) if mixed_depth is not None else range(0)
    if mixed_depth is not None and loss_config.get("tail_size") != model_config.get("tail_size", 128):
        raise ValueError("mixed_depth requires the common loss tail to equal the protected model tail")
    use_time_features = bool(model_config.get("use_time_features", False))
    num_gap_buckets = _positive_int(model_config, "num_gap_buckets", 33)
    time_unit = float(data_config.get("gap_time_unit", 1.0))
    if time_unit <= 0:
        raise ValueError("gap_time_unit must be positive")
    feature_settings = {"use_time_features": use_time_features, "time_unit": time_unit,
                        "num_gap_buckets": num_gap_buckets - 1}
    data_path = resolve_yambda_parquet_path(
        Path(str(data_config.get("path", "/kaggle/input/yambda-50m/listens.parquet")))
    )
    max_seq_len = _positive_int(model_config, "max_seq_len", 1028)

    data_started = time.perf_counter()
    print(json.dumps({"phase": "data_loading", "path": str(data_path)}, sort_keys=True), flush=True)
    if dataset is None:
        dataset = load_yambda_sequences(data_path, timestamp_unit=str(data_config.get("timestamp_unit", "seconds")))
    if use_time_features and any(split.timestamps is None for split in dataset.splits):
        raise ValueError("time features require timestamps for every sequence")
    data_seconds = time.perf_counter() - data_started
    print(
        json.dumps(
            {
                "phase": "data_loaded",
                "elapsed_seconds": data_seconds,
                "num_users": len(dataset.splits),
                "num_items": dataset.num_items,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    device_name = str(training_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"

    d_model = _positive_int(model_config, "d_model", 256)
    n_heads = _positive_int(model_config, "n_heads", 4)
    if d_model // n_heads < 32:
        raise ValueError("d_model / n_heads must be at least 32")
    depth_arguments = ({"compression_depth": model_config.get("compression_depth", 1)}
                       if model_config.get("compressor_type") == "recursive_pair_mlp" else {})
    model = SASRecModel(
        num_items=dataset.num_items,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=_positive_int(model_config, "n_layers", 4),
        ffn_multiplier=_positive_int(model_config, "ffn_multiplier", 4),
        dropout=float(model_config.get("dropout", 0.2)),
        compressor_type=str(model_config.get("compressor_type", "identity")),
        tail_size=_positive_int(model_config, "tail_size", 128),
        use_time_features=use_time_features,
        num_gap_buckets=num_gap_buckets,
        **depth_arguments,
    ).to(device)

    physical_batch_size = _positive_int(training_config, "physical_batch_size", 4)
    effective_batch_size = _positive_int(training_config, "effective_batch_size", 128)
    if effective_batch_size % physical_batch_size:
        raise ValueError("effective_batch_size must be divisible by physical_batch_size")
    accumulation_steps = effective_batch_size // physical_batch_size
    workers = int(training_config.get("workers", 2))
    if workers < 0:
        raise ValueError("workers must not be negative")

    training_loader = _loader(
        _TrainingDataset(dataset.splits, max_seq_len, **feature_settings),
        batch_size=physical_batch_size,
        shuffle=True,
        workers=workers,
        seed=seed,
        collate_fn=_collate_training,
        device=device,
    )
    max_training_batches = training_config.get("max_batches_per_epoch")
    if max_training_batches is not None:
        max_training_batches = _positive_int(training_config, "max_batches_per_epoch", 1)
    training_batches = min(len(training_loader), max_training_batches or len(training_loader))
    training_progress_interval = int(training_config.get("progress_interval_batches", 100))
    if training_progress_interval < 0:
        raise ValueError("progress_interval_batches must not be negative")
    evaluation_batch_size = _positive_int(evaluation_config, "batch_size", 32)
    validation_loader = _loader(
        _EvaluationDataset(dataset.splits, max_seq_len, "validation", **feature_settings),
        batch_size=evaluation_batch_size,
        shuffle=False,
        workers=workers,
        seed=seed,
        collate_fn=_collate_evaluation,
        device=device,
    )
    max_validation_batches = evaluation_config.get("max_validation_batches")
    if max_validation_batches is not None:
        max_validation_batches = _positive_int(evaluation_config, "max_validation_batches", 1)
    evaluation_progress_interval = int(evaluation_config.get("progress_interval_batches", 20))
    if evaluation_progress_interval < 0:
        raise ValueError("progress_interval_batches must not be negative")
    test_loader = _loader(
        _EvaluationDataset(dataset.splits, max_seq_len, "test", **feature_settings),
        batch_size=evaluation_batch_size,
        shuffle=False,
        workers=workers,
        seed=seed,
        collate_fn=_collate_evaluation,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_name = str(loss_config.get("name", "bce"))
    num_negatives = _positive_int(loss_config, "num_negatives", 1)
    loss_tail_size = loss_config.get("tail_size")
    if loss_tail_size is not None:
        loss_tail_size = _positive_int(loss_config, "tail_size", 128)
    negative_generator = torch.Generator(device=device).manual_seed(seed + 173)
    depth_generator = torch.Generator(device="cpu").manual_seed(seed + 271)
    max_epochs = _positive_int(training_config, "max_epochs", 15)
    patience = _positive_int(training_config, "early_stopping_patience", 3)
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    k = _positive_int(evaluation_config, "k", 10)
    if k != 10:
        raise ValueError("this experiment contract reports NDCG@10, so evaluation.k must be 10")
    item_chunk_size = _positive_int(evaluation_config, "item_chunk_size", 65_536)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best-model.pt"
    history: list[dict[str, Any]] = []
    best_validation = -math.inf
    best_epoch = 0
    train_loss_at_best = math.inf
    stale_epochs = 0
    training_seconds = 0.0
    validation_seconds = 0.0
    validation_examples = 0
    training_peak_allocated_gib = 0.0
    training_peak_reserved_gib = 0.0
    validation_details: dict[str, Any] = {}
    best_validation_recall = 0.0
    best_groups: dict[str, Any] = {}
    best_depth_validation: dict[str, Any] = {}
    depth_totals = {str(depth): {"batches": 0, "users": 0, "loss_sum": 0.0,
                   "encoder_padded_length_sum": 0, "encoder_padded_length_max": 0,
                   "encoder_padded_tokens": 0, "encoder_valid_tokens": 0}
                    for depth in trained_depths}

    for epoch in range(1, max_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        print(
            json.dumps(
                {"phase": "training_started", "epoch": epoch, "batches_total": training_batches},
                sort_keys=True,
            ),
            flush=True,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        epoch_depth_stats = {depth: {key: 0.0 if key == "loss_sum" else 0 for key in values}
                             for depth, values in depth_totals.items()}
        for batch_index, (item_ids, valid_mask, targets, gap_ids) in enumerate(
            islice(training_loader, training_batches), start=1
        ):
            deadline = training_config.get("deadline_unix_seconds")
            if deadline is not None and batch_index % 32 == 1 and time.time() >= float(deadline):
                raise TimeoutError("search training budget exhausted")
            item_ids = item_ids.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            gap_ids = gap_ids.to(device, non_blocking=True) if use_time_features else None
            if mixed_depth is not None:
                depth = _sample_mixed_depth(mixed_depth[0], depth_generator)
                model.set_compression_depth(depth)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                encoded, compressed_mask, source_positions = model.encode(item_ids, valid_mask, gap_ids)
                loss = next_item_loss(
                    name=loss_name,
                    model=model,
                    encoded=encoded,
                    valid_mask=compressed_mask,
                    source_positions=source_positions,
                    targets=targets,
                    num_negatives=num_negatives,
                    loss_tail_size=loss_tail_size,
                    original_lengths=valid_mask.sum(dim=1),
                    negative_generator=negative_generator,
                )
            if mixed_depth is not None:
                batch_loss = float(loss.detach().item())
                if not math.isfinite(batch_loss):
                    raise FloatingPointError(f"non-finite mixed-depth loss at epoch {epoch}, batch {batch_index}, depth {depth}")
            group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
            group_size = min(accumulation_steps, training_batches - group_start + 1)
            scaler.scale(loss / group_size).backward()
            if batch_index % accumulation_steps == 0 or batch_index == training_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if mixed_depth is None:
                batch_loss = float(loss.detach().item())
            running_loss += batch_loss
            if mixed_depth is not None:
                stats = epoch_depth_stats[str(depth)]
                stats["batches"] += 1
                stats["users"] += int(item_ids.shape[0])
                stats["loss_sum"] += batch_loss
                stats["encoder_padded_length_sum"] += int(encoded.shape[1])
                stats["encoder_padded_length_max"] = max(stats["encoder_padded_length_max"], int(encoded.shape[1]))
                stats["encoder_padded_tokens"] += int(encoded.shape[0] * encoded.shape[1])
                stats["encoder_valid_tokens"] += int(compressed_mask.sum().item())
            batches += 1
            if training_progress_interval and batch_index % training_progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "epoch": epoch,
                            "batches_done": batch_index,
                            "batches_total": training_batches,
                            "elapsed_seconds": time.perf_counter() - train_started,
                            "mean_loss": running_loss / batches,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            # Do not retain the final training activations during evaluation/benchmark.
            del item_ids, valid_mask, targets, gap_ids, encoded, compressed_mask, source_positions, loss

        train_loss = running_loss / batches
        epoch_training_seconds = time.perf_counter() - train_started
        if device.type == "cuda":
            training_peak_allocated_gib = max(
                training_peak_allocated_gib, torch.cuda.max_memory_allocated(device) / 2**30
            )
            training_peak_reserved_gib = max(
                training_peak_reserved_gib, torch.cuda.max_memory_reserved(device) / 2**30
            )
        training_seconds += epoch_training_seconds
        if mixed_depth is not None:
            for depth, values in epoch_depth_stats.items():
                for key, value in values.items():
                    if key == "encoder_padded_length_max":
                        depth_totals[depth][key] = max(depth_totals[depth][key], value)
                    else:
                        depth_totals[depth][key] += value
                values["mean_loss"] = values["loss_sum"] / values["batches"] if values["batches"] else None
            entry: dict[str, Any] = {
                "epoch": epoch, "train_loss": train_loss,
                "training_seconds": epoch_training_seconds, "validation_seconds": 0.0,
                "validation_performed": False, "mixed_depth_training": epoch_depth_stats,
            }
            if epoch % mixed_depth[1] == 0 or epoch == max_epochs:
                validation_started = time.perf_counter()
                depth_validation: dict[str, Any] = {}
                for depth in trained_depths:
                    model.set_compression_depth(depth)
                    print(json.dumps({"phase": "depth_validation_started", "epoch": epoch,
                                      "compression_depth": depth}), flush=True)
                    details: dict[str, Any] = {}
                    depth_started = time.perf_counter()
                    ndcg, examples = _full_catalog_ndcg(
                        model, validation_loader, device=device, k=k, item_chunk_size=item_chunk_size,
                        amp=amp, phase="validation", max_batches=max_validation_batches,
                        progress_interval_batches=evaluation_progress_interval, details=details,
                    )
                    if depth and examples != validation_examples:
                        raise ValueError("mixed depth validation modes must evaluate the same users")
                    validation_examples = examples
                    depth_validation[str(depth)] = {"ndcg_at_10": ndcg,
                        "recall_at_10": details["recall_at_10"], "examples": examples,
                        "seconds": time.perf_counter() - depth_started, "groups": details["groups"]}
                    entry[f"depth_{depth}_validation_ndcg_at_10"] = ndcg
                    entry[f"depth_{depth}_validation_recall_at_10"] = details["recall_at_10"]
                validation_ndcg = sum(value["ndcg_at_10"] for value in depth_validation.values()) / len(trained_depths)
                validation_recall = sum(value["recall_at_10"] for value in depth_validation.values()) / len(trained_depths)
                elapsed = time.perf_counter() - validation_started
                validation_seconds += elapsed
                entry.update({"validation_performed": True, "validation_seconds": elapsed,
                    "validation_ndcg_at_10": validation_ndcg, "validation_recall_at_10": validation_recall,
                    "validation_mean_ndcg_at_10": validation_ndcg,
                    "validation_mean_recall_at_10": validation_recall, "validation_depths": depth_validation})
                if validation_ndcg > best_validation + min_delta:
                    best_validation, best_epoch = validation_ndcg, epoch
                    train_loss_at_best, best_validation_recall = train_loss, validation_recall
                    best_depth_validation = depth_validation
                    torch.save(model.state_dict(), checkpoint_path)
            history.append(entry)
            _save_history(output_dir, history)
            print(json.dumps(entry, sort_keys=True), flush=True)
            # Mixed screening always completes its configured epoch budget.
            continue

        if epoch % validation_interval != 0 and epoch != max_epochs:
            history.append({"epoch": epoch, "train_loss": train_loss,
                "training_seconds": epoch_training_seconds, "validation_seconds": 0.0,
                "validation_performed": False})
            _save_history(output_dir, history)
            print(json.dumps(history[-1], sort_keys=True), flush=True)
            continue

        validation_started = time.perf_counter()
        print(json.dumps({"phase": "validation_started", "epoch": epoch}, sort_keys=True), flush=True)
        validation_ndcg, validation_examples = _full_catalog_ndcg(
            model,
            validation_loader,
            device=device,
            k=k,
            item_chunk_size=item_chunk_size,
            amp=amp,
            phase="validation",
            max_batches=max_validation_batches,
            progress_interval_batches=evaluation_progress_interval,
            details=validation_details,
        )
        epoch_validation_seconds = time.perf_counter() - validation_started
        validation_seconds += epoch_validation_seconds
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_ndcg_at_10": validation_ndcg,
                "validation_recall_at_10": validation_details["recall_at_10"],
                "training_seconds": epoch_training_seconds,
                "validation_seconds": epoch_validation_seconds,
                "validation_performed": True,
            }
        )
        _save_history(output_dir, history)
        print(json.dumps(history[-1], sort_keys=True), flush=True)

        if validation_ndcg > best_validation + min_delta:
            best_validation = validation_ndcg
            best_epoch = epoch
            train_loss_at_best = train_loss
            best_validation_recall = validation_details["recall_at_10"]
            best_groups = validation_details["groups"]
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    # Optimizer buffers are not part of inference memory or latency.
    del optimizer, scaler
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate_test = bool(evaluation_config.get("evaluate_test", True))
    diagnostics_enabled = bool(config.get("diagnostics", {}).get("enabled", False))
    benchmark_config = dict(config.get("benchmark", {}))
    benchmark_enabled = bool(benchmark_config.get("enabled", False))
    prediction_rows: list[dict[str, Any]] = []
    mixed_metrics: dict[str, int | float | str | bool] = {}
    label = {"pair_mlp": "pairmlp", "tail_only": "recent128"}.get(
        str(model_config.get("compressor_type", "identity")), "full")
    checksum = _checkpoint_sha256(checkpoint_path)
    if mixed_depth is not None or evaluate_test or diagnostics_enabled or benchmark_enabled:
        best_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best_state)
        del best_state
    if mixed_depth is not None:
        mixed_metrics.update({"mixed_depth_training": True, "early_stopping_enabled": False,
            "checkpoint_selection_metric": "validation_mean_ndcg_at_10",
            "validation_mean_ndcg_at_10": best_validation,
            "validation_mean_recall_at_10": best_validation_recall,
            "mixed_depth_validation_interval": mixed_depth[1], "checkpoint_sha256": checksum,
            "test_evaluated": False})
        contexts: list[np.ndarray] = []
        gaps = None
        if benchmark_enabled:
            sample_size = int(benchmark_config.get("real_sample_size", 200))
            selected = np.random.default_rng(seed).choice(len(dataset.splits),
                        size=min(sample_size, len(dataset.splits)), replace=False)
            contexts = [validation_example(dataset.splits[int(i)], max_seq_len)[0] for i in selected]
            if use_time_features:
                gaps = [evaluation_gap_ids(dataset.splits[int(i)], max_seq_len, "validation",
                        time_unit=time_unit, num_buckets=num_gap_buckets - 1) for i in selected]
        final_validation_seconds = 0.0
        for depth in trained_depths:
            model.set_compression_depth(depth)
            label = f"depth_{depth}"
            directory = output_dir / "runs" / label
            directory.mkdir(parents=True, exist_ok=True)
            details = {}
            first_row = len(prediction_rows)
            started = time.perf_counter()
            print(json.dumps({"phase": "depth_predictions_started", "compression_depth": depth,
                              "best_epoch": best_epoch}), flush=True)
            ndcg, examples = _full_catalog_ndcg(
                model, validation_loader, device=device, k=k, item_chunk_size=item_chunk_size,
                amp=amp, phase="validation", progress_interval_batches=evaluation_progress_interval,
                details=details, prediction_rows=prediction_rows, model_label=label, seed=seed,
            )
            elapsed = time.perf_counter() - started
            final_validation_seconds += elapsed
            for row in prediction_rows[first_row:]:
                row["compression_depth"] = depth
            values: dict[str, int | float | str | bool] = {
                "compression_depth": depth, "best_epoch": best_epoch,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": checksum,
                "checkpoint_selection_metric": "validation_mean_ndcg_at_10",
                "validation_ndcg_at_10": ndcg, "validation_recall_at_10": details["recall_at_10"],
                "validation_examples": examples, "final_validation_seconds": elapsed,
            }
            for group, group_values in details["groups"].items():
                for key, value in group_values.items():
                    values[f"validation_{group}_{key}"] = value
            if benchmark_enabled:
                # Each forced depth warms its own shapes; do not inherit depth0's free cache.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                options = dict(benchmark_config)
                options.update({"seed": seed, "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checksum, "compression_depth": depth,
                    "revision": os.environ.get("GITHUB_SHA", "local")})
                print(json.dumps({"phase": "benchmark_started", "model": label,
                                  "compression_depth": depth}), flush=True)
                values.update(benchmark_inference(model, contexts, device=device, amp=amp,
                    item_chunk_size=item_chunk_size, config=options, output_dir=directory, k=k, gap_ids=gaps))
            (directory / "metrics.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
            for key, value in values.items():
                mixed_metrics[f"{label}_{key}"] = value
        mixed_metrics["final_validation_seconds"] = final_validation_seconds
        for depth, values in depth_totals.items():
            for key, value in values.items():
                mixed_metrics[f"depth_{depth}_training_{key}"] = value
            if values["batches"]:
                mixed_metrics[f"depth_{depth}_training_mean_loss"] = values["loss_sum"] / values["batches"]
                mixed_metrics[f"depth_{depth}_training_encoder_padded_length_mean"] = (
                    values["encoder_padded_length_sum"] / values["batches"])
            mixed_metrics[f"depth_{depth}_training_probability"] = float(mixed_depth[0][int(depth)].item())
        (output_dir / "mixed-depth-selection.json").write_text(json.dumps({
            "criterion": "validation_mean_ndcg_at_10", "best_epoch": best_epoch,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checksum,
            "validation_mean_ndcg_at_10": best_validation,
            "validation_mean_recall_at_10": best_validation_recall,
            "validation_depths": best_depth_validation,
            "training_depths": depth_totals,
        }, indent=2) + "\n", encoding="utf-8")
    if diagnostics_enabled and mixed_depth is None:
        _full_catalog_ndcg(
            model, validation_loader, device=device, k=k, item_chunk_size=item_chunk_size,
            amp=amp, phase="validation", max_batches=max_validation_batches,
            prediction_rows=prediction_rows, model_label=label, seed=seed,
        )
    test_ndcg = 0.0
    test_seconds = 0.0
    test_details: dict[str, Any] = {}
    if evaluate_test:
        test_started = time.perf_counter()
        test_ndcg, _ = _full_catalog_ndcg(
            model,
            test_loader,
            device=device,
            k=k,
            item_chunk_size=item_chunk_size,
            amp=amp,
            phase="test",
            progress_interval_batches=evaluation_progress_interval,
            details=test_details,
            prediction_rows=prediction_rows if diagnostics_enabled else None,
            model_label=label, seed=seed,
        )
        test_seconds = time.perf_counter() - test_started
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "model-config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if diagnostics_enabled or mixed_depth is not None:
        import polars as pl
        pl.DataFrame(prediction_rows, infer_schema_length=None).write_parquet(output_dir / "predictions.parquet")
    benchmark_metrics: dict[str, Any] = {}
    if benchmark_enabled and mixed_depth is None:
        benchmark_config.update({"seed": seed, "checkpoint": str(checkpoint_path),
                                 "checkpoint_sha256": checksum,
                                 "revision": os.environ.get("GITHUB_SHA", "local")})
        sample_size = int(benchmark_config.get("real_sample_size", 200))
        selected = np.random.default_rng(seed).choice(len(dataset.splits),
                    size=min(sample_size, len(dataset.splits)), replace=False)
        contexts = [validation_example(dataset.splits[int(i)], max_seq_len)[0] for i in selected]
        gaps = ([evaluation_gap_ids(dataset.splits[int(i)], max_seq_len, "validation",
                                   time_unit=time_unit, num_buckets=num_gap_buckets - 1) for i in selected]
                if use_time_features else None)
        print(json.dumps({"phase": "benchmark_started", "model": label, "time": use_time_features}), flush=True)
        benchmark_metrics = benchmark_inference(model, contexts, device=device, amp=amp,
                             item_chunk_size=item_chunk_size, config=benchmark_config,
                             output_dir=output_dir, k=k, gap_ids=gaps)
    if not bool(training_config.get("keep_checkpoint", False)) and mixed_depth is None:
        checkpoint_path.unlink()

    metrics: dict[str, int | float | str | bool] = {
        "experiment": str(config.get("name", "yambda-sasrec")),
        "seed": seed,
        "num_users": len(dataset.splits),
        "num_training_users": len(training_loader.dataset),
        "num_items": dataset.num_items,
        "data_path": str(data_path),
        "data_seconds": data_seconds,
        "training_batches_per_epoch": training_batches,
        "training_is_subset": training_batches < len(training_loader),
        "validation_examples": validation_examples,
        "validation_is_subset": max_validation_batches is not None and max_validation_batches < len(validation_loader),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "train_loss_at_best": train_loss_at_best,
        "validation_ndcg_at_10": best_validation,
        "validation_recall_at_10": best_validation_recall,
        "validation_cold_target_fraction": cold_target_fraction(dataset.splits, target="validation"),
        "training_seconds": training_seconds,
        "training_examples_per_second": (len(history) * min(len(training_loader.dataset),
                                           training_batches * physical_batch_size)) / training_seconds,
        "validation_seconds": validation_seconds,
        "training_peak_allocated_gib": training_peak_allocated_gib,
        "training_peak_reserved_gib": training_peak_reserved_gib,
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "training_examples_per_epoch": min(len(training_loader.dataset), training_batches * physical_batch_size),
        "loss": loss_name,
        "num_negatives": num_negatives,
        "amp": amp,
        "device": str(device),
        "compressor_type": str(model_config.get("compressor_type", "identity")),
        "time_features": use_time_features,
        "loss_tail_size": loss_tail_size or 0,
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "best_at_epoch_limit": best_epoch == max_epochs,
        "checkpoint_sha256": checksum,
        "test_evaluated": evaluate_test,
        "validation_interval": validation_interval,
        "checkpoint_selection_metric": "validation_ndcg_at_10",
        "early_stopping_enabled": True,
        **benchmark_metrics,
        **mixed_metrics,
    }
    if evaluate_test:
        metrics.update(
            {
                "test_ndcg_at_10": test_ndcg,
                "test_recall_at_10": test_details["recall_at_10"],
                "test_cold_target_fraction": cold_target_fraction(dataset.splits, target="test"),
                "test_seconds": test_seconds,
            }
        )
    for phase, groups in (("validation", best_groups), ("test", test_details.get("groups", {}))):
        for group, values in groups.items():
            for key, value in values.items():
                metrics[f"{phase}_{group}_{key}"] = value
    train_lengths = np.array([min(len(split.train) - 1, max_seq_len)
                             for split in training_loader.dataset.splits], dtype=np.int64)
    eval_lengths = np.array([min(len(split.train), max_seq_len) for split in dataset.splits], dtype=np.int64)
    tail_size = int(model_config.get("tail_size", 128))
    for phase, lengths in (("train", train_lengths), ("validation", eval_lengths)):
        metrics[f"{phase}_history_length_mean"] = float(lengths.mean())
        metrics[f"{phase}_history_length_p50"] = float(np.quantile(lengths, .5))
        metrics[f"{phase}_history_length_p95"] = float(np.quantile(lengths, .95))
        metrics[f"{phase}_eligible_compression_fraction"] = float((lengths > tail_size).mean())
    (output_dir / "length-distributions.json").write_text(json.dumps({
        "train": dict(zip(*[a.tolist() for a in np.unique(train_lengths, return_counts=True)])),
        "validation": dict(zip(*[a.tolist() for a in np.unique(eval_lengths, return_counts=True)])),
    }, indent=2) + "\n", encoding="utf-8")
    return metrics


def evaluate_saved_checkpoint(config: dict[str, Any], checkpoint: Path) -> dict[str, int | float]:
    """Evaluate the validation-selected winner on test without retraining."""
    _seed_everything(int(config.get("seed", 42)), bool(config.get("deterministic", True)))
    data_path = resolve_yambda_parquet_path(Path(config["data"]["path"]))
    dataset = load_yambda_sequences(data_path, timestamp_unit=str(config["data"].get("timestamp_unit", "seconds")))
    model_config = config["model"]
    training = config["training"]
    evaluation = config["evaluation"]
    device = torch.device(training.get("device", "cuda"))
    model = SASRecModel(num_items=dataset.num_items, **model_config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    loader = _loader(
        _EvaluationDataset(dataset.splits, model_config["max_seq_len"], "test",
            use_time_features=bool(model_config.get("use_time_features", False)),
            time_unit=float(config["data"].get("gap_time_unit", 1.0)),
            num_gap_buckets=int(model_config.get("num_gap_buckets", 33)) - 1),
        batch_size=int(evaluation["batch_size"]), shuffle=False,
        workers=int(training.get("workers", 2)), seed=int(config.get("seed", 42)),
        collate_fn=_collate_evaluation, device=device,
    )
    started = time.perf_counter()
    details: dict[str, Any] = {}
    ndcg, examples = _full_catalog_ndcg(
        model, loader, device=device, k=10, item_chunk_size=int(evaluation["item_chunk_size"]),
        amp=bool(training.get("amp", True)), phase="test", progress_interval_batches=20,
        details=details,
    )
    return {
        "test_ndcg_at_10": ndcg, "test_examples": examples,
        "test_recall_at_10": details["recall_at_10"],
        "test_seconds": time.perf_counter() - started,
        "test_cold_target_fraction": cold_target_fraction(dataset.splits, target="test"),
    }
