from __future__ import annotations

import json
import math
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
    cold_target_fraction,
    load_yambda_sequences,
    make_training_pair,
    resolve_yambda_parquet_path,
    test_example,
    validation_example,
)
from rhpr.sasrec.losses import next_item_loss
from rhpr.sasrec.model import SASRecModel


class _TrainingDataset(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, splits: Sequence[SequenceSplit], max_seq_len: int) -> None:
        self.splits = tuple(split for split in splits if split.train.size >= 2)
        self.max_seq_len = max_seq_len
        if not self.splits:
            raise ValueError("no user has enough training interactions for shifted targets")

    def __len__(self) -> int:
        return len(self.splits)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return make_training_pair(self.splits[index].train, self.max_seq_len)


class _EvaluationDataset(Dataset[tuple[np.ndarray, int]]):
    def __init__(self, splits: Sequence[SequenceSplit], max_seq_len: int, split_name: str) -> None:
        if split_name == "validation":
            self.example: Callable[[SequenceSplit, int], tuple[np.ndarray, int]] = validation_example
        elif split_name == "test":
            self.example = test_example
        else:
            raise ValueError("split_name must be validation or test")
        self.splits = tuple(splits)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.splits)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.example(self.splits[index], self.max_seq_len)


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
    batch: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*batch, strict=True)
    item_ids, valid_mask = _pad_sequences(inputs)
    padded_targets, _ = _pad_sequences(targets)
    return item_ids, valid_mask, padded_targets


def _collate_evaluation(batch: Sequence[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts, targets = zip(*batch, strict=True)
    item_ids, valid_mask = _pad_sequences(contexts)
    return item_ids, valid_mask, torch.tensor(targets, dtype=torch.long)


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
) -> tuple[float, int]:
    model.eval()
    total = 0.0
    observations = 0
    catalogue_size = model.num_items
    effective_k = min(k, catalogue_size)

    started = time.perf_counter()
    batches = islice(loader, max_batches) if max_batches is not None else loader
    for batch_index, (item_ids, valid_mask, targets) in enumerate(batches, start=1):
        item_ids = item_ids.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            states = model.last_state(item_ids, valid_mask)

        best_scores = torch.empty((states.shape[0], 0), device=device, dtype=torch.float32)
        best_ids = torch.empty((states.shape[0], 0), device=device, dtype=torch.long)
        for first_item in range(1, catalogue_size + 1, item_chunk_size):
            last_item = min(first_item + item_chunk_size, catalogue_size + 1)
            embeddings = model.item_embedding.weight[first_item:last_item]
            scores = states.float() @ embeddings.float().T
            local_k = min(effective_k, scores.shape[1])
            chunk_scores, chunk_indexes = torch.topk(scores, k=local_k, dim=1)
            chunk_ids = chunk_indexes + first_item
            candidate_scores = torch.cat((best_scores, chunk_scores), dim=1)
            candidate_ids = torch.cat((best_ids, chunk_ids), dim=1)
            keep = min(effective_k, candidate_scores.shape[1])
            best_scores, indexes = torch.topk(candidate_scores, k=keep, dim=1)
            best_ids = torch.gather(candidate_ids, 1, indexes)

        matches = best_ids == targets[:, None]
        found = matches.any(dim=1)
        ranks = matches.to(torch.int64).argmax(dim=1) + 1
        gains = torch.where(found, 1.0 / torch.log2(ranks.float() + 1.0), 0.0)
        total += float(gains.sum().item())
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
    return total / observations, observations


def _positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def run_yambda_experiment(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    """Train SASRec and evaluate leave-two-out targets against the full catalogue."""

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    _seed_everything(seed, deterministic)

    data_config = dict(config.get("data", {}))
    model_config = dict(config.get("model", {}))
    training_config = dict(config.get("training", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    loss_config = dict(config.get("loss", {}))
    data_path = resolve_yambda_parquet_path(
        Path(str(data_config.get("path", "/kaggle/input/yambda-50m/listens.parquet")))
    )
    max_seq_len = _positive_int(model_config, "max_seq_len", 1028)

    data_started = time.perf_counter()
    print(json.dumps({"phase": "data_loading", "path": str(data_path)}, sort_keys=True), flush=True)
    dataset = load_yambda_sequences(data_path)
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
    model = SASRecModel(
        num_items=dataset.num_items,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=_positive_int(model_config, "n_layers", 4),
        ffn_multiplier=_positive_int(model_config, "ffn_multiplier", 4),
        dropout=float(model_config.get("dropout", 0.2)),
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
        _TrainingDataset(dataset.splits, max_seq_len),
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
        _EvaluationDataset(dataset.splits, max_seq_len, "validation"),
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
        _EvaluationDataset(dataset.splits, max_seq_len, "test"),
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
    history: list[dict[str, int | float]] = []
    best_validation = -math.inf
    best_epoch = 0
    train_loss_at_best = math.inf
    stale_epochs = 0
    training_seconds = 0.0
    validation_seconds = 0.0
    validation_examples = 0
    training_peak_allocated_gib = 0.0
    training_peak_reserved_gib = 0.0

    for epoch in range(1, max_epochs + 1):
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
        for batch_index, (item_ids, valid_mask, targets) in enumerate(
            islice(training_loader, training_batches), start=1
        ):
            item_ids = item_ids.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                encoded, compressed_mask, source_positions = model.encode(item_ids, valid_mask)
                loss = next_item_loss(
                    name=loss_name,
                    model=model,
                    encoded=encoded,
                    valid_mask=compressed_mask,
                    source_positions=source_positions,
                    targets=targets,
                    num_negatives=num_negatives,
                )
            group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
            group_size = min(accumulation_steps, training_batches - group_start + 1)
            scaler.scale(loss / group_size).backward()
            if batch_index % accumulation_steps == 0 or batch_index == training_batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().item())
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

        train_loss = running_loss / batches
        epoch_training_seconds = time.perf_counter() - train_started
        if device.type == "cuda":
            training_peak_allocated_gib = max(
                training_peak_allocated_gib, torch.cuda.max_memory_allocated(device) / 2**30
            )
            training_peak_reserved_gib = max(
                training_peak_reserved_gib, torch.cuda.max_memory_reserved(device) / 2**30
            )
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
        )
        epoch_validation_seconds = time.perf_counter() - validation_started
        training_seconds += epoch_training_seconds
        validation_seconds += epoch_validation_seconds
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_ndcg_at_10": validation_ndcg,
                "training_seconds": epoch_training_seconds,
                "validation_seconds": epoch_validation_seconds,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)

        if validation_ndcg > best_validation + min_delta:
            best_validation = validation_ndcg
            best_epoch = epoch
            train_loss_at_best = train_loss
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    evaluate_test = bool(evaluation_config.get("evaluate_test", True))
    test_ndcg = 0.0
    test_seconds = 0.0
    if evaluate_test:
        try:
            best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch before weights_only was introduced
            best_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_state)
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
        )
        test_seconds = time.perf_counter() - test_started
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if not bool(training_config.get("keep_checkpoint", False)):
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
        "validation_cold_target_fraction": cold_target_fraction(dataset.splits, target="validation"),
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "training_peak_allocated_gib": training_peak_allocated_gib,
        "training_peak_reserved_gib": training_peak_reserved_gib,
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "training_examples_per_epoch": min(len(training_loader.dataset), training_batches * physical_batch_size),
        "loss": loss_name,
        "num_negatives": num_negatives,
        "amp": amp,
        "device": str(device),
    }
    if evaluate_test:
        metrics.update(
            {
                "test_ndcg_at_10": test_ndcg,
                "test_cold_target_fraction": cold_target_fraction(dataset.splits, target="test"),
                "test_seconds": test_seconds,
            }
        )
    return metrics
