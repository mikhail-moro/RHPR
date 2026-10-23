"""Finite TIGER/GRID window epochs using GenPAS's released model components.

No GenPAS weighted augmentation is enabled. Data preparation and model math
remain upstream; finite batching and the Lightning schedule are RHPR adapters.
"""
from __future__ import annotations

from functools import partial
import math
from pathlib import Path
import random
import time
from types import MethodType

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _write


class EpochWindows:
    """Isolate upstream window sampling from dropout and validation RNG use."""
    def __init__(self, collator, seed):
        import torch
        self.collator, self.seed = collator, seed
        self.rng = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch):
        self.rng.manual_seed(self.seed + epoch)

    def __call__(self, rows):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng.get_state())
            result = self.collator(rows)
            self.rng.set_state(torch.get_rng_state())
        return result


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def _fp32_eval_step(self, *args, **kwargs):
    import torch
    # Native beam search multiplies four probabilities. Under FP16, an
    # approximately uniform path (256**-4) underflows. Mixed training retains
    # FP32 weights, so disabling autocast here preserves the native arithmetic.
    with torch.autocast(device_type=self.device.type, enabled=False):
        return type(self).eval_step(self, *args, **kwargs)


def load_author_rows(cfg):
    """Read every TFRecord row, including each file's final partial batch.

    The author row preprocessing and SID mapping are instantiated from the
    pinned config. Reading per row avoids TFRecordIterator.iter_batches' implicit
    drop_remainder=True. No split is regenerated and no target is moved.
    """
    from hydra.utils import instantiate
    import torch
    result, provenance = {}, {}
    for stage, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        data = cfg.data_loading[f"{key}_dataloader_config"].dataloader
        dataset = instantiate(data.dataset_config)
        directory = Path(data.data_folder)
        files = sorted(directory.rglob("*.tfrecord.gz"))
        if not files:
            raise FileNotFoundError(f"No author TFRecord files in {directory}")
        iterator = dataset.data_iterator
        iterator.update_list_of_file_paths([str(p) for p in files])
        iterator.should_shuffle_rows = False
        rows = []
        for raw in iterator.iterrows():
            row = raw
            for transform in dataset.preprocessing_functions:
                row = transform(row, dataset_config=dataset)
                if row is None:
                    break
            if row is None:
                raise ValueError("An author input row was filtered; epoch coverage must be explicit")
            row = {name: row[name].long() for name in ("sequence_data", "user_id")}
            if row["sequence_data"].numel() < 2 * cfg.model.num_hierarchies:
                raise ValueError("Training/evaluation history must contain context and a complete target SID")
            rows.append(row)
        identifiers = [int(row["user_id"].reshape(-1)[0]) for row in rows]
        if not rows or len(set(identifiers)) != len(rows):
            raise ValueError("Expected exactly one nonempty source history per user in each split")
        if any(not torch.isfinite(row["sequence_data"]).all() for row in rows):
            raise ValueError("Nonfinite author SIDs")
        result[stage] = rows
        provenance[stage] = {"rows": len(rows), "files": [
            {"path": str(p), "sha256": file_sha256(p)} for p in files]}
    return result, provenance


def make_datamodule(rows, train_collator, evaluation_collator, training, seed):
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, Sampler

    class SourceOrder(Sampler):
        epoch = 0

        def __iter__(self):
            return iter(epoch_order(len(rows["train"]), seed, self.epoch))

        def __len__(self):
            return len(rows["train"])

    class EpochDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.order = SourceOrder()
            self.collate = EpochWindows(train_collator, seed)

        def set_epoch(self, epoch):
            self.order.epoch = epoch
            self.collate.set_epoch(epoch)

        def loader(self, stage):
            train = stage == "train"
            # No worker prefetch can consume the next epoch's sampler state.
            # A dedicated generator also prevents DataLoader from changing dropout RNG.
            return DataLoader(rows[stage], batch_size=training["source_user_batch_size"] if train
                else training["evaluation_batch_size"], sampler=self.order if train else None,
                collate_fn=self.collate if train else evaluation_collator, num_workers=0,
                drop_last=False, generator=torch.Generator().manual_seed(seed))

        def train_dataloader(self):
            # Set state before Lightning creates/prefetches an iterator, rather
            # than in on_train_epoch_start (which may happen after prefetch).
            self.set_epoch(self.trainer.current_epoch)
            # Upstream training_step explicitly unwraps batch[0].
            return [self.loader("train")]

        def val_dataloader(self):
            return self.loader("validation")

        def test_dataloader(self):
            return self.loader("test")

    return EpochDataModule()


def make_callbacks(output, training, datamodule):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    class Exposure(pl.Callback):
        def __init__(self):
            self.curves, self.examples, self.source_rows = [], 0, 0

        def on_train_epoch_start(self, trainer, model):
            self.epoch_examples = 0
            self.epoch_started = time.perf_counter()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            inputs, _ = batch[0]
            self.epoch_examples += len(inputs.mask)

        def on_validation_end(self, trainer, model):
            if trainer.sanity_checking:
                return
            self.curves.append({"epoch": trainer.current_epoch + 1, "updates": trainer.global_step,
                "model_examples_consumed": self.examples + self.epoch_examples,
                "metrics": {name: float(value.detach().cpu()) for name, value in
                    trainer.callback_metrics.items() if name.startswith("val/")}})
            _write(Path(output) / "learning-curves.json", self.curves)

        def on_train_epoch_end(self, trainer, model):
            self.examples += self.epoch_examples
            self.source_rows += len(datamodule.order)
            _write(Path(output) / "training-exposure.json", {
                "completed_epochs": trainer.current_epoch + 1, "optimizer_updates": trainer.global_step,
                "source_rows_consumed": self.source_rows, "model_examples_consumed": self.examples,
                "last_epoch_examples": self.epoch_examples,
                "last_epoch_seconds_including_scheduled_validation": time.perf_counter() - self.epoch_started,
                "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
                "epoch_definition": "one seeded visit to every training source user; sampled TIGER/GRID windows",
                "validation_epochs": [row["epoch"] for row in self.curves]})

    checkpoint = ModelCheckpoint(dirpath=str(Path(output) / "checkpoints"),
        filename="tiger-{epoch:04d}-{step:07d}", monitor="val/recall@5", mode="max",
        every_n_train_steps=0, every_n_epochs=training["validation_interval_epochs"],
        save_on_train_epoch_end=False, save_top_k=1, save_last=True)
    stopping = EarlyStopping(monitor="val/recall@5", mode="max",
        patience=training["patience_checks"], check_on_train_epoch_end=False, strict=True)
    return [Exposure(), checkpoint, stopping]


def fit_generator(model, rows, train_collator, evaluation_collator, config, output):
    """Use native Lightning optimization, loss, metrics and checkpoint selection."""
    import pytorch_lightning as pl
    import torch
    spec = config["training"]
    if spec.get("evaluation_precision", "fp32") != "fp32":
        raise ValueError("GenPAS beam evaluation requires FP32 probability arithmetic")
    if spec.get("precision") not in {"32-true", "16-mixed", "bf16-mixed"}:
        raise ValueError("Use FP32 weights with fp32 or mixed-precision training")
    if spec.get("accelerator") not in {"cpu", "gpu"}:
        raise ValueError("This GenPAS epoch runner supports one CPU/GPU device; TPU requires a separate adapter")
    for name in ("max_epochs", "validation_interval_epochs", "patience_checks",
                 "source_user_batch_size", "evaluation_batch_size", "accumulate_grad_batches"):
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if spec["max_epochs"] % spec["validation_interval_epochs"]:
        raise ValueError("max_epochs must end on a validation boundary")
    data = make_datamodule(rows, train_collator, evaluation_collator, spec, config["seed"])
    model.eval_step = MethodType(_fp32_eval_step, model)
    callbacks = make_callbacks(output, spec, data)
    trainer = pl.Trainer(accelerator=spec["accelerator"], devices=1, logger=False,
        default_root_dir=str(output), max_steps=-1, max_epochs=spec["max_epochs"], min_epochs=0,
        val_check_interval=1.0, check_val_every_n_epoch=spec["validation_interval_epochs"],
        num_sanity_val_steps=0, precision=spec["precision"],
        accumulate_grad_batches=spec["accumulate_grad_batches"], callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, datamodule=data)
    checkpoint = callbacks[1]
    if not checkpoint.best_model_path:
        raise RuntimeError("No validation-selected checkpoint; test evaluation is forbidden")
    completed_epochs, updates, stopped = trainer.current_epoch, trainer.global_step, trainer.should_stop
    # This file was created by this fit call. Native hyperparameters contain
    # functools.partial; PyTorch's weights-only loader cannot deserialize them.
    selected = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["state_dict"])
    # The held-out split is evaluated once, after validation-only selection.
    test = trainer.test(model, datamodule=data, ckpt_path=None)[0]
    return {"completed_epochs": completed_epochs, "optimizer_updates": updates,
        "validation_checks": len(callbacks[0].curves), "model_examples_consumed": callbacks[0].examples,
        "selected_checkpoint": checkpoint.best_model_path, "test_evaluations": 1,
        "selected_epoch": selected["epoch"] + 1, "selected_update": selected["global_step"],
        "early_stopping_reached": stopped,
        **{name: float(value) for name, value in test.items()}}


def run_genpas_epochs(config, output):
    import pytorch_lightning as pl
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from rhpr.gr.genpas import GENPAS_REVISION, activate_source, compose_beauty

    source = activate_source(config["source_path"])
    cfg = compose_beauty(source, config["data_path"], output)
    # Disable only the weighted GenPAS sampler: use the released TIGER/GRID collator.
    from src.data.loading.components.collate_functions import collate_with_sid_causal_duplicate, collate_fn_train
    spec = config["training"]
    pl.seed_everything(config["seed"], workers=True)
    rows, provenance = load_author_rows(cfg)
    labels = instantiate(cfg.data_loading.train_dataloader_config.dataloader.labels)
    common = {"labels": labels, "sequence_length": 100, "padding_token": -1, "masking_token": -1,
              "oov_token": None}
    collator = partial(collate_with_sid_causal_duplicate, sequence_field_name="sequence_data",
        sid_hierarchy=4, max_batch_size=512, **common)
    evaluation = partial(collate_fn_train, **common)
    model = instantiate(cfg.model)
    sid_path = Path(config["data_path"]) / "sids/flan-t5-xxl_rkmeans_3_256_seed42.pt"
    _write(output / "input-provenance.json", {"splits": provenance, "sids": {
        "path": str(sid_path), "sha256": file_sha256(sid_path)}, "genpas_revision": GENPAS_REVISION})
    _write(output / "genpas-model-config.json", OmegaConf.to_container(cfg.model, resolve=True))
    _write(output / "training-protocol.json", {"sampling": "TIGER/GRID collate_with_sid_causal_duplicate",
        "genpas_weighted_augmentation": False, "source_users": len(rows["train"]),
        "source_batches_per_epoch": math.ceil(len(rows["train"]) / spec["source_user_batch_size"]),
        "generator_parameters": sum(p.numel() for p in model.parameters()),
        "scheduler_interval": "epoch (unchanged upstream configure_optimizers)",
        "scheduler_steps": int(cfg.optim.scheduler.scheduler_steps), "training": spec,
        "torch_version": torch.__version__, "lightning_version": pl.__version__,
        "exact_published_reproduction": False})
    return {"experiment": config["name"], "seed": config["seed"], "genpas_revision": GENPAS_REVISION,
        "exact_published_reproduction": False,
        **fit_generator(model, rows, collator, evaluation, config, output)}
