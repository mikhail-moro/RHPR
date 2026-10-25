"""Train pinned GRID's RQ-VAE and retain its differentiable components.

Only the data adapter, finite step budget and logging policy are RHPR code.
Encoder/decoder, STE, residual recursion, beta loss and initialization are GRID.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.sid import validate_item_embeddings


def build_rqvae(source_root, recipe_path, dimension, spec, *, feature_schema=None, field_dim=64):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from rhpr.gr.grid import load_grid

    load_grid(source_root)
    recipe = OmegaConf.load(recipe_path)
    recipe.embedding_dim = dimension
    recipe.num_hierarchies = spec["num_hierarchies"]
    recipe.codebook_width = spec["codebook_size"]
    recipe.model.verbose = False
    recipe.model.track_residuals = False
    recipe.model.init_buffer_size = spec["init_buffer_size"]
    recipe.model.quantization_layer.initializer.max_iter = spec["initialization_steps"]
    if feature_schema is not None:
        recipe.model._target_ = "rhpr.gr.field_rqvae_native.FieldResidualQuantization"
        recipe.model.feature_schema = feature_schema
        recipe.model.field_dim = field_dim
        recipe.model.reconstruction_loss_function = None
    if spec.get("distance_backend", "upstream") == "torch_cdist":
        # All references in this recipe use the same squared L2 metric,
        # including the initializer. Numeric rounding can differ from upstream.
        recipe.model.quantization_layer.distance_function = {
            "_target_": "rhpr.gr.distances.TorchCdistSquaredEuclideanDistance"}
    elif spec.get("distance_backend", "upstream") != "upstream":
        raise ValueError("Unknown RQ-VAE distance backend")
    model = instantiate(recipe.model)
    return model, OmegaConf.to_container(recipe.model, resolve=True)


def export_codes(model, values, device, batch_size, *, capture=None):
    import torch
    from lightning.pytorch.trainer.states import TrainerFn

    # Native forward uses TrainerFn, independently of Module.training.
    # Lightning tears down its strategy after fit and moves the module to CPU.
    # Export is outside Trainer.predict, so explicitly restore the requested
    # device before making CUDA batches (including BatchNorm buffers).
    model.to(device)
    model.trainer.state.fn = TrainerFn.PREDICTING
    model.eval()
    result, squared_error, elements = [], 0.0, 0
    continuous_error = 0.0
    if capture is not None:
        capture["encoded"] = []
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[begin:begin + batch_size], device=device)
            normalized = model.normalization_layer(batch)
            encoded = model.encoder(normalized)
            codes, _, quantized, _ = model(encoded)
            reconstruction = model.decoder(quantized)
            if capture is not None:
                capture["encoded"].append(encoded.cpu().numpy())
                continuous_error += float((model.decoder(encoded) - normalized).square().sum())
            if not torch.isfinite(reconstruction).all():
                raise FloatingPointError("nonfinite RQ-VAE reconstruction")
            squared_error += float((reconstruction - normalized).square().sum())
            elements += normalized.numel()
            result.append(codes.cpu().numpy())
    if capture is not None:
        capture["continuous_embedding_mse_normalized_input"] = continuous_error / elements
    return np.concatenate(result), squared_error / elements


def probe_grid_rqvae(model, values, device, batch_size, output, step):
    """The same catalogue occupancy/geometry probe as the structured arm."""
    import torch
    from rhpr.gr.codebook_diagnostics import diagnose_codebooks

    modes = {module: module.training for module in model.modules()}
    fn = model.trainer.state.fn
    devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    before, capture = time.monotonic(), {}
    try:
        with torch.random.fork_rng(devices=devices):
            codes, mse = export_codes(model, values, device, batch_size, capture=capture)
            books = np.stack([layer.centroids.detach().cpu().numpy() for layer in model.quantization_layer_list])
            stats = diagnose_codebooks(codes, np.concatenate(capture.pop("encoded")), books)
            stats.update(step=step, embedding_reconstruction_mse_normalized_input=mse, **capture)
            stats["codebook_loss_eval"] = sum(level["residual_mse_after"] for level in stats["levels"])
            stats["commitment_loss_eval"] = sum(float(layer.loss_function.beta) * level["residual_mse_after"]
                for layer, level in zip(model.quantization_layer_list, stats["levels"]))
            stats["native_beta_loss_eval"] = stats["codebook_loss_eval"] + stats["commitment_loss_eval"]
            stats["probe_seconds"] = time.monotonic() - before
            directory = Path(output) / "codebook-snapshots"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"step-{step:05d}.npz", semantic_codes=codes, codebooks=books,
                counts=np.array([row["counts"] for row in stats["levels"]]))
            _write(Path(output) / "codebook-diagnostics.json", stats)
            return stats
    finally:
        model.trainer.state.fn = fn
        for module, mode in modes.items():
            module.training = mode


def fit_grid_rqvae(values, spec, seed, device, source_root, recipe_path, output):
    import torch
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = validate_item_embeddings(values)
    for key in ["num_hierarchies", "codebook_size", "batch_size", "max_steps",
                "initialization_steps", "init_buffer_size"]:
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(f"quantizer.{key} must be a positive integer")
    if len(values) < max(spec["codebook_size"], spec["init_buffer_size"]):
        raise ValueError("catalogue is smaller than RQ-VAE initialization buffer")
    if spec["batch_size"] < 2 or len(values) % spec["batch_size"] == 1:
        raise ValueError("RQ-VAE BatchNorm needs at least two items in each batch")
    interval = spec.get("diagnostics_interval_steps")
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError("diagnostics_interval_steps must be a positive integer")
    torch.manual_seed(seed)
    model, resolved = build_rqvae(source_root, recipe_path, values.shape[1], spec)
    from src.data.loading.components.interfaces import ItemData

    def collate(rows):
        return ItemData(transformed_features={"input_embedding": torch.stack(rows)})

    loader = DataLoader(torch.from_numpy(values), batch_size=spec["batch_size"],
        shuffle=True, drop_last=False, num_workers=0, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed))

    class Progress(Callback):
        def __init__(self):
            self.last_loss, self.batches, self.examples = None, 0, 0
            self.last_probe, self.last_log = -1, -1
            self.curves, self.probe_seconds = [], 0.0

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("nonfinite native RQ-VAE loss")
            self.batches += 1
            self.examples += len(batch[0].transformed_features["input_embedding"])
            self.last_loss = float(loss.detach())
            step = trainer.global_step
            if interval and step > 0 and step != self.last_probe and all(
                    layer.is_initialized for layer in module.quantization_layer_list) and (
                    self.last_probe < 0 or step % interval == 0 or step == spec["max_steps"]):
                stats = probe_grid_rqvae(module, values, device, spec["batch_size"], output, step)
                self.curves.append(stats)
                self.last_probe = step
                self.probe_seconds += stats["probe_seconds"]
                _write(output / "codebook-learning-curves.json", self.curves)
                _event("rqvae_codebooks", step=step, semantic_unique_ids=stats["semantic_unique_ids"],
                    max_collision_group=stats["max_collision_group"],
                    used_codes=[row["used_codes"] for row in stats["levels"]])
            if step and step != self.last_log and step % 500 == 0:
                self.last_log = step
                _event("rqvae_progress", updates=step, loss=self.last_loss)

    progress = Progress()
    # Native loss/initialization remain intact; disable only metric emission.
    model.log_dict = lambda *args, **kwargs: None
    trainer = Trainer(accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=[torch.device(device).index or 0] if device.startswith("cuda") else 1,
        precision="32-true", max_steps=spec["max_steps"], max_epochs=-1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[progress], default_root_dir=str(output))
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=[loader])
    if trainer.global_step != spec["max_steps"]:
        raise RuntimeError("RQ-VAE did not complete its declared update budget")
    flags = [bool(layer.is_initialized) for layer in model.quantization_layer_list]
    if not all(flags):
        raise RuntimeError("RQ-VAE codebooks did not initialize")
    fit_seconds = time.monotonic() - started
    # Keep completed optimization even if a subsequent export/check fails.
    torch.save({"model": model.state_dict(), "optimizer": trainer.optimizers[0].state_dict(),
        "spec": spec, "seed": seed, "input_dim": values.shape[1], "resolved_model": resolved,
        "initialized": flags, "steps": trainer.global_step,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "source_recipe_sha256": file_sha256(Path(recipe_path)),
        "automatic_resume": False}, output / "quantizer.pt")
    _event("rqvae_fit_checkpoint_saved", steps=trainer.global_step,
           fit_seconds=fit_seconds, checkpoint="quantizer.pt")
    codes, mse = export_codes(model, values, device, spec["batch_size"])
    np.save(output / "semantic-codes.npy", codes, allow_pickle=False)
    from rhpr.gr.codebook_diagnostics import semantic_statistics
    semantic_summary = semantic_statistics(codes, spec["codebook_size"])
    _write(output / "semantic-code-diagnostics.json", semantic_summary)
    report = {"kind": "grid_rqvae", "seed": seed, "config": spec, "resolved_model": resolved,
        "input_dim": values.shape[1], "latent_dim": model.encoder.output_dim,
        "precision": "fp32", "steps": trainer.global_step, "batches": progress.batches,
        "item_exposures": progress.examples, "final_training_loss": progress.last_loss,
        "reconstruction_mse_normalized_input": mse,
        "semantic_code_statistics": semantic_summary, "diagnostic_seconds": progress.probe_seconds,
        "used_codes_per_level": [int(len(np.unique(codes[:, level]))) for level in range(codes.shape[1])],
        "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started,
        "source_recipe": "GRID configs/experiment/rqvae_train_flat.yaml",
        "source_recipe_sha256": file_sha256(Path(recipe_path)),
        "checkpoint_sha256": file_sha256(output / "quantizer.pt"),
        "notes": ["GRID BatchNorm+L2 input normalization, MLP768/256/128/64 and mirrored decoder",
            "STE + beta0.25 quantization loss + normalized-input MSE; Adagrad1e-3",
            "fixed3000 native optimizer steps, finite shuffled catalogue loader",
            "FP32 quantizer on T4; FP16 is the separate generator policy",
            "optional cdist squared L2 also applies to initializer; no hidden precision or LR fallback"]}
    _write(output / "quantizer.json", report)
    return codes, report
