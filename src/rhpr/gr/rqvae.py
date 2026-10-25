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


def build_rqvae(source_root, recipe_path, dimension, spec):
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
    if spec.get("distance_backend", "upstream") == "torch_cdist":
        # All references in this recipe use the same squared L2 metric,
        # including the initializer. Numeric rounding can differ from upstream.
        recipe.model.quantization_layer.distance_function = {
            "_target_": "rhpr.gr.distances.TorchCdistSquaredEuclideanDistance"}
    elif spec.get("distance_backend", "upstream") != "upstream":
        raise ValueError("Unknown RQ-VAE distance backend")
    model = instantiate(recipe.model)
    return model, OmegaConf.to_container(recipe.model, resolve=True)


def export_codes(model, values, device, batch_size):
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
    with torch.inference_mode():
        for begin in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[begin:begin + batch_size], device=device)
            normalized = model.normalization_layer(batch)
            encoded = model.encoder(normalized)
            codes, _, quantized, _ = model(encoded)
            reconstruction = model.decoder(quantized)
            if not torch.isfinite(reconstruction).all():
                raise FloatingPointError("nonfinite RQ-VAE reconstruction")
            squared_error += float((reconstruction - normalized).square().sum())
            elements += normalized.numel()
            result.append(codes.cpu().numpy())
    return np.concatenate(result), squared_error / elements


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

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("nonfinite native RQ-VAE loss")
            self.batches += 1
            self.examples += len(batch[0].transformed_features["input_embedding"])
            self.last_loss = float(loss.detach())
            if trainer.global_step and trainer.global_step % 500 == 0:
                _event("rqvae_progress", updates=trainer.global_step, loss=self.last_loss)

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
    report = {"kind": "grid_rqvae", "seed": seed, "config": spec, "resolved_model": resolved,
        "input_dim": values.shape[1], "latent_dim": model.encoder.output_dim,
        "precision": "fp32", "steps": trainer.global_step, "batches": progress.batches,
        "item_exposures": progress.examples, "final_training_loss": progress.last_loss,
        "reconstruction_mse_normalized_input": mse,
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
