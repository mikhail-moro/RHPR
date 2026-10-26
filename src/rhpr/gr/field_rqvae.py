"""Joint feature encoder / native GRID RQ-VAE fitting on catalogue fields."""
from pathlib import Path
import time

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.fields import FIELD_NAMES, prepare_fields, tensor_fields
from rhpr.gr.rqvae import build_rqvae


def export_field_codes(model, arrays, device, batch_size, *, capture=None):
    import torch
    from lightning.pytorch.trainer.states import TrainerFn

    model.to(device).eval()
    model.trainer.state.fn = TrainerFn.PREDICTING
    names = model.field_names
    codes, embeddings = [], []
    sums = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    continuous_sums = {name: 0.0 for name in names}
    if capture is not None:
        capture["encoded"] = []
    brand_correct, cat_tp, cat_fp, cat_fn, price_abs_error = 0, 0, 0, 0, 0.0
    title_cosine, title_squared = 0.0, 0.0
    with torch.inference_mode():
        for begin in range(0, len(arrays["brand"]), batch_size):
            fields = tensor_fields(arrays, slice(begin, begin + batch_size), device)
            features, encoded = model.encode_fields(fields)
            batch_codes, _, quantized, _ = model(encoded)
            predictions = model.field_heads(model.decoder(quantized))
            if not all(torch.isfinite(value).all() for value in [features, encoded, *predictions.values()]):
                raise FloatingPointError("nonfinite field RQ-VAE export")
            losses, observed = model.field_heads.losses(predictions, fields)
            if capture is not None:
                capture["encoded"].append(encoded.cpu().numpy())
                # Same trained decoder/heads without quantization: a diagnostic,
                # not a separately trained continuous-autoencoder baseline.
                continuous = model.field_heads(model.decoder(encoded))
                continuous_losses, _ = model.field_heads.losses(continuous, fields)
                for name in names:
                    continuous_sums[name] += float(continuous_losses[name]) * observed[name]
            brand_mask = fields["brand"] > 0
            brand_correct += int((predictions["brand"][brand_mask].argmax(-1) == fields["brand"][brand_mask] - 1).sum())
            cat_mask = fields["categories_observed"]
            predicted = predictions["categories"][cat_mask] >= 0
            actual = fields["categories"][cat_mask] > 0
            cat_tp += int((predicted & actual).sum())
            cat_fp += int((predicted & ~actual).sum())
            cat_fn += int((~predicted & actual).sum())
            price_mask = fields["price_observed"]
            price_abs_error += float((predictions["price"][price_mask] - fields["price"][price_mask]).abs().sum())
            if "title" in names:
                title_mask = fields["title_observed"]
                title_cosine += float(torch.nn.functional.cosine_similarity(
                    predictions["title"][title_mask], fields["title"][title_mask], dim=-1).sum())
                title_squared += float(fields["title"][title_mask].square().sum())
            for name in names:
                sums[name] += float(losses[name]) * observed[name]
                counts[name] += observed[name]
            codes.append(batch_codes.cpu().numpy())
            embeddings.append(features.cpu().numpy())
    metrics = {
        name: {"loss": sums[name] / counts[name] if counts[name] else None,
               "observed_items": counts[name]} for name in names}
    metrics["brand"]["accuracy"] = brand_correct / counts["brand"] if counts["brand"] else None
    metrics["categories"]["micro_f1_at_probability_0_5"] = 2 * cat_tp / max(2 * cat_tp + cat_fp + cat_fn, 1)
    metrics["price"]["mae_standardized"] = price_abs_error / counts["price"] if counts["price"] else None
    if "title" in names and counts["title"]:
        metrics["title"]["cosine_similarity"] = title_cosine / counts["title"]
        metrics["title"]["zero_predictor_mse"] = title_squared / (counts["title"] * model.feature_encoder.title_dim)
    if capture is not None:
        capture["continuous_field_losses"] = {name: continuous_sums[name] / counts[name] if counts[name] else None
            for name in names}
    return np.concatenate(codes), np.concatenate(embeddings), metrics


def probe_field_codebooks(model, arrays, device, batch_size, output, step):
    """Observe eval/serving codes without altering fit state, RNG or buffers."""
    import torch
    from rhpr.gr.codebook_diagnostics import diagnose_codebooks

    modes = {module: module.training for module in model.modules()}
    fn = model.trainer.state.fn
    cuda_devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    before = time.monotonic()
    capture = {}
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            codes, features, reconstruction = export_field_codes(model, arrays, device, batch_size, capture=capture)
            books = np.stack([layer.centroids.detach().cpu().numpy() for layer in model.quantization_layer_list])
            encoded = np.concatenate(capture.pop("encoded"))
            stats = diagnose_codebooks(codes, encoded, books)
            if hasattr(model, "ae_anchor_latents"):
                from rhpr.gr.ae_warmup import latent_spectrum
                stats["warmup_latent_drift_mse"] = float(np.square(encoded - model.ae_anchor_latents).mean())
                stats["warmup_feature_drift_mse"] = float(np.square(features - model.ae_anchor_features).mean())
                stats["latent_spectrum"] = latent_spectrum(encoded, books.shape[1], books.shape[0])
                stats["codebook_effective_dimensions_99pct"] = [
                    latent_spectrum(book, books.shape[1], 1)["effective_dimension_99pct"] for book in books]
                quantized = books[np.arange(len(books))[None, :], codes].sum(1)
                stats["quantized_effective_dimension_99pct"] = latent_spectrum(
                    quantized, books.shape[1], books.shape[0])["effective_dimension_99pct"]
            stats.update(step=step, quantized_field_reconstruction=reconstruction, **capture)
            stats["native_beta_loss_eval"] = sum(
                (1 + float(layer.loss_function.beta)) * values["residual_mse_after"]
                for layer, values in zip(model.quantization_layer_list, stats["levels"]))
            stats["codebook_loss_eval"] = sum(values["residual_mse_after"] for values in stats["levels"])
            stats["commitment_loss_eval"] = stats["native_beta_loss_eval"] - stats["codebook_loss_eval"]
            stats["field_weights"] = dict(zip(model.field_names, model.task_weights().detach().cpu().tolist()))
            stats["field_log_variances"] = model.uncertainty.loss_scale.detach().cpu().tolist()
            stats["probe_seconds"] = time.monotonic() - before
            output = Path(output)
            directory = output / "codebook-snapshots"
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"step-{step:05d}.npz", semantic_codes=codes,
                codebooks=books, counts=np.array([values["counts"] for values in stats["levels"]]))
            _write(output / "codebook-diagnostics.json", stats)
            return stats
    finally:
        model.trainer.state.fn = fn
        for module, mode in modes.items():
            module.training = mode


def fit_field_rqvae(texts, representation, spec, seed, device, source_root, recipe_path, output, *,
                    title_embeddings=None, title_observed=None, warmup_checkpoint=None):
    import torch
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    preparation_started = time.monotonic()
    names = representation.get("fields")
    if names not in (list(FIELD_NAMES), [*FIELD_NAMES, "title"]):
        raise ValueError("expected brand/categories/price with optional frozen title")
    if ("title" in names) != (title_embeddings is not None):
        raise ValueError("title field declaration and embeddings must agree")
    for key in ("num_hierarchies", "codebook_size", "batch_size", "max_steps",
                "init_buffer_size", "initialization_steps"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(f"quantizer.{key} must be a positive integer")
    count = len(texts)
    if count < max(spec["init_buffer_size"], spec["codebook_size"]):
        raise ValueError("catalogue is smaller than codebook/initialization buffer")
    if spec["batch_size"] < 2 or count % spec["batch_size"] == 1:
        raise ValueError("native BatchNorm requires at least two items per batch")
    diagnostic_interval = spec.get("diagnostics_interval_steps")
    if diagnostic_interval is not None and (type(diagnostic_interval) is not int or diagnostic_interval < 1):
        raise ValueError("diagnostics_interval_steps must be a positive integer")
    schema, arrays = prepare_fields(texts, title_embeddings=title_embeddings, title_observed=title_observed)
    _write(output / "feature-schema.json", schema)
    # Persist exact prepared inputs/targets independently of trainable embeddings.
    np.savez_compressed(output / "catalogue-fields.npz", **arrays)
    preparation_seconds = time.monotonic() - preparation_started
    torch.manual_seed(seed)
    field_dim = representation["field_dim"]
    input_dim = len(FIELD_NAMES) * field_dim + schema.get("title_dim", 0)
    model, resolved = build_rqvae(source_root, recipe_path, input_dim,
        spec, feature_schema=schema, field_dim=field_dim)
    from rhpr.gr.field_autoencoder import configure_objective
    from rhpr.gr.sid_study import state_digest
    warmup = None
    if warmup_checkpoint is not None:
        configure_objective(model, arrays, spec.get("loss_policy"))
        from rhpr.gr.ae_staging import load_warmup
        warmup = load_warmup(model, warmup_checkpoint, schema, representation, spec)
        _write(output / "warmup-transfer.json", warmup)
    elif spec.get("freeze_policy"):
        raise ValueError("freeze_policy requires a continuous AE warm-up checkpoint")
    common_prefixes = ("feature_encoder.", "normalization_layer.", "encoder.", "decoder.", "field_heads.", "uncertainty.")
    initial_digest = state_digest({k: v for k, v in model.state_dict().items() if k.startswith(common_prefixes)})
    initial_full_digest = state_digest(model.state_dict())
    if spec.get("expected_initial_state_sha256", initial_digest) != initial_digest:
        raise ValueError("initial neural weights differ from the registered AE control")
    if warmup is None:
        configure_objective(model, arrays, spec.get("loss_policy"))
    from src.data.loading.components.interfaces import ItemData

    def collate(indices):
        return ItemData(transformed_features=tensor_fields(arrays, indices, "cpu"))

    loader = DataLoader(range(count), batch_size=spec["batch_size"], shuffle=True,
        drop_last=False, num_workers=0, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed))

    class Progress(Callback):
        def __init__(self):
            self.examples, self.curves = 0, []
            self.last_loss = None
            self.last_logged_step = -1
            self.last_probe_step = -1
            self.codebook_curves = []
            self.probe_seconds = 0.0
            self.gradient_norms = {}

        def on_before_optimizer_step(self, trainer, module, optimizer):
            upcoming = trainer.global_step + 1
            if upcoming % 500 != 0 and upcoming != spec["max_steps"]:
                return
            groups = {"feature_encoder": module.feature_encoder.parameters(),
                "rq_encoder": module.encoder.parameters(), "rq_decoder": module.decoder.parameters(),
                "field_heads": module.field_heads.parameters(), "uncertainty": module.uncertainty.parameters()}
            groups.update({f"codebook_{level + 1}": [layer.centroids]
                for level, layer in enumerate(module.quantization_layer_list)})
            self.gradient_norms = {name: float(torch.stack([
                parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None
            ] or [torch.zeros((), device=module.device)]).sum().sqrt()) for name, parameters in groups.items()}

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("nonfinite field RQ-VAE training loss")
            self.examples += len(batch[0].transformed_features["brand"])
            self.last_loss = float(loss.detach())
            step = trainer.global_step
            ready = all(layer.is_initialized for layer in module.quantization_layer_list)
            if diagnostic_interval and ready and step > 0 and step != self.last_probe_step and (
                    self.last_probe_step < 0 or step % diagnostic_interval == 0 or step == spec["max_steps"]):
                stats = probe_field_codebooks(module, arrays, device, spec["batch_size"], output, step)
                self.codebook_curves.append(stats)
                self.probe_seconds += stats["probe_seconds"]
                self.last_probe_step = step
                _write(output / "codebook-learning-curves.json", self.codebook_curves)
                _event("field_rqvae_codebooks", step=step,
                    used_codes=[value["used_codes"] for value in stats["levels"]],
                    perplexity=[value["perplexity"] for value in stats["levels"]],
                    semantic_unique_ids=stats["semantic_unique_ids"], max_collision_group=stats["max_collision_group"],
                    suffix_representable=stats["suffix_representable"], quantization_mse=stats["quantization_mse"])
            if step > 0 and step != self.last_logged_step and (step % 500 == 0 or step == spec["max_steps"]):
                self.last_logged_step = step
                weights = module.task_weights().detach().cpu().tolist()
                components = {key: float(value) for key, value in module.last_loss_components.items()}
                row = {"step": step, "total_loss": self.last_loss,
                    "loss_components": components,
                    "gradient_norms": self.gradient_norms,
                    "raw_field_losses": module.last_field_losses,
                    "field_observed_counts": module.last_field_counts,
                    "field_weights": dict(zip(names, weights))}
                if hasattr(module, "task_balancer"):
                    row["balancing_diagnostics"] = {"complete_catalogue_passes": int(module.task_balancer.passes),
                        "multipliers": module.task_balancer.multipliers().detach().cpu().tolist(), **module.task_balancer.diagnostics}
                self.curves.append(row)
                _write(output / "field-learning-curves.json", self.curves)
                _event("field_rqvae_progress", **row)

    progress = Progress()
    model.log_dict = lambda *args, **kwargs: None
    trainer = Trainer(accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=[torch.device(device).index or 0] if device.startswith("cuda") else 1,
        precision="32-true", max_steps=spec["max_steps"], max_epochs=-1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        callbacks=[progress], default_root_dir=str(output))
    started = time.monotonic()
    trainer.fit(model, train_dataloaders=[loader])
    flags = [bool(layer.is_initialized) for layer in model.quantization_layer_list]
    if trainer.global_step != spec["max_steps"] or not all(flags):
        raise RuntimeError("field RQ-VAE did not finish its declared initialized fit")
    fit_seconds = time.monotonic() - started
    if warmup is not None:
        from rhpr.gr.ae_staging import finish_stage
        warmup = finish_stage(model, warmup)
        _write(output / "warmup-transfer.json", warmup)
    torch.save({"model": model.state_dict(), "optimizer": trainer.optimizers[0].state_dict(),
        "spec": spec, "representation": representation, "feature_schema": schema, "warmup": warmup,
        "resolved_model": resolved, "initialized": flags, "seed": seed,
        "loss_policy": model.loss_policy, "loss_normalizers": model.loss_normalizers,
        "initial_state_sha256": initial_digest, "initial_full_state_sha256": initial_full_digest,
        "reconstruction_updates": model.reconstruction_updates,
        "initialization_only_batches": model.initialization_batches,
        "steps": trainer.global_step, "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "source_recipe_sha256": file_sha256(Path(recipe_path)), "automatic_resume": False},
        output / "quantizer.pt")
    codes, embeddings, reconstruction = export_field_codes(model, arrays, device, spec["batch_size"])
    from rhpr.gr.codebook_diagnostics import semantic_statistics
    semantic_summary = semantic_statistics(codes, spec["codebook_size"])
    _write(output / "semantic-code-diagnostics.json", semantic_summary)
    if reconstruction["price"]["mae_standardized"] is not None:
        reconstruction["price"]["mae_original_units"] = reconstruction["price"]["mae_standardized"] * schema["price_scale"]
    if "title" in names:
        targets = arrays["title"][arrays["title_observed"]]
        reconstruction["title"]["catalogue_mean_predictor_mse"] = float(np.square(targets - targets.mean(0)).mean())
    np.save(output / "semantic-codes.npy", codes, allow_pickle=False)
    np.save(output / "item-embeddings.npy", embeddings, allow_pickle=False)
    weights = model.task_weights().detach().cpu().tolist()
    report = {"kind": "grid_field_rqvae", "seed": seed, "config": spec,
        "resolved_model": resolved, "precision": "fp32", "steps": trainer.global_step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_dim": input_dim, "latent_dim": model.encoder.output_dim,
        "item_exposures": progress.examples, "final_training_loss": progress.last_loss,
        "field_reconstruction": reconstruction, "field_weights": dict(zip(names, weights)),
        "loss_policy": model.loss_policy, "loss_normalizers": model.loss_normalizers,
        "initial_state_sha256": initial_digest, "initial_full_state_sha256": initial_full_digest,
        "reconstruction_updates": model.reconstruction_updates,
        "initialization_only_batches": model.initialization_batches,
        "optimizer_param_groups": [{"name": g.get("name", "all"), "lr": g["lr"]} for g in trainer.optimizers[0].param_groups],
        "field_log_variances": model.uncertainty.loss_scale.detach().cpu().tolist(),
        "semantic_code_statistics": semantic_summary, "diagnostic_seconds": progress.probe_seconds,
        "final_batch_loss_components": {key: float(value) for key, value in model.last_loss_components.items()},
        "used_codes_per_level": [int(len(np.unique(codes[:, level]))) for level in range(codes.shape[1])],
        "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": file_sha256(output / "quantizer.pt"),
        "feature_schema_sha256": file_sha256(output / "feature-schema.json"),
        "source_recipe_sha256": file_sha256(Path(recipe_path)),
        "objective": "sum_l codebook+beta0.25 commitment; " + model.loss_policy["mode"] + " field reconstruction",
        "normalization": "catalogue price z-score then learned per-field projections; native GRID BatchNorm+L2 before MLP",
        "joint_training_scope": ("warm-started RQ; freeze policy=" + warmup["freeze_policy"] + "; no NIP gradients")
            if warmup is not None else "feature encoder and RQ-VAE by field reconstruction; no NIP gradients",
        "warmup": warmup}
    _write(output / "quantizer.json", report)
    representation_report = {**representation, "seconds": preparation_seconds,
        "shape": list(embeddings.shape), "sha256": file_sha256(output / "item-embeddings.npy"),
        "training": ("warm-started field encoder; freeze policy=" + warmup["freeze_policy"])
            if warmup is not None else "joint field projections with RQ-VAE; optional frozen title concatenated directly before BatchNorm/L2",
        "feature_schema_sha256": report["feature_schema_sha256"]}
    _write(output / "representation.json", representation_report)
    return codes, report, representation_report
