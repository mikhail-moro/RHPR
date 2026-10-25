"""Continuous control: native GRID neural modules, no residual quantizer."""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _event, _write
from rhpr.gr.fields import prepare_fields, tensor_fields
from rhpr.gr.rqvae import build_rqvae


class ContinuousFieldAutoencoder(nn.Module):
    def __init__(self, native):
        super().__init__()
        # Construct the full native model first: common parameters retain the
        # RQ arm's initialization/RNG ordering. Unused books are then discarded.
        for name in ("feature_encoder", "normalization_layer", "encoder", "decoder", "field_heads", "uncertainty"):
            setattr(self, name, getattr(native, name))
        self.field_names = native.field_names

    def forward(self, fields):
        features = self.feature_encoder(fields)
        latent = self.encoder(self.normalization_layer(features))
        return features, latent, self.field_heads(self.decoder(latent))

    def objective(self, predictions, fields):
        losses, counts = self.field_heads.losses(predictions, fields)
        active = torch.tensor([counts[name] > 0 for name in self.field_names], device=fields["brand"].device)
        total = self.uncertainty(torch.stack([losses[name] for name in self.field_names]), active)
        return total, losses, counts


def build_field_autoencoder(source_root, recipe_path, schema, spec, field_dim):
    dimension = 3 * field_dim + schema.get("title_dim", 0)
    native, resolved = build_rqvae(source_root, recipe_path, dimension, spec,
        feature_schema=schema, field_dim=field_dim)
    return ContinuousFieldAutoencoder(native), native.optimizer, resolved


def evaluate_autoencoder(model, arrays, device, batch_size, *, capture=None):
    modes = {module: module.training for module in model.modules()}
    names = model.field_names
    sums, counts = dict.fromkeys(names, 0.0), dict.fromkeys(names, 0)
    correct, tp, fp, fn, price_error, cosine, title_squared = 0, 0, 0, 0, 0., 0., 0.
    norm_sum, examples = 0., 0
    if capture is not None:
        capture.update(features=[], latent=[])
    try:
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith("cuda") else []):
            model.eval()
            with torch.inference_mode():
                for begin in range(0, len(arrays["brand"]), batch_size):
                    fields = tensor_fields(arrays, slice(begin, begin + batch_size), device)
                    features, latent, predicted = model(fields)
                    if not all(torch.isfinite(t).all() for t in [features, latent, *predicted.values()]):
                        raise FloatingPointError("nonfinite continuous autoencoder evaluation")
                    losses, observed = model.field_heads.losses(predicted, fields)
                    for name in names:
                        sums[name] += float(losses[name]) * observed[name]
                        counts[name] += observed[name]
                    valid = fields["brand"] > 0
                    correct += int((predicted["brand"][valid].argmax(-1) == fields["brand"][valid] - 1).sum())
                    valid = fields["categories_observed"]
                    p, t = predicted["categories"][valid] >= 0, fields["categories"][valid] > 0
                    tp += int((p & t).sum());fp += int((p & ~t).sum());fn += int((~p & t).sum())
                    valid = fields["price_observed"]
                    price_error += float((predicted["price"][valid] - fields["price"][valid]).abs().sum())
                    if "title" in names:
                        valid = fields["title_observed"]
                        cosine += float(nn.functional.cosine_similarity(predicted["title"][valid], fields["title"][valid]).sum())
                        title_squared += float(fields["title"][valid].square().sum())
                    norm_sum += float(latent.norm(dim=-1).sum());examples += len(latent)
                    if capture is not None:
                        capture["features"].append(features.cpu().numpy());capture["latent"].append(latent.cpu().numpy())
    finally:
        for module, mode in modes.items():
            module.training = mode
    result = {name: {"loss": sums[name] / counts[name] if counts[name] else None,
        "observed_items": counts[name]} for name in names}
    result["brand"]["accuracy"] = correct / counts["brand"] if counts["brand"] else None
    result["categories"]["micro_f1_at_probability_0_5"] = 2 * tp / max(2 * tp + fp + fn, 1)
    result["price"]["mae_standardized"] = price_error / counts["price"] if counts["price"] else None
    if "title" in names and counts["title"]:
        result["title"].update(cosine_similarity=cosine / counts["title"],
            zero_predictor_mse=title_squared / (counts["title"] * model.feature_encoder.title_dim))
        titles = arrays["title"][arrays["title_observed"]]
        result["title"]["catalogue_mean_predictor_mse"] = float(np.square(titles - titles.mean(0)).mean())
    return {"field_reconstruction": result, "latent_norm_mean": norm_sum / examples,
        "field_weights": dict(zip(names, model.uncertainty.weights().detach().cpu().tolist())),
        "field_log_variances": model.uncertainty.loss_scale.detach().cpu().tolist()}


def fit_field_autoencoder(texts, representation, spec, seed, device, source_root, recipe_path, output, *,
                          title_embeddings=None, title_observed=None):
    from torch.utils.data import DataLoader

    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    for key in ("max_steps", "batch_size", "diagnostics_interval_steps"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(f"autoencoder {key} must be a positive integer")
    if spec["batch_size"] < 2 or len(texts) % spec["batch_size"] == 1:
        raise ValueError("BatchNorm requires at least two items per batch")
    before = time.monotonic()
    schema, arrays = prepare_fields(texts, title_embeddings=title_embeddings, title_observed=title_observed)
    if schema["fields"] != representation["fields"]:
        raise ValueError("continuous control fields differ from its declaration")
    _write(output / "feature-schema.json", schema)
    np.savez_compressed(output / "catalogue-fields.npz", **arrays)
    preparation_seconds = time.monotonic() - before
    torch.manual_seed(seed)
    model, optimizer_factory, resolved = build_field_autoencoder(source_root, recipe_path, schema, spec, representation["field_dim"])
    model.to(device)
    optimizer = optimizer_factory(params=model.parameters())
    data_rng = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(len(texts)), batch_size=spec["batch_size"], shuffle=True,
        drop_last=False, num_workers=0, generator=data_rng)
    curves, evaluations = [], []
    started, updates, exposures = time.monotonic(), 0, 0

    def evaluate(step):
        row = {"step": step, **evaluate_autoencoder(model, arrays, device, spec["batch_size"])}
        evaluations.append(row)
        _write(output / "reconstruction-learning-curves.json", evaluations)
        _event("continuous_autoencoder_evaluation", **row)

    evaluate(0)
    while updates < spec["max_steps"]:
        for indices in loader:
            model.train()
            fields = tensor_fields(arrays, indices.numpy(), device)
            optimizer.zero_grad(set_to_none=True)
            _, _, predictions = model(fields)
            loss, losses, counts = model.objective(predictions, fields)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite continuous autoencoder loss")
            loss.backward()
            norms = {}
            if (updates + 1) % spec["diagnostics_interval_steps"] == 0 or updates + 1 == spec["max_steps"]:
                for name in ("feature_encoder", "encoder", "decoder", "field_heads", "uncertainty"):
                    values = [p.grad.detach().square().sum() for p in getattr(model, name).parameters() if p.grad is not None]
                    norms[name] = float(torch.stack(values).sum().sqrt()) if values else 0.0
                if not all(np.isfinite(value) for value in norms.values()):
                    raise FloatingPointError("nonfinite continuous autoencoder gradients")
            optimizer.step()
            updates += 1;exposures += len(indices)
            if updates % spec["diagnostics_interval_steps"] == 0 or updates == spec["max_steps"]:
                row = {"step": updates, "total_loss": float(loss.detach()),
                    "raw_field_losses": {name: float(value.detach()) for name, value in losses.items()},
                    "field_observed_counts": counts, "scope": "last_training_batch", "gradient_norms": norms,
                    "field_weights": dict(zip(model.field_names, model.uncertainty.weights().detach().cpu().tolist()))}
                curves.append(row);_write(output / "field-learning-curves.json", curves)
                _event("continuous_autoencoder_progress", **row)
                evaluate(updates)
            if updates == spec["max_steps"]:
                break
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "spec": spec,
        "representation": representation, "feature_schema": schema, "resolved_model": resolved,
        "seed": seed, "steps": updates, "data_rng": data_rng.get_state(),
        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "quantization_enabled": False, "automatic_resume": False}, output / "autoencoder.pt")
    capture = {}
    final = evaluate_autoencoder(model, arrays, device, spec["batch_size"], capture=capture)
    np.save(output / "item-embeddings.npy", np.concatenate(capture["features"]), allow_pickle=False)
    np.save(output / "latent-embeddings.npy", np.concatenate(capture["latent"]), allow_pickle=False)
    if final["field_reconstruction"]["price"]["mae_standardized"] is not None:
        final["field_reconstruction"]["price"]["mae_original_units"] = final["field_reconstruction"]["price"]["mae_standardized"] * schema["price_scale"]
    report = {"kind": "continuous_field_autoencoder", "quantization_enabled": False,
        "seed": seed, "steps": updates, "precision": "fp32", "item_exposures": exposures,
        "parameters": sum(p.numel() for p in model.parameters()), "elapsed_seconds": time.monotonic() - started,
        "input_dim": model.feature_encoder.output_dim, "latent_dim": model.encoder.output_dim,
        "optimizer": type(optimizer).__name__, "learning_rate": optimizer.param_groups[0]["lr"],
        "objective": "same uncertainty-weighted field reconstruction; no codebook/commitment loss",
        "checkpoint_sha256": file_sha256(output / "autoencoder.pt"),
        "source_recipe_sha256": file_sha256(Path(recipe_path)), **final}
    _write(output / "autoencoder.json", report)
    representation_report = {**representation, "seconds": preparation_seconds,
        "shape": [len(texts), model.feature_encoder.output_dim],
        "sha256": file_sha256(output / "item-embeddings.npy")}
    _write(output / "representation.json", representation_report)
    return None, report, representation_report
