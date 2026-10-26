"""Transfer a trained continuous AE into native RQ without changing its books."""
from pathlib import Path

from rhpr.gr.data import file_sha256

AE_MODULES = ("feature_encoder", "normalization_layer", "encoder", "decoder", "field_heads", "uncertainty")


def module_hashes(model):
    from rhpr.gr.sid_study import state_digest
    return {name: state_digest(getattr(model, name).state_dict()) for name in AE_MODULES}


def load_warmup(model, checkpoint, schema, representation, spec):
    import torch
    from rhpr.gr.sid_study import state_digest

    path = Path(checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("quantization_enabled") is not False or state["feature_schema"] != schema:
        raise ValueError("warm-up must be continuous AE with identical field schema")
    if state["representation"] != representation:
        raise ValueError("warm-up representation mismatch")
    if state["loss_policy"] != model.loss_policy or state["loss_normalizers"] != model.loss_normalizers:
        raise ValueError("warm-up objective mismatch")
    for key in ("num_hierarchies", "codebook_size", "latent_dim", "distance_backend"):
        if state["spec"].get(key) != spec.get(key):
            raise ValueError(f"warm-up architecture mismatch: {key}")
    # Strictly load every AE parameter and buffer (including BN), and nothing
    # from a discarded quantizer. Shape/key errors fail before spending updates.
    expected = {k for k in model.state_dict() if k.split(".")[0] in AE_MODULES}
    if set(state["model"]) != expected:
        raise ValueError("warm-up neural state keys mismatch")
    for name in AE_MODULES:
        prefix = name + "."
        getattr(model, name).load_state_dict({k[len(prefix):]: v for k,v in state["model"].items()
            if k.startswith(prefix)}, strict=True)
    mode = spec.get("freeze_policy", "joint")
    if mode not in {"joint", "ae", "embeddings"}:
        raise ValueError("freeze_policy must be joint, ae or embeddings")
    frozen = list(AE_MODULES) if mode == "ae" else ["feature_encoder"] if mode == "embeddings" else []
    for name in frozen:
        getattr(model, name).requires_grad_(False)
    model.ae_freeze_policy = mode
    model.ae_frozen_modules = frozen
    root = path.parent
    import numpy as np
    model.ae_anchor_latents = np.load(root / "latent-embeddings.npy", allow_pickle=False)
    model.ae_anchor_features = np.load(root / "item-embeddings.npy", allow_pickle=False)
    return {"checkpoint_sha256": file_sha256(path), "steps": state["steps"], "seed": state["seed"],
        "neural_state_sha256": state_digest(state["model"]), "freeze_policy": mode, "frozen_modules": frozen,
        "module_hashes_before": module_hashes(model),
        "optimizer_transfer": "fresh native RQ Adagrad; AE accumulators not reused",
        "initializer": "unchanged native mini-batch residual KMeans; eval AE during initialization"}


def enforce_stage_modes(model):
    if not hasattr(model, "ae_freeze_policy"):
        return
    # Native initialization changes readiness inside forward. Keep the first
    # reconstruction batch in eval too, so all arms see identical initial books.
    initializing = model.reconstruction_updates == 0
    for name in AE_MODULES:
        getattr(model, name).train(model.training and not initializing and name not in model.ae_frozen_modules)


def finish_stage(model, provenance):
    if provenance is None:
        return None
    after = module_hashes(model)
    for name in provenance["frozen_modules"]:
        if provenance["module_hashes_before"][name] != after[name]:
            raise RuntimeError(f"frozen AE module changed parameters/buffers: {name}")
    return {**provenance, "module_hashes_after": after, "frozen_state_verified": True}
