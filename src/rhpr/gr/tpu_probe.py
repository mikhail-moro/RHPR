"""Bounded native-TPU compatibility test, not recommendation training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from rhpr.gr.experiment import _event, _write


def install_probe_dependencies(resources: Path, target: Path, expected_manifest: str):
    """Reuse compatible hashed wheels without replacing native torch/XLA."""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    from rhpr.gr.data import file_sha256
    from rhpr.offline_runtime import prepare_wheel_links

    source = resources / "wheels"
    manifest_path = source / "runtime-manifest.json"
    if file_sha256(manifest_path) != expected_manifest:
        raise ValueError("TPU probe wheel manifest mismatch")
    manifest = json.loads(manifest_path.read_text())
    tags = set(sys_tags())
    selected = []
    for row in manifest["wheels"]:
        wheel_tags = parse_wheel_filename(row["filename"])[3]
        pure = all(tag.abi == "none" and tag.platform == "any" for tag in wheel_tags)
        needed_abi = row["name"].replace("_", "-").lower() in {"tokenizers", "safetensors"}
        if (pure or needed_abi) and wheel_tags & tags:
            if row["name"].replace("_", "-").lower() in {"torch", "torch-xla", "libtpu"}:
                raise ValueError("probe must retain the native torch/XLA installation")
            selected.append(row)
    links = target.parent / "probe-wheel-links"
    prepare_wheel_links(source, links, {"wheels": selected})
    for row in selected:
        if file_sha256(links / row["filename"]) != row["sha256"]:
            raise ValueError("TPU probe wheel digest mismatch")
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
               "--target", str(target), *[str(links / row["filename"]) for row in selected]]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    sys.path.insert(0, str(target))
    return [{"name": row["name"], "version": row["version"]} for row in selected]


def run_tpu_probe(config, output):
    report = {"quality_not_evaluated": True, "full_pipeline_supported": False,
              "python": platform.python_version(), "checks": {}, "stage": "native_xla_import"}
    started = time.monotonic()
    compatible = False

    def stage(name):
        report["stage"] = name
        _write(output / "tpu-compatibility.json", report)
        _event("tpu_probe_stage", stage=name)

    try:
        # Never silently turn a TPU check into a CPU execution.
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        report.update(torch=torch.__version__, torch_xla=torch_xla.__version__,
                      xla_device_type=xr.device_type())
        if xr.device_type() != "TPU":
            raise RuntimeError("compatibility must be tested on actual TPU hardware")
        device = xm.xla_device()
        stage("native_matrix_test")
        matrix = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        actual = matrix.to(device) @ matrix.to(device).T
        xm.mark_step()
        torch.testing.assert_close(actual.cpu(), matrix @ matrix.T)
        report["checks"]["native_tpu_matrix"] = True
        stage("gr_dependencies")
        from rhpr.gr.resources import unpack_grid, verify_resources
        from rhpr.offline_runtime import resolve_dataset_directory
        resources = resolve_dataset_directory(config["resources"]["path"])
        verify_resources(resources, config["resources"]["manifest_sha256"])
        scratch = Path(config["resources"]["scratch_path"])
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false", HF_HOME=str(scratch / "huggingface"))
        report["installed_compatible_wheels"] = install_probe_dependencies(
            resources, scratch / "probe-site", config["wheels_manifest_sha256"])
        from rhpr.gr.grid import build_grid_generator, prepare_grid_source
        from rhpr.gr.experiment import _loss
        from rhpr.gr.paper_protocol import collate_paper_rows
        from rhpr.gr.data import UserSequence
        import numpy as np

        root = prepare_grid_source(unpack_grid(resources / "grid-source.bin", scratch / "original"),
                                   scratch / "runtime")
        stage("grid_cpu_reference")
        sids = np.array([[i, 0, 0, 0] for i in range(32)], dtype=np.int64)
        users = [UserSequence(i, tuple((i + j) % 32 for j in range(8)), 0, 0) for i in range(4)]
        batch = collate_paper_rows(users, sids, [0, 1, 2, 3], sequence_length=120, max_augmented_batch=8)
        model = build_grid_generator(sids, config["generator"], seed=42, source_root=root)
        model.eval()
        with torch.no_grad():
            reference_loss = float(_loss(model, batch))
        model = model.to(device)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        stage("grid_xla_forward")
        with torch.no_grad():
            loss = _loss(model, batch)
        xm.mark_step()
        actual_loss = float(loss.cpu())
        if abs(actual_loss - reference_loss) > 1e-3:
            raise FloatingPointError("GRID TPU FP32 forward differs from CPU reference")
        report["checks"]["fp32_loss_absolute_difference"] = abs(actual_loss - reference_loss)
        stage("grid_bf16_update")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-6)
        first = next(model.parameters())
        before = first.detach().cpu().clone()
        losses = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("xla", dtype=torch.bfloat16):
                loss = _loss(model, batch)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            losses.append(float(loss.detach().cpu()))
        if not all(np.isfinite(losses)) or torch.equal(first.detach().cpu(), before):
            raise FloatingPointError("TPU BF16 did not produce finite, effective updates")
        report["checks"].update(bf16_updates=3, bf16_losses=losses, weights_changed=True)
        stage("grid_beam_search")
        model.eval()
        with torch.no_grad():
            ids, scores = model.generate(input_ids=batch["input_ids"][:2],
                attention_mask=batch["attention_mask_encoder"][:2], user_id=batch["user_id"][:2])
        xm.mark_step()
        ids, scores = ids.cpu(), scores.cpu()
        if tuple(ids.shape) != (2, config["generator"]["beam_width"], 4) or not torch.isfinite(scores).all():
            raise RuntimeError("invalid TPU beam-search outputs")
        report["checks"]["fp32_beam_search"] = True
        compatible = True
        stage("complete")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:1500]}
    report.update(generator_compatible=compatible, elapsed_seconds=time.monotonic() - started,
                  note="One TPU device; runtime/forward/BF16 updates/beam only. Full epoch runner, Flan encoding, RK fitting and multi-device scaling are not validated.")
    _write(output / "tpu-compatibility.json", report)
    return {"experiment": config["name"], "probe_complete": True, "quality_not_evaluated": True,
        "generator_tpu_compatible": compatible, "full_pipeline_tpu_supported": False,
        "last_stage": report["stage"], "probe_seconds": report["elapsed_seconds"]}
