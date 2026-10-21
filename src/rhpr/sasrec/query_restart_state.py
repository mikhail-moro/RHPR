"""Verified weight-only restarts and validation stopping for the LR diagnostic.

The parent pilot did not save Adam or RNG state. Loading its FP32 weights and
resetting RNG therefore starts a new optimization trajectory, not exact resume.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from rhpr.sasrec.query_pilot import PILOT_KIND, query_variants, validate_query_pilot_config


RESTART_KIND = "yambda_query_lr_restart"
PARENT_KERNEL = "carasik/rhpr-query-ssm-block-pilot"
PARENT_SOURCE_SHA = "d460ddd439212440ab524e37a3fdd04769782349"
PARENT_NAMES = ("recent128", "query_cross_attention_m4")
_SHARED_TRAIN_KEYS = (
    "device", "amp", "cpu_threads", "raw_context_length", "raw_tail_size",
    "supervised_targets", "effective_batch_size",
    "num_negatives", "weight_decay", "grad_clip_norm", "initial_loss_scale",
    "serving_context_length", "old_prefix_length", "query_context_size",
)


def _valid_digest(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _checkpoint_spec(config: dict[str, Any], parent_name: str) -> dict[str, Any]:
    if parent_name not in PARENT_NAMES:
        raise ValueError("unregistered parent checkpoint")
    spec = config.get("restart", {}).get("checkpoints", {}).get(parent_name)
    if not isinstance(spec, dict):
        raise ValueError(f"missing checkpoint specification for {parent_name}")
    filename = spec.get("file")
    path = PurePosixPath(filename) if isinstance(filename, str) else PurePosixPath()
    if (not isinstance(filename, str) or path.is_absolute() or ".." in path.parts
            or path.parts != ("checkpoints", f"{parent_name}.pt")):
        raise ValueError("checkpoint file must be the registered relative checkpoints/*.pt path")
    if not _valid_digest(spec.get("sha256"), 64):
        raise ValueError("checkpoint sha256 must be a SHA256 digest")
    if type(spec.get("size_bytes")) is not int or spec["size_bytes"] < 1:
        raise ValueError("checkpoint size_bytes must be a positive integer")
    return spec


def restart_arms(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Family-major arm order; each LR shares exactly the same parent weights."""
    result = []
    for name in PARENT_NAMES:
        for lr in config["restart"]["learning_rates"]:
            result.append({"name": f"{name}_lr{format(lr, '.3f').replace('.', 'p')}",
                "kind": "recent" if name == "recent128" else "query_cross_attention",
                "summary_tokens": 0 if name == "recent128" else 4,
                "learning_rate": lr, "parent_name": name})
    return result


def validate_restart_config(config: dict[str, Any]) -> None:
    """Keep the accepted four-arm experiment distinct from the original pilot."""
    if config.get("kind") != RESTART_KIND:
        raise ValueError(f"kind must be {RESTART_KIND}")
    restart = config.get("restart", {})
    if (restart.get("parent_kernel") != PARENT_KERNEL
            or restart.get("source_code_sha") != PARENT_SOURCE_SHA
            or restart.get("parent_epoch") != 48):
        raise ValueError("restart must pin the completed pilot's source and pass48")
    if restart.get("learning_rates") != [.001, .003]:
        raise ValueError("the diagnostic registers exactly learning_rates [.001,.003]")
    root = restart.get("checkpoint_root")
    if not isinstance(root, str) or Path(root).name != PARENT_KERNEL.split("/")[1]:
        raise ValueError("checkpoint_root must name the registered parent kernel")
    if config.get("variants") != list(PARENT_NAMES):
        raise ValueError("the restart uses only Recent128 and Cross Attention")
    if set(restart.get("checkpoints", {})) != set(PARENT_NAMES):
        raise ValueError("exactly the two parent checkpoints are required")
    for name in PARENT_NAMES:
        _checkpoint_spec(config, name)
    if config.get("kaggle", {}).get("kernel_sources") != [PARENT_KERNEL]:
        raise ValueError("the completed private pilot must be the sole kernel source")
    train = config.get("training", {})
    expected = {"max_additional_passes": 48, "validation_every_additional_passes": 8,
        "early_stopping_min_additional_passes": 24, "early_stopping_patience": 3}
    if any(type(train.get(key)) is not int or train[key] != value for key, value in expected.items()):
        raise ValueError("restart requires max48, validation8, minimum24 and patience3")
    if train.get("physical_batch_sizes") != [32]:
        raise ValueError("restart fixes the successful parent physical batch at32")
    # Reuse the mature shared protocol validator without treating its original
    # fixed-pass budget as this diagnostic's early-stopping policy.
    common = copy.deepcopy(config)
    common["kind"] = PILOT_KIND
    common["variants"] = [row["name"] for row in query_variants(common)]
    common["training"].update(min_passes=30, max_passes=60, validation_interval=10)
    validate_query_pilot_config(common)


def resolve_parent_checkpoint(config: dict[str, Any], parent_name: str,
        input_root: str | Path = "/kaggle/input") -> Path:
    """Resolve only the known kernel slug within four bounded mount layouts.

    Kaggle has used both /input/slug and /input/kernels/owner/slug. An explicit
    root is a hint, not permission to search unrelated data or accept duplicates.
    """
    spec = _checkpoint_spec(config, parent_name)
    input_root = Path(input_root).resolve()
    slug = PARENT_KERNEL.split("/")[1]
    configured = Path(config["restart"]["checkpoint_root"])
    if not configured.is_absolute():
        configured = input_root / configured
    candidates = {input_root / slug}
    # Do not recurse through mounted datasets or enumerate their files.
    for pattern in (f"*/{slug}", f"*/*/{slug}", f"*/*/*/{slug}"):
        candidates.update(input_root.glob(pattern))
    if configured.name != slug:
        raise ValueError("checkpoint_root must name the registered parent kernel")
    try:
        configured.resolve().relative_to(input_root)
    except ValueError as exc:
        raise ValueError("checkpoint_root must stay inside input_root") from exc
    candidates.add(configured)
    matches: set[Path] = set()
    for root in candidates:
        if not root.is_dir():
            continue
        resolved_root = root.resolve()
        try:
            resolved_root.relative_to(input_root)
        except ValueError as exc:
            raise ValueError("parent kernel mount escapes input_root") from exc
        checkpoint = root / spec["file"]
        if checkpoint.is_file():
            checkpoint = checkpoint.resolve()
            try:
                checkpoint.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError("checkpoint file escapes its parent kernel mount") from exc
            matches.add(checkpoint)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one mounted checkpoint for {parent_name}; found {len(matches)}")
    return next(iter(matches))


def _validate_parent_configuration(parent: dict[str, Any], current: dict[str, Any]) -> None:
    validate_query_pilot_config(parent)
    for key in ("seed", "summary_budgets", "group_arity", "model", "compressor"):
        if parent.get(key) != current.get(key):
            raise ValueError(f"parent checkpoint configuration mismatch: {key}")
    # Paths can differ when the same content is mounted on another machine.
    for section, ignored in (("data", {"prepared_path", "evaluation_path"}),
                              ("runtime", {"wheel_path", "target_site"})):
        expected = {k: v for k, v in current[section].items() if k not in ignored}
        actual = {k: v for k, v in parent[section].items() if k not in ignored}
        if actual != expected:
            raise ValueError(f"parent checkpoint configuration mismatch: {section}")
    for key in _SHARED_TRAIN_KEYS:
        if parent["training"].get(key) != current["training"].get(key):
            raise ValueError(f"parent checkpoint configuration mismatch: training.{key}")


def load_parent_checkpoint(config: dict[str, Any], parent_name: str, *, num_items: int,
        model: torch.nn.Module | None = None, input_root: str | Path = "/kaggle/input") -> dict[str, Any]:
    """Hash before safe deserialization; reject all metadata or tensor mismatch.

    A supplied model is changed only after the complete state has passed key,
    shape, dtype and finite-value checks. Optimizer state is intentionally absent.
    """
    spec = _checkpoint_spec(config, parent_name)
    checkpoint = resolve_parent_checkpoint(config, parent_name, input_root=input_root)
    with checkpoint.open("rb") as stream:
        if checkpoint.stat().st_size != spec["size_bytes"]:
            raise ValueError("parent checkpoint size_bytes mismatch")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != spec["sha256"]:
            raise ValueError("parent checkpoint SHA256 mismatch")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported parent checkpoint format")
    if payload.get("source_code_sha") != config["restart"]["source_code_sha"]:
        raise ValueError("parent checkpoint source_code_sha mismatch")
    if type(payload.get("best_epoch")) is not int or payload["best_epoch"] != config["restart"]["parent_epoch"]:
        raise ValueError("parent checkpoint best_epoch mismatch")
    if type(payload.get("num_items")) is not int or payload["num_items"] != num_items:
        raise ValueError("parent checkpoint num_items mismatch")
    expected_variant = {"name": parent_name, "kind": "recent" if parent_name == "recent128"
        else "query_cross_attention", "summary_tokens": 0 if parent_name == "recent128" else 4}
    if payload.get("variant") != expected_variant:
        raise ValueError("parent checkpoint variant mismatch")
    if payload.get("model_config") != config["model"]:
        raise ValueError("parent checkpoint model_config mismatch")
    parent_config = payload.get("experiment_config")
    if not isinstance(parent_config, dict):
        raise ValueError("parent checkpoint experiment_config is required")
    _validate_parent_configuration(parent_config, config)
    score = payload.get("selection_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("parent checkpoint selection_score must be finite")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state or any(not isinstance(key, str)
            or not isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("parent checkpoint must contain a tensor state_dict")
    for value in state.values():
        if value.is_floating_point():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("parent checkpoint floating state must be finite FP32")
    if model is not None:
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError("parent checkpoint state_dict keys mismatch")
        if any(state[key].shape != value.shape or state[key].dtype != value.dtype
                for key, value in expected.items()):
            raise ValueError("parent checkpoint state_dict shape or dtype mismatch")
        model.load_state_dict(state, strict=True)
    return payload


def reset_restart_rng(seed: int) -> None:
    """Reset every RNG after loading/profiling; deliberately discard parent RNG."""
    if type(seed) is not int or seed < 0:
        raise ValueError("restart seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ValidationEarlyStopping:
    """Strict validation improvement, with the parent at additional pass zero."""

    baseline_score: float
    minimum_additional_passes: int = 24
    patience: int = 3
    validation_interval: int = 8
    max_additional_passes: int = 48
    best_score: float = field(init=False)
    best_additional_passes: int = field(init=False, default=0)
    bad_checks: int = field(init=False, default=0)
    last_additional_passes: int = field(init=False, default=0)
    stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._check_score(self.baseline_score)
        values = (self.minimum_additional_passes, self.patience,
                  self.validation_interval, self.max_additional_passes)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("early-stopping counts must be positive integers")
        if (self.minimum_additional_passes > self.max_additional_passes
                or self.max_additional_passes % self.validation_interval):
            raise ValueError("invalid early-stopping horizon")
        self.best_score = self.baseline_score

    @staticmethod
    def _check_score(score: float) -> None:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("validation score must be finite")

    def observe(self, additional_passes: int, score: float) -> dict[str, Any]:
        self._check_score(score)
        if (self.stopped or type(additional_passes) is not int
                or additional_passes != self.last_additional_passes + self.validation_interval
                or additional_passes > self.max_additional_passes):
            raise ValueError("validation checks must follow the registered schedule before stopping")
        self.last_additional_passes = additional_passes
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_additional_passes = score, additional_passes
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        self.stopped = additional_passes >= self.minimum_additional_passes and self.bad_checks >= self.patience
        return {"improved": improved, "should_stop": self.stopped,
            "best_additional_passes": self.best_additional_passes,
            "best_score": self.best_score, "bad_checks": self.bad_checks}
