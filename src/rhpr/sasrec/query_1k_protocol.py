"""Registered 1K learning-rate search with validation-only length transfer.

Models, training examples, targets, negative draws and initialization remain
those of the pinned pilot. The shorter prefix is a compute proxy; each chosen
checkpoint is also evaluated on the full observed validation prefix, and that
full-length score selects the recipe. No held-out test set is used here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from numbers import Real
from typing import Any

from rhpr.sasrec.query_proxy_protocol import HPO_KIND, validate_proxy_hpo_config


SEARCH_1K_KIND = "yambda_query_prefix_1k_search"
_CONFIG_FIELDS = frozenset({
    "benchmark", "compressor", "data", "hpo", "kaggle", "kind", "model",
    "name", "parent", "runtime", "seed", "summary_budgets", "training", "variants",
})
_PARENT_FIELDS = frozenset({
    "kernel", "source_code_sha", "checkpoint_root", "manifest_sha256",
    "context_manifest_sha256", "experiment_config", "baselines", "checkpoints",
})
_KAGGLE_FIELDS = frozenset({
    "competition_sources", "dataset_sources", "kernel_sources", "model_sources",
})
_TRIALS = (
    {"name": "current_lr", "backbone_lr": .001, "compressor_lr": .001},
    {"name": "lower_lr", "backbone_lr": .0003, "compressor_lr": .0003},
    {"name": "faster_compressor", "backbone_lr": .001, "compressor_lr": .003},
)


def validate_1k_config(config: dict[str, Any]) -> None:
    """Fail closed on unregistered changes, including all pinned pilot inputs."""
    if not isinstance(config, dict) or set(config) != _CONFIG_FIELDS:
        raise ValueError("the registered 1K configuration fields are required exactly")
    if (config.get("kind") != SEARCH_1K_KIND
            or config.get("name") != "yambda500m-query-prefix-1k-search"):
        raise ValueError("the named query-prefix 1K search configuration is required")
    parent = config.get("parent")
    if not isinstance(parent, dict) or set(parent) != _PARENT_FIELDS:
        raise ValueError("the registered pinned parent fields are required exactly")
    if parent.get("checkpoint_root") != "/kaggle/input/rhpr-sasrec-prefix-query-pilot":
        raise ValueError("the completed private parent mount must remain pinned")
    original = parent.get("experiment_config")
    if not isinstance(original, dict) or not isinstance(original.get("training"), dict):
        raise ValueError("parent.experiment_config.training is required")
    expected_training = dict(original["training"], min_passes=30, max_passes=60,
                             budget_seconds=14400)
    if config.get("training") != expected_training:
        raise ValueError("1K search preserves parent optimization with 30–60 passes and a14400-second budget")
    expected_hpo = {
        "proxy_old_length": 1024, "min_passes": 30, "max_passes": 60,
        "validation_interval": 10, "patience": 3, "bootstrap_draws": 5000,
        "physical_batch_size": 32, "full_length_validation": True,
        "test_evaluations": 0, "trials": [dict(trial) for trial in _TRIALS],
    }
    if config.get("hpo") != expected_hpo:
        raise ValueError("1K search requires six registered trials, full-length validation and zero test evaluations")
    kaggle = config.get("kaggle")
    if not isinstance(kaggle, dict) or set(kaggle) != _KAGGLE_FIELDS:
        raise ValueError("the registered Kaggle source fields are required exactly")

    # Reuse the older registration's exact parent/config/artifact hashes and
    # source/runtime checks, replacing only the explicitly changed search plan.
    registered_proxy = copy.deepcopy(config)
    registered_proxy["kind"] = HPO_KIND
    registered_proxy["training"] = dict(original["training"], max_passes=30,
                                         budget_seconds=21600)
    registered_proxy["hpo"] = {
        "proxy_old_length": 4096, "diagnostic_old_lengths": [1024, 2048, 4096],
        "proxy_passes": 20, "final_passes": 30, "validation_interval": 10,
        "bootstrap_draws": 5000, "physical_batch_size": 32,
        "trials": [dict(trial) for trial in _TRIALS],
    }
    validate_proxy_hpo_config(registered_proxy)


def select_1k_candidate(trials: Sequence[dict[str, Any]],
                        results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a completed recipe by finite full-validation NDCG@10.

Call separately for each architecture after all its trials have run. Failed or
unfinished records cannot win, even if they contain a proxy score. Exact ties
use registered recipe order, independent of execution/result order. An empty
eligible set returns None so the runner can finish the other architecture
before reporting that the search was incomplete.
"""
    if (not isinstance(trials, Sequence) or isinstance(trials, (str, bytes))
            or not isinstance(results, Sequence) or isinstance(results, (str, bytes))):
        raise ValueError("trials and results must be sequences of named records")
    names = []
    for trial in trials:
        name = trial.get("name") if isinstance(trial, Mapping) else None
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("registered trial names must be nonempty and unique")
        names.append(name)
    by_name = {}
    for result in results:
        name = result.get("trial_name") if isinstance(result, Mapping) else None
        if name not in names or name in by_name:
            raise ValueError("results must identify unique registered trials from one architecture")
        by_name[name] = result
    selected, best_score = None, -math.inf
    for trial in trials:
        result = by_name.get(trial["name"], {})
        if result.get("status") != "success":
            continue
        metrics = result.get("best_full_validation")
        value = metrics.get("ndcg10") if isinstance(metrics, Mapping) else None
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 <= value <= 1):
            continue
        if value > best_score:
            selected, best_score = dict(trial), float(value)
    return selected
