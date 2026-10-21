from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def candidate_config(base: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key in ("d_model", "n_layers", "ffn_multiplier", "dropout"):
        result["model"][key] = params[key]
    result["model"]["n_heads"] = params["d_model"] // params["head_dim"]
    for key in ("learning_rate", "weight_decay"):
        result["training"][key] = params[key]
    return result


def ranked_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: r["metrics"]["validation_ndcg_at_10"], reverse=True)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_search(config: dict[str, Any], output_dir: Path) -> dict[str, int | float | str | bool]:
    import optuna
    import torch

    from rhpr.runner import execute
    from rhpr.sasrec.experiment import evaluate_saved_checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    search = config["search"]
    base = copy.deepcopy(config["base"])
    base["kind"] = "yambda_sasrec"
    base["seed"] = int(config.get("seed", 42))
    base["evaluation"]["evaluate_test"] = False
    base["training"]["physical_batch_size"] = 8
    start = time.time()
    deadline = start + float(search["training_budget_seconds"])
    base["training"]["deadline_unix_seconds"] = deadline
    records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    planned_runs = 3 + int(search["n_trials"]) + int(search["n_finalists"])
    budget_reached = False

    def persist() -> None:
        _atomic_json(output_dir / "search-results.json", {"runs": records, "failures": failures})
        progress: dict[str, int | float | str | bool] = {
            "experiment": config["name"], "seed": base["seed"],
            "search_complete": False, "completed_runs": len(records),
            "failed_runs": len(failures), "planned_runs": planned_runs,
            "search_elapsed_seconds": time.time() - start,
        }
        if final_records:
            winner = ranked_candidates(final_records)[0]
            progress["validation_ndcg_at_10"] = winner["metrics"]["validation_ndcg_at_10"]
            _atomic_json(output_dir / "best-config.json", winner["config"])
        _atomic_json(output_dir / "metrics.json", progress)
        _atomic_json(output_dir / "run-manifest.json", {
            "git_sha": os.environ.get("GITHUB_SHA", "local"), "config": config,
            "started_at": datetime.fromtimestamp(start, UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(), "search_complete": False,
            "optuna_version": optuna.__version__, "torch_version": torch.__version__,
        })
        (output_dir / "summary.md").write_text(
            "# SASRec search in progress\n\n" +
            f"Completed {len(records)} / {planned_runs} runs. See search-results.json.\n",
            encoding="utf-8",
        )

    def run_one(candidate: dict[str, Any], label: str, epochs: int, keep: bool) -> dict[str, Any]:
        if time.time() >= deadline:
            raise TimeoutError("search training budget exhausted")
        candidate = copy.deepcopy(candidate)
        candidate["name"] = label
        candidate["training"]["max_epochs"] = epochs
        candidate["training"]["keep_checkpoint"] = keep
        directory = output_dir / "runs" / label
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        _atomic_json(path, candidate)
        print(json.dumps({"phase": "search_run_started", "run": label, "epochs": epochs}), flush=True)
        try:
            result = execute(path, directory)
            record = {"name": label, "config": candidate, "metrics": result["metrics"],
                      "checkpoint": str(directory / "best-model.pt") if keep else None}
            records.append(record)
            if keep:
                final_records.append(record)
            persist()
            return record
        except (RuntimeError, TimeoutError) as exc:
            failures.append({"name": label, "reason": str(exc)[:500]})
            persist()
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    persist()
    study = optuna.create_study(
        study_name=config["name"], direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base["seed"], n_startup_trials=3),
        storage="sqlite:///" + str(output_dir / "optuna.sqlite3"),
    )
    try:
        for name, negatives in (("bce", 1), ("sampled_softmax", 128), ("sampled_softmax", 512)):
            candidate = copy.deepcopy(base)
            candidate["loss"] = {"name": name, "num_negatives": negatives}
            try:
                run_one(candidate, f"loss-{name}-{negatives}", int(search["full_epochs"]), True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
        if not final_records:
            raise RuntimeError("all loss candidates failed")
        selected = ranked_candidates(final_records)[0]
        selected_base = selected["config"]
        model = selected_base["model"]
        training = selected_base["training"]
        study.enqueue_trial({
            "d_model": model["d_model"], "n_layers": model["n_layers"],
            "head_dim": model["d_model"] // model["n_heads"],
            "ffn_multiplier": model["ffn_multiplier"], "dropout": model["dropout"],
            "learning_rate": training["learning_rate"], "weight_decay": training["weight_decay"],
        })

        def objective(trial: Any) -> float:
            params = {
                "d_model": trial.suggest_categorical("d_model", [128, 256]),
                "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
                "head_dim": trial.suggest_categorical("head_dim", [32, 64]),
                "ffn_multiplier": trial.suggest_categorical("ffn_multiplier", [2, 4]),
                "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
                "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
                "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
            }
            result = run_one(candidate_config(selected_base, params), f"trial-{trial.number}",
                             int(search["trial_epochs"]), False)
            trial.set_user_attr("run_name", result["name"])
            return float(result["metrics"]["validation_ndcg_at_10"])

        study.optimize(objective, n_trials=int(search["n_trials"]),
                       timeout=max(0.0, deadline - time.time()), catch=(torch.cuda.OutOfMemoryError,))
        successful = [r for r in records if r["name"].startswith("trial-")]
        for index, record in enumerate(ranked_candidates(successful)[:int(search["n_finalists"])]):
            try:
                run_one(record["config"], f"finalist-{index}", int(search["full_epochs"]), True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
    except TimeoutError:
        budget_reached = True

    if not final_records:
        raise RuntimeError("search budget expired before a complete candidate was available")
    winner = ranked_candidates(final_records)[0]
    best_config = copy.deepcopy(winner["config"])
    best_config["training"].pop("deadline_unix_seconds", None)
    _atomic_json(output_dir / "best-config.json", best_config)
    shutil.copyfile(winner["checkpoint"], output_dir / "best-model.pt")
    test = evaluate_saved_checkpoint(best_config, output_dir / "best-model.pt")
    _atomic_json(output_dir / "study-trials.json", [
        {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
        for t in study.trials
    ])
    return {
        "experiment": config["name"], "seed": base["seed"], "search_complete": True,
        "optuna_version": optuna.__version__, "torch_version": str(torch.__version__),
        "search_budget_reached": budget_reached or time.time() >= deadline,
        "completed_runs": len(records), "failed_runs": len(failures), "planned_runs": planned_runs,
        "optuna_trials_completed": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
        "search_seconds": time.time() - start, "best_run": winner["name"],
        "best_epoch": winner["metrics"]["best_epoch"],
        "train_loss_at_best": winner["metrics"]["train_loss_at_best"],
        "validation_ndcg_at_10": winner["metrics"]["validation_ndcg_at_10"],
        "loss": best_config["loss"]["name"], "num_negatives": best_config["loss"]["num_negatives"],
        "d_model": best_config["model"]["d_model"], "n_layers": best_config["model"]["n_layers"],
        "n_heads": best_config["model"]["n_heads"], "max_seq_len": best_config["model"]["max_seq_len"],
        **test,
    }
