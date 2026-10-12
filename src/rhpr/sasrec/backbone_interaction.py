"""Offline paired architecture-by-context analysis of two completed kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rhpr.sasrec.depth_diagnostics import _bootstrap_means


IDENTITY_FIELDS = ("user_id", "split", "context_hash", "target", "seed", "time_enabled",
                   "observed_length_raw", "window_length", "old_length", "tail_length",
                   "old_unique_fraction", "target_seen_tail", "target_seen_old",
                   "target_seen_only_old", "target_not_in_window",
                   "short_input_context_hash", "long_input_context_hash")


def analyze_interaction(sasrec_rows: list[dict[str, Any]], hstu_rows: list[dict[str, Any]], *,
                        samples: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Resample the same users jointly for all four independently trained models."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    indexed = {}
    for architecture, rows in (("sasrec", sasrec_rows), ("hstu", hstu_rows)):
        indexed[architecture] = {}
        for row in rows:
            user = row["user_id"]
            if user in indexed[architecture] or row.get("architecture") != architecture:
                raise ValueError("rows must have unique users and the expected architecture")
            if row.get("split") != "validation" or row.get("seed") != seed or row.get("time_enabled") is not False:
                raise ValueError("interaction requires the common validation/seed/Time OFF protocol")
            indexed[architecture][user] = row
    if not indexed["sasrec"] or indexed["sasrec"].keys() != indexed["hstu"].keys():
        raise ValueError("architecture comparison requires identical nonempty user sets")
    joined = []
    for user in sorted(indexed["sasrec"]):
        sasrec, hstu = indexed["sasrec"][user], indexed["hstu"][user]
        if any(sasrec[field] != hstu[field] for field in IDENTITY_FIELDS):
            raise ValueError(f"architectures differ in evaluation identity for user {user}")
        for row in (sasrec, hstu):
            for mode in ("short", "long"):
                for metric in ("ndcg_at_10", "recall_at_10"):
                    value = row[f"{mode}_{metric}"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("per-user scores must be finite numbers in [0,1]")
        joined.append((sasrec, hstu))
    cohorts = {"all": joined,
               "old_prefix_present": [pair for pair in joined if pair[0]["old_length"] > 0],
               "history_gt512": [pair for pair in joined if pair[0]["window_length"] > 512],
               "target_seen_only_old": [pair for pair in joined if pair[0]["target_seen_only_old"]]}
    analyses = []
    for name, pairs in cohorts.items():
        result = {"cohort": name, "users": len(pairs), "metrics": {}}
        for metric in ("ndcg_at_10", "recall_at_10"):
            if not pairs:
                result["metrics"][metric] = None
                continue
            values = np.asarray([[s[f"short_{metric}"], s[f"long_{metric}"],
                                  h[f"short_{metric}"], h[f"long_{metric}"]]
                                 for s, h in pairs], dtype=np.float64)
            stable = hashlib.sha256(json.dumps([seed, name, metric]).encode()).digest()
            draws = _bootstrap_means(values, samples, np.random.default_rng(int.from_bytes(stable[:8], "little")))
            means = values.mean(axis=0)
            interaction = (means[3] - means[2]) - (means[1] - means[0])
            interaction_draws = (draws[:, 3] - draws[:, 2]) - (draws[:, 1] - draws[:, 0])
            stats = {"sasrec_short": float(means[0]), "sasrec_long": float(means[1]),
                     "hstu_short": float(means[2]), "hstu_long": float(means[3]),
                     "sasrec_long_gain": float(means[1] - means[0]),
                     "hstu_long_gain": float(means[3] - means[2]),
                     "interaction_absolute": float(interaction),
                     "interaction_absolute_ci95": np.quantile(interaction_draws, [.025, .975]).tolist(),
                     "interaction_relative_gain_difference_pp": None,
                     "interaction_relative_gain_difference_pp_ci95": None}
            if means[0] > 0 and means[2] > 0:
                stats["interaction_relative_gain_difference_pp"] = float(
                    100 * (means[3] / means[2] - means[1] / means[0]))
            if np.all(draws[:, 0] > 0) and np.all(draws[:, 2] > 0):
                stats["interaction_relative_gain_difference_pp_ci95"] = np.quantile(
                    100 * (draws[:, 3] / draws[:, 2] - draws[:, 1] / draws[:, 0]), [.025, .975]).tolist()
            result["metrics"][metric] = stats
        analyses.append(result)
    return {"schema_version": 1, "comparison": "(HSTU1028-HSTU128)-(SASRec1028-SASRec128)",
            "bootstrap_unit": "joint_paired_validation_user", "bootstrap_samples": samples,
            "seed": seed, "ci": "percentile95_unadjusted", "cohorts": analyses,
            "limitations": ["Conditional on four chosen checkpoints from one training seed; training and selection uncertainty excluded.",
                            "Common fixed15epoch recipe is a screen, not converged per-architecture HPO.",
                            "Target-seen-only-old uses a held-out label for offline diagnosis; unavailable to a production router.",
                            "HSTU reference-kernel speed does not establish optimized production-kernel speed."]}


def compare_outputs(sasrec_dir: Path, hstu_dir: Path, output_dir: Path, *,
                    samples: int = 10000) -> dict[str, Any]:
    import polars as pl

    manifests = []
    for architecture, directory in (("sasrec", sasrec_dir), ("hstu", hstu_dir)):
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("backbone_context_complete") is not True or metrics.get("architecture") != architecture:
            raise ValueError("compare only completed backbone-context kernels")
        manifest = json.loads((directory / "run-manifest.json").read_text())
        base = json.loads(json.dumps(manifest["config"]["base"]))
        if base["model"].pop("architecture") != architecture:
            raise ValueError("manifest architecture mismatch")
        base.pop("name", None)
        manifests.append((manifest, base))
    if manifests[0][1] != manifests[1][1] or manifests[0][0]["git_sha"] != manifests[1][0]["git_sha"]:
        raise ValueError("kernels must use identical common recipe and code revision")
    result = analyze_interaction(pl.read_parquet(sasrec_dir / "paired-predictions.parquet").to_dicts(),
                                 pl.read_parquet(hstu_dir / "paired-predictions.parquet").to_dicts(), samples=samples)
    result["git_sha"] = manifests[0][0]["git_sha"]
    result["sources"] = {"sasrec": str(sasrec_dir), "hstu": str(hstu_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backbone-interaction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Backbone × context interaction", "", result["comparison"], "",
             "Positive interaction means that longer history gives HSTU a larger absolute gain.", "",
             "| Cohort | Users | SASRec NDCG gain | HSTU NDCG gain | Interaction | 95% CI |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for cohort in result["cohorts"]:
        stats = cohort["metrics"]["ndcg_at_10"]
        if stats is None:
            continue
        lines.append(f"| {cohort['cohort']} | {cohort['users']} | {stats['sasrec_long_gain']:.6f} | "
                     f"{stats['hstu_long_gain']:.6f} | {stats['interaction_absolute']:.6f} | "
                     f"{stats['interaction_absolute_ci95']} |")
    lines.extend(["", *result["limitations"]])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sasrec", type=Path, required=True)
    parser.add_argument("--hstu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    compare_outputs(args.sasrec, args.hstu, args.output, samples=args.bootstrap_samples)


if __name__ == "__main__":
    main()
