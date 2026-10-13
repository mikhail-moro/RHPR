from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate long-history quality and routing results")
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output stem for PNG and PDF")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = json.loads((args.run_output / "long-quality.json").read_text())
    analysis = json.loads(args.analysis.read_text())
    metric_k = analysis.get("metric_k", 10)
    metric_name = f"ndcg{metric_k}"
    metric_label = f"Test NDCG@{metric_k}"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    budgets = (4, 8, 16)
    colors = {"pair_mlp": "#1665a8", "qformer": "#cd6227", "uniform": "#51864e", "mean": "#827aa1"}
    labels = {"pair_mlp": "Hierarchical MLP", "qformer": "Learned queries", "uniform": "Uniform sample", "mean": "Region mean"}
    ax = axes[0, 0]
    for kind in colors:
        values = [run["models"][f"{kind}_m{k}"]["test"][metric_name] for k in budgets]
        ax.plot(range(3), values, "o-", color=colors[kind], label=labels[kind], linewidth=2)
    ax.axhline(run["models"]["recent128"]["test"][metric_name], color="#555555", linestyle="--", label="Recent128")
    ax.set(xticks=range(3), xticklabels=budgets, xlabel="Old-prefix summary tokens", ylabel=metric_label,
           title="A  Static quality: one training seed")
    if metric_k == 10:
        ax.set_ylim(.0215, .0333)
    ax.legend(frameon=False, fontsize=9, ncols=2, loc="upper left")
    ax.grid(axis="y", alpha=.15)

    ax = axes[0, 1]
    for kind in colors:
        for budget in budgets:
            model = run["models"][f"{kind}_m{budget}"]
            milliseconds = model["trained_inference"]["metrics"]["full_b32_wall_mean_ms"]
            value = model["test"][metric_name]
            ax.scatter(milliseconds, value, color=colors[kind], s=40)
            offset = (-12, -10) if budget == 4 else (4, 3)
            if kind == "mean" and budget == 8:
                offset = (-10, -10)
            ax.annotate(str(budget), (milliseconds, value), xytext=offset, textcoords="offset points", fontsize=8, color=colors[kind])
    recent = run["models"]["recent128"]
    ax.scatter(recent["trained_inference"]["metrics"]["full_b32_wall_mean_ms"], recent["test"][metric_name], color="#555555", marker="s", s=45)
    ax.annotate("Recent128", (recent["trained_inference"]["metrics"]["full_b32_wall_mean_ms"], recent["test"][metric_name]),
                xytext=(0, -14), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Uncached full inference, B32 (ms)", ylabel=metric_label,
           title="B  Compression and catalogue scoring included")
    if metric_k == 10:
        ax.set_ylim(.020, .033)
    ax.grid(alpha=.15)

    ax = axes[1, 0]
    contrasts = ("pair16_minus_pair4", "pair16_minus_pair8", "pair8_minus_pair4")
    features = analysis["core_features_fixed_before_joining_outcomes"]
    rows = {(r["feature"], r["outcome"]): r for r in analysis["primary_compression_correlations"]}
    values = np.array([[rows[f, c]["spearman_rho"] for c in contrasts] for f in features])
    corrected_signal = any(row["bh_q_within_family"] < .05 for row in rows.values())
    heatmap = ax.imshow(values, cmap="RdBu_r", vmin=-.1, vmax=.1, aspect="auto")
    for i in range(len(features)):
        for j in range(len(contrasts)):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    ax.set(xticks=range(3), xticklabels=("16 − 4", "16 − 8", "8 − 4"), yticks=range(5),
           yticklabels=("Old item uniqueness", "Old item entropy", "Recent → old overlap", "Old duration", "Train activity"),
           xlabel="Paired NDCG difference", title="C  History associations" + (": corrected signal" if corrected_signal else ": no corrected signal"))
    fig.colorbar(heatmap, ax=ax, label="Spearman ρ", shrink=.8)

    ax = axes[1, 1]
    router = analysis["heldout_router"]
    estimates = [router["vs_validation_selected_fixed_ci"], router["vs_assignment_matched_random_mixture_ci"]]
    means = np.array([row["mean"] for row in estimates])
    errors = np.array([[row["mean"] - row["ci_low"] for row in estimates], [row["ci_high"] - row["mean"] for row in estimates]])
    ax.errorbar(means, [1, 0], xerr=errors, fmt="o", color="#1665a8", capsize=5)
    ax.axvline(0, color="#555555", linestyle="--")
    ax.set(yticks=[1, 0], yticklabels=["vs fixed 8", "vs matched mixture"], ylim=(-.5, 1.5),
           xlabel=f"Held-out Δ NDCG@{metric_k} (95% paired CI)", title=f"D  Router: {router['heldout_users']:,} held-out users")
    ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"HSTU + 16K history: quality, cost and budget routing at NDCG@{metric_k}", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
