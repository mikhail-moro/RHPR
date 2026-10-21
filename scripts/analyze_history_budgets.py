from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rhpr.sasrec.history_budget_analysis import run_history_budget_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore whether past history predicts compression utility")
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--metric-k", type=int, choices=(10, 100), default=10,
                        help="Saved NDCG cutoff; top-10 records cannot recover NDCG@100")
    args = parser.parse_args()
    report = run_history_budget_analysis(args.run_output, args.features, args.output,
        seed=args.seed, permutations=args.permutations, bootstrap_draws=args.bootstrap_draws,
        metric_k=args.metric_k)
    print({"metric_k": report["metric_k"], "users": report["users"], "oracle": report["oracle"],
           "heldout_router": report["heldout_router"]})


if __name__ == "__main__":
    main()
