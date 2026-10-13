"""Prepare private daily next-item anchors and past-only routing descriptors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhpr.sasrec.daily_evaluation_data import (
    iter_daily_evaluation_examples, load_prepared_daily_evaluation,
    prepare_daily_evaluation_data, write_prepared_daily_evaluation,
)
from rhpr.sasrec.long_evaluation_data import iter_long_evaluation_examples, load_prepared_long_evaluation
from rhpr.sasrec.long_history_data import load_prepared_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--train-cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-end-timestamp", required=True, type=int)
    parser.add_argument("--test-end-timestamp", required=True, type=int)
    parser.add_argument("--reference-context-length", type=int, default=16_384)
    parser.add_argument("--day-seconds", type=int, default=86_400)
    parser.add_argument("--previous-evaluation", type=Path,
                        help="Verify each split's first daily anchor equals the previous period anchor")
    args = parser.parse_args()
    started = time.perf_counter()
    cohort = load_prepared_cohort(args.train_cohort)
    print(json.dumps({"phase": "training_cohort_verified", "users": len(cohort.users),
                      "catalogue_items": cohort.num_items}), flush=True)
    data = prepare_daily_evaluation_data(
        args.source, cohort, validation_end_timestamp=args.validation_end_timestamp,
        test_end_timestamp=args.test_end_timestamp, day_seconds=args.day_seconds,
        reference_context_length=args.reference_context_length,
    )
    print(json.dumps({"phase": "daily_anchors_selected", "validation_anchors": data.summary["validation_anchors"],
                      "test_anchors": data.summary["test_anchors"]}), flush=True)
    summary = write_prepared_daily_evaluation(data, cohort, args.output)
    loaded = load_prepared_daily_evaluation(args.output, cohort)
    checks = {"fixed_train_prefix_matches": summary["source_train_prefix_exact_matches"],
              "round_trip_evaluation_content_sha256": loaded.summary["evaluation_content_sha256"],
              "reference_context_length": args.reference_context_length}
    if args.previous_evaluation:
        previous = load_prepared_long_evaluation(args.previous_evaluation, cohort)
        for split in ("validation", "test"):
            first = {}
            for example in iter_daily_evaluation_examples(
                cohort, loaded, split_name=split, max_seq_len=args.reference_context_length,
            ):
                first.setdefault(example.user_id, example)
            old = {example.user_id: example for example in iter_long_evaluation_examples(
                cohort, previous, split_name=split, max_seq_len=args.reference_context_length)}
            if first.keys() != old.keys():
                raise ValueError("first daily anchor user support differs from previous evaluation")
            for uid, example in first.items():
                prior = old[uid]
                if (example.target != prior.target or example.target_timestamp != prior.target_timestamp
                        or not np.array_equal(example.context, prior.context)
                        or not np.array_equal(example.timestamps, prior.timestamps)):
                    raise ValueError(f"first daily anchor differs from previous evaluation: {split}, UID {uid}")
            checks[split + "_first_anchor_exact_previous_matches"] = len(first)
    checks["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "preparation-audit.json").write_text(
        json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "checks": checks}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
