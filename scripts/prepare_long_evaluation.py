"""Prepare private held-out anchors without changing the train cohort/catalogue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhpr.sasrec.long_evaluation_data import prepare_long_evaluation_data, write_prepared_long_evaluation
from rhpr.sasrec.long_history_data import load_prepared_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--train-cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-end-timestamp", required=True, type=int)
    parser.add_argument("--test-end-timestamp", required=True, type=int)
    args = parser.parse_args()
    cohort = load_prepared_cohort(args.train_cohort)
    data = prepare_long_evaluation_data(args.source, cohort,
                                      validation_end_timestamp=args.validation_end_timestamp,
                                      test_end_timestamp=args.test_end_timestamp)
    summary = write_prepared_long_evaluation(data, cohort, args.output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
