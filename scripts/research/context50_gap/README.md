# Replay: context50 split-gap diagnostics

These scripts inspect saved rankings; they do not train or call a model.
The evidence bundle supplies all inputs. Extract it under `/tmp` and run with
RHPR dependencies and `PYTHONPATH=src:.` from the RHPR checkout used in the study
(executable source `d8027d0d84d375659335680f63468423718c4a37` plus these scripts).

Run in order:

1. `analyze.py`: input/target checks, cohorts, attribution, checkpoint diagnostic.
2. `timestamp_targets.py`: timestamp-group label sensitivity from saved rankings.
3. `cohort_bands.py`: simultaneous conditional bands for the 72 inspected groups.
4. `window_checks.py`: unchanged visible histories and Comedy/Mystery comparisons.
5. `robustness.py`: support and threshold sensitivity for the long-gap example.

Results are written to `/tmp/context50-split-gap`. The scripts keep fixed original
checkpoints, indices, users and ranks. Alternate-target and checkpoint-partition
results are diagnostics, not replacement benchmark scores or independent evals.
The input catalogue, actor features/vectors and embargoed event fingerprints are
preserved. Report and conditional-inference limitations are described in
`docs/gr-movielens100k-context50-split-gap.md` and the bundled README.
