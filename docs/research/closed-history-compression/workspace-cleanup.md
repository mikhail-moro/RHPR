# Closure workspace cleanup

On 2026-10-21, **177** completed-research files/directories were moved from the
workspace root into `/workspace/archive/history-compression-2026-10-21/`.
These include old bundles, server snapshots, integration outputs, logs,
downloaded prediction artifacts and original one-off analysis sources.
Research files were not deleted. The local archive records every original and
new path in `workspace-cleanup-manifest.json`, so the moves can be reversed.

The Git archive contains the useful reports, aggregate evidence, analysis
snapshots and plots. Raw events, model weights, per-user predictions and
source downloads remain private and are not committed.

Prepared data remain at the workspace root for possible reuse:

- `plateau-cohort/`
- `baseline-evaluation/`
- `music-metadata/`
- `music-origin/`

The local scratch archive has aliases to these inputs. The analysis snapshots
can select its layout with `RHPR_RESEARCH_WORKSPACE`; see their
[reproduction notes](analysis/README.md#closure-workspace).

Shared Python environments, Kaggle client configuration, unrelated workspace
directories and environment-managed dot-directories were left in place.
Only reproducible Python cache directories were removed from the checkout.

Experiment code, configuration, tests, vendor provenance and licenses remain
at their original repository paths. Documentation marks this research as
closed; it does not schedule SID/GR or revive historical experiment plans.
