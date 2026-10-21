# Historical analysis sources

These are the ten one-off analysis/download scripts used for the final
QSSM2 and music-data review on 20 October 2026. They are retained as research
source snapshots, not as a new supported runner or SID/GR implementation.

Only executable path literals were adapted: `RHPR_RESEARCH_WORKSPACE` selects
the scratch directory (default `/workspace`), and the repository's `src`
directory is located relative to this archive. A scratch directory inside the
checkout is rejected. Seeds, registered recipes, formulas, assertions and
analysis logic are unchanged. Historical printed messages and generated
Markdown links can still mention the original `/workspace` layout.

[source-manifest.json](source-manifest.json) records the original and archived
SHA256 for each script, the rewritten path literals and the AST-equivalence
check after reversing those path substitutions. Scripts were compiled without
executing their data/network operations during archival.

**Do not import these snapshots as modules.** The original scripts execute at
top level. Execute an individual file deliberately after restoring its inputs.
The two `fetch` scripts perform public HTTP downloads when executed; the other
scripts run CPU analysis only. Nothing here is invoked automatically by the
experiment runner or CI.

## Inputs and original scratch layout

Prepared event data, per-user ranks, input-context fingerprints, sparse Parquet
projections, source downloads and model weights are not committed. Restore the
private prepared cohort and matching completed-run outputs from the provenance
recorded in the [archive index](../README.md), retaining their original hashes.
The scripts intentionally assert exact catalogue, schedule and context support;
do not remove those checks to accommodate unrelated inputs.

```text
$RHPR_RESEARCH_WORKSPACE/
  plateau-cohort/                 train.parquet, catalog.parquet, cohort-summary.json
  baseline-evaluation/            daily-evaluation-summary.json, daily-anchors.parquet,
                                 daily-future.parquet, history-features.parquet
  ssm2-state128-final-output/      query-ssm2-training.json, metrics.json,
                                 ssm2-training-coverage.json, selected p60 ranks/contexts
  rotating-final-output/          query-rotating-windows.json, metrics.json,
                                 rotating-window-coverage.json, selected p60 ranks/contexts
  music-metadata/                 artist_item_mapping.parquet, album_item_mapping.parquet,
                                 fetch-manifest.json
  music-origin/                   listens-organic.partial.parquet, fetch-manifest.json
  qssm2-repeat-heuristic-recipes.json
  music-data-audit-recipe.json
```

Copy the two registered recipe JSON files byte-for-byte from
[aggregates/](../aggregates/) to the scratch root. The heuristic script verifies
the original recipe digest
`5ed810de5aeeb15b5181d383c28d191589fb1476477e103f4042321bf7a7efb9`.

The selected rank/context filenames are explicit in the source. They include
`query_ssm2_m32_s128-fixed_lr-p60-validation-*`,
`query_ssm_m32-rotating-p60-validation-*` and
`fuxi_linear-rotating-p60-validation-*`. No neural checkpoint is needed for
these final analyses; saved predictions are used without neural inference.

Metadata and origin downloads pin public Yambda revision
`dd6f3a19eef5866e346c3270e098baa641a44948`. The origin downloader stores only
UID/`is_organic` column chunks in a sparse container. Its approximately 4 GB
logical size is not the transferred/allocated size, and undeclared source
columns are unreadable. Do not treat it as a complete source dataset.

## Script dependencies and outputs

Use the repository's long-history Python dependencies (`numpy`, `pyarrow`,
`polars`, `torch`, `einops`) for data analyses, plus `matplotlib` for the curve
plot. The scripts resolve the current checkout for `rhpr` imports; the numerical
environment used for the archived final CPU analyses included Python 3.12 and
Torch 2.10 CPU. Native CUDA/Mamba is not required for these analyses.

| Script | Reads | Writes in scratch workspace |
|---|---|---|
| `analyze-qssm2-research-results.py` | Both completed-run outputs, selected ranks/contexts | `qssm2-research-results.json` |
| `audit-qssm2-saved-ranks.py` | Both completed-run reports and selected ranks | `qssm2-saved-ranks-audit.json` |
| `analyze-qssm2-repeat-slices.py` | Prepared data and selected ranks/contexts | `qssm2-research-repeat-slices.json` |
| `evaluate-qssm2-repeat-heuristics.py` | Registered recipe, prepared data and selected ranks/contexts | `qssm2-repeat-heuristics.json` |
| `plot-qssm2-research-curves.py` | `qssm2-research-results.json` | `qssm2-research-curves.png` and `.svg` |
| `analyze-music-data.py` | Data-audit recipe, prepared data, heuristic result | `music-data-audit.json` |
| `music-metadata-fetch.py` | Public pinned Yambda HTTP files/footer | `music-metadata/` mappings and fetch manifest |
| `analyze-music-metadata.py` | Prepared data, mappings and QSSM2 input contexts | `music-metadata-audit.json` and `.md` |
| `fetch-music-origin.py` | Public pinned Yambda UID/origin column ranges/footer | `music-origin/` projection and fetch manifest |
| `analyze-music-origin.py` | Prepared data, origin projection, saved ranks/contexts, heuristic result | `music-origin-audit.json` and `.md` |

Existing scratch outputs can be overwritten. A minimal plotting example, after
copying the archived aggregate JSON into an external scratch directory, is:

```bash
export RHPR_RESEARCH_WORKSPACE=/tmp/rhpr-history-research
mkdir -p "$RHPR_RESEARCH_WORKSPACE"
cp docs/research/closed-history-compression/aggregates/qssm2-research-results.json \
  "$RHPR_RESEARCH_WORKSPACE/"
python docs/research/closed-history-compression/analysis/plot-qssm2-research-curves.py
```

## Interpretation limits

- The repeated validation set and one training seed do not establish final-test
  performance, training-seed uncertainty or a converged quality ceiling.
- Repeat and origin groups use the eventual target only after prediction. They
  are diagnostic strata, not ready-to-use online routing inputs.
- The active long-history cohort is selected, and logs do not directly measure
  intent, mood, satisfaction or a causal recommendation effect. An item absent
  from the visible window can still have appeared earlier in the full history.

Read the [archive reports](../reports/) for the final corrected interpretation.
The scripts preserve the calculations, including their historical scope; they
do not start new training or revise the experiment protocol.

## Closure workspace

The completed run outputs and one-off files from this workspace were moved
reversibly to `/workspace/archive/history-compression-2026-10-21/`. To read
those retained inputs here, set:

```bash
export RHPR_RESEARCH_WORKSPACE=/workspace/archive/history-compression-2026-10-21
```

That local archive contains aliases to the prepared cohort, evaluation and
metadata directories that remain at the workspace root for possible reuse.
The local `workspace-cleanup-manifest.json` records every old/new path. None of
those private inputs or aliases is part of this Git archive.
