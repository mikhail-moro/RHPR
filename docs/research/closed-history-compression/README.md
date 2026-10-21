# Closed history-compression research

Closed **2026-10-21** by the user's decision to pursue a new research direction concerning **SID for GR**. This archive preserves the earlier compression/QSSM work and the subsequent music-data investigation. It does not implement SID/GR, choose a new objective, or assume a particular tokenizer, quantizer or generative model.

All proposed future runs in the historical reports are **cancelled**. Full SASRec, new CrossAttention budget comparisons, completion-based analyses and any further QSSM tuning are historical proposals, not a pending experiment queue. No new GPU run was launched for the final analysis or closure.

## Read first

- [Music data and research review](reports/music-research-review.md): final interpretation, objective/task distinctions, data audits and related literature.
- [QSSM2 results and architecture review](reports/qssm2-research-review.md): matched rotating-window comparison, uncertainty, architecture and budget caveats.
- [Music objectives and product scenarios](reports/music-objectives-literature.md) and [repetition literature](reports/music-repetition-literature.md): primary sources and limitations, not a systematic review or a novelty guarantee.
- [Archived CPU analysis scripts](analysis/README.md): historical reproduction code and private input requirements.
- [Historical result ledger](results-ledger.md): nine earlier run snapshots, selections and pinned provenance, separated by protocol.
- [Archive manifest](archive-manifest.json): original versus archived SHA256, exact-byte recipes and transformations.
- [Original PR history](pull-requests.json), [closure checks](closure-validation.json) and [workspace cleanup](workspace-cleanup.md).

## Final matched comparison

All three fresh models used H1024 + R128, 128 parallel causal targets per user/window, matched rotating train windows/negatives, seed 42, the same full catalogue and first-event-per-occupied-day validation. No new final test comparison was performed.

| Model, selected pass 60 | Macro-user NDCG@10 | NDCG@100 | Mean training pass |
| --- | ---: | ---: | ---: |
| FuXi | 0.02425166 | 0.04038278 | 39.04 s |
| QSSM state16 | 0.01830040 | 0.03361334 | 27.29 s |
| QSSM2 state128 | 0.01755053 | 0.03303159 | 59.20 s |

All reached the 60-pass budget cap, 2100 optimizer updates and 33,999,360 unique supervised target positions; all saved quality curves were still increasing. These are fixed-budget results, not demonstrated convergence ceilings. A pass presents one window per user and is not an exhaustive epoch over every training event.

Validation: 26,335 anchors, 4384 users, catalogue 1,049,612 items, 1400 OOV targets retained with zero credit. Paired user-bootstrap for QSSM2 minus FuXi NDCG@10: -0.0067011, 95% interval [-0.0080731, -0.0053823]. QSSM2 minus original QSSM interval includes zero. This describes sampling of frozen, validation-selected models and excludes training-seed and checkpoint-selection uncertainty.

Source revisions: QSSM2 training `248b1aebcb77cc0fc400138cd2b15ebadc0ee33f`; fresh rotating QSSM/FuXi `9f73990679c6730dfc4c57b6db20e1ea5e4411a1`. [QSSM2 workflow](https://github.com/mikhail-moro/RHPR/actions/runs/39334027184). Configuration and implementation remain in the repository: [QSSM2 training registration](../../query-ssm2-training.md), [rotating protocol](../../query-rotating-windows.md), [QSSM2 source](../../../src/rhpr/sasrec/query_ssm2.py). The cited run outputs and per-anchor ranks remain external private artifacts; Git does not contain their event arrays or weights.

## What changed the interpretation

| Original daily-first validation | NDCG@10 | NDCG@100 |
| --- | ---: | ---: |
| Global popularity within cohort train | 0.00192780 | 0.00592471 |
| Frequency in recent128 | 0.09724503 | 0.12615553 |
| Most-recent unique item in recent128 | 0.08791442 | 0.12103285 |

The recent-frequency NDCG@10 was independently reproduced without the global popularity fallback. Simple repetition is a strong signal; its advantage alone does not diagnose a broken neural pipeline. Direct related work reports similar music-domain behaviour, under different protocols.

- About 86% of the FuXi–QSSM2 NDCG@10 gap comes from targets already present in recent128. These target-conditioned slices are post-hoc diagnostics, not routing features.
- A fixed alternative sample, one random event per same user/day, retains 26,335 anchors. Known-ID any-past-repeat remains about 81% versus 80% for daily-first. Daily-first does amplify recent-repeat (31.41% versus 26.80%). Existing neural metrics apply only to the original anchors.
- Daily-first median preceding gap is 14.26 hours versus 3 minutes for random-event targets. The recent128 input itself spans a median 37 hours; 1152 inputs span about 16 days. Event-count cropping is not a verified session/intent boundary.
- Organic targets show 51.47% recent128-repeat versus 6.22% for recommendation-driven targets. Their within-group recent-frequency NDCG@10 is 0.163124 versus 0.010786. Source groups differ in users and contexts; this is not a causal effect of recommendation origin.
- In 25.31% of original anchors the track is absent from recent128 but its artist is present. Artist/album familiarity differs from exact-track recurrence. Album relations can include compilations; neither mapping identifies mood or playlist intent.

The final data investigation supports separating observable music scenarios before further architecture conclusions. It does not establish that long history, NIP, sequential models or ML are generally useless.

## Scientific and data limits

The cohort deliberately selects 4427 very active users with at least roughly 16K training events (89,191,505 train events). It is not representative of all listeners. The dataset revision is `dd6f3a19eef5866e346c3270e098baa641a44948`, source `yandex/yambda`, `sequential/500m/listens.parquet`.

- Existing training treats retained listens alike. Origin was restored for the final CPU audit. Completion and track duration were confirmed in the pinned source schema but **not downloaded or analysed**.
- Organic is an observed source flag, not randomized assignment, guaranteed manual choice of each track, or satisfaction. Target-origin strata are not automatically available as online prediction features.
- OOV mappings lose original identity; known-ID any-past-repeat is a lower bound. Absence from a cropped window does not establish that a track is new to the user.
- Daily bins use anonymous, globally shifted timestamps, not verified local morning/day-of-week labels. Five-second ties were excluded from strict-past contexts for alternative anchors.
- All comparisons are exploratory, single training seed, and repeatedly inspected validation. User-bootstrap intervals do not cover model-selection/training variation. No causal uplift, mood labels or validated novelty claim follows.
- FuXi used zero timestamps in these runs; CA/QSSM queries were formed from the old H suffix, not the subsequent scored raw tail. CA reads all H, and FuXi also compresses memory. The reports explain these distinctions.

## Archive contents and reproduction

`reports/` holds the final reviews, focused data/repeat/origin reports and literature notes. `aggregates/` contains numeric summaries, registered recipes, hashes and source-download metadata only. `figures/` keeps PNG and SVG for both plots. `analysis/` keeps historical one-off CPU source snapshots separately from the active runner.

`run-summaries/` preserves nine finite-scalar metrics snapshots with source selections and whitelist provenance. Its [index](run-summaries/index.json) records source hashes. These historical runs use different protocols and are not a single matched leaderboard.

Private per-user histories, per-anchor predictions, input arrays, trained weights, Parquet files, sparse download containers, chunks, credentials, environments, kernel bundles and transient logs are intentionally outside Git. Reproducing the CPU scripts requires those external inputs and the registered original working layout described in `analysis/README.md`; the archive is not a self-contained data release.

Markdown machine-local links were converted to repository/archive-relative links. Unarchived private inputs are labelled as external artifacts rather than linked to nonexistent local files. JSON machine paths were converted to `archive:` or `external-private-artifact:` provenance labels; they are not executable paths. Registered recipe files and their bytes were preserved unchanged so their embedded SHA checks remain valid. `archive-manifest.json` records the exact original and archived hashes. Original notes retain their dated scientific history beneath an explicit closure banner; their former proposed actions do not authorize new compute.
