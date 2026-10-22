# Research documentation

The history-compression / Query SSM track was closed on **2026-10-21**.
The [closure archive](research/closed-history-compression/README.md) contains
the final interpretation, aggregate evidence, analysis snapshots and literature.
Earlier plans below are historical; no experiment is queued by these documents.

The active SID/GR study is described in the [research brief](gr-research-brief.md),
[framework audit](gr-framework-audit.md), and [Beauty baseline](gr-baseline.md).
The stages below belong to the closed history-compression study.

| Stage | Documents |
| --- | --- |
| Pairwise and recursive compression | [Pilot](compression-pilot.md), [convergence](compression-convergence.md), [MixedDepth](mixed-depth.md), [deep budgets / Recent128](deep-mixed-depth-and-recent128.md), [frozen tail ablation](tail-ablation.md) |
| Predictors and compute | [SASRec / HSTU](backbone-context-comparison.md), [long-training profile](long-training-profile.md), [measured costs](long-training-profile-results.md), [runtime dependencies](profile-runtime-dependencies.md) |
| Long-history data and evaluation | [Training cohort](long-history-data.md), [temporal evaluation](long-history-evaluation-data.md), [daily anchors](daily-evaluation-data.md) |
| HSTU compression and budgets | [Pilot](hstu-long-compression-pilot.md), [results](hstu-long-compression-results.md), [MixedBudget](mixed-budget-daily-experiment.md), [aggregate JSON](hstu-long-compression-aggregate.json), [paired comparisons](hstu-long-compression-paired-comparisons.json) |
| Query-conditioned compression | [Initial QSSM](query-ssm-block-pilot.md), [prefix / parallel targets](sasrec-prefix-query-pilot.md), [LR diagnostic](query-lr-restart.md), [history ablation](query-history-ablation.md), [QueryOnly](query-only-control.md) |
| Fixed-window screening and HPO | [Proxy HPO](query-proxy-hpo.md), [1K search](query-1k-search.md), [compressor architecture](query-compressor-architecture.md), [matched baselines](query-baseline-screen.md), [QSSM memory search](query-qssm-search.md) |
| Later history and mechanism checks | [Full-history continuation](query-history-continuation.md), [Mamba features](query-mamba-ablation.md), [rotating targets](query-rotating-windows.md), [FuXi / CA interpretation](fuxi-and-cross-attention.md), [serving / caching](cached-compressor-serving.md) |
| QSSM2 | [Native benchmark](query-ssm2-benchmark.md), [state128 training](query-ssm2-training.md), [final audit](research/closed-history-compression/reports/qssm2-research-review.md) |
| Musical data and objectives | [Final review](research/closed-history-compression/reports/music-research-review.md), [data / selection audit](research/closed-history-compression/reports/music-data-audit.md), [origin audit](research/closed-history-compression/reports/music-origin-audit.md), [literature](research/closed-history-compression/reports/music-objectives-literature.md) |

## Reuse boundaries

Experiment JSON, `src/`, tests and upstream licenses remain at their original
paths. Frozen model weights, event histories, per-user ranks and downloaded
source files stay outside Git in private storage. The archive preserves
aggregate results and hashes, rather than copies of those private datasets.

The original 50M leave-last-two baseline, fixed-block 1K studies, rotating-block
studies and daily-first evaluation are different protocols. Compare results
within the stated protocol; read the final archive before reusing old claims.
