# Fixed mini-TIGER transfer: two-seed average preserves the best indexes

The preselected narrow4+4 proxy at epoch10has Spearman0.80 on the four seed202
indexes and selects the same best index and top2as full TIGER at epoch30.
This is a positive limited transfer observation,not a validated universal proxy:
individual mini generator seeds give rho1.00and−0.20,and only four related
indexes with one full-generator seed each are available.

## Completion and fixed protocol

Private `carasik/rhpr-ml1m-mini-transfer/1` completed all8fits in27.823minutes,
finished2026-11-06T06:55:41.629635UTC. Executable
`31d120c8c962adfb7c9f1f4db30e18c1f8d6d153`;submission workflow42174475061.
All fits completed10epochs,2960optimizer updates,2VALchecks,151166examples per
epoch and0AMPskips. No retries,TEST evaluation,tokenizer/full-model training,
new data upload or additional compute were made during collection.

The architecture (narrow4+4,d64,d_ff512,6heads,d_kv32,1,375,040parameters),
epoch10and the two mini generator seeds31415/27182were fixed before the run.
Full references are the existing seed202generators at epoch30. Indexes were
not used to select the proxy architecture or epoch,although their full outcomes
were already part of earlier MTL research. The same VALcohort is reused.

## Primary result: VAL NDCG@10

| Index | Full TIGER epoch30 | Mini epoch10,seed31415 | Mini epoch10,seed27182 | Mini mean |
|---|---:|---:|---:|---:|
|SUM|0.071987|0.036535|0.040104|0.038320|
|UW|0.067196|0.026371|0.040631|0.033501|
|GradNorm|0.071192|0.029739|0.039645|0.034692|
|PCGrad|0.068147|0.028113|0.036160|0.032136|

Full ranking:SUM>GradNorm>PCGrad>UW.
Two-seed mini ranking:SUM>GradNorm>UW>PCGrad.
Five of six pairwise orderings agree;the top2and best candidate match exactly.
Target NDCG regret for the mini-selected candidate is0.

The primary mean-score Spearman is0.80;against best-of30fullVAL (secondary),
it is also0.80. This is within a single index-seed group,so the observation
cannot be explained solely by separating the pilot's seed42/101groups.
It does not resolve full-generator randomness or evaluation uncertainty.

## Single-generator instability

Mini seed31415has rho1.00and selects SUM. Seed27182has rho−0.20and selects UW,
the lowest full-target score,with target NDCG regret0.004792 (about6.66%of the
full-target winner). The successful two-seed average does not establish that
an arbitrary pair of seeds will reliably rank these indexes.

At epoch5(diagnostic only),the mean proxy correlation is0.40. It is not used
to select or replace the predeclared epoch10score.

## Uncertainty and descriptive baselines

Shared paired-user bootstrap5000draws,RNG20261106,resamples2786VALusers jointly
across the fixed proxy/full predictions. The conditional95%interval for rho
is[−0.60,1.00]. This excludes training randomness,index-selection uncertainty,
target-item dependence and transfer to other tokenizers/datasets.

An exact two-sided label permutation over the24possible four-index orderings
gives p=0.333. This descriptive exchangeability check is not a causal test of
tokenizer quality;four candidates give little precision. No significance claim
is supported. The full-target gap between SUM and GradNorm is only0.000795,
so the identity of the observed winner should not be treated as certain.

The existing full generator at epochs5and10has rho0.40on these four indexes,
selects SUM,and overlaps one of the full-target top2. These checkpoints share
the full target's trajectory;they are not independent generator replications.

## Speed and verification

Actual mean mini training time is38.827seconds per epoch;mean10epoch fit wall
time including evaluation is403.96seconds (6.73minutes). Two workers finish the
eight fits in27.82minutes. The historical full-model training mean is57.91seconds
per epoch,so width reduction gives about1.49xper-epoch speedup. Averaging two
mini fits doubles compute per index;parameter count is not a runtime guarantee.

The CPU replay independently reconstructed all16mini and4full epoch30VAL
prediction files:identical users/targets,NDCG/Recall@5/@10,invalid paths and
duplicates. It checked full SID/reference hashes,case digests,initialization
pins,architecture,budgets,workers,finite top-level metrics,manifest and summary.
Spearman was independently recomputed using SciPy. TEST files were not opened.
Repository regression:1009tests,91optional skips,0failures (119.835seconds).

## Decision

Retain narrow4+4/epoch10with two-generator-seed averaging as a candidate for
prioritizing tokenizer experiments. It is not yet reliable enough to discard
close candidates or replace full-generator confirmation. Single-seed scoring
is visibly unstable in this run. The data do not yet justify further shrinking
or claims that joint optimization through this proxy will transfer.

No follow-up was launched. All raw numerical outputs,full reference predictions,
verification results,plots and replay code are preserved locally in an evidence
ZIP. Generator weights remain in private Kaggle outputs.

Replay:`scripts/research/analyze_mini_transfer.py --root <download>
--data <clean-input> --output <analysis>`. The download's `full-validation`
directory contains the four original full epoch30VALnpz files. Required input
cases are TRAIN/VALonly. CPU replay requires NumPy,SciPy and Matplotlib.
