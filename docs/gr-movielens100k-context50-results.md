# MovieLens100K context 50: completed seed-42 screening

Collected on 2026-11-04 (Europe/Moscow). Private Kaggle run
`carasik/rhpr-ml100k-context50/1`, executable commit
`d79fdcee647729dfc1335e690aaf09b724439ab4`. Both models completed successfully
in **66.87 minutes**. No new training or GPU inference was launched to collect
these results.

## Main finding

On context 50, the actor graph improves validation NDCG@10 by **8.94%**, but
reduces test NDCG@10 by **10.83%** relative to the no-people control.
Both checkpoints were selected at epoch 26 by validation, without test selection.

| Context | Control validation | Graph validation | Control test | Graph test |
|---|---:|---:|---:|---:|
| 20 | 0.066559 | 0.063452 | 0.045298 | Not budget matched |
| 30 | 0.065442 | 0.064609 | 0.047900 | 0.043613 |
| 40 | 0.070235 | 0.065990 | 0.045638 | 0.058873 |
| 50 | 0.062088 | 0.067638 | 0.047234 | 0.042117 |

All entries use seed 42 and original corresponding frozen SID indices.
Validation selection is restricted to the same first 30 epochs and check schedule.
The historical context-20 graph test used its epoch-31 checkpoint; it is excluded
from the matched test comparison. Its matched validation checkpoint is epoch 24.
Contexts 30/40 are the previously completed models, not retrained controls.

For the graph, validation increases monotonically across these selected windows:
20 -> 30 -> 40 -> 50. Context 50 improves graph validation over 40 by **2.50%**
and over 20 by **6.60%**. However, graph test falls **28.46%** from 40 to 50.
Control 40 -> 50 changes by **-11.60%** on validation and **+3.50%** on test.
Thus this run does not establish that longer windows improve heldout retrieval,
or identify an optimal window. The large seed-42 context-40 test spike remains
unreliable: its earlier replication on seeds 101/202 averaged only +0.56% graph
versus control, with mixed individual outcomes.

## Fixed epochs and uncertainty

The context-50 validation lead is not solely caused by choosing different best
checkpoints. Graph/control validation differences are **+13.44% at epoch 20**,
**+35.92% at epoch 25**, **+8.94% at epoch 26**, and **+8.17% at epoch 30**.
Epoch 25 also illustrates pronounced control volatility (NDCG@10 0.04562,
versus 0.06044 at 24 and 0.06209 at 26). All curves are preserved.

Paired user bootstrap, 10,000 resamples conditional on the fitted models:

| Contrast (A - B) | Absolute NDCG@10 delta | Conditional 95% interval |
|---|---:|---:|
| Graph50 - Control50, validation | +0.005550 | [-0.00597, +0.01729] |
| Graph50 - Control50, test | -0.005117 | [-0.01628, +0.00578] |
| Graph50 - Graph40, validation | +0.001648 | [-0.01089, +0.01426] |
| Graph50 - Graph40, test | -0.016757 | [-0.02898, -0.00501] |

These intervals do not include training/index seed variability, repeated holdout
inspection, window selection or multiple-comparison correction. The negative
last interval describes these particular fitted models; it does not prove that
context 50 is generally worse. Graph50 versus Control50 test has 57 users with
higher NDCG@10, 58 with lower, and 828 tied (943 total). Counts ignore the differing
magnitudes of ranking changes and are not a significance test.

## Repetition, speed and memory

Selected-checkpoint top-10 occupancy by already-known films remains unfiltered:

| Split | Control50 | Graph50 |
|---|---:|---:|
| Validation | 38.35% | 40.49% |
| Test | 38.93% | 40.55% |

Graph repeats remain slightly higher. For seed 42 at context 40, test occupancy
was 38.82% / 40.25%. Longer context has not removed repetition. This does not
establish popularity bias or explain the quality differences.

Mean training epoch: **130.30 s control / 122.37 s graph**, versus
92.86 / 97.83 s at context 40 (+40.3% / +25.1%). Full runner elapsed time was
66.87 minutes with both models in parallel; this is not the sum of GPU time.
Peak PyTorch allocated fit+validation memory was **2.400 GiB per model**,
versus 1.839 GiB at context 40. It excludes some CUDA/library allocations and
is not total/reserved device memory. Physical batch remained 128, effective 256.
No OOM, hardware/batch fallback, nonfinite failure or skipped AMP update occurred.

## Protocol and audit

Ordinary eager T5, TIGER 4 encoder + 4 decoder layers, d_model 128, d_ff 1024,
6 heads with d_kv 64, ReLU, learned T5 relative position bias; 50 films become
201 encoder tokens including the prepended user token. FP16 AMP training,
FP32 unfiltered beam-20 validation/test. The graph weight 0.1 belongs to the
previous frozen-index construction; no encoder, graph or quantizer fitting
was performed here. Fresh generators match original seed-42 initial parameters.

Each model completed 30 epochs, 10,020 optimizer updates and 14 validation checks:
5/10/15/20, verified optimizer/scaler/RNG resume, then every epoch 21-30 with the
original plateau LR schedule. Both best epochs are 26; no test-driven checkpoint
selection or extension. The epoch cap does not establish convergence.

Independent CPU replay verified exact configs and executable SHAs, original
frozen index fingerprints/initialization, 85,470 training targets, all 943 heldout
user/target orders, last-40 context suffixes and all three split schedule hashes.
**94 distinct prediction files were independently rescored to 1e-12**, including
all 28 new validation files and 2 new test files, prior context-30/40 predictions
and selected context-20 references. New root metrics match trial records;
all new test predictions have zero invalid paths and duplicates. Curves have
30 epochs with zero AMP skips, and optimizer/scaler/RNG restoration proofs pass.
The unchanged runtime was validated before dispatch: 957 repository tests passed
(91 skipped), plus 5 native runner tests. No production code changed on collection.

Caveats: one pipeline seed; previously examined shared validation/test; ambiguous
rating-time interpretation; model fitting and dropout trajectories change with
input width. Fixed left padding and prepended user token also change relative
positions when the maximum width changes. The window comparison does not isolate
the causal value of the added older events.

## Preserved evidence

`gr-movielens100k-context50-results.json` stores metrics, contrasts, fixed epochs,
new training curves and file verification. Temporary `/tmp/context50-results`
contains CSV/JSON tables, per-user cases, charts and replay scripts. The evidence
ZIP contains raw numerical outputs and reference inputs, without model weights
or credentials. Full checkpoints remain on the private Kaggle runs. Temporary
files may not survive environment replacement; this report and summary are
committed to the research branch. No new fits, inference runs or retries pending.
