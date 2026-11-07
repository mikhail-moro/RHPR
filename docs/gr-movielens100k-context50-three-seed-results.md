# Context50 replication: three pipeline seeds

Collected 2026-11-04 (Europe/Moscow). Private Kaggle replication
`carasik/rhpr-ml100k-context50-replication/1`, executable
`d8027d0d84d375659335680f63468423718c4a37`, completed four models in **127.75 min**.
Reuse completed seed42 context50 models from `carasik/rhpr-ml100k-context50/1`
(`d79fdcee647729dfc1335e690aaf09b724439ab4`); seed42 was not retrained.
No new GPU training or inference was launched during result collection.

## Main result: validation and test disagree

NDCG@10, original unfiltered beam20 protocol, selection within the same 30 epochs:

| Seed | Control validation | Graph validation | Relative graph/control |
|---|---:|---:|---:|
| 42 | 0.062088 | 0.067638 | +8.94% |
| 101 | 0.064834 | 0.062695 | -3.30% |
| 202 | 0.063707 | 0.071869 | +12.81% |
| Mean | **0.063543** | **0.067401** | **+6.07%** |

| Seed | Control test | Graph test | Relative graph/control |
|---|---:|---:|---:|
| 42 | 0.047234 | 0.042117 | -10.83% |
| 101 | 0.042283 | 0.048720 | +15.22% |
| 202 | 0.050273 | 0.037783 | -24.84% |
| Mean | **0.046596** | **0.042873** | **-7.99%** |

Graph wins validation on 2/3 seeds, but test on 1/3. The graph/control sign reverses
between validation and test on each of the three seeds. This is a descriptive
pattern, not a demonstrated mechanism or evidence of a split implementation bug.
On the two new seeds alone, the mean graph/control differences are +4.69% on
validation and -6.54% on test. The overall validation lead therefore is not solely
the reused seed42 outcome, but it does not establish a heldout test gain.

Sample standard deviations across pipeline seeds: validation control.001380,
graph.004591; test control.004033, graph.005508. Each seed uses its original frozen
seed-specific index and fresh generator initialization. This does not separate
index variability from generator variability on one common SID mapping.

## Context length signal

Original context20 validation is restricted to the same first30 epochs and
check schedule. Context40 uses the already completed corresponding models.

| Window | Control validation mean | Graph validation mean | Control test mean | Graph test mean |
|---|---:|---:|---:|---:|
| 20 | 0.068808 | 0.062344 | Not budget matched | Not budget matched |
| 40 | 0.067885 | 0.065963 | 0.046939 | 0.051527 |
| 50 | 0.063543 | 0.067401 | 0.046596 | 0.042873 |

For graph20 -> 50, validation changes are **+6.60% / +14.61% / +4.35%** for
seeds42/101/202: positive 3/3, aggregate **+8.11%**. Control20 -> 50 is negative 3/3:
-6.72% / -6.09% / -10.06%, aggregate **-7.65%**.

Graph40 -> 50 validation is +2.50% / +5.62% / -0.93% (aggregate **+2.18%**),
so per-seed validation is not universally monotonic with width.
Graph40 -> 50 test changes are -28.46% / +2.38% / -21.48% (aggregate **-16.80%**).
Control40 -> 50 test changes are +3.50% / -13.03% / +7.97% (aggregate -0.73%).
On the two new seeds alone graph test falls9.62% from40 to50, also with mixed
per-seed directions. The old context40test mean is elevated by the seed42 spike,
whose magnitude was not reproduced on101/202.

The validation pattern is worth preserving, but the test outcomes do not support
claiming that longer context establishes an overall graph advantage or that50 is
an optimal width. Historical context20 test checkpoints were often selected after
30 epochs; no matched missing test checkpoints were reconstructed or generated.

## Uncertainty and fixed epochs

Shared-user bootstrap,20,000 resamples conditional on the fitted models:

| Contrast | Absolute NDCG@10 delta | Conditional95% interval |
|---|---:|---:|
| Graph50 - Control50, validation | +0.003858 | [-0.00375, +0.01154] |
| Graph50 - Control50, test | -0.003723 | [-0.01003, +0.00252] |
| Graph50 - Graph20, validation | +0.005057 | [-0.00144, +0.01176] |
| Graph50 - Graph40, validation | +0.001438 | [-0.00536, +0.00811] |
| Graph50 - Graph40, test | -0.008654 | [-0.01539, -0.00214] |

These do not cover training/index seed uncertainty, repeated inspection of the
same heldouts, window selection or multiple-comparison correction. The final
negative interval describes these fitted models and is not a general theorem
that50 is worse. For the two new seeds alone that last interval crosses0.

At fixed validation epochs20/25/30, graph mean advantages over control are
+4.70% / +7.32% / +1.34%. Individual signs remain mixed. Thus the mean validation
lead is not solely best-checkpoint selection; its magnitude and robustness remain
uncertain. Every trajectory and all predeclared validation checks are retained.

## Repetition and invalid paths

At context50, mean known-film occupancy of top10 is40.10% / 40.26% on validation
and **40.42% / 40.35% on test**, control/graph respectively. Repetition remains
substantial and the mean graph excess has disappeared; this is not an explanation
of the ranking differences or evidence that popularity bias has been resolved.

One exception to otherwise valid new test outputs: control101 emits **63 invalid
SID paths out of18,860 beam paths (0.334%)**, including20 top10 slots across20users.
No new test outputs contain duplicate paths. Official evaluation keeps the
original ranks and invalid paths as misses, exactly as in previous runs.
A CPU-only diagnostic removing invalid paths from the already saved beam and
compacting its ranks leaves control101test NDCG@10 **unchanged at0.0422826283**.
No new generation was performed and this diagnostic does not replace the official
metrics. It does not explain the observed graph/control NDCG difference.

## Training, timing and integrity

All six context50 models completed30epochs/10,020optimizer updates/14valchecks
with **zero skipped AMP updates**. New selected epochs: control10125, graph10122,
control20229, graph20228; original42both26. Test is evaluated once at the selected
validation checkpoint. Per-model7200s and session180min guards were not reached.
This capped experiment does not establish convergence.

New mean training epoch times: control101124.72s/control202124.99s,
graph101123.32s/graph202123.46s. Peak PyTorch allocated fit+validation memory is
2.400-2.402GiB. These are allocated tensor memory figures, not total/reserved
device memory. The four-model runner took127.75min with two models in parallel.
No OOM, precision/batch/hardware fallback or numerical training failure occurred.

Unchanged architecture/protocol: ordinary eager T5, TIGER4+4/d128/FF1024,
6heads*d_kv64, ReLU, learned relative bias;201 encoder tokens including the user token. FP16AMP/FP32parameters and scaler, physical128/effective256, FP32unfiltered
beam20eval. Original20epoch prefix validates5/10/15/20; verified optimizer/scaler/
RNGresume continues21-30 with validation each epoch and the original LR schedule.
No graph/encoder/quantizer training; corresponding historical SID indices frozen.

Independent CPU audit rescored **119 distinct prediction files to1e-12**, including
all 84 validation + 6 test files across the six context50models,12 original context-20 references,
and17 distinct selected/fixed-epoch-30 context-40 files. Exact configs/executableSHAs, parent
index/lineage fingerprints, initializers, all943user/target orders,85,470training
examples, split hashes and last40context suffixes verified. New root aggregates
match trial results. Every resume proof restores optimizer/scaler/RNG. The runtime
was validated before launch:958repository tests passed,91skipped; four full-size
constructor/backward checks passed. Collection changed no production runtime code.

Caveats: previously examined common heldouts; one target per split/user; only
three pipeline seeds; fixed input padding and prepended user token change relative
positions when width changes. Fitting and dropout trajectories change as well.
This comparison does not isolate the causal value of added older interactions.

## Evidence

Numerical summary:`docs/gr-movielens100k-context50-three-seed-results.json`.
Temporary `/tmp/context50-three-seed-analysis` stores metrics/aggregates/contrasts/
curves/fixedepochs inJSON/CSV, per-user cases, fingerprints, diagnostics and replay
scripts. The evidence ZIP contains raw numerical outputs/reference inputs without
model weights or credentials. Large checkpoints remain on the original private
Kaggle runs. Temporary files may not survive environment replacement; report and
summary are committed. No further training, GPU inference, dispatch or tuning
was started during collection.
