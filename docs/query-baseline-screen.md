# Matched short-history baseline screen

This run adds four fresh models to the completed CA/QSSM compressor comparison.
It uses the same H1024+R128 training proxy selected for the earlier search. It is
a fixed-recipe screen, not an independently tuned reproduction of each paper or
a benchmark of models trained on 16–28K events.

| Family | Actual training/primary validation input | New training | Physical batch | Full-transfer batch |
|---|---|---|---:|---:|
| CA baseline + SASRec | H1024+R128, compressed to K4+R128 | Reused |32|32|
| QSSM inner64 + SASRec | H1024+R128, compressed to K4+R128 | Reused |32|32|
| SASRec Recent128 | R128 | Yes |32|32|
| SASRec QueryOnly | Old-query128+R128, queries become K4 | Yes |32|32|
| Mamba4Rec | H1024+R128, standalone predictor | Yes |32|8|
| FuXi-Linear | H1024+R128, standalone predictor | Yes |16|4|

## Common targets and selection

Retain the original users, full 1,049,612-item training catalogue, global daily
split, seed42, final training-window endpoints, 128 shifted targets per user and
512 identity-bound catalogue negatives. A pass visits one final window for each
of 4,427 users; it is not an exhaustive epoch over every event. CPU-side input
crops preserve the target and negative identities. Occupied OOV events retain
their chronological slots.

Effective batch128, AdamW LR0.001, weight decay1e-5, AMP and gradient clipping
stay fixed. QueryOnly has backbone/compressor groups; standalone models and
Recent128 have one nonempty backbone group. The inherited all-parameter decay
convention also applies to native SSM parameters and is recorded explicitly.
Different physical batches and native initializers mean optimization trajectories
need not match across architectures even with the same seed.

Train at least20 and at most60 passes, validate every10, and stop after two
checks without strict improvement in macro-user NDCG@10. Select the earliest
strictly best primary-validation checkpoint independently for every fresh model.
NDCG@100 is also recorded. Evaluate the exact full catalogue on the same daily
validation anchors, including OOV targets scored as zero. No test ranks are read
and no new test predictions computed.

**Primary comparison uses the registered short inputs.** Mamba4Rec and FuXi then
receive a separate frozen-weight full-history validation. A numerical/OOM failure
of that optional transfer is recorded without discarding the primary result.
Recent128 and QueryOnly have identical effective inputs in both modes, so their
selected validation results are reused instead of computing duplicate forwards.
Report paired user bootstrap comparisons at both input settings separately.

## Architecture fidelity and interpretation

Recent128 and QueryOnly retain the exact parent SASRec initializer and32768 PE
table. Mamba4Rec retains its pinned official recommendation blocks, native Mamba1
CUDA scan/convolution, specialized state initialization and absence of absolute
PE. FuXi retains the official three-channel chunkwise blocks with chunk128,
eight temporal heads and positional width32. Its Time OFF adapter supplies zero
timestamps to the temporal channel; this is not the paper's temporal recipe.
No future timestamps are provided.

Mamba/FuXi preserve their own native embedding/block initializers. Their width128
and two blocks do not equalize parameter count, FLOPs or convergence. Neither
baseline receives a per-family hyperparameter search here; the CA/QSSM controls
were selected in earlier searches. Relative quality in this screen cannot be
claimed as the methods' independently tuned optimum.

FuXi's positional channel is a trainable sinusoid-initialized table. Rows beyond
the short training width receive no training signal, apart from optimizer decay.
Its full-history result therefore tests extrapolation into mostly untrained
positions. Full-length training of all long-input families remains a separate
experiment after this screen.

## Cost, safeguards and artifacts

Every fresh model receives two real effective Adam updates and short-context
catalogue-inference profiling before the budget gate. Long-input families also
receive full-context capacity/rate profiling at their separately registered
inference batches. Native Mamba CUDA is mandatory; no Python GPU scan substitutes
for it. No automatic LR, precision or batch fallback is applied.

The six-hour safety budget prices all four fresh models at60 passes plus all
validation checks, supported full transfers, checkpoints and statistics. Early
stopping is a possible saving, not funding for unregistered extra trials. A
primary numerical/OOM failure is isolated and reported after other arms resolve;
unexpected programming or data errors still abort the run.

Report parameters, physical/effective batches, memory, training/inference wall
rates and arithmetic estimates with their scopes. SASRec compressor observations
and native-family major-matmul formulas are not identical total-FLOP measures;
scan/conv/activation/packing and other exclusions remain explicit. These are
uncached computation profiles, not a cached-serving latency experiment.

The CA and inner64 QSSM references come from the completed private architecture
kernel at source `a8f6f0dc760624e83d1f8261bd19333cfbd4dc02`. Verify source/config,
manifest/report hashes, actual selected weight bytes, validation ranks, original
contexts, Q/R preservation and equivalent stopping-policy replay. The CA weights
originate in the preceding1K kernel; both origins are pinned. Copy the two verified
selected weights into the new output so its checkpoint paths are portable. No
control training is repeated.

All new best FP32 weights and rolling complete Adam/scaler/RNG checkpoints are
retained. Ragged sequence-preserving models gather each row's own final occupied
states before loss/FP32 conversion, preserving real targets under right padding.
The previous fixed-width compressed loss and gradients remain identical.
