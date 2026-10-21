# MixedDepth compression screen

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

Run `experiments/yambda_mixed_depth_15ep.json` through the existing Kaggle
workflow with GPU and private kernel override `carasik/rhpr-mixed-depth-15ep`.
The experiment embeds the exact Git revision, attaches `ploshkin/yambda`, keeps
Internet disabled and sends no Telegram messages.

## Architecture and training

Train one SASRec backbone and one shared PairMLP from fresh initialization for
15 epochs. Reuse the common recipe from the compression pilots: raw context1028,
tail128, d256, four layers/heads, FFN x4, dropout0.2, LR0.001129959000566945,
weight decay1e-5, physical batch8, effective batch128, sampled softmax512 and
seed42. Time features are OFF; no hyperparameter search is performed.

The same `mean(a,b) + GELU(Linear([a;b]))` projection is reused zero to three
times on adjacent representations of the old prefix. Its residual starts at
zero. Each pass retains an odd singleton; the most recent128 observed events
remain raw. At raw length1028, encoder lengths are1028/578/353/241 for depths
0/1/2/3. The factors1/2/4/8 describe the old prefix, not the whole input.
Ragged final groups can contain fewer than2/4/8 raw events; repeated unweighted
mean can give unequal raw weights in these boundary groups.

Select one depth independently for every physical batch with categorical
probabilities `[0.25,0.25,0.25,0.25]`. A dedicated CPU generator seeded at
`seed+271` isolates this sampling from user order, negatives and dropout. All
users in that physical batch execute the same number of passes. Reshuffling
and resampling expose users to different depths across epochs. This avoids
padding mixed-depth batches to their longest mode; it is stochastic training,
not a learned adaptive policy.

Carry each summary's right endpoint in the original raw sequence through
every pass. Apply learned recency PE only once, after the final compression,
using `n-1-source_endpoint`. Track the number of raw events in each summary
for diagnostics. No scale embedding, halting head, distillation, local merge
policy, summary cache or KV cache is added.

All modes supervise the same last min(n,128) raw states with the same sampled
softmax targets. The older summaries receive gradients through attention; the
encoder remains causal. Compression reads only the old prefix, which precedes
all supervised tail queries. Train15 is a fixed screening budget; mixed-depth
early stopping is disabled. Batch/user depth counts, losses and actual encoder
lengths are logged. Future convergence and multiple seeds are separate checks.

## One checkpoint and four evaluations

At epochs5/10/15, force deterministic depth0, then1,2,3 on all validation
users. Choose **one** checkpoint by the unweighted mean of their four
NDCG@10 values. The final four modes all load that checkpoint; independent
per-depth best checkpoints would describe a different experiment. Keep test
reserved. Each user has one chronological held-out next item, queried from
the last valid state against the full catalogue in FP32; seen items remain
eligible. Evaluation physical batch32 and catalogue chunks65536 are unchanged.

The depth0 mode shares jointly trained parameters and is not a separately
trained Full control. Old30-epoch Full/Pair results are external reference
runs with different training budgets and objectives. A weak deep output can
reflect interference between modes; a separately trained fixed deep model
is the next control before concluding that its compression factor fails.

## Cost and diagnostics

Benchmark each forced depth with the same checkpoint SHA256, hardware,
precision, sampled histories and strict B32. Use30 warmups,200 synchronized
B1 wall-time samples and20 throughput batches. Record history (including
compression and packing), scoring/top10, combined call, p50/p95, users/s,
peak allocated/reserved memory and nominal dense matmul FLOPs from actual
executed shapes. Recursive projection FLOPs include all passes. Optional
operator profiling is disabled. CPU data preparation, transfer, serving
queues and network are excluded. Homogeneous-depth results do not establish
mixed-batch production throughput or a cached-compression benefit.
Allocator cache is cleared before each depth benchmark, outside timing;
allocated/incremental allocated memory describes live tensors, while reserved
memory also depends on caching and is not a minimum VRAM requirement.

Frozen history features use the selected joint checkpoint's item embeddings.
Quartile thresholds are fitted once on validation, shared by all depths and
saved. Strict paired joins verify user, split, seed, time mode, context hash
and target across all four modes. Report the paired contrasts depths1/2/3
versus0 and depths2/3 versus1, fixed length groups, actual compression and
history-feature quartiles with user-bootstrap intervals. Cohorts are
exploratory and overlapping; intervals do not include training-seed or
checkpoint-selection uncertainty. Two misses for one user do not establish
that their history is safe to compress. No router is fitted or evaluated.

On the same fixed sample of up to200 validation histories, inspect all four
compressor outputs on CPU FP32 before PE and LayerNorm. Log raw endpoints,
actual raw span counts, old-summary norms, centered variance and angular
spread. These diagnostics run outside all latency timers.

Artifacts include flat finite `metrics.json`, `summary.md`, `run-manifest.json`,
`comparison.md`, `training-config.json`, `model-config.json`, `history.json`,
one `best-model.pt`, `depth-results.json`, per-depth `runs/depth_N/metrics.json`
and benchmarks, common prediction/history-feature JSON and Parquet,
`diagnostic-reference.json`, `feature_bins.json`, `depth-cohorts.json`,
`depth-paired_predictions.parquet`, `depth-cohort_results.parquet`, and
`compression-representations.json`/Parquet. Partial completion status and
epoch history are persisted so failures remain visible.

## Verification

Run `python3 -m unittest discover -s tests -v` and compile the package before
publishing. Tests check tied weights, depths0..3, ragged endpoints/span counts,
unchanged raw tail/targets, causal future perturbations, gradient flow through
three passes, independent RNG, common checkpoint selection, strict paired
joins, recursive FLOPs and real CPU training/evaluation/benchmark artifacts.
