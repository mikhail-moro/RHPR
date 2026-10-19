# QSSM2 native training-step benchmark

This short experiment checks whether Mamba2's official SSD backend can reduce
the cost of this repository's query-conditioned history compressor on a Kaggle
T4. It performs no quality training, validation, test evaluation, inference
benchmark, checkpoint selection or hyperparameter search.

The three fixed arms are the selected original QSSM with native Mamba1 scan,
and two fresh QSSM2 compressors with head dimensions 32 and 64. Both SSD arms
use chunk size 64, inner width 64, state width 16, conditioning rank 32, K32,
and initial learned event steps in [1e-4, 1e-2]. Queries are generated from the
last 128 events of H1024. B and delta depend on the event and query; C depends
on the query, preserving the existing readout choice. No convolution, D-skip
or output gate is added.

Mamba2 uses scalar-times-identity transitions within each head, whereas the
original QSSM has separate transitions for state components. The SSD arms are
new architectures. A speed improvement would justify a subsequent quality
experiment, not establish quality equivalence or a better recommender.

All arms reconstruct the same selected fresh SASRec initializer, including
the full 32768-position table and untrimmed 1,049,612-item catalogue. Source
registration embedded in the configuration records the exact initializer;
its checkpoint references are metadata only. No source kernel or weights are
mounted or read. The prepared training cohort and offline runtime are the only
Kaggle dataset sources; evaluation data are unnecessary.

Seven complete effective batches of 128 distinct real users are selected from
the existing seed42 first-pass training schedule. Every arm receives exactly
the same item, shifted target, occupied-mask and identity-bound 512-negative
arrays. Array and descriptor hashes are recorded. Each window is H1024+R128,
with 128 parallel causal next-item losses; physical batch size is 32. Data are
collated once outside the timing region.

Each arm first checks tiny actual native forward/backward against the CPU
reference recurrence. Nonfinite, missing or zero critical B/delta/query
gradients fail the arm. Two full effective updates warm kernels and allocate
real dense AdamW state; five further updates are timed. This includes H2D,
compression, SASRec, sampled loss, accumulated backward, gradient clipping,
AdamW and CUDA synchronization. The optimizer, FP16 autocast, initial loss
scale 1024, LR .001 and decay 1e-5 follow the existing recipe. No overflow
replay, loss-scale reduction, reduced batch or reference GPU fallback is
permitted. Initialization and cold compilation are excluded from steady
timings and included in the recorded audit/warmup/arm duration.

An additional two warmup and five measured compressor-only forward/backward
calls operate on detached real item embeddings. This excludes embedding lookup,
SASRec, sampled loss and optimizer, and helps locate any saving. Absolute peak
allocation still includes the live whole model and dense Adam state; a separate
allocation baseline and incremental peak are reported. Neither measurement
should be read as inference latency.

Both synchronized host wall time and CUDA stream elapsed time are retained.
Stream events include launch gaps and do not measure summed kernel busy time.
Five samples and sequential arm order support a practical screening decision,
not a precise statistical claim. The runner has a 600-second ceiling checked
between operations; a single already-running kernel/compilation is not
preempted. Offline runtime installation precedes that ceiling. Failed or
unsupported SSD arms retain their concrete errors and the native1 baseline;
there is no automatic retry in another precision or backend.

Deploy this screening run with `submit_only=false` and `timeout_minutes=15`.
The existing workflow collector requests cancellation when that external
deadline expires, covering runtime setup and an already-running JIT call that
the runner's cooperative budget cannot interrupt.

The report is `query-ssm2-benchmark.json`, with raw samples, audits, GPU/runtime
versions, source revision, configuration and data provenance. The readable
table is `query-ssm2-benchmark.md`; standard execution also writes flat finite
`metrics.json`, `summary.md` and `run-manifest.json`. No model weights are saved.
