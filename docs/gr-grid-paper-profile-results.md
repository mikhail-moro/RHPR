# GRID Beauty paper-recipe runtime profile

Completed on Tesla T4, native Torch 2.11.0+cu128, source
`b390cf6cf095286ca645b56c974c2cc41b912f22`. The profiler took 370.3 seconds;
the Kaggle worker used approximately 435.1 seconds of GPU quota including startup.
[Workflow](https://github.com/mikhail-moro/RHPR/actions/runs/39912619214),
[configuration](../experiments/amazon_beauty_grid_paper_profile.json),
[aggregate measurements and artifact hashes](gr-grid-paper-profile-results.json).

This is a runtime profile, not a quality result. The encoder is actual Flan-T5-XL;
the 13,128,064-parameter GRID generator is fresh and uses structural runtime SID
assignments. Training uses the unchanged author collator on real Beauty histories.
Test targets are not used. Short timing samples cannot predict convergence.

## Batches and precision

All training cases preserve effective batch 256. Peak memory below is allocated
memory in GiB; allocator-reserved cache is not intrinsic per-case demand.

| Physical train batch | FP32 seconds/update | FP16 seconds/update | FP32 / FP16 GiB |
| --- | ---: | ---: | ---: |
| 32 | .650 | .492 | .80 / .63 |
| 64 | .583 | .320 | 1.39 / 1.02 |
| 128 | .550 | .266 | 2.54 / 1.79 |
| 256 | .540 | .250 | 4.79 / 3.29 |

Batch 256 is the sensible training choice on T4. Its gain over 128 is only
1.9% in FP32 and 6.1% in FP16. All measured updates applied successfully;
FP16 training is 2.16× faster in this short diagnostic. This does not establish
equivalent convergence or final quality. The FP32 source reference remains
available without changing any architecture or effective batch setting.

GPU-resident versus synchronous full-pipeline updates have similar FP32 timing.
The main generator cost is model computation. The separate instrumented step
measured forward/loss .192 seconds, backward .337 seconds, cold Adam .011
seconds, CPU augmentation .040 seconds and H2D .0004 seconds. Instrumentation
changes timings; these stages must not be added to the normal update cost.

Validation beam generation on 512 users extrapolates to approximately 31.9
seconds for all 22,363 users in FP32 at batch 256. Batch 128 is nearly as fast
(32.3 seconds) with 1.85 GiB allocated rather than 3.64 GiB. FP16 batch 256
extrapolates to 16.7 seconds, but changes top1 for 4/512 users even on the fresh
model. Keep FP32 validation as the reference for checkpoint selection.

Flan-T5-XL FP32 batch 8 reached 13.0 items/second, projecting to approximately
15.5 minutes for 12,101 items. Batches 16 and 32 offered no speed gain. All three
FP16 encoder cases failed finite-output checks. Use FP32 batch 8 for embeddings;
cache the resulting matrix across generator runs. Whole-catalogue encoding was
not performed by this profile.

## The expensive RK implementation

The source's eight rank-local update contributions, executed sequentially on
one T4, took 2.087 seconds per logical step including all three residual
assignment stages. For 1,000 steps per level and three levels, the steady
operator projection is approximately 104.4 minutes. Per-level initialization,
dataset iteration and logging are additional. This is not measured full fitting
or an estimate of eight physical GPUs' elapsed time.

The cost is consistent with the source distance implementation, rather than
extra helper overhead: 24 assignment stages times the measured local .08661
seconds gives 2.079 seconds. Source `SquaredEuclideanDistance.compute` constructs
chunked `[256,256,2048]` broadcast differences, with 512 MiB intermediates.
The code moves much more data than a matrix-based distance calculation needs.

Before a full RK run, test the mathematically equivalent
`||x||² + ||c||² - 2 x @ c.T` distance implementation against the source. FP32
rounding, nearest-centroid ties and centroid trajectories require explicit
checks. No replacement or speedup is claimed by this profile. Another potential
optimization is skipping future residual stages that do not affect the active
layer's loss/counts, after an upstream-equivalence check.

## Cost at fixed exposure budgets

With a validation pass every 100 updates and FP32 validation throughout:

| 10,000 generator updates / 2.56M examples | Generator + 100 validations | Plus encoder and current RK operator projection |
| --- | ---: | ---: |
| FP32 training | 143.1 min | 263.0 min / 4.38 h |
| FP16 training | 94.9 min | 214.8 min / 3.58 h |

The totals exclude some initialization, I/O, checkpoint and test costs. They
are exposure-budget estimates, not time to convergence. After profiling the
reported weekly GPU remainder was approximately 4.11 hours. A literal FP32
10k-update run with the current RK implementation exceeds that remainder.

Recommended next step: improve and verify the RK distance kernel first, retain
cached FP32 embeddings/tokenization, use train batch 256, and retain FP32
validation (batch 128 or 256). No full quality training was submitted with this
profile. The article's training settings take priority; any precision or kernel
change must remain a declared numerical implementation choice.
