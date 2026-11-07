# T4 batch-throughput pilot

Authorized2026-11-05 after the native SDPA pilot. Private kernel
`carasik/rhpr-t5-batch-pilot`, config `experiments/t5_batch_t4_pilot.json`.
This run measures throughput/capacity and short FP16 stability, not recommendation
quality. User explicitly accepts that effective batches512/1024 change training.

Sixteen predeclared attempts, each in a fresh process on physical T4 GPU0:

- Backends: historical Transformers4.47 eager and native5.22.1 SDPA.
- Windows:20and50items.
- Effective/physical batches:256/128,256/256,512/512,1024/1024.
- Order alternates backends inside each window/batch pair, reducing the fixed
  all-eager-before-all-SDPA order confound of the first pilot. One sweep only;
  no randomized repetitions, independent timing confidence intervals or tuning.

Reuse the SAME byte-pinned trained sum42 checkpoint and frozen SIDs. No new
quantizer, embeddings or private dataset uploads. Existing private weights
mount directly from `carasik/rhpr-ml100k-singleton-mtl`. Public5.22.1 wheels
mount from the previously verified runtime dataset. Torch/CUDA stay unchanged.

At each width, sample1024strict-past TRAIN examples using a fixed NumPyseed42.
Each effective batch uses a prefix; paired backends and both256cases get exactly
the same examples. Record SHA256 of histories, users, targets; paired backend
hashes must match when both attempts complete. No validation/test targets are
trained. Full dataset/split/provenance hashes remain pinned.

Each attempt reloads trained weights, resets optimizer/scaler/RNG, uses
FP16/dropout.1, AdamLR.001/wd.0001,clip1,GradScaler1024. NOautomatic LR scaling.
10warmup,50timed optimizer updates with resident input tensors, then10separate
end-to-end steps including SID batch building/H2D. Losses scale by actual
microbatch fraction so256/128and256/256 optimize the same mean objective.
Different dropout random masks, floating point reduction and kernel selection
can still produce different trajectories.

Save per-step loss/time, examples/second, end-to-end examples/second, actual
attention-config flags, peak CUDA allocated/reserved memory, input hashes,
updates/examples consumed. Larger effective batches consume proportionally
more examples in the same70updates; this is NOT a quality comparison or matched
sample-budget training study. LR, epochs and production batch defaults are not
changed by the pilot.

Expected OOM is a feasibility result: record the original requested batch,
exception and memory counters, then continue the remaining PREDECLARED attempts
in fresh processes. Never lower/retry that batch. Nonfinite loss/gradient or
AMPskip is recorded as numerical_failed, not a valid throughput candidate.
Other exceptions fail the run rather than being silently reclassified. Total
child execution capped1200seconds, installation bounded180seconds. No additional
batch sizes, seeds, hardware fallback, retries, Telegram or ML1M training.

Quota before submission12.01/45GPUhours remaining. Estimate8–12minutes including
setup; larger feasible batches may vary. The first pilot's CPU/CUDA parity and
native integration checks established backend correctness; current tests verify
new accumulation semantics, matrix bounds, dispatch and private/offline bundle.
See `gr-t5-batch-preflight.json` and submission receipt for actual validation and
run status. Results require completed Kaggle metrics/summary/manifest.
