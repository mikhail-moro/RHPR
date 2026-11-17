# MovieLens1M: Last-8 and ordinary TIGER context windows

Authorized 2026-11-17: exactly six fresh generator fits, seed42, windows
8/16/32/64/128/256. ML10M is deferred. No pair compression or additional HPO.
Registration: `experiments/ml1m_last8_windows_2t4.json`.

Use the original GroupLens `ml-1m.zip`, SHA256
`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
Keep all1,000,209 ratings/all6040users/all rating values; native title(year)
and genres, all3706 rated films as a fixed static catalogue. No TMDB/Zenodo
coverage filter, positive-rating threshold, additional core or long-user
selection. Native metadata for un-rated films is not a candidate universe.

Sort by(user,timestamp,movieID). Last interaction is TEST, penultimate is VAL.
For each user, the last eight positions of the remaining TRAIN prefix are
fixed TRAIN targets. Context contains only earlier positions, cropped to the
trial window. Thus exactly48,320 TRAIN examples and6040 VAL/TEST queries;
TEST observes VAL. No resampling of TRAIN targets between epochs/windows.
For length L, target positions are zero-based L-10 through L-3; VAL=L-2,
TEST=L-1. No TRAIN target observes either heldout event.

Timestamp ties remain artificial order:2483 TRAIN/VAL boundaries and2544
VAL/TEST boundaries are tied.21794/48320 TRAIN targets tie with the preceding
event. Save chronological event arrays and per-anchor positions/timestamps
for later strict-boundary sensitivity analysis; do not equate rating order
with viewing order or interpret ties as verified temporal causality.
Fixed Last-8 labels cover3085/3706 films;56VAL and73TEST targets are absent
from the selected TRAIN labels. Across all TRAIN events, only2TEST targets
are unseen. Keep these cases and report zero ranking credit if missed.

Frozen Sentence-T5-base, FP32/max256 tokens, title+genres; native GRID
normalized residual KMeans3x256,10 catalogue passes/level, seed42; one
collision-ordinal suffix on EVERY film (3+1, fixed catalogue row order).
This matches our current frozen RQ-KMeans pipeline, not an exact reproduction
of original TIGER's RQ-VAE tokenizer. Recompute embeddings/index once on GPU
from native text; no transfer of cached embeddings, codebooks or fitted models.

Ordinary GRID TIGER4encoder+4decoder blocks,d_model128,d_ff1024,d_kv64,
6heads,dropout.1,ReLU,no extra MLP/SEP,USER2000 integer-remainder bins.
No pair-input embedding table or encoder monkeypatch. All window models
start with identical parameters. The USER precedes left padding as in GRID;
maximum width also changes its relative distance to observed events.
One source/core/runtime/constructor and fixed SID index across all six fits.

Training:FP16,SDPA,torch.compile reduce-overhead,Adam.003/wd1e-4/clip1,
effective batch512. Physical microbatches8/16/32/64/128/256 windows:
256/256/128/64/32/16 respectively; accumulate gradients to the same512
examples/update (95 attempted updates/epoch). Dropout consumption can differ
with microbatch shape; shared seed is not identical stochastic execution.
Warmup2epochs,cosine through60epochs,floor.001;minimum40,maximum120epochs.
VAL every5,EMA(alpha.5,min_delta.0005),patience20epochs; select checkpoint by
RAW VAL itemNDCG@10. Save best/last optimizer/scaler/scheduler/RNG states.
Epoch or time caps are censored, not demonstrated convergence.

Validation/selected-checkpoint TEST:FP32,beam20,full3706-candidate catalogue,
no sampled negatives,no seen-item filtering. Evaluation batches64/64/32/32/16/8
respectively, including TEST. Save per-user paths/scores/targets for paired
analysis and post-hoc length/timestamp strata. Never select using TEST.

Six fits in two isolated T4 workers, longest first. Per-trial pretraining
profile:one warmup and two effective512 updates,180s profile deadline,
restore weights/RNG; log estimated eager train time and peak memory. This
is not the compiled throughput estimate or a serving latency benchmark.
CPU mathSDPA profiler records fixed-width B1 forward supported FLOPs,
teacher-forced4code/FMA2; excludes beam/backward/text encoding. Record actual
first/warm epoch times, validation time and peak memory. No extra profile
kernel, trials, automatic LR/precision fallbacks or retries.

Session deadline8h;per-fit training budget20,000s,save partial completed
epochs if capped. GPU quota checked before dispatch. Intended private
kernel:`carasik/rhpr-ml1m-last8-windows`; two offline runtime datasets plus
a new private input containing only the original ML1M archive/provenance.
The archive is stored byte-identically as `ml-1m-source.bin` to prevent Kaggle
from automatically extracting ZIP inputs. A download-back SHA256 check passes.
