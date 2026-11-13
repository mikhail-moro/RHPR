# Intermediate Recent20: LR0.003→0.001 and EMA stopping

The user corrected the desired LR floor to0.001 and accepts some quality loss
for intermediate experiment speed. Exactly one new seed42 Recent20 fit on the
same pinned item index and matched initializer as previous Kindle controls.
No Sentence-T5/quantizer fitting, extra seeds, Pair20 runs or TEST evaluation.

Keep batch512, FP16/nativeSDPA, model4+4/d64/FF512, Adamwd1e-4/clip1 and the
same source windows/order. Warm-up2epochs to0.003 then cosine toward0.001 by
60nominal epochs. Minimum40, maximum60, VAL every5. Causal EMA0.5 initialized
by first raw VAL NDCG@10; reset its significant-best reference only after an
absolute improvement>0.0005. Stop after15epochs without this improvement,
subject to minimum40. Checkpoint selection remains highest RAW NDCG. Save
raw/EMA/reference/improvement epoch, all VAL predictions, LR/loss/AMP counts,
and exact optimizer/scaler/RNG/scheduler/EMA state for resume. No hidden retry.

This combines a new LR floor and a cheaper stopping policy; it does not isolate
which change affects final quality. The previous estimated1.7% or2.1% quality
losses are not guarantees on this run. Same-seed historical reference curves
are retained; TEST stays closed for schedule/stopping selection.

A short hardware profile precedes this fit. Baseline train batch512 and VAL32
are preserved for the clean schedule comparison unless an explicitly documented
prediction-preserving speed change is adopted after its measurements. Larger
training batches/bucketing alter the optimization and are future choices.

One worker on GPU0; no unnecessary second model is trained to fill GPU1. Expected
~17–26minutes of training plus startup, depending on EMA stop. Per-fit3300s,
server80min. Time-budget exit before minimum40 is a failed run. No automatic
follow-up/extra seeds/epochs. Profile and training are separate private kernels.

The completed profile found eager batch512 0.12593s/update versus compiled
0.07220s/update on a resident batch (about1.75x), cold case including compilation
45.34s. GPU-resident input gave no throughput gain. Trainbatch1024/2048 only
+8.5%/+10.2%examples/sec and change optimizer batch. Adopt compile for standard
training forward only; keep original module for optimizer/checkpoint state and
uncompiled generation. Static shapes512 and485 can compile separately, so
first-epoch/cold overhead and actual whole-run speed must be measured. Compiled
FP16/dropout numerics can differ; this is not an isolated LR-floor experiment.
Use VALbatch256 after exact SID/ranking agreement across all2625VAL cases on
the profiled checkpoint, with two timed passes4.29/4.31s versus5.55/5.48s at32.
No bucketing or data-order changes; no GPU-residency implementation needed.
EMA state is saved and exact CPU resume tested. Compiler runtime caches/CUDA
graph state are not serialized; cross-session compiled CUDA bitwise identity
is not asserted. Parent/raw-index/initial-parameter pins remain enforced.

Final verification:1087regression tests passed,152optional skips. Actual native
GRID CPU forward/backward, validation and EMA-checkpoint training passed, including
epochs before the first validation. EMA uninterrupted/resumed trajectories and
raw-best selection match. A compiled-forward CPU test confirms optimizer updates
reach original parameters and checkpoints contain no _orig_mod wrapper prefix.
