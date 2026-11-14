# Kindle capacity pilot: wider Recent20 and Recent32

Question: does the current generator capacity limit predictive quality and the
benefit of additional history? This is one fixed seed42 screening experiment,
not a definitive scaling study or proof that older history is useful.

Reuse the completed Kindle long-window parent55350fd1a301, exactly its
frozen seed42 item SID, same958users/16177books/141137train examples and VAL958.
No text embeddings or tokenizer are recomputed. Pin32 parent artifacts:manifest,
configuration,index,study summary and both narrow seed42 trajectories including
all20 per-validation prediction files. Copy these as two existing baselines;
train only two new models, one perT4 GPU, with windows20 and32.

Architecture changes only widths, preserving4encoder+4decoder layers,6heads,
dropout.1 and the rest of the generator construction:

| Shape | d_model | FFN | d_kv | Total parameters |
|---|---:|---:|---:|---:|
| Existing mini |64|512|32|1440576|
| Wide pilot |128|1024|64|5108992|

Totals include the historical attached pair-input embedding table, unused by
these single-item controls. Excluding that table gives1375040 and4977920.
Actual CPU math-SDPA B1 full-context teacher-forced four-code forward profiling:
wide20=.427087952GF,wide32=.704769968GF, including output heads, FMA=2;
no backward,beam search or text encoding. Timing and CUDA memory are measured
in the real fits rather than inferred from these operation counts.

All training/evaluation settings equal the parent configuration, enforced at
runtime:FP16/SDPA/compile,physical=effective batch512,Adam wd.0001,clip1,
LR.003 with2epoch warmup then cosine to.001 at60,VAL5/FP32/beam20/batch64,
EMAalpha.5/min_delta.0005/patience15/min40/max60. Raw best VAL NDCG@10 selects
weights; EMA controls stopping and is not a weight average. Each model gets
up to7200s; a time-budget exit is flagged incomplete rather than treated as
convergence. Full best/last checkpoints,all curves/predictions,VAL CE,FLOPs,
first/warm epoch timing and peak CUDA allocated/reserved bytes are retained.
TEST remains closed. Private/offline kernel,no notifications.

Within the wide pair, same index,initializer,supervised targets,epoch shuffle,
optimizer schedule and stopping policy; only visible context differs. Narrow
baselines reuse their already completed independent training. Their initial
weight shape necessarily differs from the wide model; they share the declared
seed and data/index. The LR is unchanged, not retuned for the wider model.

Compare four cells:narrow20,narrow32,wide20,wide32. Inspect common fixed40 epoch,
best checkpoints within a common observed budget,raw selected checkpoints and
curves. Increased NDCG for both wide windows alone indicates a size/optimization
effect; a larger32−20 gap suggests size may affect the use of older context.
Neither is a statistical confirmation with one seed and sparse958targets.
Do not attribute a gain solely to capacity independent of optimization or claim
convergence from hitting60. New seeds/windows require a later decision.

Preflight on the real frozen SID and actual VAL contexts:finite native GRID
forward/backward for both windows,identical wide initial parameter SHA256
34e7397ed277c78befd6254cbb736794df4907a011ed032bbd0735d9b7272286,
no parameter mutation from profiling. The pilot uses the already exercised
kindle_windows worker/trial and its tested parent event contract.
