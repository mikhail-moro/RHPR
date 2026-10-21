# Yambda 500M: four-backbone GPU cost results

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

All four requested backbones completed the cost screen on one Tesla T4.
The best tested physical batches estimate about **21 seconds/pass for SASRec
Recent128, 24 seconds/pass for HSTU Recent128, 130 seconds/pass for Mamba4Rec16K,
and 450 seconds/pass for FuXi-Linear16K**. These passes mean one training window
per user; this experiment measured neither recommendation quality nor HPO.

Run provenance:

- [GitHub workflow 38350079365](https://github.com/mikhail-moro/RHPR/actions/runs/38350079365).
- Private [Kaggle kernel](https://www.kaggle.com/code/carasik/rhpr-long-training-profile).
- Source commit `9ce5e7fe42e73c4e25547c1050122b566ad04c20`, matching both
  `training-profile.json` and `run-manifest.json`.
- Config: [yambda_long_training_profile.json](../experiments/yambda_long_training_profile.json).
- Protocol: [long-training-profile.md](long-training-profile.md).
- Both `complete` and flat root `profile_complete` are `true`.

## Workload and validation

The train-only Yambda 500M cohort contains **4,427 users, 89,191,505 events and
1,049,612 catalogue items**, retaining all listens, repeats and skips.
Eligibility/catalogue use only events before timestamp 24,710,400; held-out
targets were not inspected. Training histories contain 16,386–27,709 events.

The seed-42 schedule contains 128 distinct users and full shifted 16K input
windows. SASRec/HSTU crop the last 128 events before embedding lookup; Mamba/FuXi
keep all 16,384. Each model receives the same last 128 next-item labels and
512 catalogue-uniform negatives per window. Training is causal multi-position
prediction: short-model labels see 1–128 events, whereas the corresponding
long-model labels see 16,257–16,384. Inference anchors have the entire stated
128 or 16,384 input events.

Schedule SHA256:
`ae463406ab53eaa028693768d871c9a2f4248cb1028727a6574b244c9f1b6e11`.
An independent PyArrow audit recomputed complete file/content fingerprints,
schedule, targets and negatives. **585/585 checks passed**, including exact
source/runtime provenance and strict completion acceptance.

All 15 successful training shapes process 128 windows/16,384 targets in one
successful update, with finite loss/gradient norm and **zero measured AMP
skips**. Eleven inference shapes succeed. FuXi training B=8/inference B=32
record OOM; successful smaller shapes satisfy the acceptance rule.

Common settings: d=128, two blocks, dropout=0.2, Time OFF, effective batch 128,
AdamW `foreach=False`; remaining settings are in the linked protocol/config.
Physical B=1/2/4/8 uses accumulation 128/64/32/16. Only GPU 0 of two available
CUDA devices is used: a Tesla T4 with 14.562 GiB memory. Runtime: Python 3.13.15,
Torch 2.10.0+cu128, CUDA 12.8, mamba-ssm 2.3.1 and causal-conv1d 1.7.0.

## Training cost and budget

The following throughput includes CPU batch preparation, H2D and synchronization.
Peak allocated memory includes parameters, gradients, Adam state and temporary
tensors. B is the **best tested** physical batch, not an optimized batch search.

| Backbone | Context | B | Wall windows/s | Peak allocated GiB | Approx. seconds/pass | Approx. minutes/20 passes |
|---|---:|---:|---:|---:|---:|---:|
| SASRec Recent128 | 128 | 8 | 211.64 | 3.042 | 20.9 | 7.0 |
| HSTU Recent128 | 128 | 8 | 187.13 | 3.041 | 23.7 | 7.9 |
| Mamba4Rec | 16,384 | 8 | 34.18 | 4.691 | 129.5 | 43.2 |
| FuXi-Linear | 16,384 | 4 | 9.845 | 8.869 | 449.7 | 149.9 |

A pass is **4,427 windows, one per user**, approximately 35 updates and 566,656
targets, not an exhaustive epoch over 89 million events. Estimates are
`4427 / wall_windows_per_second`, excluding startup, validation/test and unknown
passes-to-quality. Partial batches/pipeline overlap can change actual time.
Twenty passes are a budget example, not a tested training recipe.

The full measured training grid is below. Pure throughput excludes CPU/H2D;
wall throughput includes them. Reserved memory also reflects allocator caching.

| Backbone | Physical B | Pure windows/s | Wall windows/s | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| SASRec Recent128 | 1 | 29.91 | 29.33 | 3.029 | 3.043 |
| SASRec Recent128 | 2 | 58.11 | 56.83 | 3.030 | 3.047 |
| SASRec Recent128 | 4 | 114.34 | 111.78 | 3.034 | 3.051 |
| SASRec Recent128 | 8 | 217.44 | 211.64 | 3.042 | 3.057 |
| HSTU Recent128 | 1 | 25.64 | 25.21 | 3.025 | 3.041 |
| HSTU Recent128 | 2 | 50.94 | 50.03 | 3.028 | 3.045 |
| HSTU Recent128 | 4 | 99.65 | 97.61 | 3.032 | 3.049 |
| HSTU Recent128 | 8 | 191.58 | 187.13 | 3.041 | 3.555 |
| Mamba4Rec | 1 | 18.26 | 17.99 | 3.236 | 3.795 |
| Mamba4Rec | 2 | 25.65 | 25.30 | 3.444 | 3.795 |
| Mamba4Rec | 4 | 31.40 | 31.01 | 3.861 | 4.297 |
| Mamba4Rec | 8 | 34.54 | 34.18 | 4.691 | 5.305 |
| FuXi-Linear | 1 | 4.733 | 4.702 | 4.541 | 4.836 |
| FuXi-Linear | 2 | 7.565 | 7.519 | 5.982 | 6.592 |
| FuXi-Linear | 4 | 9.896 | 9.845 | 8.869 | 9.596 |
| FuXi-Linear | 8 | OOM | OOM | — | — |

The shared embedding has 134,350,464 parameters: 0.5005 GiB FP32 weights,
2.002 GiB weights+gradient+Adam m/v. Short-model backward p50 remains about
29 ms (SASRec)/32 ms (HSTU) as B grows; dense embedding gradients cost substantially.
Larger short-model batches could improve throughput. Memory increment includes
gradients/optimizer temporaries as well as activations.

## Inference cost

Complete prefill + exact FP32 catalogue top10, chunks of 65,536; p50 is **per
entire batch**, eight repetitions after four warmups. Inputs already reside on
GPU; CPU/H2D/serving, caching and incremental state are excluded.
Adam/scaler/gradients are freed first.

| Backbone | B | History prefill p50 ms | Catalogue top10 p50 ms | Full batch p50 ms | Mean users/s | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| SASRec Recent128 | 1 | 2.78 | 4.04 | 6.82 | 146.20 | 0.520 |
| SASRec Recent128 | 8 | 2.72 | 6.19 | 8.96 | 893.43 | 0.524 |
| SASRec Recent128 | 32 | 2.83 | 9.74 | 12.60 | 2,525.50 | 0.539 |
| HSTU Recent128 | 1 | 4.76 | 4.12 | 8.93 | 110.72 | 0.520 |
| HSTU Recent128 | 8 | 4.74 | 6.21 | 11.00 | 725.48 | 0.529 |
| HSTU Recent128 | 32 | 4.77 | 9.69 | 14.53 | 2,194.10 | 0.560 |
| Mamba4Rec | 1 | 6.25 | 4.55 | 10.89 | 91.05 | 0.595 |
| Mamba4Rec | 8 | 53.10 | 7.61 | 60.82 | 131.54 | 1.121 |
| Mamba4Rec | 32 | 250.49 | 11.77 | 262.21 | 121.91 | 2.923 |
| FuXi-Linear | 1 | 51.35 | 4.15 | 55.58 | 18.00 | 1.061 |
| FuXi-Linear | 8 | 259.35 | 6.92 | 266.22 | 30.00 | 4.813 |
| FuXi-Linear | 32 | OOM | — | OOM | — | — |

Component p50s need not sum to full p50; throughput uses mean full time.
Mamba's best tested inference throughput is B=8. Seen items remain eligible.
FuXi is recreated after training OOM and uses fresh inference weights; other
models retain measured-update weights. This compares computation paths/finite
outputs, not quality between those states.

## FLOPs, implementation and interpretation

Multiply-add = two FLOPs. These are **forward major matrix multiplies/user
only**, excluding scan/conv/activation/norm/packing/lookup/loss/backward/optimizer
and catalogue. They are not total training FLOPs or cross-family latency
predictors. Catalogue matmul separately adds 0.2687 GFLOPs/user before top-k.

| Backbone | Context | Forward major matmul GFLOPs/user | Non-item parameters |
|---|---:|---:|---:|
| SASRec Recent128 | 128 | 0.1174 | 413,184 |
| HSTU Recent128 | 128 | 0.0587 | 181,246 |
| Mamba4Rec | 16,384 | 16.3746 | 522,240 |
| FuXi-Linear | 16,384 | 30.6352 | 1,802,580 |

Common width/two blocks do not equalize capacity. HSTU has UVQK/gating without
a Transformer FFN. Mamba uses native initialization and **fused CUDA scan/conv**,
with no Python reference fallback. FuXi uses official PyTorch chunkwise retention
and AMP-compatible scan backward; HSTU uses reference PyTorch attention/packing.
FuXi computes all three channels with zero temporal payload, differing from its
published temporal recipe.

Different fusion/maturity makes this a comparison of present implementations
on T4, not architectural speed limits. FuXi is about 3.5 times slower than
Mamba at their best tested training shapes and requires more memory; this
supports budgeting, not accuracy conclusions.

Each shape has **one measured optimizer update**, versus 128/64/32/16 microbatch
forward/backward samples. Each B reuses the same windows and ongoing state,
after four warmup physical-batch updates. Synchronization limits overlap;
eight inference repetitions do not establish production p95. Near ties need
repeated measurements.

Offline installation took 171.2 seconds; runner execution took 147.1 seconds,
including cohort verification and `profile_seconds=136.2`. The whole workflow took about 46m50s,
with substantial waiting before profiling; that wait is not FuXi computation.
GitHub artifact `11791494613` matches Kaggle's metrics/profile/summary/manifest
byte-for-byte. Startup and queueing are separate from the pass budget.

This run provides no NDCG/Recall, convergence result, HPO ranking or evidence
that old history improves recommendations. A matched quality experiment must
use new window endings across passes and a shared evaluation protocol before
comparing the usefulness of 16K history or claiming adaptive compression gains.
It must also control the short-model training contexts described above: they
contain 1–128 events per label in this screen. To isolate history beyond 128,
an exact Recent128 control can use separate 128-event windows and supervise
only their final states for the same selected targets.
