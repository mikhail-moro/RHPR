# Four-backbone training cost screen

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

`experiments/yambda_long_training_profile.json` profiles SASRec Recent128,
HSTU Recent128, native Mamba4Rec at 16K and FuXi-Linear at 16K on one fixed dataset.
It runs neither HPO nor recommendation-quality evaluation.

The prepared Yambda 500M cohort includes 4,427 users, 89,191,505 training events
and 1,049,612 catalogue items. Eligibility requires 16,385 events strictly before
timestamp 24,710,400 seconds (day 286); all listens, repeats and skips are retained.
Only eligible users' pre-cutoff events enter the catalogue. This is a new
training-cost workload, not the previous 50M evaluation protocol. The closed
Kaggle cohort includes no validation/test event arrays.

Common settings: model width 128, two blocks, FFN multiplier 4 where present,
dropout 0.2, Time OFF, AdamW with learning rate 0.001, weight decay 1e−5 and
`foreach=False`, gradient clipping at 1, and effective batch size 128.
HSTU has its native UVQK/gating blocks rather than a Transformer FFN.
Mamba1 has state dimension 32, convolution width 4 and expansion factor 2;
it preserves native SSM initialization.
FuXi retains all three official channels with zero temporal payload; it does
not receive future timestamps and creates no global 16K × 16K mask.
These dimensions do not match parameter capacities and are not optimal tuning.

For all architectures, the same seed-42 schedule identifies 128 real full 16K
training windows. Input ends are within train and shifted target/end boundaries
are verified. Short models slice the last 128 events before embedding lookup.
All models receive the same 128 last-position labels and 512 negative IDs per
sequence, tied to the window identity and independent of architecture,
batch segmentation and model RNG.
At standard causal multi-position training, early short-model positions see
1–128 events. A full context of 128 events is available for inference anchors.

Each physical batch size (1, 2, 4 or 8) receives warmup updates to allocate
the real Adam states. The timed cycle accumulates a complete update with
effective batch size 128. CPU preparation, H2D, forward, FP32 sampled loss,
backward, unscale/clip, optimizer/scaler update and `zero_grad` are measured
separately using synchronized wall timing. Peak CUDA allocated/reserved memory
includes the model and Adam states.
The additional synchronizations make this a controlled component timing, not
a claim about a fully overlapped production input pipeline. One complete update
per shape gives many microbatch samples but only one optimizer sample; report
the actual sample count and repeat later if near-tied decisions depend on it.
Warmup and measured training do not provide accuracy evidence.

After freeing Adam states, gradients and scaler, measure inference at batch
sizes 1, 8 and 32: full prefill, then FP32 exact catalogue top-10.
Inputs are already on GPU; preprocessing, H2D and serving orchestration are
excluded. No caching/incremental inference is claimed.
All models must complete at least one native training and inference shape;
larger OOM shapes are recorded without changing their history length or width.
OOM recovery discards partial model/optimizer/scaler state before another shape.

FLOPs cover forward major matrix multiplies only. Elementwise scan, nonlinear
operations, normalization, packing, lookup, loss, optimizer and catalogue costs
are excluded; these are not total training FLOPs. FuXi counts actual three-channel
chunkwise GEMMs, Mamba counts fused projection/FFN weights explicitly.

All four use a private offline bundle of 47 wheels: Torch 2.10.0+cu128,
CPython 3.13, mamba-ssm 2.3.1 and causal-conv1d 1.7.0.
See [runtime notes](profile-runtime-dependencies.md).
No network setting is changed; wheels are hash-locked and installed before
Torch import into a disk-backed target. Native CUDA scan is mandatory.
The installed site and pip temporary directory are under `/kaggle/temp`, outside
the collected kernel-output directory; only small install/provenance logs remain.
CPU-only reference scans are test fixtures, never timing fallbacks.

Artifacts: `training-profile.json`, flat finite `metrics.json`, cohort summary,
window schedule, runtime-bootstrap JSON, installation log and runner manifest.
No model checkpoint or raw event data is emitted as kernel output.
Extrapolated seconds per pass use one new window per user and measured training
windows per second; they exclude setup/evaluation and do not predict convergence
or HPO quality.

The first completed T4 measurements are recorded in
[long-training-profile-results.md](long-training-profile-results.md).
