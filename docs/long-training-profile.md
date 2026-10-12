# Four-backbone training cost screen

`experiments/yambda_long_training_profile.json` profiles SASRec Recent128,
HSTU Recent128, native Mamba4Rec16K and FuXi-Linear16K on one fixed dataset.
It runs neither HPO nor recommendation-quality evaluation.

The prepared Yambda500M cohort includes4,427users,89,191,505training events
and1,049,612catalogue items. Eligibility requires16,385events strictly before
timestamp24,710,400seconds(day286); all listens/repeats/skips are retained.
Only eligible users' pre-cutoff events enter the catalogue. This is a new
training-cost workload, not the previous50M evaluation protocol. The closed
Kaggle cohort includes no validation/test event arrays.

Common settings: d128, two blocks, FFN4× where present, dropout.2, TimeOFF,
AdamW lr.001/weight_decay1e−5/foreach=False, clip1, effectivebatch128.
HSTU has its native UVQK/gating blocks rather than a Transformer FFN.
Mamba1 has state32/conv4/expand2 and preserves native SSM initialization.
FuXi retains all three official channels with zero temporal payload; it does
not receive future timestamps and creates no global16K×16K mask.
These dimensions do not match parameter capacities and are not optimal tuning.

For all architectures, the same seed42 schedule identifies128real full16K
training windows. Input ends are within train and shifted target/end boundaries
are verified. Short models slice the last128before embedding lookup. All
models receive the same128last-position labels and identity-bound512negative
IDs per sequence, independent of architecture/batch segmentation/model RNG.
At standard causal multi-position training, early short-model positions see
1…128events. Full context128 is available for inference anchors.

Each physicalbatch1/2/4/8 receives warmup physical-batch updates to allocate
the real Adam states. The timed cycle accumulates a complete effective128
update. Measure CPU preparation, H2D, forward, FP32 sampledloss, backward,
unscale/clip, optimizer/scaler update and zero_grad separately using synchronized
wall timing. Peak CUDA allocated/reserved includes model and Adam states.
The additional synchronizations make this a controlled component timing, not
a claim about a fully overlapped production input pipeline. One complete update
per shape gives many microbatch samples but only one optimizer sample; report
the actual sample count and repeat later if near-tied decisions depend on it.
Warmup and measured training do not provide accuracy evidence.

After freeing Adam/grad/scaler, measure inferenceB1/8/32: fullprefill, then
FP32 exact catalogue top10. Inputs are already on GPU; preprocessing/H2D and
serving orchestration are excluded. No caching/incremental inference is claimed.
All models must complete at least one native training and inference shape;
larger OOM shapes are recorded without changing their history length or width.
OOM recovery discards partial model/optimizer/scaler state before another shape.

FLOPs cover forward major matrix multiplies only. Elementwise scan, nonlinear
operations, normalization, packing, lookup, loss, optimizer and catalogue costs
are excluded; these are not total training FLOPs. FuXi counts actual three-channel
chunkwise GEMMs, Mamba counts fused projection/FFN weights explicitly.

All four use a private offline47wheel bundle: Torch2.10.0+cu128, CPython3.13,
Mamba2.3.1, causal-conv1d1.7.0. See [runtime notes](profile-runtime-dependencies.md).
No network setting is changed; wheels are hash-locked and installed before
Torch import into a disk-backed target. Native CUDA scan is mandatory.
The installed site and pip temporary directory are under `/kaggle/temp`, outside
the collected kernel-output directory; only small install/provenance logs remain.
CPU-only reference scans are test fixtures, never timing fallbacks.

Artifacts: `training-profile.json`, flat finite `metrics.json`, cohort summary,
window schedule, runtime-bootstrap JSON, installation log and runner manifest.
No model checkpoint or raw event data is emitted as kernel output.
Extrapolated seconds/pass use one new window/user and measured trainingwindows/s;
they exclude setup/evaluation and do not predict convergence or HPO quality.
