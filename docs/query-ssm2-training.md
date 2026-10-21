# One-arm QSSM2 state128 quality run

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This experiment trains one fresh QSSM2 compressor plus the existing two-layer
SASRec predictor. It checks recommendation quality after the successful native
SSD speed screening. It is a single fixed configuration, not an HPO run or a
new comparison grid.

The compressor uses K32, inner width64, **state dimension128**, head dimension64,
chunk64, conditioning rank32, and initial learned event steps in [1e-4,1e-2].
There is one scalar-A head per query; d_state128 is the width of B/C and the
state, independently of the head's64 value channels. Queries use H[-128:],
B and delta depend on event and query, and C depends on the query. Convolution,
D-skip and output gate remain absent. This architecture does not accept old
QSSM checkpoints.

Increasing state width does not create128 independent decay timescales: this
configuration still has one scalar-A head per query. Its event-conditioned B
projection also retains rank32. The run tests a larger write/read state under
those constraints, rather than a general128-timescale Mamba architecture.

The selected original fresh SASRec, item embeddings and full32768 positional
table are reconstructed, then only the compressor is replaced. Source model
weights and optimizer state are never loaded. Original pilot manifests,
evaluation-context hashes and saved validation-rank metadata remain mounted
for the existing evaluation audit. Selected QSSM/FuXi checkpoint kernels are
not mounted, and no controls or other models are trained or re-evaluated.

The cohort, untrimmed catalogue, seed42, daily validation anchors, train/date
boundaries, negatives and training schedule match the prior rotating-window
experiment. Each pass presents one H1024+R128 input per user and128 parallel
causal next-item targets. The first pass uses the latest train block; later
passes visit other non-overlapping128-target blocks in the same deterministic
per-user order. Queries are generated entirely within H, before the scored
raw tail. Occupied OOV events keep their positions and masks.

Training uses physical32/effective128,512 uniform negative items, original
AdamW LR.001 and decay1e-5, FP16 autocast, loss scale1024 and gradient clip1.
The stopping policy is minimum20, maximum60 passes, validation every10,
patience2. Earliest strictly best macro-user NDCG@10 chooses the checkpoint;
NDCG@100 and the other existing full-catalogue metrics remain secondary.
Training uses the established bounded AMP-overflow recovery; no automatic
batch, dtype, LR, model or pass-cap fallback is introduced.

Before quality training, native FP16 SSD output and final-output backward
must match the CPU reference at state128 across a full chunk and partial next
chunk. A disposable full model then performs two real effective Adam updates,
requiring finite nonzero B/delta/query/read/A gradients and no AMP retry.
The second update measures training throughput. Exact full-catalogue validation
throughput is measured with the real dense optimizer state still allocated.

The cost gate prices all60 training passes, six validation passes and120 seconds
of checkpoint/statistics overhead using measured rates and safety factor1.25.
It refuses to start if the estimated total exceeds7200 seconds. No registered
work is automatically removed to fit the budget. Native compilation and data
preparation are included in elapsed runner time. Full-history inference,
training-length transfer and test evaluation are not registered.

Best FP32 model weights, last completed-pass Adam/scaler/RNG resume, daily
validation ranks, input hashes, pass curves, preflight timings/memory and major
matmul/logical-recurrence FLOP estimates are retained. `ssm2-training-coverage.json`
separately records the maximum schedule, actual completed exposure and exposure
of the selected checkpoint; `ssm2-window-descriptors.json.gz` records endpoints
and negative seeds without raw events. Standard finite scalar metrics and
source/configuration provenance follow the existing experiment contract.

The appropriate historical reference is the completed **rotating-window** QSSM
run on the same H1024+R128/daily validation protocol. Previous fixed-last-block
metrics answer a different training-distribution question. Any cross-run
quality or speed comparison remains exploratory and single-seed; this run
alone does not isolate the effect of the larger state from the SSD transition
structure.
