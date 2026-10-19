# Matched rotating-window QSSM and FuXi experiment

The fixed-block experiments repeatedly supervised the same final 128 targets
per user. This experiment changes the positive training windows while retaining
the selected K32/slow Query SSM, the native FuXi architecture, the catalogue,
seed 42, sampled loss, daily validation anchors and optimizer settings. It tests
whether more diverse supervision helps either architecture; it is not a new
architecture or a full-history training experiment.

The registration is
[`experiments/yambda_query_rotating_windows.json`](../experiments/yambda_query_rotating_windows.json).
It trains exactly two models from scratch: Query SSM and FuXi. Their previously
selected fixed-block checkpoints and validation ranks are verified and reused;
neither historical control is trained or evaluated again.

## Train-only windows

For a user with mapped train sequence `T` and `n = len(T)`, define endpoints
`e_j = n - 1 - 128*j`, retaining all endpoints with `e_j >= 1152`. Each window
contains inputs `T[e-1152:e]`, shifted labels `T[e-1151:e+1]`, and loss only on
the last 128 input states. Thus its compressor history has 1024 events, its raw
tail has 128 events and the 128 positive labels are `T[e-127:e+1]`.

These target blocks are disjoint. The first pass uses the latest block, with
the exact historical user order and negative identities. Remaining blocks are
permuted using hashes bound to seed, UID, train length and cycle. Each block is
used once before a new independently permuted cycle begins. Reordering users
or changing physical batch size does not change a user's endpoint or negatives.
Only block descriptors are cached; item and timestamp arrays are not cached.

One pass remains one window for each of 4427 users: 566656 target presentations
and 35 effective optimizer updates. Both models see the same windows and uniform
512-item negative draws, including the original positive-collision convention.
Physical training batches are 32 for QSSM and 16 for FuXi; effective batch is 128.

At least 120 complete target blocks are available per user in the 16K cohort.
Consequently none repeats within 60 passes: 20 passes expose 11333120 distinct
target positions and 60 expose 33999360. The counts describe positions, not
unique items. This is still not an exhaustive event epoch: the initial 1024
transitions are needed to provide a full context, fewer than 128 additional
early positions are excluded by the complete-block grid, and the stopping
policy may finish before all remaining blocks are sampled.

## Optimization and evaluation

All other source settings stay fixed, including AdamW LR 0.001/decay 1e-5,
gradient clipping, dropout, AMP and the 32768-position model initializer.
Training runs 20–60 passes with validation every 10 passes and patience 2 checks
for strict macro-user NDCG@10 improvement. It is not resumed from the saved
controls. Disposable preflight models perform two real effective Adam updates;
they are discarded before fresh quality-model initialization.

Validation continues to use the original first-listen-per-user-day anchors,
global time boundaries, full catalogue, OOV handling and H1024+R128 suffix crop.
NDCG@10 is primary; NDCG@100 and first/later validation-anchor slices remain
diagnostics. No test ranks or full-history evaluation are requested. A
three-hour ceiling is checked against measured maximum-pass costs before
quality training; there is no automatic trial, precision or epoch reduction.

## Interpretation and artifacts

Compare each fresh rotating model with its own verified historical fixed-block
control and compare the two fresh models on their shared validation anchors.
One seed and repeated validation inspection make the results exploratory.
Bootstrap intervals over users do not include training-seed or model-selection
uncertainty.

Early stopping and selected checkpoints may occur at different passes across
models and schedules. The update count is matched per pass and the same maximum
and stopping policy apply; realized total updates and selected-checkpoint target
coverage can differ. Report both completed-training and selected-checkpoint
coverage. Historical comparisons describe the resulting training procedures,
including their stopping and checkpoint selection, rather than a universal
causal effect of new target blocks.

FuXi additionally uses the current FP32 recurrence and reads. Its historical
fixed-block run used the earlier mixed-precision implementation, so its
historical-to-fresh difference combines schedule and numerical execution
changes. It cannot isolate the effect of rotating positives. No additional
fixed-block control is retrained because of the user's GPU budget.

Outputs retain both source checkpoints and hashes, actual new checkpoints,
training passes, per-checkpoint validation ranks and input audits. The common
window artifact records ordered and order-invariant schedule hashes, endpoint
positions, events behind the latest block, target presentations, exact unique
target-position coverage and the excluded early positions. Actual input width
is always 1152 and is reported separately from an endpoint's position in the
source train history. Flat finite scalar results remain in `metrics.json`.
