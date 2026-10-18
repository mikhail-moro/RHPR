# Paired history-length continuation

The experiment asks whether either selected model benefits from additional
observed history when given the same extra optimization budget. It follows the
H1024+R128 search; the earlier frozen full-history inference diagnostic did not
train either family on long inputs.

Each private kernel contains one family and two arms:

| Family | Shared starting weights | Short arm | Full arm |
| --- | --- | --- | --- |
| QSSM K32/slow | selected pass20, inner64/state16/rank32 | H1024+R128 | all observed H+R128 |
| FuXi-Linear | selected pass60, native two-layer model | H1024+R128 | all observed H+R128 |

Configurations are `experiments/yambda_history_continuation_qssm.json` and
`experiments/yambda_history_continuation_fuxi.json`. They pin source revisions,
report/config/manifest hashes, selected weight bytes and validation artifacts.
The selected weights, rather than a later optimizer resume, initialize both arms
within each family. AdamW, AMP scaler and training RNG are reset independently
for each arm with seed42. Parameter shapes and model architecture are fixed.

## Matched training

Both arms run exactly twenty additional passes with no early stopping. A pass
uses one original final training prefix per user and parallel loss on its last
128 shifted labels; it is not an exhaustive sliding-window sweep over all
historical events. Schedule epochs61 through80 provide identical user order,
endpoints, target identities and 512 uniform negative IDs across arms and
families, avoiding reuse of the original1–60 negative schedules.

The short arm applies only a CPU-side left crop to H. The full arm retains every
observed input event. The original32768-event safety limit is enforced by
rejecting oversized training examples; it is not a new truncation schedule.
Queries remain the last128 events of H, and R remains the final128 input events.
The unknown-item ID0 remains an occupied event when its validity mask is true.

Effective batch128, LR0.001, original weight decay, clipping, labels and catalogue
remain fixed. Physical training batches are32/1 for short/full QSSM and16/1 for
short/full FuXi. Gradient accumulation preserves the effective batch; physical
batching and floating-point operation order can still affect optimization.
Independent RNG reset provides reproducibility, not identical dropout masks for
different input shapes.

## Evaluation and interpretation

Validate at continuation passes0,10 and20 using each arm's training input length,
the same daily anchors, full catalogue and macro-user aggregation. Evaluation
batches are32 for short and4 for full. No held-out test targets or ranks are
read. Primary quality is endpoint20 NDCG@10; endpoint20 NDCG@100 and the best
observed validation score within the fixed budget are secondary. Best checkpoint
selection must not replace the fixed endpoint in the primary comparison.

Record the full-minus-short difference at initialization and at endpoint20.
Their difference indicates whether long-input continuation improves the
length-transfer gap beyond the two input conditions' initial gap. Use paired
user bootstrap over the shared anchor support, retaining the same resampled
users for both conditions and checkpoints. A single training seed does not
measure training-seed uncertainty. Report the direct endpoint difference as
well; the two quantities answer different questions.

This experiment measures continuation from a selected short-history checkpoint.
It does not establish from-scratch long-history optimality or equal-compute
superiority between model families: the full arm spends more computation per
matched optimizer update, and the two families have different starting pass
counts. Subsequent architecture tuning is outside this run.

## Runtime and numerical gate

Each arm first profiles two effective AdamW updates plus the longest training row
at its actual physical batch. Profile evaluation at the declared input length
before estimating the complete two-arm budget. Apply the original1.25 safety
factor. QSSM has a21600-second kernel budget; FuXi has28800 seconds. A failed
budget or numerical gate must stop that family explicitly rather than silently
reducing the cohort, labels, history, updates or precision requirements.

FuXi's full-history numerical check is required because earlier frozen transfer
produced nonfinite final states. Apply the same numerical implementation to
both arms. Preserve FP32 state accumulation and check the residual output path
before spending the long-run budget; the initial short validation provides a
record of any change from the pinned source checkpoint's reported metric.

Save source provenance, model config, complete per-arm trajectories, initial and
endpoint ranks/context hashes, training time, peak memory and arithmetic scope.
Keep best and endpoint model weights plus an endpoint optimizer/scaler/RNG resume
for each arm. `metrics.json` remains flat and finite; richer paired statistics
and failure information belong in separate JSON artifacts. These kernels remain
private and use the pinned offline CUDA scan runtime.

Launch each family through `kaggle-experiment.yml` with `submit_only=true`.
Long training and Kaggle queue time can outlive a GitHub-hosted job. This mode
records a versioned submission and returns; a green submission workflow means
only that Kaggle accepted the kernel, not that training or quality evaluation
finished. Inspect Kaggle and collect the completed version's outputs later.
