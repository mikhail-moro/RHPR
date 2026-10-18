# History-length continuation

The experiment follows the H1024+R128 search. The earlier frozen full-history
inference diagnostic did not train either family on long inputs. QSSM has a
paired short/full continuation to isolate history length at a matched update
count. FuXi trains only on full history to conserve GPU time and compares with
the saved historical short-history result.

Each private kernel contains one family:

| Family | Starting weights | Short comparison | Full arm |
| --- | --- | --- | --- |
| QSSM K32/slow | selected pass20, inner64/state16/rank32 | H1024+R128 | all observed H+R128 |
| FuXi-Linear | selected pass60, native two-layer model | saved H1024+R128 metrics/ranks only | all observed H+R128 |

Configurations are `experiments/yambda_history_continuation_qssm.json` and
`experiments/yambda_history_continuation_fuxi.json`. They pin source revisions,
report/config/manifest hashes, selected weight bytes and validation artifacts.
The selected weights, rather than a later optimizer resume, initialize every
fresh training arm. AdamW, AMP scaler and training RNG reset independently for
each arm with seed42. Parameter shapes and model architecture are fixed. FuXi
has no newly trained short-history model. Its configuration explicitly sets
`continuation.short_control="historical_only"`; the existing QSSM configuration
remains byte-identical and retains both training arms.

## Matched training

Every fresh arm runs exactly twenty additional passes with no early stopping. A pass
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
remain fixed. Physical training batches are32/1 for short/full QSSM and2 for
full-history FuXi. Gradient accumulation preserves the effective batch; physical
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

For QSSM, record the full-minus-short difference at initialization and at endpoint20.
Their difference indicates whether long-input continuation improves the
length-transfer gap beyond the two input conditions' initial gap. Use paired
user bootstrap over the shared anchor support, retaining the same resampled
users for both conditions and checkpoints. A single training seed does not
measure training-seed uncertainty. Report the direct endpoint difference as
well; the two quantities answer different questions.

For FuXi, compare the full-history endpoint with the pinned historical short
validation ranks from source pass60. That historical result used the older FP16
output path, while the new continuation retains FP32 residual/state arithmetic.
The comparison includes twenty additional optimization passes, input length and
the numerical implementation change. It is an exploratory quality comparison,
not a causal estimate of the benefit of history. Do not report a paired
short/full difference-in-differences for FuXi: no fresh short arm exists. The
full-history pass0 and pass20 validations also show its continuation trajectory.

This experiment measures continuation from a selected short-history checkpoint.
It does not establish from-scratch long-history optimality or equal-compute
superiority between model families: full inputs spend more computation per
optimizer update, the two families have different starting pass counts, and
FuXi lacks a matched fresh short control. Subsequent architecture tuning is
outside this run.

## Runtime and numerical gate

Each arm first profiles two effective AdamW updates plus the longest training row
at its actual physical batch. Profile evaluation at the declared input length
before estimating the complete registered-arm budget. Apply the original1.25 safety
factor. QSSM has a21600-second kernel budget; FuXi has32400 seconds (nine hours). A failed
budget or numerical gate must stop that family explicitly rather than silently
reducing the cohort, labels, history, updates or precision requirements.

FuXi's first measured paired run passed numerical/capacity checks but stopped
before quality training: its conservative estimate exceeded the original
eight-hour ceiling. The revised full-only run removes fresh short training and
uses physical batch2, which must pass a new measured capacity and timing gate
before training starts. The1.25 safety factor and all twenty passes are retained.
No automatic batch-size or precision fallback is permitted. The already running
QSSM package and configuration are unaffected.

FuXi's full-history numerical check is required because earlier frozen transfer
produced nonfinite final states. Preserve FP32 state accumulation and check the
residual output path before spending the long-run budget. Report the numerical
implementation change explicitly alongside comparisons with historical FP16
short-history metrics.

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
