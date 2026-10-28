# Original versus ALS-swapped RQ-KMeans: longer generator training

The user authorized two 60-epoch fits on 2026-10-27 and explicitly requested
validation every 10 epochs. This supersedes the earlier screening budget only
for these two fits. No other candidates, graph fitting, seeds or automatic retries.

## Question and prior evidence

The proposed mechanism is that a more useful item-to-SID arrangement may become
visible after longer generator training. Ten epochs did not establish a plateau:
the original RQ-KMeans validation NDCG@10 was 0.03060930 at epoch 10 and
0.03699076 at epoch 20. This motivates the check; it does not predict its outcome.

Reuse the frozen original T5 / normalized-residual RQ-KMeans index and the
**ALS uniform** candidate from the latest screen. It had the highest epoch-10
NDCG@10 among the three new RQ-KMeans variants: 0.03024668, versus original
0.03060930. Selection used validation, so this is exploratory follow-up rather
than an independent confirmatory comparison. The earlier cooccurrence candidate
belongs to a different experiment and is not silently substituted here.

The candidate was produced by 256 accepted whole-path swaps. It changes 480 of
12,101 item paths, including 401 first-level assignments. It preserves the exact
set of complete four-token paths, tree occupancies and collision counts. Its
optimization-user CF LCP score improved 4.08%, but independent TRAIN-user CF
LCP improved only 0.50% and first-prefix sequential gain 0.59%. Similar downstream
scores after this small intervention do not imply that SID structure is irrelevant.

Longer training can reveal a late advantage, or allow both mappings to be learned
equally well, or reveal overfitting. If neither trajectory plateaus by epoch 60,
the experiment still cannot establish asymptotic quality. ALS-neighbor LCP,
first-prefix predictability, full-path prediction and beam retrieval are different
objectives; no causal explanation is assumed in advance.

## Fixed protocol

`experiments/amazon_beauty_sid_als_long_2t4.json`:

- Exactly two fresh fits, one per T4: original and frozen ALS uniform.
- Preserve generator/data seed 42, exact original neural initialization and TRAIN
  case/order hashes, GRID 4+4 / d128 / 4,977,920 parameters, FP16 training,
  Adam LR0.001 / weight decay0.0001 / clip1, effective batch256 / physical128.
- All 131,413 TRAIN prefixes exactly once per epoch, latest20-item contexts.
  Per model: 60 complete epochs, 30,840 optimizer updates, 7,884,780 examples.
- Validation at 10/20/30/40/50/60, FP32 batch32, free beam20, full catalogue.
  Save predictions and user/target alignment at every check, including the
  original model, unlike the historical aggregate-only control.
- Primary endpoint: epoch-60 validation NDCG@10. Also report the full six-point
  trajectory, Recall/NDCG@5/10 and best-validation selection separately. Do not
  choose the reporting epoch post hoc or stop when one model looks better.
- No early stopping before60; per-fit time guard4hours is a failure guard, not
  a plan to consume4hours. No test construction/evaluation in this comparison.
- Save best/last weights and complete last optimizer/scaler/RNG states. Both
  fits restart from the same initialization; do not mix an older checkpoint
  lacking complete RNG state with a freshly trained control.

The existing private dataset `carasik/rhpr-sid-als-screen` already contains both
arrays and their hashes; no new upload is required. The loader's explicit
`use_original_paths` control reads the manifest-pinned original array of the
selected candidate and generates identity ownership. It verifies the source
candidate too, keeps original .npy bytes and suffixes, and records original
provenance without claiming the candidate's search diagnostics for the control.

Expected wall time on the measured two-T4 setup is about85–95minutes after GPU
allocation (approximately170–190GPU-minutes). Keep the kernel private/offline,
and retain its outputs. No Telegram. The Field RQ-VAE branch remains closed.
