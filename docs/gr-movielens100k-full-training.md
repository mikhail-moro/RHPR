# Frozen-index full-training pilot, seed42

Four continuations: TIGER text+poster+plot without people; TIGER with graph people;
TIGER with names in text instead of graph people; item-ID-only SASRec. Continue
from each existing20epoch `generator-last.pt`, not from selected best weights.
Config: `experiments/movielens100k_full_training_seed42_2t4.json`.
New private kernel: `carasik/rhpr-ml100k-full-training-seed42`.

## Source lineage

Reuse completed private `carasik/rhpr-ml100k-multimodal-ae` v1 trials
`seed42_stage_03/04` and `carasik/rhpr-ml100k-names-sasrec-controls` v1 trials
`tiger_names_seed42/sasrec_id_seed42`, mounted through `kernel_sources`.
Parent manifests/executables, last/best checkpoints, curves, validation files,
SID maps and associated metadata are SHA256-pinned in the config. All4last
checkpoints have epoch20/6680updates; all preserve one visible CUDA RNG state.
The local source audit verified64downloaded files(~321MB); no new feature upload.

Freeze each exact saved quantizer/index. Do not repeat AE, RQ or item encoders.
Reconstruct SID suffixes and assert equality with saved fullSIDs, then compare
index SHA256 with the checkpoint. Same1273films/943users, timestamp-group holdout,
strict-past history20 and85,470train targets/epoch. Verify all case digests and
source-film order before workers start. SASRec keeps its exact catalogue-ID map.

## Continuation policy

- Epochs1–20 and validation5/10/15/20 are historical prefix evidence, retained.
- Resume training at21; minimum total30epochs, maximum total100.
- Validation every new epoch; checkpoint criterion remains raw validationNDCG10.
- From21, ReduceLROnPlateau(max): factor0.5 after5consecutive failed checks,
  minLR1e-5, absolute improvement threshold1e-4. The user-facing5checks maps to
  Torch `patience=4` because Torch reduces after `num_bad_epochs>patience`.
- New scheduler is seeded with the historical best validation metric and epoch20,
  with zero bad checks at the phase boundary. It does not retroactively reinterpret
  sparse5epoch validation as every-epoch observations. It first monitors epoch21.
- Early stopping:20total epochs since the best raw validationNDCG10 improvement,
  subject to minimum30. Small strict improvements may update the checkpoint/ES
  while not satisfying the scheduler's1e-4threshold; those policies are explicit.
- Test once after selection. If the epoch100cap is reached before ES, mark the
  result budget-censored; do not declare convergence. Per-model continuation
  time guard7200sec; no automatic extensions or precision/LR fallback.

Keep model architecture, data/order, Adam0.001initial LR/weight_decay0.0001,
effective/physical batch256/128, clip1 and FP16train/FP32eval. New LR scheduling
and longer training change together; this is not an epochs-only ablation.
SASRec may stop earlier under the same rule. Match predictor capacity, not total
FLOPs or identical attained epoch counts; active counts remain4.59M/4.61M.

## Resume correctness

Restore last model parameters/buffers, Adam moments and step counters, scaler,
CPU/CUDA/Python/NumPy RNG and global epoch/update/overflow counts. Preserve the
old best checkpoint and metric, prior curves and validation prediction files.
Check seed/data/architecture, frozen map and optimizer settings before mutation.
New resume checkpoints additionally store case digests, cumulative timings and
scheduler state. Restoring an existing scheduler rejects a changed schedule.
Restore RNG immediately before the next training epoch, after object construction.
Record verification hashes and successful RNG restoration. CPU uninterrupted vs
split/resumed tests cover real dropout, every RNG stream, Adam, LR reductions,
prior best retention, epoch minima and stopping; native GRID tests cover all4SID
loss heads. RNG restoration does not enforce bitwise deterministic CUDA kernels.

The shared trainer's resume/scheduler hooks are opt-in; historical configs retain
their old fresh-training behavior. Resume failure raises rather than reverting to
a fresh optimizer or weights-only warm start. Source GPU checkpoints remain on
Kaggle; mounted parent files are read-only and new outputs use a distinct kernel.

## Diagnostics and reporting

Record training loss, LR used/next LR, free beam20 itemRecall/NDCG5/10, invalid
SID rate, and FP32 teacher-forced validation CE per SID level (including suffix).
For SASRec, log its full-catalogue item CE separately. Losses of different token
indices/objectives are not directly comparable. Additional diagnostic evaluation
runs in eval mode and an isolated Torch RNG context. Preserve selected/last
checkpoints with optimizer/scaler/scheduler/RNG, complete prefix+new curves,
all new validation predictions, source checksums and final selected test.

Report lifetime epochs/updates/timings separately from the new continuation
portion. Checkpoint selection uses validation only; earlier exploratory test
results are not an untouched holdout claim. Same users/targets/catalogue, no
seen-item mask or sampled negatives. Existing generator numerical variability
and the distinction between next MovieLens rating and VOD viewing remain.

## Execution boundary

One seed42pilot,4continuations on2×T4: worker0 no-people+names TIGER, worker1 graph
TIGER+SASRec. Worst-case roughly2–2.5h plus setup/queue, based on~50sec perTIGER
epoch; ES may shorten it substantially. No new seeds, HPO, index changes,
quantizer retraining, automatic retries, TPU or Telegram. Parent studies preserved.
