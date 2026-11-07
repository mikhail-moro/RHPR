# Frozen MovieLens actor models: unseen-item serving evaluation

Authorized2026-11-03. No fitting, quantizer changes, checkpoint reselection,
new seeds or graph modifications. Reuse six original validation-selected models
from private `carasik/rhpr-ml100k-actor-full-training v1` (source70a0c16).
Every checkpoint, SID matrix, reference prediction and parent manifest is
SHA256-pinned in `experiments/movielens100k_actor_seen_serving.json`.

Both control and actor.1 use the same943users/targets, original20-event inputs,
FP32 inference and the same full-catalogue scoring. Known-item eligibility is
all training history; test additionally includes the selected validation item.
Embargoed timestamp-group events are excluded. Targets must never be known.
The encoder continues receiving the original20visible events; masking uses
external historical metadata, not additional encoder features.

For each frozen model and each split:

1. Independently reconstruct original beam20NDCG from saved predictions.
2. Reproduce the first original32-user batch at beam20, requiring exact paths
   and numerically matching original probabilities before interpreting changes.
3. Generate a common beam100, preserving GRID's raw probability arithmetic.
   Report unmasked beam100 separately to distinguish width from eligibility.
4. Decode complete valid SIDs, remove known movies and duplicates, retain top10.
   Common prefixes are never blacklisted; no child-local renormalization.
5. If a user has fewer10eligible candidates, regenerate that user with beam256.
   Any remaining shortage uses exact raw autoregressive catalogue probabilities,
   evaluated in catalogue chunks for that user only. Explicitly count both paths.
   Exact fallback may change beam ranking; save complete log probabilities.
6. Require10valid unseen outputs per user; persist their item IDs and original
   model probabilities, raw beam100, refill256 and all per-user metadata.
   Check learned-weight hashes before/after. No model or checkpoint is updated.

This is a new serving policy on old checkpoints, not a new best-checkpoint
selection protocol or an independent holdout. Report original, wider unmasked,
and refilled eligible metrics separately. Validation/test have already been
examined in exploratory work. Their shared-user history does not constitute
independent replication. The previously reported retained-beam20filtering
was only a diagnostic and must not be mislabeled as this evaluation.

Two GPU workers, three models each. Private new kernel
`carasik/rhpr-ml100k-unseen-serving`; do not overwrite completed parents.
Quota before dispatch:7.90GPU-hours remaining. Runtime is inference only;
startup can dominate. Save progress immediately at each completed split.

Validation:952 repository tests passed(90skipped), plus native tiny-GRID check
that exact fallback's full-path teacher-forced log probabilities match cached
beam-path raw probabilities. Original training/evaluation defaults unchanged.
