# Context-length robustness across retained 10/20-epoch models

On 2026-10-27 the user chose all ten retained models, inference only, about
15–20 minutes on two T4s. Configuration:
`experiments/amazon_beauty_sid_context_robustness_2t4.json`.

Reuse four epoch20 checkpoints: original/cooccurrence-swapped RQ-KMeans and
original/cooccurrence-swapped native RQ-VAE. These are two direct comparisons
against original semantic indices. The existing read-only validation replay
provides their byte-pinned reference prediction files in a separate parent.

Reuse six epoch10 checkpoints: ALS uniform, ALS priority and random reassignment
for each of the two index families. Original epoch10 checkpoints were overwritten
by later best/last states and are not available. Compare ALS variants with the
same-family random control, separately from original-index comparisons. Random
matches the priority-changed item set/available paths; uniform changes a somewhat
different item set. No new original control is trained, and no later checkpoint
is mislabeled as epoch10. Shared random baselines make comparisons dependent.

## Population and intervention

All four passes use the same original 22,363 validation users, target items,
FP32 batch32, free beam20 and padded width20. Mask old events to PAD for caps
3/5/9/20; never move surviving positions or change weights, user IDs or SIDs.

Report two distinct estimands:

- **All users:** policy "up to K recent events", effective length `min(K,h)`.
  This retains the same full population at each cap. For 7,162 users with h=3,
  every cap produces the same input; the all-user effect includes these zeros.
- **Matched cohorts:** the same 1,019 users with h>=20 at exact lengths3/5/9/20;
  the same4,242 with h>=9 at3/5/9;the same10,980 with h>=5 at3/5. These estimate
  sensitivity when all compared history lengths are available.

For each registered comparison/cohort, report model differences at each cap and
the paired interaction `(candidate_long-control_long)-(candidate3-control3)`.
Use joint user resampling across both models and lengths. Treat robustness
comparisons as exploratory, with no pooling of different control types or
independence claim across shared seeds/models. All models use generator seed42;
these are not ten independent training-seed replications. Index construction,
CF signal and training budgets differ across experiments, so an epoch contrast
does not isolate training duration alone.

Epoch60 data already contain all-user predictions and are reused locally without
rerunning those models. Across all users the original/ALS changes at caps3/5/9/20
are−1.96%/+1.34%/+2.00%/+2.25%. The policy interaction is+0.00161533 with paired
user interval[+0.00038944,+0.00287528], conditional on that fitted pair.

## Integrity and budget

All parents/manifests/checkpoints/SIDs/selections/reference predictions are
SHA256/size pinned. Each trial declares its exact expected epoch/training config;
per-trial overrides support mixing existing 10/20 checkpoints without relaxing
verification. Verify original layout replay and exact unchanged-input predictions;
hash model states before/after. Save all40beam files and compact user matrices.

No fitting, optimizer, new embeddings/indices, test pass, data upload, seed search,
automatic retries or Telegram. Exactly one new private inference session with
five retained models per GPU. Large checkpoints remain mounted from existing
private parent notebooks, rather than downloaded/reuploaded locally.
