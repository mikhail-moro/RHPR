# Fixed-user context truncation on the frozen epoch-60 models

The user authorized this inference-only check on 2026-10-27. Restore exactly the
original and ALS-uniform RQ-KMeans generators from the completed 60-epoch run,
source `10e716dfe69a7afef766e7647c876a07478c2714`. Both selected checkpoints are
epoch60. Pin checkpoints, complete SIDs, selection reports, reference predictions
and the parent manifest by SHA256/size. No fitting or optimizer is constructed.

`experiments/amazon_beauty_sid_context_ablation_2t4.json` reuses the private parent
notebook output and the existing prepared Beauty/runtime dataset. No data upload,
index generation, extra checkpoints, test pass or training is authorized here.

## Intervention

For every original validation user, keep user ID and target fixed. Replace older
history events with PAD to leave the latest 3,5,9 or20 events visible. The tensor
width stays20events, surviving positions do not move and every pass uses the
same full22363-user order, FP32 batch32 and free beam20. Evaluate one model per T4.
Users with shorter histories retain all available events; their input is unchanged
for lengths at least their observed history. Their predictions must remain exact
within this fixed-layout sweep, or the run fails its numerical control.

The full20context baseline is evaluated first and compared with the saved parent
epoch60 predictions and aggregates. Record any parent replay drift explicitly;
all intervention contrasts use the newly replayed baseline, not mixed old/new
predictions. Hash model state before/after and preserve exact complete SID paths.

## Fixed cohorts and contrasts

- **Primary:** the same1019users with at least20observed TRAINevents at every
  length3/5/9/20. Primary interaction is
  `(swapped20-original20) - (swapped3-original3)`, averaged over these users.
  A positive interaction means the swap advantage grows with visible context.
- **Secondary:** the same4242users with at least9TRAINevents, at lengths3/5/9;
  compare the change in model difference from3to9.
- **Secondary:** the same10980users with at least5TRAINevents, at lengths3/5;
  compare the change in model difference from3to5.

Use paired user uncertainty, preserving dependence across both models and all
lengths. These are one pair of trained models; intervals exclude fitting/seed
uncertainty. Secondary cohorts and other pointwise comparisons are exploratory.
All primary cohort masks depend on TRAINhistory length alone, not ranking results.

Save all8full validation beam files, compact per-user ranks/NDCG/Recall matrices,
user/target IDs, original history lengths, cohort hashes, restored-state hashes,
unchanged-input controls and parent-prediction agreement. Analyse both the actual
model differences and their within-user changes across lengths. Do not infer a
context interaction from different users evaluated at different lengths.

## Interpretation

This isolates the effect of available recent events for these frozen models. It
does not simulate a genuinely new user: learned parameters, user-bin token and
training exposure remain fixed. Nor does it directly compare a semantic model
with a CF model: both were trained for NIP with different frozen item-to-SID maps.
Less context may make inputs differ from their original inference distribution.

Expected inference is about3–4minutes per worker plus notebook setup, typically
5–8minutes total. No Telegram, automatic retries or further experiments.
