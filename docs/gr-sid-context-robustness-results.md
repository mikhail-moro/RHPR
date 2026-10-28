# Ten retained models: completed context robustness

Original session `carasik/rhpr-context-robustness-replay`, source
`d4305b9c563a77d967ff1e2b2dcefaec33f07c1e`, completed all ten models and forty
inference passes. All parent predictions replay exactly; all unchanged-input
controls and model-state hashes pass. Forty raw beam files independently match
the compact per-user ranks. No fitting, test pass, replacement run or cancellation.

## Timing

The persisted log starts with runtime-ready at34.65seconds; dependency installation
and imports take33.86seconds. Workers start at43.13seconds, replay body takes
1077.88seconds, and notebook log ends at1117.78seconds (18min38sec). This matches
the15–20minute compute estimate. The user/API observed a much longer interval
without live stdout; the completed log does not identify that external delay.
There is no evidence here of a30minute pip hang. Prepared startup diagnostics
and lazy parent lookup were not used to generate these results.

## All 22,363 users: original semantic-index controls

Caps retain min(K,observed history), with the same users and targets throughout.
The interaction is `(candidate20-original20)-(candidate3-original3)` in NDCG@10.

| Models / signal | Swap effect at cap3 | At cap20 | Interaction | Paired-user 95% interval |
|---|---:|---:|---:|---|
|RQ-KMeans / ALS /60epochs (earlier run)|-1.96%|+2.25%|+0.001615|[+0.000389,+0.002875]|
|RQ-KMeans / cooccurrence /20epochs|+2.23%|-3.88%|-0.002195|[-0.003255,-0.001106]|
|RQ-VAE / cooccurrence /20epochs|-1.35%|-1.74%|-0.000144|[-0.001234,+0.000924]|

For the20epochRQ-KMeans pair, target-item block resampling also gives a negative
interval[-0.003402,-0.000946]. Thus the60epochALS finding does not replicate as
a universal increase in swap advantage with more visible context.

Within the same1,019users with at least20events, interactions are+0.0140825 for
the60epochALSpair, -0.0103253 for20epochRQ-KMeans/cooccurrence, and+0.0002426
for20epochRQ-VAE/cooccurrence. The last is indistinguishable from0; the negative
RQ-KMeans estimate has a user interval below0 but a target-block interval
overlapping0. Full-population and matched-cohort conclusions are consistent.

## Epoch10 controls are different

Six retained10epochmodels provide uniform/priorityALS and random reassignment
in each family. Original10epoch checkpoints do not survive. All four ALS/random
all-user context interaction intervals include0, as do their1,019-user intervals.
Higher point quality than random does not establish growing advantage with context.
Do not treat these as original-index replications or independent seed repeats.

The compared20and60epochindices differ in CF signal and accepted-swap budget.
This does not isolate training duration and does not disprove the positive effect
for the exact60epochALSpair. All generator seeds are42; controls are shared.
The justified conclusion is configuration-dependent context sensitivity, requiring
further controlled replication before a semantic/CF switching rule.

## Retained artifacts

`/workspace/artifacts/sid-context-robustness/analysis/README.md`, figures,
all paired cap/interaction JSON with user and target-cluster sensitivity, model
verification, timing evidence and target contributions. The long-form Parquet
`all-model-user-context-metrics.parquet` has894,520rows with model, epoch, user,
target, context cap, effective length, rank, NDCG and Recall. Full raw beams
remain on the private Kaggle output; their independent hashes are retained locally.

No additional experiment is pending or automatically authorized by these findings.

## Clarification: ALS20 metrics do exist

The same long original/ALS trajectories retained epoch20 validation predictions:
original0.03642522, swapped0.03637116, effect−0.15%. At60the corresponding scores
are0.03985706/0.04075540,+2.25%. These ordinary full-context metrics should be
considered alongside the cooccurrence20effect−3.88%;the previous context-sweep
summary omitted ALS20because its weights were overwritten, not because its
validation outputs were absent. Joint user-bootstrap intervals for ALS-minus-cooc
swap effects at20and the ALS effect change20to60both include0. Budgets also
remain different (256vs62accepted swaps),so this is not a pure signal ablation.

The checkpoint-retention limitation prevents a counterfactual3/5/9context sweep
at ALS20without new training. For future duration studies, retain the specified
intermediate weight snapshots as well as validation predictions.

Cooccurrence was normalized cosine on binary TRAIN-user incidence, with global
weight normalization in the LCP objective. The known concern is sparse evidence
(88.3%of top20edges have only1common user),not a demonstrated normalization bug.
Full audit:`/workspace/artifacts/sid-context-robustness/als20-audit/README.md`.
