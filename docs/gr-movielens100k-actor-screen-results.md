# Actor-only graph and reconstruction-weight screen: completed

Collected2026-11-03 MSK. Private carasik/rhpr-ml100k-actor-screen v1
completed all10fits successfully on2T4 in157.86min (2h37m52s).
Every pipeline finished exactly30epochs with6validation checks, no LR
scheduler or early stopping. No new compute was launched during collection.

## Primary selection: validation

Each generator checkpoint is selected by its validation NDCG10. The candidate
is selected by mean validation over the two pipeline seeds, not by test.

| Candidate | Mean val NDCG10 | Mean test NDCG10 | Test sample SD | Test Recall10 |
|---|---:|---:|---:|---:|
| No people | .05695 | .04223 | .00123 | .08802 |
| Old shared graph, cast readout, weight1 | .06021 | .03957 | .00236 | .08908 |
| New actor-only graph, weight1 | .06047 | .04308 | .00719 | .09067 |
| New actor-only graph, weight.3 | .06117 | .04247 | .00031 | .08908 |
| New actor-only graph, weight.1 | **.06325** | **.04535** | .00235 | **.09491** |

Weight.1 is best on validation on BOTH seeds. Within the new graph,
validation improves1→.3→.1 on both seeds, although seed101 differences are
small. Test is not monotonic in the coefficient. SD from two points is
not a reliable estimate of training stability.

## Per-seed selected test NDCG10

| Candidate | Seed42 | Seed101 |
|---|---:|---:|
| No people | .04136 | .04310 |
| Old shared graph, cast readout, weight1 | .03790 | .04124 |
| New actor graph, weight1 | .03799 | **.04817** |
| New actor graph, weight.3 | .04269 | .04224 |
| New actor graph, weight.1 | **.04701** | .04369 |

Validation-selected weight.1 beats no-people on both test seeds: +13.7%
and+1.4%, mean+7.4% (+.003124 absolute). Its mean validation gain is+11.1%.
Mean Recall10 grows7.8%. Paired-user bootstrap95%CI for mean testNDCG delta
is[-.00450,+.01083], crossing0. This conditions on these two fitted models,
excludes seed/runtime/graph-seed uncertainty and multiple comparisons, and
cannot establish significance/equivalence or a generally optimal coefficient.

Weight.1 versus new-graph weight1: validation+4.6%(2/2seeds), descriptive
test+5.3%(1/2). Seed101 actually prefers weight1 on test. Thus weaker graph
reconstruction is a candidate supported by this validation screen, not a
proved universal downstream improvement.

New actor-only graph at weight1 versus old shared-graph cast: validation
+0.4%(1/2), test+8.9%(2/2,one nearly tied). Weight.1 versus old shared-graph
cast gives test+14.6%(2/2). These do not isolate the new graph's effect at
weight.1 because the old graph was not tested at that coefficient. This was
not a full topology×weight factorial.

The large difference between actor-only weight1 seeds(.03799/.04817),
despite excluding director-role edges, also cautions against attributing
previous variability specifically to directors. No identity/random-feature
control was included: benefits cannot yet be credited to DeepWalk over
shared actor identity or another auxiliary representation.

## What the reconstruction intervention changed

New actor graph weight1→.1, raw per-head RQ MSE:

| Seed | Text | Poster | Plot | People |
|---|---|---|---|---|
|42|.8050→.7891|.8484→.8271|.8469→.8463|.7601→.8655|
|101|.8169→.7881|.8480→.8246|.8633→.8423|.7770→.8650|

Other channels reconstruct better while graph reconstruction worsens on both
seeds. This is consistent with a changed tradeoff; it does not establish a
causal NDCG mechanism or a validated reconstruction proxy. Weighted total
losses across coefficients represent different objectives.

## Budget and comparability

Fixed3500AE/3000nativeRQ, same pretrained text/poster/plot, original1273film/
943user timestamp-group split,85470strict-past examples per epoch, context20,
TIGER4+4/d128 FP16train/FP32eval and matched initial neural weights.
Each seed runs all5candidates on the same assigned GPU. Actor graph seed42
is fixed across pipelines; the graph is not re-trained per generator seed.

The new graph has15998cast nodes and ONLY cast-role film membership edges;
64D mean actor features occupy the first half of the old128D schema with
zero director padding. Selected64coordinates are catalogue normalized and
alone contribute people MSE. Coefficient applies to BOTH AE and RQ; all
other active reconstruction weights1 and native quantization losses unchanged.
Other frozen modalities are unchanged. No-people excludes the people block
and loss entirely. The old-shared cast arm uses exactly the old vectors.

Best epochs(42/101): nopeople10/30; oldsharedcast30/30; newactor1 25/20;
newactor.3 15/25; newactor.1 30/25. Weight.1 seed42 selects at the30epoch
boundary, so convergence is not established. Prior full-training/ES metrics
are historical references, not matched controls for this fixed30epoch screen.

This is exploratory next-rating prediction on a previously examined split,
not observed VOD viewing order or an untouched final benchmark. Two seeds
share the holdout. A longer or independent replication would require a new
explicitly authorized experiment; none was submitted.

## Verification and saved evidence

- Official Kaggle COMPLETE; all10results/workers successful, each worker
  holds one seed. Executable/config exactly match submission.260numerical,
  metadata and log files downloaded and hashes recorded; listing has no
  next page. Large .pt weights remain private Kaggle.
- All70saved val/test predictions independently decoded/rescored: same943
  users/targets, NDCG/Recall5/10, invalid/duplicate beam slots agree exactly.
- Reconstructed all train/val/test cases and verified their digests against
  the prepared split. No changed targets, sampled negatives or seen masks.
- Actor input feature/provenance/sourceID hashes and original source order
  match. Recomputed each arm's normalization from the raw frozen inputs;
  selected role columns and inactive zero padding verified.
- Quantizer/generator seed-specific neural initialization pins, raw/weighted
  AE/RQ losses,3500fixedAE/3000RQ plus initialization, neural transfer/fresh
  RQ optimizer/initial latent checks verified. Every epoch has85470examples,
  334updates; all finish10020optimizer updates with ZERO AMP skips.
- Checks at5/10/15/20/25/30, validation-best checkpoint selection and constant
  LR agree. No resume/scheduler/ES was used in this screen.
- One invalid test SID among18860slots in actor_w03_seed42, zeroothers;
  no test duplicates. Selected weight.1 has none on either seed.

Executable5858df9f4682a15fdc740d3717469c8d4c6e22c5.
Configuration experiments/movielens100k_actor_screen_2t4.json.
Full analysis: /tmp/movie-actor-analysis (CSV, replay, per-user ranks, plots).
Workspace artifact link: artifacts/movielens100k-actor-screen/analysis.
Raw data: /tmp/movie-actor-completed. Standalone numerical evidence ZIP:
/tmp/movielens100k-actor-screen-numerical-evidence.zip. Temporary files may
not survive environment replacement; original outputs/weights are on Kaggle,
new actor graph on its private dataset, and the replay/report in the repo.
