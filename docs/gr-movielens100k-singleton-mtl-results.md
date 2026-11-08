# Singleton holdouts: three-seed MTL results

Collected2026-11-05MSK. Private carasik/rhpr-ml100k-singleton-mtl v1 COMPLETE.
Executable35ce22dd1f07aaf02251af7847eb242f91187413. All12fresh pipelines
completed30epochs,9420updates,6VALchecks and0AMPskips. Runner195.26min/3h15m
on2T4. No new GPU/model calls, training or dispatch during collection.

## Validation-selected checkpoints

| Method | Mean VAL NDCG10 | Mean TEST NDCG10 ± sample SD | TEST vs sum | TEST wins |
|---|---:|---:|---:|---:|
|sum|0.07089|0.04665 ± 0.00240|+0.0%|0/3|
|uw|0.06572|0.04958 ± 0.00203|+6.3%|2/3|
|gradnorm|0.06826|0.04604 ± 0.00283|-1.3%|1/3|
|pcgrad|0.06475|0.04778 ± 0.00423|+2.4%|2/3|

| Seed | Sum test | UW | GradNorm | PCGrad |
|---|---:|---:|---:|---:|
|42|0.04938|0.04938|0.04862|0.05216|
|101|0.04490|0.04765|0.04648|0.04746|
|202|0.04567|0.05170|0.04302|0.04372|

UWtest seed effects -0.01/+6.13/+13.22%;42is practically tied. UW is the strongest
TEST point estimate, not a confirmed downstream winner: mean VAL is7.29%lower,
winning1/3VALseeds. GradNormVAL-3.72%,PCGradVAL-8.66%(loses3/3). A method chosen
by mean validation would remain SUM. Do not select UW on inspected TEST results.

All5000draw paired-user conditional delta95%CIs cross0. TEST UW+.002927,
CI[-.003411,+.009175];GradNorm-.000611,[-.007148,+.005944];PCGrad+.001128,
[-.005000,+.007149]. Users resampled jointly after averaging the3fitted pairs;
this excludes training/runtime seed uncertainty, target clustering, multiplicity
and the prior exploratory history. Three pipeline seeds share the same903heldouts.
Sample SD is not a confidence interval. No equivalence/general MTL-harm claim.

TEST Recall10 sum.10114,UW.10188(+0.73%),GradNorm.09376,PCGrad.09967. UW's NDCG
lead does not imply a similarly large increase in number of top10hits. Its delta
splits into+.000630on common hits and+.002296net gained/lost-hit contribution;
these are descriptive rank decompositions, not attribution of a causal mechanism.

Best epochs42/101/202:sum30/25/20,UW20/30/25,GradNorm25/30/10,PCGrad30/15/20.
Four select the30epoch cap; this is the requested fixed-budget screen, not proof
of convergence. Fixed-epoch mean validation at20/25/30 also favors sum over every
MTL alternative; the disagreement is not solely best-checkpoint selection.

## Target ambiguity was removed; val/test disagreement remains

Independently reconstructed all91469original retained interactions using pinned
sequence and embargo files, without trusting reported group sizes. The last two
singleton timestamp groups and>=2earlier training timestamp groups reproduce all
903kept users, targets, training and intervening contexts exactly. No random or
SHA-selected movie inside a multi-film holdout group. All40exclusions preserved.

Verified82234train events,80268strict-past examples,903VAL/TEST each,3173between
VAL/TESTevents only in TESTcontext,3098events after TESTdiscarded. All original
source-file byte hashes match and all3case digests match pinned preflight. No
training target sees same-timestamp or future events. TEST context includes all
strictly earlier available interactions before truncation to20. Historical ties
still have item-ID order within context; timestamps denote ratings, not views.

Thus this run's val/test disagreement cannot be explained by arbitrary choice of
one item WITHIN a tied holdout group. It does not establish a faulty split or
explain the remaining disagreement. 903single-target users still give sparse,
variable quality measurements. Changed endpoints/cohort mean old and new absolute
scores are not directly comparable; this is not a new independent dataset.

## MTL behavior

Across156saved probes,468pair cosines on the last shared encoder weight, none
negative; lowest.306. This is sampled and layer-specific. PCGrad now also logs
cumulative full-shared-coordinate projections on ALL6500optimizer calls/model:

| PCGrad seed | Calls with projection /6500 | Pair projections | Already in first500calls |
|---|---:|---:|---:|
|42|12|29|23|
|101|21|65|65|
|202|11|35|31|

Only44/19500calls(.226%)need projection. 119/129pair projections occur in the
first500AEcalls;10occur later. Some directional conflicts EXIST, but they are
rare under this recipe, not persistent. The projection-based mechanism therefore
has little work to do. This does not rule out capacity tradeoffs, loss-scale
imbalance or reconstruction/NIP objective mismatch.

GradNorm consistently favors plot:final text/poster/plot weights42 .952/.962/1.086,
101 .936/.930/1.133,202 .960/.938/1.103. This does not yield consistent NIP gains.
UW weights remain close to one another, with sums3.553/3.477/3.584 vs3control:
UW changes global reconstruction/quantization balance as well as relative weights.
A positive UW point estimate cannot be credited specifically to modality balancing.
Summed losses remain the validation-supported default in this experiment.

## Verification and evidence

- All84saved prediction files independently decoded/rescored:72VAL+12TEST,
  same903users/targets, NDCG/Recall5/10 and original invalid/duplicate ranks.
- Only GradNorm101TEST has2invalid paths/18060;other11TESTs0,all0duplicates.
  Intermediate validation path counters also independently matched.
- Semantic maps reconstructed incl suffix, catalogue/book counts consistent;
  all full SIDs unique. Initial generator and AE parameter pins checked per seed,
  frozen feature hashes identical,3500AE/3000RQ/freshoptimizer/neuraltransfer verified.
- Bothworkers complete with6fits and no failures. Exact executable/config match.
  Replayed all6checkpoint checks and best-VAL selection for every model.
- Runtime training unchanged in collection; CPU replay invariants pass. No need
  to rerun969prelaunch tests for analysis-only additions.

Replay scripts/research/analyze_singleton_mtl.py RAW OUTPUT --original ORIGINAL.
Summary docs/gr-movielens100k-singleton-mtl-results.json; local analysis
/tmp/singleton-analysis contains curves, per-user CSV, metrics, verification andPNG/PDF.
Full portable numerical bundle /tmp/movielens100k-singleton-mtl-numerical-evidence.zip
contains original replay inputs, saved predictions/indices/MTL curves and source.
Large weights remain in the private Kaggle notebook. No further compute queued.
