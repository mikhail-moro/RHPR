# Where random32 gains: saved-prediction diagnosis

Collected 2026-11-15. CPU analysis only: no training, inference, checkpoint
reselection, probability tuning, new seed, or Kaggle quota use. All five
checkpoints remain raw-VAL-selected epoch55. Same958users and targets per
split, frozen indexes and evaluation settings. TEST has already been examined;
these are exploratory observations on one fitted seed, not confirmation.

## Main finding: a plausible Pair32 recovery, fragile Recent superiority

|Model|VAL NDCG10|VAL hits1 / hits10|TEST NDCG10|TEST hits1 / hits10|
|---|---:|---:|---:|---:|
|Random32|.017308|7 / 30|.014961|6 / 26|
|Pair32|.015966|6 / 31|.010651|2 / 21|
|Previous Mixed32|.018937|7 / 35|.009874|2 / 20|
|Recent32|.016899|6 / 30|.014447|3 / 28|
|Recent20|.020492|9 / 38|.014722|7 / 24|

Random's TEST advantage over Pair32 is+.004310NDCG, or4.1287summed
discounted credits. Its17new hits contribute+7.8808credits,12lost hits
contribute negative credit, and shared ranks contribute the remainder;
the exact additive accounting is in the aggregate JSON. All gains combined
are10.4354credits versus6.3068lost credits. This is a change in which targets
are predicted, not simply five extra hits on the same list of successes.

The count of rank1hits increases2→6relative toPair. Accounting by rank,
rank1 contributes+.004175of the total+.004310NDCG difference (96.9%);
ranks2–10contribute+.000134. This does not mean four individual cases explain
all model differences: targets enter, leave, and change rank simultaneously.

Relative toRecent32,the advantage is much smaller:

* VAL: identical30top10hits, but7versus6rank1hits; net+.000409NDCG.
* TEST:26versus28top10hits, but6versus3rank1hits; net+.000514NDCG.
* TEST rank1 contributes+.003132; ranks2–10contribute−.002618.
* Removing just one+1credit contribution from the numerator, retaining the
  original958denominator, flips either selected VAL/TEST difference negative.
  This is a sensitivity calculation, not permission to drop an evaluation case.

AgainstRecent20,TEST's margin is+.000238NDCG, also smaller than one rank1
contribution. VAL is−.003184, with30versus38hits. Random32 inference costs
20.93% more forward FLOPs thanRecent20
(.516501vs.427088GF). A useful Recent20 replacement has not been established.

## Concentration, uncertainty, and concrete cases

All eight selected VAL/TEST conditional paired-user95%intervals cross0.
Resampling target-item clusters rather than individual users also leaves
all eight intervals crossing0. For TEST Random−Pair the target-cluster
interval is[−.002324,+.011201]; Random−Recent32[−.006736,+.007687].
Neither procedure includes seed/index/checkpoint-selection uncertainty;
they address different dependence assumptions, not independent replications.

The26Random TEST hits involve23distinct target books. Therefore gains are
not all repeats of one target book. The largest positive target-level
contribution relative toPair is1.5178credits,36.8%of the net4.1287credits;
subtracting it still leaves+.002725NDCG at the original denominator. Against
Recent32,one target's+1credit exceeds the entire net+.4924credit margin.

Examples of TEST Random improvement overPair:

* *Renewing the Soul: Souls of Chicago series*: outside saved beam20→rank1.
  This restores one of the original Pair32 failures; a related book was already
  present in the uncompressed recent tail. It does not prove loss of old content.
* *Fallen Jester (Clifton Forge Book5)*:rank11→1.
* *Marrying Mr. Wrong (Dirty Martini Running Club Book3)*:rank8→1.
* Counterexample:*Verity*:rank1→6.

Per-user ranks/features, the top five gains AND losses for every comparator
and split, and catalog features are retained privately. Saved beam absence
does not identify whether generator logits or beam pruning caused a miss.

## Two leads that have the same sign across splits

First, recovery on low-support targets relative toPair32:

|Targets with1–5TRAIN next-item positions|Users|Pair NDCG / hits10|Random NDCG / hits10|
|---|---:|---:|---:|
|VAL|402|.004571 / 5|.009609 / 7|
|TEST|425|.000784 / 1|.005162 / 5|

This group contributes+.002114to the global VAL delta, larger than the net
global+.001343:other target-support groups offset part of the gain. AtTEST
it contributes+.001942of+.004310(45.1%);6–20support contributes+.001726,
and>20support+.000642. Thus the TEST improvement is not confined to rare
targets. Cold targets remain0hits. Counts of successful rare targets are tiny,
and Recent32's TEST rare-target NDCG is still higher(.006773,5hits).

Second, recent-context popularity separates Random versusRecent32 consistently:

|Split/group|Users|Recent32 NDCG|Random NDCG|Delta|
|---|---:|---:|---:|---:|
|VAL, less popular recent context|479|.016051|.010120|−.005931|
|VAL, more popular recent context|479|.017747|.024497|+.006749|
|TEST, less popular recent context|502|.011788|.010416|−.001372|
|TEST, more popular recent context|456|.017374|.019963|+.002590|

Popularity is mean log1pTRAINtarget frequency among the latest16books.
Threshold2.30082044was fixed in the earlier Pair diagnosis, not chosen for
this run. Holding the original VAL low/high user membership fixed onTEST
preserves both signs. This is a descriptive lead, not a validated router or
causal evidence of popularity bias. In particular,Random loses toRecent20
in the high-popularity group on BOTH splits;the lead does not supply a clear
winning group against the stronger, cheaper baseline.

## Negative findings retained

Eight exhaustive group families were inspected; all cells are in JSON/CSV.
New continuous thresholds are medians of VAL covariates, not quality-optimal
cutoffs. These remain additional exploratory comparisons without multiplicity
correction. The history threshold112is TRAINexample count per user, not the
current visible context length (which is32for everyone here).

* Total user history length does not consistently separate performance versus
  Recent32:shorter histories gain onVAL but lose onTEST, longer do the reverse.
* Average cosine similarity of the8pair constituents flips its association
  across splits. No reliable “merge similar books” rule follows from these data.
* Target similarity to older versus recent items also reverses versusRecent32.
  This uses the future target and could not be a serving gate even if stable.
* Users with a previously seen complete pair show no consistent advantage
  versusRecent32:VAL−.000206,TEST+.014358(n128/111). Most users have a novel
  pair;the opposite signs do not support a simple novelty explanation.
* Same-user improvement does not persist strongly between different targets:
  Random−Pair improves the same user on both splits only once;879of958users
  have zero NDCG delta on both. Random−Recent32 likewise improves both once.
  Sparse hits and different target books limit this diagnostic;it does not
  establish that personalization is absent.
* Random beatsRecent32at9/12VALchecks,but only2/5checks at40–60and has
  mean late delta−.000641. VersusPair it wins4/12overall,1/5late,with mean
  late delta−.000055. A positive selected epoch is not a uniformly better
  late-training curve. All models reached the60epoch cap.

## Pair novelty, leakage checks, and what the seed means

Independently replayed10selected prediction files and matched all saved
user/target IDs and ranks. All10input hashes remain pinned. In BOTH splits,
0targets occur in their own input context or among the same user's TRAIN
targets. The earlier full audit verified every sampled TRAINepoch, raw targets,
initializer, index and VAL-only selection. These checks found no target-format
or same-user target-reuse explanation for the TEST gain. They do not turn this
static, transductive, already-inspected cohort into a temporal deployment test.

The114727potential adjacent TRAINpairs for random tiling are EXACTLY the
same set as the114727pairs available under fixed tiling over all rolling TRAIN
windows. Changing window endings already exposes both parities in that set.
Random's benefit therefore cannot be attributed to enlarging this set.
Actual exposure frequency and combinations of pairs/singles change;potential
membership does not show how often a pair received an optimizer update.

SID assignments themselves are NOT random during training:both single and
pair indexes remain frozen at quantizer seed42. What is sampled is the number
and placement of merges in each input. The current pilot fixes all seed bases
at42,with separate NumPy RNGstreams7401(count)and7402(placement);these draws
do not consume the Torch initialization/dropout stream. The same class seed
also determines its expected seed+epoch training permutation. The fixed runner
and its contract currently support only42, so do not claim a configurable
factorial seed experiment has already been implemented.

A future harness should expose distinct `index_seed`, `training_seed`
(initializer/shuffle/dropout),and `augmentation_seed`(count/placement).
Changing training and augmentation seeds together estimates variability of
the whole method;holding training/index fixed and varying only augmentation
estimates sensitivity to the sampled tiling. Both are useful but answer
different questions. Matching seed labels across methods does not imply
identical dropout masks when tensor shapes differ.

## Next experiment, not launched

The evidence supports testing the SAME stochastic within-history training
rule at a20item window before adding schedules, feature gates or stronger
compression. This is the proposed “Mixed20”, not the previous whole-batch
Full32/Pair32method. At inference,older10items become5pairs,recent10remain
single:15SIDblocks×4+USER=61encoder positions,versusRecent20's81.

Use the same cohort/splits/targets and frozen item/pair centroids,3+1codes
for every input and target,original deterministic Pair20inference,and retain
Recent20as the primary baseline. If the different window requires uncached
pair keys,encode them with frozen models and append unique suffixes without
refitting old assignments. Later24/28windows can bracket the tradeoff.
Do not select lengths,probabilities or seeds by the already-inspectedTEST.
No new fit or inference was launched during this diagnosis.

## Reproduction and artifacts

`scripts/research/diagnose_kindle_random_tokenization.py` loads the saved
predictions, cases and ST5cache; verifies rank identity; computes all groups,
exact gain/loss accounting, conditional intervals and concentration checks.
All partition contributions sum exactly to the global deltas. No new model
outputs are computed. Full regression:1132tests PASS,152optional skips,
0failures,81.577s;the analyzer was also executed end to end successfully.

Aggregate evidence:`gr-kindle-random-tokenization-diagnosis.json`.
Private directory:`/workspace/artifacts/kindle-pilot/random-tokenization-diagnosis`
contains`analysis.json`,`groups.csv`,`metrics.csv`,`user-features.csv`,
`extreme-examples.json`,and`diagnosis.png/.pdf`.
