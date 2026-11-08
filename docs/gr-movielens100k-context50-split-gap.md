# Context50 validation/test gap: saved-prediction audit

2026-11-04, Europe/Moscow. Existing context40/50 predictions, seeds42/101/202.
No model training, forward pass, GPU quota, new checkpoint selection for the
benchmark, ranking changes or experiment dispatch. Official metrics stay intact.

## Main finding: target choice materially changes the measured effect

The original split selects ONE film from the penultimate/last distinct timestamp
group using a SHA256 tie-break. Other co-timestamp ratings are embargoed. This
is deterministic and leakage-safe; it still leaves an arbitrary choice of label
when multiple films share the timestamp. There are2,042 embargoed events.

Validation has417 multi-film groups and526 singletons; test409 and534.
Group sizes are1-10 (means1.99 and2.17). Every alternate film is verified against
the canonical catalogue, the user's heldout timestamp and exclusion from train.

For each already saved ranking, compute the ONE-target NDCG@10 for each possible
film in that group, then average uniformly within the group and across users.
This is the exact expected one-target score under uniform label choice. It does
not require re-generation: beam generation takes only history/mask/user ID.
Use the original fitted checkpoints and the original test history, including
only the originally selected validation film. This is a sensitivity diagnostic,
not a replacement benchmark or a replay of fitting under other labels.

Relative graph/control NDCG@10, ratios of three-seed means:

| Window/split | Original selected target | Uniform within-group expectation |
|---|---:|---:|
| 40 validation | -2.83% | -1.31% |
| 40 test | +9.77% | +1.54% |
| 50 validation | +6.07% | +1.85% |
| 50 test | -7.99% | -4.80% |

At50, the graph-minus-control validation/test gap changes from **-0.007581**
to **-0.003428**, a54.8% reduction in this fixed-model diagnostic. This does not
causally attribute54.8% to a single mechanism: checkpoints and LR trajectories
were selected using the original validation labels. The residual gap remains,
and uniform expectation still does not establish a graph test advantage.
Conditional user intervals for all four expected graph/control differences
include0. The original40test gain was particularly sensitive to which film was
chosen as the one heldout label.

A further4,000 uniform label draws keep the same ranking and assign one common
label per user to all models/seeds. For50test, the graph-minus-control difference
ranges from-0.00574 to+0.00125 at the2.5/97.5% draw quantiles;10.7% of draws are
positive. These are label-choice sensitivity ranges, not confidence intervals,
p-values, independent test splits or separately trained runs.

All ratings1-5 are interactions in the current protocol. MovieLens timestamps
record rating submission, not film viewing. Multi-film timestamp groups should
therefore not be interpreted as ordered consumption events or universally liked
films. Multi-relevant NDCG/recall are archived separately as different objectives;
no comparison of their absolute values with the official one-target score is made.

## Broad target/user features do not explain most of the gap

Both splits have the same943 users; targets cover494/503 films with286 common.
Median training next-item frequency is117 for validation and105 for test; mean
138.69/132.33. Actor-overlap frequencies with the visible last50 history are
75.19%/76.35%; mean log actor catalogue frequency1.25705/1.25757. These measures
are calculated from train/catalogue metadata, not heldout outcomes.

An exact descriptive Kitagawa decomposition separates changes in group proportions
from changes of graph effect within groups. Bins use pooled metadata quantiles,
shared between splits, without looking at quality. Popularity terciles explain
only0.2% of the signed gap via composition. Popularity x actor-overlap six-stratum
composition contributes **3.7%**; after equalizing their proportions, **96.3%** of
the original gap remains. This is a coarse composition check, not causal adjustment
or proof that every possible item feature is irrelevant.

No monotonic actor-hub/graph-cosine/repeated-coacting rule appears. Absolute partial
Spearman correlations of the inspected actor features with per-user graph gain
are below0.04 after rank controls for target popularity, cast size and history.
Different functional forms, unmeasured attributes and low power remain possible.

Common-title standardization is unstable with support: using all286 shared titles
(637validation/643test cases), the pooled-title-weighted gap is-0.00982, so broad
label identity matching alone does not remove it. Requiring at least3cases per
split leaves48films/210validation/208test cases and nearly removes the gap;
requiring5 leaves only7films and gives a large negative gap again. These changed
subpopulations do not justify a general explanation by item identity.

## Checkpoint selection is not the whole explanation

Validation chooses the best of14 checks whereas test is evaluated once. Fixed
validation epoch30 reduces the graph delta to+0.000765 versus selected+0.003858,
but choosing that fixed epoch post hoc is not a correction to test quality.

As a separate diagnostic, divide the943 validation users into two halves, choose
each model's epoch on one half and score it on the other, then reverse the halves.
Across200 predefined random partitions, mean control/graph scores are
0.05687/0.06096, graph **+7.19%**, absolute delta+0.00409. Both absolute scores
fall from their in-sample selected values; the mean graph lead does not disappear.
Partition deltas span[-0.00122,+0.00979] at the2.5/97.5% quantiles. This is partition
sensitivity, not an independent validation result: the fitted LR schedules used
the full original validation set and the checkpoints were already inspected.

## Concrete users, films and a counterexample to a genre-only explanation

Per-user mean-seed graph deltas on validation and test have Pearson r=0.021.
For503users the average delta is0 in both splits; for219 only validation differs,
for132 only test differs, and for89 both differ. In495users neither model on any
seed hits either heldout target. These are target-score statements, not equality
of the recommendation lists. A single next rating is a sparse measure here.

Signed title contributions to delta(test)-delta(validation) sum exactly to the
aggregate gap. Largest negative contributions include:

| Film | Val/test target cases | Contribution to mean gap |
|---|---:|---:|
| Men in Black (1997) | 6 / 2 | -0.001677 |
| The Saint (1997) | 4 / 6 | -0.001448 |
| The Truth About Cats & Dogs (1996) | 7 / 1 | -0.000881 |
| Terminator2 (1991) | 4 / 0 | -0.000775 |
| The Rock (1996) | 5 / 7 | -0.000772 |

The two largest signed contributions equal41.2% of the net gap; top5 equal73.3%.
These titles were chosen after observing outcomes. Positive contributions offset
negative ones, so these fractions are not causal explained variance. Supports are
small and no single film accounts for all the discrepancy. All titles, user cases,
features and per-seed contributions are preserved, including counterexamples.

The **same293 Comedy test targets** supply a useful configuration counterexample:

| Window | Control NDCG10 | Graph NDCG10 | Graph/control | Seed signs |
|---|---:|---:|---:|---|
| 40 | 0.04684 | 0.05955 | +27.1% | + / + / + |
| 50 | 0.05226 | 0.03656 | -30.0% | - / - / - |

This contradicts a simple invariant rule that graph features are harmful for
these films. The models and maximum context differ, so it does not identify the
mechanism. For103of these users, the actual visible film list is identical at40
and50; their graph/control means still change from+12.0% to-29.6%. In total360test
users have identical unpadded item histories at both widths. Different fitting,
dropout and user-token relative positions remain confounded. This is not an
ablation proving that padding is responsible or that old events are harmful.

## Attractive subgroup leads fail robustness checks

A pause over1h before the test rating gives84users: control0.00661/graph0.02111,
positive deltas on all3seeds. But this comes from **4improved users,1worsened and
79tied**; total hits across252 user-seed cases are4control versus8graph.
At context40, the same subset has mixed seed signs; validation does not reproduce
the effect. No user improves on all3seeds individually. This is a lead to preserve,
not a demonstrated production scenario. Pauses are between ratings, not views.

The audit retains72 subgroup rows (history, popularity, actor overlap/hubness,
coacting, graph cosine, timestamp gap and overlapping genre groups). No subgroup
has strictly positive effects on all3seeds on both validation and test.
Approximate simultaneous paired-user bootstrap bands over these72 subgroup-
versus-complement interactions all include0 (4,000 draws, max-standardized-deviation
critical value3.398). These bands cover only this inspected family, not earlier
research, subsequent probes, training-seed uncertainty or movie clustering.

## Evidence and practical implication

Official model weights, checkpoints, SID indices, predictions and metrics remain
unchanged. The analysis rechecks90distinct context50prediction files, case orders,
all14validation scores per model and selected epochs against archived results;
context40comparisons use the already verified corresponding predictions/case table.
The target-choice diagnostic separately checks all official40/50scores, canonical
IDs, embargo fingerprint and target timestamps. Portable CPU replay passed.

A sensible next evaluation step is to retain the canonical one-target report and
add declared sensitivity to timestamp-group label choice or multiple relevance
labels before more architecture tuning. This diagnosis does not establish a useful
graph router or remove the need for a fresh, predeclared evaluation later.

Persistent numerical summary accompanies this report. Full case/film tables,
all72groups, correlations, contribution decompositions, label diagnostics,
checkpoint partitions, input fingerprints and replay scripts are in the evidence
bundle. It contains no model weights or credentials. Raw checkpoints remain on
private Kaggle. Temporary `/tmp/context50-split-gap` files may not survive an
environment replacement; the report and replay sources are committed.
