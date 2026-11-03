# Actor graph: counterexamples and avoidable ranking losses

2026-11-03. Exploratory CPU audit of existing predictions from the completed
three-seed full-training run. No new training, inference, checkpoint selection,
router or production evaluation changes. Original full-catalogue evaluation
has **no known-item mask**. Original results remain unchanged.

## What can actually be isolated

There are six test users strictly worse with the actor graph on all three
seeds, but no established rule for identifying them beforehand. A measurable
source of ranking loss is recommending already-rated films. Mean known-past
occupancy of test top10 is **41.35% without people / 42.83% with the graph**.
In the 216 user-seed cases where the graph reduces NDCG10, these fractions
are **37.18% / 42.50%**. These are slots, not percentages of unique users.

We removed known movies from each model's **saved beam20**, preserving order,
invalid slots and all previously selected checkpoints. This is an intervention
on ranking, not on graph construction or model weights:

| Retained-beam diagnostic | Validation control | Validation graph | Test control | Test graph |
|---|---:|---:|---:|---:|
| Original, no filtering | .070774 | .065210 | .048112 | .047215 |
| Remove visible last20 movies | .091116 | .088087 | .062967 | .066071 |
| Remove all known past movies | .104387 | .100386 | .071584 | .077037 |

Test graph/control difference changes from -1.9% to +7.6% after all-past
filtering. Equivalently, filtering improves graph NDCG by .029821 and control
by .023472: the differential removable penalty is .006349. This identifies a
ranking effect within saved candidates; it **does not establish graph benefit**.
Validation still favors control (-3.8%); seed101 still favors control on test.
The fixed-model shared-user bootstrap95 interval of the filtered test delta
is [-.00214,+.01329], includes zero and excludes retraining variance.

Importantly, only **13/216 originally harmed test user-seed cases become ties**
after all-past filtering, and none becomes a strict graph win. The aggregate
change therefore does not mean this explains most individual graph failures;
filtering also benefits cases where graph already wins or both originally miss.

Known past = training events, plus the selected validation item on test.
Embargoed same-timestamp events are not leaked into this mask. Targets were
verified absent from known past; user/movie ratings have no duplicates.
Only20 saved candidates are available: no replenishment, no recovery of targets
absent from beam20, and no re-decoding with masked prefixes. After all-past
filtering, 657 control and673 graph lists have fewer than10 valid retained
candidates (of2829 user-seed lists). Last20 filtering has33/42 such lists.
These are diagnostics, **not replacement official metrics** or a measurement
of an exhaustive unseen-catalogue decoder.

## Concrete counterexamples

Ranks below are unfiltered test ranks; outside20 means absent from saved beam.

| User / target | Control ranks, seeds42/101/202 | Graph ranks |
|---|---|---|
| 206 / L.A. Confidential | 3 / 5 / 5 | 10 / outside20 /16 |
| 105 / Alien Resurrection | 1 / 1 /6 | 7 /7 /8 |
| 424 / The Birdcage | 6 /9 /6 | 13 / outside20 / outside20 |
| 241 / Scream2 | 1 /3 /1 | 4 / outside20 /4 |
| 820 / Alien Resurrection | 2 /3 /8 | 9 /16 /10 |
| 614 / The Nutty Professor | 2 /2 /1 | 4 /3 /2 |

For **user206, seed42**, target L.A. Confidential has rating5. Graph top3:
Wag the Dog (unseen), Good Will Hunting (already rated5), Blues Brothers2000
(already rated**1**, in visible history). Its next two recommendations are
also known movies. Removing all known movies moves the graph target from10
to3 and the control from3 to1, so a graph deficit remains. All ratings are
currently treated as interaction events, without a negative-feedback feature;
a rated1 movie is not explicitly labeled undesirable to the model.

For **user105**, graph top1 is The Devil's Advocate, already rated2, while
the target Alien Resurrection is rank7. The target itself also has rating2:
next-rated-item accuracy is not positive-preference prediction.
For **user424**, graph top1 is Mr. Holland's Opus, the immediately preceding
rated movie. These examples show real undesirable repetition, but cannot
prove an actor edge caused the model to favor that particular film.

For **user241**, graph top1 Liar Liar was rated at exactly the same timestamp
as the held-out Scream2. This is an ambiguous event-order example, not clean
evidence of a bad user recommendation. Graph top2 I Know What You Did Last
Summer is known history, rated2. Over all cases graph does **not** increase
same-timestamp alternatives: test top10 occupancy1.05% versus control1.11%.
The split uses strictly distinct timestamp groups; ambiguity remains because
only one target from each final group is evaluated.

## Does actor similarity explain the remaining harm?

Exclude known movies before interpreting actor overlap: otherwise recommending
the same movie trivially shares all its actors with history. Among new wrong
movies promoted into graph top10 versus wrong movies displaced from control,
conditional on graph-harmed test cases:

- Any shared cast member with visible history: **67.5% vs63.4%**.
- Shared cast but no shared top5-cast pair: **35.6% vs30.4%**.
- Shared cast and no target-genre overlap: **30.1% vs28.0%**.

These are means of per-case fractions over nonempty candidate sets, not a
causal estimate or a paired matching experiment. Validation does not repeat
the main contrast: any-cast overlap61.5% vs61.6%, graph cosine .0310 vs.0329.
Test graph cosine rises slightly (.0374 vs.0345), but T5 similarity also rises
slightly (.0407 vs.0400). We therefore have **no validated filter for harmful
actor links**. Removing supporting actors solely on this observation would
be premature; the current result could reflect general context similarity.

Other proposed explanations have no supporting global signal: graph top10
has lower mean log training popularity (4.954 vs4.994), approximately equal
actor hubness (1.2541 vs1.2537), and does not consistently worsen reconstruction
of the three original modalities across all seeds. Target ratings4–5 also
fail to rescue a graph advantage (test .04676 vs.04177; validation .06797 vs
.06213). Do not claim a popularity-bias mechanism or an explicit-preference gain.

## Practical next step

If further evaluation is requested, first implement the **same known-past mask
for both models**, refill candidates with a documented decoder/scoring policy,
and use validation under that policy. This does not require retraining the
existing generators. Preserve the original unmasked results as a separate
protocol. It tests whether repeats obscure graph utility, not whether any
specific graph edge is harmful. A causal graph-edge claim would require a
matched representation/quantizer retraining ablation and additional seeds.
No such GPU work was launched by this audit.

## Evidence and replay

Workspace evidence: `artifacts/movielens100k-actor-full/harm` links to
`/tmp/actor-harm-audit`; `cases.csv.gz` contains all5658 paired split/user/seed
rows, including filtered and original NDCG. JSON/CSV tables preserve all
profiles, rating groups, uncertainty, detailed movie examples and12 prediction
SHA256s. `audit_actor_harm.py` and `actor_harm_uncertainty.py` are replay scripts
for the recorded cache layout; upstream raw numerical outputs and metadata
are included in the ZIP, with a SHA256 manifest. No model weights are needed.

Inputs are the six selected-checkpoint validation and six test prediction
files of `carasik/rhpr-ml100k-actor-full-training v1`, source/catalogue metadata,
fixed actor vectors and official ML100K ratings. Exact943 case IDs, temporal
case hashes and official ratings SHA256 were verified. Original NDCG was
independently reconstructed to1e-12 before any filtering. The conditional user
bootstrap uses20,000 shared-user resamples; it is exploratory, unadjusted for
multiple comparisons, and does not include model/quantizer retraining variance.
