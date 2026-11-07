# Where the actor graph helps: exploratory subgroup audit

2026-11-03. Existing predictions only; no GPU inference/training, new
checkpoint selection, fitted router or model changes. Compared the latest
full3seed actor-graph.1 versus no-people models, and checked the prior
30epoch screen2seed models as a robustness reference. All use the same943
users/targets in each split; these are not independent holdouts.

## Main finding

No tested group has a positive effect on all3full-training seeds on BOTH
validation and test. There are descriptive leads, but none supplies a
validated rule for when to use graph information. All104group rows and32
complement contrasts are retained, including null/unfavorable results.

## Director overlap: interesting but split-specific

A target shares a director with at least one visible last20history film:

| Full-training subset | n | No people NDCG10 | Actor graph .1 | Change |
|---|---:|---:|---:|---:|
| Validation |54|.06204|.05596|-9.8%|
| Test |57|.08034|.09906|+23.3%|

Test deltas by seed42/101/202: +.000245,+.032857,+.023060. Thus all3are
positive, but42is approximately tied. Across171fitted user-seed cases,
top10hits are25control versus33graph. The conditional user95%CI for the
subgroup difference[-.01804,+.05530] includes0. The group-versus-complement
interactionCI[-.01721,+.05693] also includes0; the subgroup's positive point
estimate does not establish a distinct treatment effect.

The aggregate test gain survives removal of any one target film. However,
removing Batman & Robin reverses the very small seed42gain. Largest positive
contributions also include Jackie Brown, Back to the Future and The Abyss.
This is not simply one hit, but remains a small, post-hoc selected group.

On the older30epoch screen the test-group mean also favors graph(.06657→
.09161,+37.6%), but only1of2seeds improves. Validation has a negative mean
there too(-4.0%,one seed each direction). These are repeated fits on the same
57test cases, not independent evidence across datasets or users.

The trained graph here contains only actor-role edges. Director overlap
could be a proxy for recurring casts, franchises or related film content;
this analysis does not establish that interpretation or a director mechanism.
The target is needed to compute overlap, so this is not a serving-time gate.

## Rare targets: a very small count, not established tail benefit

Rare means <=20actual supervised next-item positions in training, not total
ratings and not a globally unseen item. Full test has84such targets:
NDCG10 .001709→.003897. The apparent+128% is only1→3top10hits across252
fitted user-seed cases. Positive2/3seeds; conditionalCI includes0. Raw examples
include Home Alone3, The Believers and Robocop3. Removing any one target
keeps the mean sign positive, but the absolute effect and count are tiny.

Full val has76cases,.003017→.006459,2→4hits across228fitted cases. The old
screen's validation mean instead declines30.6%; screen test has1→2hits.
No reliable rare-item/cold-start improvement is established.

## Input-only history signals did not yield a stable rule

- At least100available past events: fullval .03656→.04093(+12.0%,3/3),
  fulltest .02871→.02794(-2.7%,2/3). The model input remains last20events;
  this variable measures user activity, not a longer supplied context.
- High mean pairwise actor-vector cosine within history: fulltest
  .05267→.04783(-9.2%,negative3/3); fullval negative2/3. The older screen
  shows the opposite, +18.7%val/+10.6%test and positive2/2 on both. This
  illustrates sensitivity across fitted indices/training regimes rather
  than a transferable history heuristic.
- Repeated actors within history, target actor support across catalogue
  films, cast size and graph/text top20-neighbor agreement do not show a
  stable positive fullval/test pattern. Direct target/history shared cast
  gives fullval .07982→.07050 and test .05257→.05087; signs are mixed on test.
  Restricting overlap to the top5credited actors does not resolve this.

## Secondary confidence probe, after the metadata audit

Per baseline model, define low confidence as a top1-minus-top2beam score gap
at or below its validation median. No labels optimize the threshold; this
is a finite-beam score diagnostic, not calibrated catalogue uncertainty.
Group membership differs by seed.

Substituting graph predictions in the low-gap group gives fulltest+6.5%,
but fullval-12.3%(negative3/3). Substituting another no-people seed instead
has a mean test effect+10.3%. This latter number averages individual seed
substitutions; it is not a score-merged ensemble or an oracle. Conditional
intervals include0. This check does not establish a graph-specific effect.

A hypothetical fixed-median gate uses graph only in low-gap cases. Its
meanval is .06751 versus .07077control, negative3/3; meantest .04947 versus
.04811. Such a rule fails validation and must not be selected from test.
No router was fitted, deployed or recommended from these results.

## Stable individual cases

Only6test users strictly improve on all3full seeds, and6strictly worsen;
72have effects of both signs across seeds. Examples (ranks42/101/202;
"outside20" means the target was absent from the retained beam):

- User85, As Good As It Gets: outside20/outside20/outside20→3/1/1;
  no directly shared actor with the last20history films.
- User105, Alien: Resurrection:1/1/6→7/7/8 despite shared actor(s).
- Same validation film Liar Liar improves for user143(12/5/2→6/2/1),
  worsens for user155(5/1/2→16/13/5), on all3seeds in each case.

These are post-hoc examples, not a causal explanation. They show that neither
shared actors nor a film's fixed metadata alone reliably describes the effect.
Complete consistent-case lists and leave-one-target-out results are saved.

## Definitions and limitations

Primary descriptive families: actor/top5/director target overlap; user-history
actor pair overlap, graph-vector coherence and graph/text top20overlap;
cast-only catalogue node support; training target popularity(<=20,21–100,>100);
cast size(<=10,11–30,>30); available past events(<50,50–99,>=100);
untrained graph mean-similarity target rank(top100 versus outside).

History cuts use validation feature medians, fixed before reading model
outcomes and applied to test: actor pair-overlap .05714, graph cosine .02785,
graph/text top20overlap .15. Support cut is catalogue median .58824. This
analysis is exploratory, not prospectively preregistered; feature families
were chosen after earlier research. Target-dependent diagnostics are labeled
separately from available-before-target history features.

All20selected prediction files were decoded/rescored and matched recorded
NDCG, exact943users/targets, catalogue IDs and reconstructed case digests.
Frozen feature hashes verified. Shared-user bootstrap intervals condition on
these fits, exclude model/quantizer/runtime uncertainty and are unadjusted for
multiple subgroup comparisons. Validation was used to select checkpoints;
test was already examined in research. The screen also helped select the
coefficient, so repeated signs there are not independent confirmation.

Nothing here resolves the previously observed AE/RQindex repeat variation.
The data currently support several weak leads and substantial fitted-model
sensitivity, without a reliable positive subgroup or validated graph router.
No additional compute is authorized by this analysis. All949repository tests
pass(90skipped); no model code changed.

## Reproducibility

Full outputs/replay: /tmp/actor-full-subgroups. groups.csv contains all104
rows; interactions.csv all32contrasts; cases.csv.gz the individual ranks and
features. confidence-probe.py/json reproduces the secondary probe. The report
and scripts are preserved in the repo. Standalone bundle:
/tmp/movielens100k-actor-subgroups-evidence.zip. Restore the named raw
/tmp directories and metadata paths listed in the bundle README for replay.
