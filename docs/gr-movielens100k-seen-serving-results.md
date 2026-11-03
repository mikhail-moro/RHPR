# Unseen-item serving and actor-graph popularity audit

2026-11-03. Completed private `carasik/rhpr-ml100k-unseen-serving v1`, source
`8f0b782b387a5f464bdd514fa6ae5d10f735b764`, workflow41649512819.
Six frozen original validation-selected checkpoints, two T4 workers, FP32.
**Zero training updates**, no index changes or checkpoint reselection.
Runner duration76.44seconds; notebook log completed around115seconds including
bootstrap. The workflow submitted only; Kaggle separately verified COMPLETE.

## Serving result

Mean NDCG@10 across the same three fitted pipeline seeds:

| Split/policy | Control | Actor graph .1 |
|---|---:|---:|
| Validation original beam20 |.070774|.065210|
| Validation unmasked beam100 |.070836|.065073|
| Validation eligible/refilled |**.108692**|.103901|
| Test original beam20 |**.048112**|.047215|
| Test unmasked beam100 |.047585|**.047851**|
| Test eligible/refilled |.076682|**.082005**|

Eligible test graph/control difference is+6.94%, validation-4.41%.
Test graph wins seeds42and202, loses101. Fixed-model shared-user bootstrap95
interval for test absolute difference .005323 is[-.00258,+.01324]; includes0,
excludes retraining uncertainty and multiple-testing adjustments. Validation
absolute difference-.004791 has interval[-.01389,+.00434]. This does not
establish a reliable overall graph advantage or equivalence.

| Seed | Validation control | Validation graph | Test control | Test graph |
|---|---:|---:|---:|---:|
|42|.102352|.106693|.075832|.081975|
|101|.113465|.096353|.080996|.074730|
|202|.110259|.108657|.073218|.089310|

Every one of12model/split outputs contains exactly10unique valid **unseen**
items for all943users. Known means the complete training past, plus only the
selected validation item for test. Embargoed events are excluded. Original
20-event encoder inputs and targets are unchanged. All target-not-seen checks
passed. Original32-user beam20anchors reproduced exact paths; scores matched.
All learned-weight hashes were unchanged. All reported original and wider-beam
NDCG values were independently reconstructed from saved predictions to1e-12.

The common initial beam is100. Of11316user/model/split cases,13needed beam256
(7validation/6test); **none needed the exact-catalogue fallback**. The saved
raw beam100, refill256 and eligible top10allow independent recomputation.
No local child-probability renormalization or common-prefix blacklisting.
Expanded beam alone barely changes aggregate unmasked quality; the large
increase comes from eligibility filtering and filling the vacated slots.
This is a new serving policy on the old checkpoints. Retain the original
unmasked results separately; no assertion of globally exact catalogue ranking.
Validation/test are previously examined holdouts with strongly shared history.

## Are popular actors or strong edges causing graph promotion?

Reconstructed the frozen cast graph:15998people,1273movies,475238undirected
nonzero edges. Edge weight is **sum of1/(cast_size-1)** over shared films,
not raw coappearance counts. Excluding self-loops, weighted degree equals
the number of non-singleton-cast films; only4films have singleton casts.
Consequently catalogue frequency and weighted degree are nearly the same
structural property here; they are not independent tests.

Popularity here means catalogue appearances or training-event exposure of an
actor's films. We have **no external celebrity/popularity measurement**.
Movie features average log1p person frequencies/neighbors/exposure across cast.
Strong repeated-coacting edges subtract the movie's own clique contribution,
so they measure additional joint appearances in other catalogue movies.

Primary outcome: per-movie change in mean top10recommendation count, graph
minus control over three fitted seeds. Spearman correlations across1273movies:

| Movie-level predictor | Validation rho | Test rho | Test partial rho |
|---|---:|---:|---:|
| Mean actor catalogue frequency |.022|.024|.033|
| Mean actor unique-neighbor count |.000|.019|.014|
| Actor training-interaction exposure |-.012|.006|.017|
| Additional repeated-coacting edge weight |-.001|.031|.031|
| Fraction of repeatedly coacting pairs |-.002|.031|.031|

Partial Spearman residualizes rank-transformed predictors/outcome on movie
training popularity and cast size. These are tiny descriptive associations,
not evidence for a substantial monotonic actor-hub preference. Individual-seed
signs are not uniform. For increased **repeats** specifically, test partial rho
is.008with actor frequency and.003with actor interaction exposure. Stronger
repeated-coacting weights give.022. No significance/causality claim is made.

Recommendation-weighted mean log movie training frequency is **lower** with
graph on both splits: test4.954vs4.994, validation4.943vs4.989. Mean log actor
catalogue frequency is nearly equal: test1.2541vs1.2537. Actor raw SGNS vector
norm correlates negatively with catalogue frequency (rho=-.219), which also
fails to support a simple larger-norm-for-hubs explanation. Angular or nonlinear
embedding bias is not ruled out by these tests.

## Context-conditioned and quality checks

A global correlation could hide a context-specific effect. We therefore
constructed degree-normalized actor affinity between each candidate film and
the visible20-film history, using mean-pooled normalized graph edges. Compare
new incorrect movies promoted into graph top10 with new incorrect movies
removed from control top10; **exclude known movies and the target** to avoid
trivial self-overlap or using target relevance as a serving feature.

After residualizing on user identity, movie train popularity and cast size,
pooled descriptive Pearson correlations of graph promotion with this affinity
are[-.019,+.031]on validation and[+.001,+.023]on test across the three seeds.
Paired promotion/demotion affinity signs are mixed on validation; around half
of users have a positive difference. There is no strong stable context-edge
mechanism established. These are outcome-selected candidates, not randomized
matched-item comparisons; shared films/actors create dependence.

We also checked whether target graph features predict the **per-user NDCG
difference**, averaged over seeds. Test partial Spearman is-.047for actor
catalogue frequency, -.031for actor interaction exposure, +.039for repeated
edge weight. Corresponding validation values are-.012,-.032,-.007. Thus neither
an actor-popularity quality gain nor a stable strong-edge gain is supported.
All quality feature terciles and seed rows are retained, not just favorable
examples. These analyses use the original unmasked outputs; their purpose is
to explain the earlier observed differences, not select a new graph or gate.

## Interpretation and artifacts

The known-item policy removes a clear serving limitation for both models.
However **repetition is not by itself proof of PopBias**, and current measures
do not explain the graph/control difference via famous actors or strong ties.
Specific StarWars actor effects reported earlier overlap through the same
films and must not be treated as independent actor-level evidence.
No hub removal, graph reweighting or additional fitting is justified by this
analysis alone. Graph representation/index changes versus generator adaptation
remain unseparated in the original six-pipeline comparison.

`docs/gr-movielens100k-seen-serving-summary.json` preserves aggregate results,
per-seed outcomes, independent verification and all graph correlations.
Workspace outputs live under `artifacts/movielens100k-unseen-serving` and
`artifacts/movielens100k-actor-graph-bias` (temporary-cache links). The complete
prediction/checkpoint lineage remains durable on the private source Kaggle
runs. Temporary archives may disappear between environments; this document,
summary and replay sources are committed to GitHub.

Replay `scripts/analyze_actor_graph_bias.py` with `PYTHONPATH=src:.` and flags
`--predictions`(downloaded full-training outputs), `--sequences`, `--people`
(movie-people.json.gz), `--catalogue`(catalogue-fields.json.gz), `--vectors`
(frozen actor dataset people-vectors.npz), `--output`. It produces all movies,
correlations, context probes, quality groups and seed-level exposure profiles.
No training or model calls. `rhpr.gr.movie_seen_serving` implements the separate
frozen-predictor inference policy; config SHA256pins every parent artifact.
Repository952tests passed(90skipped), native tiny-GRID raw/exact-score parity
passed, and actual CUDA anchors/weights/output-invariants passed for all6models.
