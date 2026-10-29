# MovieLens rich-semantics research: data audit and initial design

Started2026-10-29. Previous SID-swap/teacher-age work is archived in its existing
reports; do not resume it automatically. New branch:`codex/movielens-rich-semantics`.
This phase downloads/audits public data and defines a comparison. No GPU training,
Kaggle submission, new embedding generation or production-data access occurred.

## Practical question

Given the same item content, does adding plot/keywords/cast/director improve
generative retrieval more than semantic dense retrieval? Measure both the
absolute quality and the incremental effect of richer content. This is a
decision about whether a generative pipeline is useful for a richly described
movie catalogue, not an assumption that SID indexing is intrinsically superior.

The user's motivating application is item-to-item recommendations. Their proposed
TIGER/SASRec comparison naturally targets personalized held-out-item retrieval.
These are different tasks; an optional clarification is pending. The proposal
below assumes the latter for the initial model comparison. Do not present a
next-item result as proof of item-to-item usefulness.

## Closest literature: the broad question already has related evidence

- [LIGER:Unifying Generative and Dense Retrieval for Sequential Recommendation](https://arxiv.org/html/2411.18814v2),
  Meta, compares implemented dense/generative methods with the same item content.
  Its dense implementation adds item-content representations both to history
  inputs and candidate representations. The reported dense method outperforms
  the reported TIGER implementation on in-set and cold-start targets on their
  small academic benchmarks. This is highly relevant prior evidence, not a
  universal impossibility result for GR. It does not by itself settle the
  incremental effect of plot/cast/keywords/posters in MovieLens.
- [MMGRec](https://arxiv.org/html/2404.16555v1) already studies multimodal generative
  recommendation. It uses a different graph quantizer, CF/multimodal fusion and
  non-sequential formulation, so its gains do not isolate our proposed factor.
- [Original TIGER section4.3](https://arxiv.org/html/2305.05065v3) has a specific
  cold-start scheme:5%test items removed from training, semantic-prefix expansion
  to cold items, and an epsilon cap on their share of retrieved candidates. The
  comparison is against semantic KNN. This is not proof that ordinary exact-SID
  generation beats a strong content-aware dense retriever on cold items. Our
  existing exact-SID adapter does not implement that special scheme.

This is a targeted literature check, not an exhaustive novelty search. Do not
claim no prior work, or claim a publishable gap before a broader review.

## Audited data, not just advertised counts

Official [MovieLens1M](https://grouplens.org/datasets/movielens/1m/):6040users,
1000209ratings,3883catalogue rows,3706movies with at least one rating. Ratings
span2000-04-25 through2003-02-28. Median history96ratings; no interaction filtering.

Downloaded only selected metadata files from
[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset):
`links.csv`,`movies_metadata.csv`,`keywords.csv`,`credits.csv`.
This is an older TMDB snapshot (movies through2017). No API key is needed to
read the snapshot. IMDb/TMDB IDs are linked through the supplied mapping;
metadata is accepted only where the record's IMDb ID agrees with the link.
No fuzzy title joins. Title/year aliases are checked separately and flagged.

Coverage denominator below is the3706 rated movies, not all3883 catalogue rows:

|Field|Nonempty items|Coverage|Use|
|---|---:|---:|---|
|Overview|3648|98.43%|Semantic plot representation|
|Keywords|3256|87.86%|Themes/topics beyond coarse genre|
|Cast|3617|97.60%|Actors; ordered top cast subset for fixed text budget|
|Director|3644|98.33%|Creative/style identity|
|Poster path|3649|98.46%|A historical path only; not verified image availability|

3658rated movies have a metadata record under agreeing external IDs;3237 have
all four rich-text fields nonempty. Keep the full rated catalogue and missingness
masks/fallback title/genres, rather than dropping incomplete films and changing
the interaction benchmark between content conditions. Overview/cast/director
cover about99.6–99.7%of rating events; keywords97.98%. Completeness is therefore
correlated with popularity and must be logged in quality slices.

Keywords here are TMDB keywords, not IMDb plot keywords:median5,p90=16,max81
among linked rated movies. Median overview52words,median cast14names. This
snapshot does not supply100IMDb tags per film. The official
[IMDb public TSV schema](https://data.imdb.com/non-commercial-datasets/) supplies
basic titles/year/runtime/genres,principals,crew and names, but no plots,posters
or plot-keyword dump. It is explicitly a non-commercial data source. No IMDb
website scraping or multi-GB TSV download was performed.

Data-quality flags:3malformed metadata rows skipped;29duplicated TMDB IDs in
the full snapshot.10ML catalogue links disagree between IMDb and TMDB records;
those enrichments are withheld.214additional title/year mismatches require
review (including aliases/country release years,not necessarily wrong joins).
These are preliminary coverage figures,not a certified training join.

## Images and richer sources

Historical poster paths are not equivalent to cached images. A fixed-seed sample
of20paths yielded1image and19HTTP404s from TMDB's public image CDN. This small
sample is diagnostic,not an estimate of full-catalogue coverage. Do not launch
a poster experiment against these paths without refreshing/validating images.

Ready research sources worth reusing:

- [See the Movie,Hear the Song,Read the Book](https://zenodo.org/records/15403972),
  University of Bari:ML1M plots3667,poster coverage3197,plus cached image features
  from ResNet152/VGG/ViT and text features from MiniLM/MPNET. Those counts are the
  authors' claims; local per-ID/vector alignment has not been audited. The Zenodo
  API file inventory/checksums were retrieved;ML1M-only NPY archive is91.10MB,
  source text24.32MB,mapping0.70MB. No need to download the entire2GB multi-domain
  release. Raw coverage and row mapping must be checked before using embeddings.
- [Weixin Chen's multimodal ML1M](https://weixinchen.com/dataset_ml1m_multimodal)
  provides visual/text/audio arrays for3706items according to its author page.
  These are a second candidate, not yet a locally verified dataset. Visual
  trailer features are not automatically equivalent to poster features.

GroupLens Tag Genome is a possible later signal, but its crowd tags/relevance
scores incorporate behavioral information. Treat it as a separate content/CF
condition,not freely available metadata for a genuinely new item.

## A small comparison that can answer the question

Use two shared text conditions:

- Basic:title,year,MovieLens genres.
- Rich:the same plus overview,TMDB keywords,cast,director.

Same frozen language encoder,token budget/truncation policy and cached vectors
for both retrieval families. Protect field budgets so title/overview cannot
silently truncate away every added actor/keyword. No popularity,vote counts,
vote averages,revenue or collection membership inferred from test interactions.

Five configurations across the user's three model families:

|Family|Basic text|Rich text|
|---|---|---|
|SASRec ID-only|One shared control|Same control|
|SASRec + content|Fit|Fit|
|TIGER-style semantic-ID retrieval|Fit|Fit|

Compare `rich-basic` within each family,then compare those gains as well as
absolute quality. A single rich-TIGER vs ID-SASRec comparison cannot establish
that GR uses semantic information better. Existing dense SASRec has no content
path yet; add frozen-content projection to BOTH item inputs and candidate
representations (for exampleID embedding + projection(content)). Simply
initializing an embedding table and discarding the content path is weaker.

The existing native GRID generator/quantizer adapters and SASRec backbone are
reusable. The Beauty-specific ingestion and historical Yambda training protocols
must be replaced by one shared MovieLens case manifest. Pin histories,targets,
train/validation/test IDs,catalogue,seen-item policy and metrics. Do not silently
retain TIGER user-hash embeddings if SASRec has no user-ID input. Report model
capacity and training cost; comparable tuning budgets are more meaningful than
forcing one common learning rate across architectures.

At3706candidates exact dense ranking is cheap. Use full-catalogue NDCG/Recall,
not sampled100-negative evaluation. Warm and rare-item slices can be in the
first protocol; truly unseen items require a separately defined cold-start
split and indexing/decoding contract. Rare is not cold. A new content-aware
dense item's ID residual must be disabled/zero if it had no training evidence.
Current exact-SID TIGER is not automatically an inductive cold-start system.

Start with text enrichment; add one image condition once its cached vectors
and masks are verified. Cached multimodal features must be provided equally
to both retrieval families. Never send a poster URL string into Sentence-T5 and
call that a visual representation. No quantizer seed sweeps or new tree methods
are needed for the first practical comparison.

## One protocol issue to resolve before spending GPU quota

ML1M contains rating-submission times,not viewing times.774028ratings(77.3866%)
belong to a user/second group with multiple ratings. Among adjacent per-user
events sorted by time,53.2149%have the same timestamp.2544/6040users' last two
ratings have the same second. A naive last-two split cuts such a group for many
users and a movie-ID tie-break invents an order.

Keep tied timestamp groups together across splits. Decide whether the evaluation
is held-out preference retrieval or next rating-group prediction; do not call
either a verified next-viewing task. Multiple targets per held-out group can be
averaged per user. Training target/context rules must also handle ties without
implying observed order within a group.6040users have at least2groups;one has
only2groups,so a strict3-way temporal group split cannot retain everyone without
an explicit exception. This choice and task scope remain open; no training run
was queued while they were unresolved.

## Reproduction and artifacts

Script:`scripts/audit_movielens_semantics.py`;source archives and selected CSVs
under`/tmp/rhpr-movie-audit`. Audit aggregates,source hashes,join flags,compressed
per-film audit and poster probe results under`/workspace/artifacts/movielens-semantics`.
Raw data were not published to GitHub. A snapshot of aggregate results and source
hashes is stored beside this report as`gr-movielens-rich-semantics-audit.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/rhpr-check/bin/python \
  scripts/audit_movielens_semantics.py --input /tmp/rhpr-movie-audit
```
