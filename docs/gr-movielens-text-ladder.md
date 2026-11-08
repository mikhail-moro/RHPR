# MovieLens: cumulative text fields → Sentence-T5 → RQ-VAE → TIGER

User request, 31 October 2026. Seven cumulative feature configurations on the
same previously accepted 2,875-movie subset (TMDB rich4 + cached Zenodo ViT).
This document records completed preparation and the proposed training contract.
**Stopped by user. Superseded by the MovieLens-100K ladder; do not restart this ML-1M run.**

## Completed preparation

`scripts/prepare_movielens_text_ladder.py` verifies the retained item IDs and
prepares source fields plus seven catalogues in `/tmp/ml-text-prepared`.
The929,776retained source ratings remain unchanged; the approved split is
packaged separately with explicit embargo and excluded-user records.
Four downloaded TMDB snapshot files match the previous audit's SHA256 exactly.
IMDb/TMDB identity agreement is required; the old title/year review flags remain.

| Stage | Added field | Median tokens | p95 | Maximum | >256 | >512 |
|---|---|---:|---:|---:|---:|---:|
|1|Title including year|10|16|36|0|0|
|2|Genres|19|26|42|0|0|
|3|Explicit year|23|30|46|0|0|
|4|Runtime|29|37|53|0|0|
|5|TMDB keywords|54|103|298|1|0|
|6|Cast|135|324|1582|266|22|
|7|Directors|142|330.3|1587|292|24|

Counts use the pinned Sentence-T5 tokenizer with special tokens and no truncation.
All named cast members are retained in original cast ranking; no top-N cap.
People and keywords are deduplicated by source identity, not by their names.
Genres come from ML-1M; year is the ML title's release year. The original title INCLUDING year is
retained at every stage, per the user clarification; stage3 adds an explicit
Year field containing information already present in the title. Runtime,
keywords, cast and directors come from the verified TMDB snapshot. Unknown
runtime is explicit. Texts exclude overview, user tags, reviews, popularity,
ratings, item IDs and user IDs. Keywords mean TMDB keywords, not IMDb/user tags.

## Encoder capacity decision

Sentence-T5's 256-token limit is the released SentenceTransformer wrapper
configuration, not a hard absolute-position table restriction. Keeping256
would truncate10.16%of the complete texts and preferentially remove the last
fields; simply adding directors could become a test of truncation.

Proposed policy: **same frozen Sentence-T5-base weights and max_seq_length2048
for all seven variants**, with raw input lengths saved. All current texts fit.
A real CPU forward on the longest1,587-token text (movie952, original title
including year) succeeded, producing
a finite768D vector through Transformer→Pooling→Dense→Normalize. This verifies
technical execution, not semantic accuracy at lengths beyond the released
wrapper's default. A larger model is unnecessary merely to fit these inputs.
Future encoder comparisons should be separate from this first feature ladder.

## Temporal audit: approved group split

There are **zero duplicate user–movie pairs**, but2,483/6,040users have several
ratings at their last timestamp. Ordinary leave-last-two would put validation
and test at the same time for these users. Source file order is a deterministic
tie-breaker, not evidence of viewing chronology.

Approved split:

- Last distinct timestamp group: choose one deterministic target for test.
- Penultimate distinct timestamp group: choose one target for validation.
- Other ratings in those two heldout groups do not enter training/history.
  Deterministic target choice must not use rating value or model predictions.
- Exclude user5258:15retained ratings, only two distinct timestamps, no preceding
  training group. All other users have at least two training events available.
- For training, each target receives only events with a strictly earlier
  timestamp. Multiple targets in one timestamp group share the same past;
  do not invent within-second order. First timestamp group supplies context
  but no next-item targets without a preceding history.
- Test context can include the selected observed validation item, which has
  a strictly earlier timestamp than test. Other embargoed group items stay out.

This would retain6,039evaluation users;906,679training events and892,574strictly
past-only training targets per epoch. Extra embargoed ratings among these users:
11,004; the separately excluded user's15ratings are not counted twice.
Alternative presented to the user: conventional leave-last-two with source
file order for timestamp ties, explicitly described as an artificial order.
The user selected the timestamp-group option; the conventional option is historical only.

## Intended training after the split is resolved

Seven fits, shared data/order/model seed42 and identical catalogue/targets.
Frozen encoder; native GRID RQ-VAE3×256, latent64, previous3000-update recipe (initialization buffer2875, matching the full smaller catalogue)
and FP32 quantizer. Same GRID TIGER4encoder+4decoder layers,d128/FF1024,
previous FP16 generator, latest20 history items, Adam1e-3/wd1e-4, batch256
(physical128), beam20. Twenty complete epochs each, no automatic extension;
validation only5/10/15/20, selection by validation item NDCG@10, one selected
checkpoint test. Save input hashes, all embeddings/codes/quantizer reports,
checkpoints, curves and predictions. No expensive SID study telemetry.

The existing trainer now accepts an optional example builder; only this run
uses the new strict temporal-case builder. All historical defaults stay intact. Generation/evaluation otherwise keep the previous full-catalogue,
raw-probability, no seen-item-mask contract so the ladder changes feature inputs
only. These settings are a study configuration, not published-score replication.

Fresh quota check:30GPUhours available, none used/reserved. This is substantially
more data per epoch than Beauty. At the old training throughput, seven full
20epoch fits could consume roughly19GPUhours, with large uncertainty; a two-T4
schedule has four fits on its longer worker. Actual first-epoch timing must be
reported. Do not shorten unsuccessful configurations without a new comparison
protocol, and do not count an epoch-boundary time stop as a complete20epoch fit.

## Saved evidence

`/workspace/artifacts/movielens-text-preflight/` retains `preflight.json`,
`long-text-forward-check.json`, verified source hashes, and all parsed fields in
`catalogue-fields.json.gz`. Seven stage texts can be regenerated from those
fields with `field_texts`; their expected SHA256 values are in the preflight.
Full stage JSON and token-length CSV are currently in `/tmp/ml-text-prepared`.
The preparation script and three focused tests cover source-person identity,
feature leakage across stages and the timestamp-tie/duplicate distinction.
The temporal adapter is `rhpr.gr.movie_data`; the two-worker runner is
`rhpr.gr.movie_suite`. Config: `experiments/movielens_text_ladder_2t4.json`.
All seven catalogues and the temporal schedule are in the private input dataset
`carasik/rhpr-ml1m-text-ladder-input`. No poster vectors are consumed in this
text-only ladder; their coverage determines the common future multimodal subset.


## Submission receipt

- [Private Kaggle](https://www.kaggle.com/code/carasik/rhpr-ml1m-text-ladder),version1,
  observed RUNNING. Submitted31October2026at21:43Moscowtime.
- [Actions41230771384](https://github.com/mikhail-moro/RHPR/actions/runs/41230771384)
  succeeded,submit-only,noTelegram. Executablecommit
  `c4d31fb32146376fb65f4e043787d2ecbb62c609`.
- Actual uploaded config and all182embeddedpackagefiles match the committed
  Git blob hashes. Private/GPU/T4/offline/two declared datasets confirmed.
  PackageSHA256`6f7e96b2ee38787d4961fb29139bb7a57c8612598ce80165d13cc77f8c2626f4`.
- All912repositorytests passed(12skipped),plus2targeted movie-suite tests
  including real encoder/RQ/tiny20epochtraining. Longest final text1587tokens
  verified through actual tokenization and CPUforward without truncation.
- Receipts,actual source audit,config,split report,initialization/case hashes
  and test evidence:`/workspace/artifacts/movielens-text-ladder`.

Do not resubmit,extend,retune,add seeds or continuously poll. Wait for user
completion/status message;collect this existing run's outputs when requested.


CI note:the first PR validation job exposed missing analysis/checkpoint dependencies in
its existing install command (SciPy and safetensors imports from historical tests). Add
`sid-analysis` and safetensors to the CI install list. This does not change the submitted Kaggle
runtime or executable;do not resubmit GPU training for the CI fix.
