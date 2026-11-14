# Kindle: long-reader cohort and Recent20/32/64 screen

## Dataset definition

The source is the pinned Amazon Reviews2023 official5core Kindle Store rating-only
file and raw Kindle metadata. This is a static transductive benchmark construction,
not a historical catalogue-availability or population-representative evaluation.

The selection is defined without reference to model scores or heldout warmth:

1. Retain rating >=4 events. Candidate users have >=66 such events in the full
   catalogue. Sort source user IDs lexically, permute with NumPy default_rng(42),
   and take the nearest whole-user prefix to800000 events. Preserve whole histories
   at this sampling stage:4788 users,799952 events,258507 books.
2. A book must have a nonblank title in the pinned metadata.
3. On the retained induced graph, each user has >=66 events and each book has
   >=5 DISTINCT users. Repeated events by one user do not increase book support.
4. Take the unique inclusion-wise maximal subgraph satisfying2–3, not an arbitrary
   subset that happens to satisfy the two degree constraints. Iterative pruning
   computes this fixed mathematical definition. A single filtering pass generally
   fails because removing books changes user degrees and vice versa.

Implementation may compute the maximal core before joining titles and then prune
again; by monotonicity this equals taking the maximal core on title-present data.
Every one of16177 retained books has title metadata, so the join removed0 events.
Final:958 users,16177 books,144011 events,141137 training examples. Whole retained
history lengths min66,median115,max1087. Filtering books removes events from the
original full-catalogue histories; no claim of contiguous original histories.
No arbitrary top-K popularity catalogue cap is applied.

Sort events by timestamp then parent_asin. Per user: last=test, penultimate=VAL,
all preceding noninitial positions=train next-item targets. Every VAL user has
>=64 past items; TEST includes the observed VAL event. Boundaries have0 tied
VAL/TEST timestamps. Warmth is measured after splitting, never an exclusion rule:
11/958 VAL targets(1.15%) and39/958 TEST targets(4.07%) are absent from train.
All cold targets remain. Full packed histories/timestamps and64-item cases are
retained in the private input for future windows and independent reconstruction.

Source digests, exact prepared counts and per-file digests are in
`gr-kindle-long-cohort-preparation.json` and the experiment configuration.
Preparation: `PYTHONPATH=src python3 scripts/prepare_kindle_long_cohort.py ROOT OUT`.
The script verifies both source files before deriving this cohort.

## Nine paired generator fits

- Windows20/32/64, pipeline seeds42/101/202. Each seed independently fits normalized
  residual GRID RQ-KMeans3x256 on the same static catalogue of Sentence-T5 vectors
  (title+categories). Export a unique collision suffix: every book has3+1 tokens.
  The same frozen index is used for all three windows within seed. Tokenizer and
  generator seeds are coupled, not a crossed variance decomposition.
- Mini-TIGER4encoder+4decoder,d_model64,d_ff512,d_kv32,6heads,dropout.1,userbins2000,
  native SDPA. Retain the previously attached unused pair-input embedding table
  so the model construction remains consistent; there is NO pair compression.
- Within seed all windows have identical parameter initializer, user/target arrays,
  number/order of training examples and epoch shuffle. Only visible context changes.
  Natural early prefixes remain; no supervision is removed for shorter histories.
- FP16 train,512physical/effective batch,Adam peakLR.003,wd.0001,clip1. Warmup2nominal
  epochs then cosine to.001 at60. Scheduler advances only on applied updates.
  torch.compile(mode=reduce-overhead,fullgraph=False,dynamic=False), training only.
- VAL every5epochs,FP32 free beam20,batch64 for ALL windows. No seen-item filtering.
  Save per-case predictions,NDCG/Recall5/10,teacher-forced VAL loss and complete curves.
  EMA(alpha.5) with absolute min_delta.0005 controls patience15epochs,min40,max60.
  RAW best VAL NDCG@10 selects the checkpoint; EMA is not a weight average.
  Epoch cap means censoring, not demonstrated convergence. A trial time-budget exit
  is recorded as incomplete, not silently accepted as an early-stopping result.
- Best and last full resumable checkpoints saved. TEST is closed during screening;
  only data integrity/warmth audit reads heldout labels. No TEST model metrics.
- TwoT4 GPUs, one fit per GPU. Longest fits scheduled first with balanced queues;
  reset compiler caches between fits. Private/offline kernel, no notifications.

## Runtime and FLOP accounting

Per fit, record full-shape B1 CPU math-SDPA profiler-supported forward FLOPs
(teacher-forced4-code forward with output heads,FMA=2,not total beam/backward work).
Record total parameters, first-epoch train time including lazy compilation,
median warmed epoch train time, cumulative train/VAL/wall time, CUDA peak allocated
and reserved bytes, optimizer/AMP skip counts. Actual epochs determine observed
compute cost; theoretical FLOPs must not be presented as measured latency.

Use the same users/targets for all windows and pair results within seed. Evaluate
quality gains against additional compute, including the training curves rather
than just selected maxima. This screen does not establish a capacity bottleneck
or convergence and does not select a compression model. Other windows need a
separate decision after these results.
