# Shared-codebook TIGER: partial failed experiment

Private carasik/rhpr-ml1m-shared-codebooks/1, executable
f6a299f5bb60c5273745248e3f6ad89e41f49054. Kaggle ERROR. Whole experiment is
incomplete: codebook_init completed30epochs,live_codebook completed29epochs and
failed at the START of epoch30 during catalogue reassignment. No top-level
metrics.json/summary.md/run-manifest.json were emitted. Do not report a completed
joint-training comparison or invent live TEST metrics. No retry authorized.

## Which logs belong to which arm

worker0 = codebook_init = copiedRQcentroids into TIGERembedding table, fixed SIDs.
worker1 = live_codebook = same initial model,liveembedding rows used as centroids,
new SID assignments at each epoch start. The original worker omitted a trial-start
event,so the outer log attached null trial tags. Raw separate worker logs identify
the arms; analysis/labeled-events.jsonl retrospectively labels them without
altering raw outputs. Future runner emits explicit trial-start/complete/failure
with worker+trial and preserves failed candidate codes/books/index diagnostics.
These logging changes are not deployed to another GPU session.

## Ranking results

| Arm | Completed epochs | Best observed VAL NDCG10 | Selected TEST NDCG10 |
|---|---:|---:|---:|
| Historical random-init SUMseed42 control |30|.06665966|.06165244|
| Codebook initialization, fixed SIDs |30|.05971500|.05188044|
| Live codebook with reassignment |29|.02015296|not evaluated|

Both completed models select epoch25. Staticinitialization is -10.42%VAL and
-15.85%TEST relative to the historical control. Same fixedSID/data/seed/architecture,
but one seed is not a replication study. Paired-user bootstrap5000draws for TEST
absolute delta: [-.01539,-.00436],conditional on fitted models;excludestraining
randomness. Staticinitialization fit took1737.10s training and1804.68s including
validation/checkpoints. All8880updates,6VALchecks,0AMPskips. Live29epochs8584updates,
5VALchecks,0AMPskips. Its best observed VAL occurs at epoch15; no best-selected
TEST was run because training failed before the selection routine completed.

Live VAL at epochs5/10/15/20/25:
.01268958/.01847976/.02015296/.01820257/.01981165.

## Index failure and geometry

Median fraction of complete SIDs changed between epochs2–29:95.81%; all2874items
change after the first epoch. Original occupiedcodes255/252/231,largestsemantic
collision2. Epoch29 occupied29/57/63,largestcollision141. Reconstructing the attempted
epoch30 from the saved last checkpoint gives21/46/64 occupiedcodes,244unique
semanticIDs,total2874items;one264itemcollision group exceeds suffixcapacity256.
The exception is a collision safety check,not a CUDA/nonfinite-loss failure.
Increasing suffix capacity would mask the terminal error without restoring index
occupancy or predictive quality.

The learned table is no longer in the frozen encoder's original geometry:
RMScentroidnorm level1 falls7.3202->.6265,level2 .6508->.5463,level3 .1704->.5447.
RQoriginalvectors andlatents64D were zero-lifted to128D to match TIGER;initial
assignments verifiedexactly. The additional64coordinates subsequently learn,
RMSnorms .3479/.3727/.3734. Dropping those learnedcoordinates for an offline
reassignment of the same29epoch checkpoint changes1440items;still only270unique
semanticIDs and20/54/63occupiedcodes,largestcollision254. Thus expansion contributes
to the geometry and exact overflow but does not alone explain the collapse.
This is a post-hoc assignment diagnostic,not retraining or a corrected-method
quality estimate. No causal attribution solely to expansion is justified.

Only NIP cross-entropy trained the livecentroids. There was no simultaneous RQ
reconstruction/commitment objective. NIP gradients optimize input lookup vectors;
they do not differentiate through nearest-neighbor assignments or preserve the
fixed latent-space distances. Updating centroids also changes history/target
labels at each epoch. LR.001 and Adam coupled weightdecay.0001 apply to the table
as to other TIGERparameters;lowerbookLR,zero bookdecay and geometry anchoring were
not separately tested. The evidence supports failure of this specific unconstrained
sharing scheme,not impossibility of all shared/joint tokenizers.

A conservative next design,not launched: keep original codebooks/SIDs fixed and
learn a projection/adapter from code vectors to TIGERinputspace. This preserves
a common semantic base while allowing generator adaptation. Full freezing of
alllookupvectors is a distinct hypothesis and was not tested by the completed
staticinitialization arm,whose vectors were trainable.

## Verification and artifacts

Independent replay verified19predictionfiles:17VAL across control/static/live
and2TEST for completed arms,with matching user/target arrays,per-epochSID snapshots,
NDCG/Recall5/10 and invalid/duplicatecounts. Initialmodel pins match,allstaticSID
snapshots unchanged,lastlivecheckpoint matches epoch29activeSID snapshot. Failed
candidate recomputed from thatcheckpoint andfrozenlatents;no newmodel training
or generator inference. Prediction hashes and checkpoint hash retained.

Replay scripts/research/analyze_shared_codebooks.py; JSON report
 docs/gr-shared-codebooks-results.json. Local plot
/tmp/shared-books-analysis/shared-codebooks.png; evidenceZIP
/tmp/shared-books-evidence.zip includes numericaloutputs,labeledlogs,failedcandidate,
checkpoint,selectedhistoricalreferencefiles andTRAIN/VAL/TESTcasearrays. Fullmodels
for completedarm remain on privateKaggle. NoGPUresults pending or automatic retry.
