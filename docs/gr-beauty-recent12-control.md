# Beauty: one matched Recent12 truncation control

User authorization2026-11-17: run one Beauty control on one seed. Exactly ONE
fresh Recent12 generator,seed42;compare with completed Beauty/1 Recent20,
Recent3 and RandomTail3. No additional fits,tokenizer variants or seed search.

## Protocol

Config`experiments/beauty_recent12_control.json`,existing`rhpr.gr.beauty_pairs`
runner. Full original GRID/P5Amazon2014Beauty22363users/12101items,
131413TRAINnext-item prefixes,22363VALandTESTtargets. Latest12raw items,
mandatory3+1SID,USERprefix:49fixed encoder positions. Targets/case order and
all TRAINtarget positions remain identical to Beauty/1. Controls do not merge.

Same Sentence-T5-base/full supplied metadata,max256,normalized-residual native
GRID RQ-KMeans3x256,seed42,10passes/level,batch2048,init3072,torch_cdist.
Recompute from existing original resource datasets ONLY;no cachedvector export,
parent kernels or new datasets. Save vectors/centroids/SIDs;require exact
original SIDfileSHA before training. Skip pair embedding and pair quantizer
entirely:empty pairlookup artifacts are explicitly unused. Retain original
unused pair-input parameter table to match initializer and parameter count.
Check original rawTRAIN/VAL/TEST case hashes and initializer before fitting.

Same wideT5 4+4/d128/FF1024/dkv64/6heads/dropout.1/USER2000/noSEP/SDPA,
5108992parameters. FP16compiled512/512,Adam.003/wd1e-4/clip1,warmup2/cosine
60to.001,min40/max120,VALevery5,EMAalpha.5/delta.0005/patience20epochs.
RawVALNDCG10selects checkpoint;TESTonce after selection,FP32beam20/batch64,
all12101items,no seenfilter/samplednegatives. Save all VALpredictions/losses,
bestweights,lastfull optimizer/scaler/scheduler/RNGstate,curves,timings,FLOPs.

Exactlyone isolated worker oncuda:0 in existing KaggleT4machine shape;no
second fit to fill the other device. Existing launcher supports one explicit
assignment. Parent still verifies twoT4hardware as existing recipe requires.
Fitlimit10000s/server21600s,no automatic fallback/retry. Estimate40-70minutes
including preparation and validation,not a measured promise. GPUquota26.83h
before submission. Private/offline/two existing resource datasets only.

CPUpreflight uses actual SAVED original index locally,without transmitting it.
All raw cases/targets match,real model finite forward/backward,initializer
matches,49positions/257705640supportedFMA2FLOPs. This equals RandomTail3
fixed maximum inference width;TRAINsampling and actual lengths differ. It is
not a beam-serving latency or equal-training-time claim. CPUreproducer:
`scripts/research/preflight_beauty_recent12.py`.

## Original TIGER comparison

Primary source:https://arxiv.org/html/2305.05065v3,Section4andAppendixC.
AppendixCstates a20-item history cap;no history-length sweep or explicit
justification is reported there. Do not attribute later reproduction findings
to the original paper. Explicit generator dimensions match ours:4+4,d128,
FF1024,6heads/dkv64,dropout.1. Paper reports~13Mparameters,our actual attached
model5.109M;dimension agreement does not resolve that discrepancy. Quantizer,
training budget/optimizer/schedule/framework differ,so no exact reproduction
claim. Paper trains Beauty200ksteps;our Recent20completed~19.3kupdates,
selected at55epochs/~14.1kupdates (batch512versus original256).

## Existing Recent3 versus Recent20: length analysis

All22363TESThistories contain>3events;context buckets are actual latest20input
lengths,not full user histories. One fixed index/generator seed only:

| TESTcontext | Users | Recent3NDCG10 | Recent20NDCG10 |
|---|---:|---:|---:|
|4-6|14063|.0262622|.0292371|
|7-10|4753|.0297020|.0317430|
|11-20|3547|.0450880|.0416347|
|All|22363|.0299793|.0317361|

Recent20global+5.86%relative toRecent3 (equivalentlyRecent3-5.54%toRecent20).
Longest group reverses direction onVALtoo:Recent3.0664093vsRecent20.0606317,
2989users. No evidence that global gain is confined to longest histories.
TESTonly876users get higherNDCGwithRecent20,789lower,20698equal;equalincludes
many both-miss cases and does not establish identical predictions. Separate
training means even identical<=3VALinputs can differ:7162users,.0412108vs
.0397581. These are not causal per-example effects of adding old events.
No proof of ignored history,no length router,one seed/post-hoc grouping.
Local audit:`/workspace/artifacts/beauty-recent3-vs20-lengths.json`.

Full regression:1169tests total,152optional skips,0failures,124.075seconds.
Private/offline bundle built successfully;CPUpreflight passed.
