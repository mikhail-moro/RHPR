# Frequency-prioritized compression and a stronger compression control

Authorized 2026-11-15: protect rare context items where possible, quantify lost
merges/FLOPs, and try compressing more than the older half. Three fresh seed42
fits, on two isolated T4 workers; no other fits/seeds or notifications. Private
kernel `carasik/rhpr-kindle-frequency`. This is an exploratory test of a
hypothesis, not a claim that rare contextual items caused earlier rare-target
errors. Existing TEST results have already been examined.

## TRAIN-only frequency and the cost of a hard prohibition

Use the number of distinct users with the item in their original TRAIN events,
including the first event and excluding val/test, before prefix expansion.
An item is rare if this count is <=5 (the lower catalog quartile, including ties).
This is 4327/16177=26.75% of the catalog. The threshold is fixed without model
quality selection. A new item with zero observed users is rare. Production would
use prior-only counts, optionally updated periodically; no assumption that future
frequencies are known. This experiment does not test robustness to distribution
shift or strict globally chronological splitting.

A strict no-rare-pair rule admits the following maximum adjacent matchings on
VAL. Exact greedy path matching computes the maximum; no event is deleted.
The per-query column uses each example's native length; the padded column uses
the minimum merge count per original-order batch64, not an optimistic average.

| Eligible prefix | Mean pairs | Mean native B1 GFLOPs | Padded batch64 GFLOPs/request |
|---|---:|---:|---:|
| 10 / 20 | 3.605 | 0.3493 | 0.4140 |
| 12 / 20 | 4.338 | 0.3339 | 0.4125 |
| 14 / 20 | 5.070 | 0.3185 | 0.4010 |
| 16 / 20 | 5.756 | 0.3042 | 0.3923 |
| 18 / 20 | 6.426 | 0.2904 | 0.3866 |
| 20 / 20 | 7.068 | 0.2772 | 0.3823 |

Original Pair20/Random20 inference has5pairs,61positions,.319748392GF.
Hard protection in old10 loses1.395pairs (27.9%) and raises native forward cost
by9.25%; batch padding raises it by29.5%. Thus a small NDCG increase alone would
not establish a better compute/quality tradeoff. Bucketing could recover some
padding cost but changes execution and is not included in this pilot.

## Three predeclared fits

All keep the latest20raw events, preserve chronological order and every event,
and keep the latest4events single. Every single AND pair uses complete3+1SID;
targets are always single-item SID. No future target affects pair placement.

| Fit | Inference placement | Pairs | Encoder positions incl.USER | B1 forward GFLOPs |
|---|---|---:|---:|---:|
| `frequency20_budget5_42` | Minimum rare-item exposure in first16 | 5 | 61 | .319748392 |
| `random20_budget5_42` | Uniform size5 matching in first16 | 5 | 61 | .319748392 |
| `random20_budget8_42` | Eight adjacent pairs in first16 | 8 | 49 | .257705488 |

The first two are exactly matched in pair count, padding and forward FLOPs for
**every** training/evaluation example. The objective is the number of rare
items included in pairs; solve an exact suffix dynamic program, then uniformly
sample among its minimizing matchings. If five pairs cannot preserve every rare
item, minimize the violation while retaining the fixed compute budget. This is
soft priority, not an absolute prohibition. On VAL all rare items stay single
in77.66%of cases; the minimum is397merged rare occurrences, versus1779under the
fixed uniform evaluation matching (77.7%fewer). TEST geometry is also retained,
but no downstream TEST quality was used to choose the rule or checkpoint.

The third fit explicitly tests *more merges*, rather than just a wider eligible
region:16old items->8summaries +4recent singles. It saves39.66%forward FLOPs
against Recent20 and19.40%against the five-pair inference. A perfect matching
of all16positions is unique, so a separate frequency-budget8arm would be an
identical duplicate and is not trained.

TRAIN uses the same constant stochastic strategy as Random20:
M~Binomial(trial_budget,.5)per optimizer batch, then independent matching per
history. For short prefixes, a shared uniform M-of-budget virtual-slot subset
is restricted to the available min(budget,floor(floor(4n/5)/2))slots. This yields
the proper Binomial(available_capacity,.5), not clipping. The two budget5arms
share the exact same count schedule, shuffle and initial weights; only pair
placement differs. They use six native lengths80/76/.../60, while budget8uses
nine80/76/.../48. No full-window padding is added for full TRAIN examples.

VAL/TEST use a fixed seed8141, no epoch or target input. Budget8evaluation is
deterministic and unique. Static input arrays, pair-start masks and per-example
merged-rare counts are saved for both splits. Epoch logs contain exposure
counts, odd-start counts, padding, input hashes and complete target coverage.

## Unchanged training and frozen index provenance

Same958users/16177items/141137TRAINtargets per epoch, wide4+4/d128/FF1024/
dkv64/H6,FP16SDPAcompile512. Same Adam.003->.001,warmup2/cosine60,
min40/max60,VAL5/EMA.5delta.0005pat15; raw VALNDCG10selects best.
Only the selected checkpoint gets final TEST. SameFP32beam20,batch64,
no seen-item filter. Full best/last checkpoints retain optimizer/scaler/LR/RNG.
Seed roles index42/training42/augmentation42. Counts are evidence from past
TRAIN, not a future-distribution estimate. No optimizer/HPO/annealing changes.

Frozen item SIDs and RQ-KMeans centroids from long-pairs33754d7; reuse the full
121412pair cache from Random20(source4026db0,legacy kernel mixed20). Encode
5055missing pair descriptions with the same frozen Sentence-T5 max2048 and
original normalized centroids. Append suffixes TRAINthenVALthenTEST without
renumbering any existing item/pair SID. Same input-table vocabulary/parameter
count. Parent69pins and data10pins are checked. Cache geometry includes all
eligible adjacencies, not only the selected evaluation matching; no held-out
behavioral signal or quality is used to fit an encoder/quantizer.

Offline overhead for cache building is recorded separately from model FLOPs.
FLOP scope matches all earlier reports: B1CPU mathSDPA,supported matmul ops,
teacher-forced4code forward,FMA2; excludes beam/backward/ST5 and CPU policy cost.
It is not an online serving latency measurement. Report warm/total training
seconds, compile cold cost, peak GPU memory and beam evaluation time separately.

## Verification and collection

CPU planner:`scripts/research/plan_kindle_frequency.py`, full table and hashes
`gr-kindle-frequency-geometry.json`. Preflight:
`scripts/research/preflight_kindle_frequency.py`, evidence
`gr-kindle-frequency-preflight.json`: exact initializer, all native real-model
forward/backward shapes, TRAIN-only counts, original SID preservation, replayed
first epochs and all60budget5count schedules. Missing semantic codes in CPU
geometry tests are explicitly synthetic; actual vectors/codes are computed and
saved on Kaggle, with no claim that their semantic values were prevalidated.

Focused tests exhaustively check path optimum/counts, uniform tie sampling,
event/suffix/order/recent preservation, matched per-example counts/padding,
microbatch replay, held-out exclusion, budget8endpoint, actual runner paths,
TEST closure during fit, and strict three-fit contract. Full repository regression passed:1145tests,152optional skips,0failures,80.667s.
No additional broad retest is needed for documentation-only submission receipts.

Quota inspected:35.55GPUhours. Expected wall time about60–90minutes after
bootstrap, estimate only; GPU0frequency5thenrandom8,GPU1random5. Per-fit7200s,
suite10800s/serverhard180minutes including startup. Submit once and return;
collect when the user reports completion. No duplicate retries or additional
models. Preserve raw predictions for paired-user uncertainty and split-specific
rare-target/context diagnostics. Primary comparison is frequency5vsrandom5;
random8is the more aggressive point. Reuse Random20,Recent20,Recent10results.
