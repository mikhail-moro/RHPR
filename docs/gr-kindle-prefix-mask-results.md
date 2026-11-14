# Frozen prefix-mask probe: completed results

Private `carasik/rhpr-kindle-prefix-mask/1`,source
`fdb04d44caa03b16f8a08482fffff4b8d4cac864`. COMPLETE8/8evaluations,2saved
models,71.260seconds session,zero failures. No fitting,optimizer,tokenizer work,
new checkpoint selection or weight changes. Same raw-VAL-selected55/seed42,
958users perVAL/TEST,FP32/SDPA/beam20/batch64,no seen filter. These already
inspected targets make this an exploratory mechanism check.

## Aggregate result: masking does not rescue Pair32

| Model / split | Full NDCG@10 | Old prefix masked | Relative change | Full top10 hits | Masked hits |
|---|---:|---:|---:|---:|---:|
|pair32 / validation|0.015966|0.007811|-51.1%|31|13|
|pair32 / test|0.010651|0.002602|-75.6%|21|7|
|recent32 / validation|0.016899|0.016775|-0.7%|30|29|
|recent32 / test|0.014447|0.012434|-13.9%|28|22|

Pair32 relies strongly on having the old prefix under these fitted weights.
A simple explanation "the compressed prefix only interferes with the recent
clue,so removing it rescues the model" is not supported globally. This checks
conditional dependence on the prefix's presence;it does NOT by itself prove
that its personalized semantic information is useful or causes better learning.

Pair32 mask delta:VAL−.008155,user95%CI[−.013344,−.003607];
TEST−.008049,CI[−.012696,−.004028]. Recent32 deltas−.000124/−.002012 have
intervals crossing0. Difference in mask responses(Pair effect−Recent effect)
is−.008031VAL,CI[−.014980,−.001574],and−.006036TEST,
CI[−.012680,−.000135]. All intervals condition on these specific checkpoints,
5000paired user resamples,and exclude seed/index/selection uncertainty and
multiple-comparison corrections. Not seed-robust statistical conclusions.

## The less-popular recent-context group is not rescued

The previous fixed VAL popularity threshold is reused,not tuned here. For
TEST,the less-popular context group(n502)goes.004209→.001914,5→3hits;
the more-popular group(n456).017742→.003360,16→4hits. VAL groups also both
decline:.009888→.004834(n479)and.022043→.010788(n479).
Do not infer stronger semantic dependence for one group from absolute loss
alone:the groups start at very different baseline quality and have few hits.

## Individual recovery does occur,with larger losses elsewhere

Of16TEST targets hit by original Recent32 but missed by original Pair32,
masking recovers ONE. The original large wrong-volume example
`Dark Hunter’s Query (... Book56)`moves from outside Pair32's beam20 to
rank6. Its original Recent32 rank was1. The other largest case,
`Renewing the Soul: Souls of Chicago series`,remains outside beam20.
Thus conditional interference is plausible for an individual case,but it is
not the global remedy:masking loses16ofPair32's original21TEST top10hits.
Only5original hits remain,plus2new hits,giving7total. Only2users improve
NDCG while20worsen. OnVAL it recovers0of14original lost control hits and
loses20of31original Pair hits,giving13total after2newhits.

## Verification

All8saved prediction files independently replayed:NDCG/Recall5/10,invalid/
duplicate accounting,users/targets/case hashes. All16parent file pins verified,
including selected weights. All4full baselines reproduce EVERY ranked SID and
beam score exactly(max score difference0). Weights unchanged,gradients disabled.
Invalid paths do not explain the loss:Pair32TEST18→17invalid outof19160beam
candidates after masking;VAL24→18. Across8evaluations135invalid/153280paths,
0duplicates. Original ranks are preserved.

Pair32 masks32old SID positions;Recent32 masks64. Recent64SIDpositions,
USERtoken and encoder lengths97/129are fixed within each model. Existing masks
block old keys in all encoder layers and decoder cross-attention. Preflight
changes every masked ID and obtains bit-identical visible encoder states and
beams/scores,so the hidden prefix cannot leak into the preserved tail. The
USER representation can change when its access to old keys is removed.

## What this does and does not establish

The fitted Pair32 is sensitive to prefix removal and its removal usually hurts.
An inference-time mask creates a distribution shift and changes attention
normalization,even with fixed token positions. Therefore this is NOT a clean
estimate of a retrained Recent16 model,proof of an optimal pair representation,
or evidence that longer semantic history improves downstream quality.

To separate personalized old-history content from structural reliance on having
some prefix,a next control would preserve prefix length/mask and substitute
other users' summaries. That control has NOT been run. No followup runs,seeds,
training,extra mask variants or notifications were started.

Source/execution notes:`gr-kindle-prefix-mask.md`. Reproduce CPU analysis with
`scripts/research/analyze_kindle_prefix_mask.py --results RESULTS --pair-parent
LONG_PAIR_RESULTS --control-parent CAPACITY_RESULTS --data LONG_INPUT`.
Private saved evidence:`/workspace/artifacts/kindle-pilot/prefix-mask-results`,
including8predictionNPZs,analysis.json,user-ranks.npz,logs and immutable manifests.
