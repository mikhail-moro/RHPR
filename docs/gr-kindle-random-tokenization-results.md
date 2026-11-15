# Random pair/single tokenization: completed results

Private `carasik/rhpr-kindle-random-tokenization/1`,source
`a7f82485857265d0cb704d54cbb9be892d076519`. COMPLETE,one fit,seed42,
60epochs/raw-VAL-selected55,epoch cap reached. Measured experiment execution
3629.208s(60.49min) after initial runtime bootstrap.
No new GPU work during collection. No additional fit,probability schedule,
checkpoint selection or inference variant launched.

## Matched selected checkpoints

All five runs completed60epochs and selected55using rawVALNDCG10. Same
958users/split,original targets,indexes and generator initializer. Pair,
Mixed and Random all use identical Pair32inference:8oldpairs+16recent single
items,3+1codes each,97encoder positions includingUSER. FP32/SDPA,beam20,
batch64,no seen-item filter. Full catalog evaluation,not a sampled candidate set.

|Model|VAL NDCG10|TEST NDCG10|TEST top10 hits|Forward GFLOPs|Warm epoch seconds|
|---|---:|---:|---:|---:|---:|
|pair32|0.015966|0.010651|21|0.516501|37.59|
|mixed32|0.018937|0.009874|20|0.516501|49.48|
|random32|0.017308|0.014961|26|0.516501|48.63|
|recent32|0.016899|0.014447|28|0.704770|55.84|
|recent20|0.020492|0.014722|24|0.427088|31.25|

Random TEST point improvements:Pair32+40.46%,Mixed+51.52%,Recent32+3.56%,
Recent20+1.62%. VAL is+8.41%vsPair32,+2.42%vsRecent32,but-8.60%vsMixed
and-15.54%vsRecent20. This is not uniform superiority across splits.

Relative toRecent32,forward FLOPs fall26.71%while TESTNDCG increases3.56%in
this one run. However Recall10 decreases:26hitsvs28. Therefore describe it
as a promising NDCG/compute point,not confirmed noninferiority or a robust
Pareto improvement across quality measures. Recent20is still cheaper
(.427088vs.516501GF);itsNDCG is only1.62%below Random.

All report5.108992Mnominal parameters;Recent controls include an unused
131072-parameter pair-input table. The nonembedding Transformer architecture
and initial trainable values are unchanged. Forward counts are B1supported
CPU math-SDPA teacher-forced4code forward,FMA2;exclude beam/backward/offline
Sentence-T5 encoding. This is not an end-to-end serving latency benchmark.

## Uncertainty and hit accounting

|Random minus comparator,TEST|Delta NDCG10|Conditional paired-user95%CI|Gained hits|Lost hits|
|---|---:|---|---:|---:|
|mixed32|+0.005087|[-0.000744, +0.011141]|15|9|
|pair32|+0.004310|[-0.001834, +0.010421]|17|12|
|recent32|+0.000514|[-0.006225, +0.007349]|15|17|
|recent20|+0.000238|[-0.007365, +0.007770]|17|15|

All8selectedVAL/TEST comparison intervals include0. The apparent+40%relative
gain vsPair comes from only26hits rather than21among958users. Random adds17
Pair-missed hits,loses12,and improves shared-hit ranks in aggregate;this is
not five simple extra hits on top of an otherwise identical result.
These5000paired-user resamples condition on fitted selected checkpoints.
They do not represent seed/index/item-clustering or model-selection uncertainty
and are not corrected for multiple comparisons. TESTwas examined in earlier
experiments,so this follow-up is exploratory. No probability chosen byTEST.

## Training curves and cost

Random beatsPair32on only4of12VALchecks. At epochs40/45/50/55/60:
Random.012375/.010172/.008558/.017308/.013919;
Pair.012525/.010739/.009185/.015966/.014195. Thus Random is not consistently
ahead throughout training. Several controls also peak at55. Minimum Random
VALCE9.148912occurs at60;do not call the60epochcap a proven plateau.
Both best and final states retained for later analysis or authorized continuation.

Actual successful updates16558plus2normalAMPskips=60x276;skip epochs56/58.
No precision/LR fallback,training failure or compile-limit fallback detected.
Input targets8,468,220,exactly one original example per epoch,matching controls.
Random optimizer-training time3501.87s(58.36min)vsPair2357.15s(39.29min):
+48.56%,including cold compilation. VersusMixed3247.00s:+7.85%;historical
Mixed includes a second cold compilation after its epoch30resume. Warm epoch
Random48.63s vsPair37.59s(+29.35%),Mixed49.48s(-1.73%),Recent3255.84s.
These are historical nonsimultaneous T4timings,not a controlled speed test.
Random first epoch388.63s;additional>80sspikes at2/3/6/9/17/30are compatible
with first encounters of native/final-batch compiled shapes. Peak allocated
3,344,786,944bytes,reserved7,572,815,872bytes.

## Previously defined groups

Use original TRAINtarget frequencies and previous VAL-fixed recent-context
popularity threshold2.300820444821813,without searching for a new cutoff.

|TEST group|Users|Pair32 NDCG10 / hits|Mixed NDCG10 / hits|Random NDCG10 / hits|Recent32 NDCG10 / hits|
|---|---:|---:|---:|---:|---:|
|less_popular_recent|502|0.004209 / 5|0.004994 / 7|0.010416 / 10|0.011788 / 11|
|more_popular_recent|456|0.017742 / 16|0.015246 / 13|0.019963 / 16|0.017374 / 17|
|target_1_5|425|0.000784 / 1|0.003217 / 4|0.005162 / 5|0.006773 / 5|
|target_6_20|415|0.017692 / 15|0.012319 / 10|0.021676 / 16|0.022231 / 19|
|target_over20|79|0.031998 / 5|0.037714 / 6|0.039786 / 5|0.021968 / 4|
|target_cold|39|0.000000 / 0|0.000000 / 0|0.000000 / 0|0.000000 / 0|

The previously weaker less-popular recent-context group improves atTEST
(.004209->.010416,5->10hits)but barely atVAL(.009888->.010120,10->12hits).
Rare-target group also improves versusPair:VAL5->7hits,TEST1->5. Counts remain
very small;these are descriptive leads,not proof of a popularity-bias mechanism.
Random recovers5of16original TESTRecent32hits missed byPair32(3of14atVAL),
while also losing other control hits.

## What changed and what was verified

Constantp=.5duringTRAIN. Per-batchBinomial count,per-history random adjacent
placements in older half;bothparities;recent half singles. No field encoder,
quantizer,catalog,training target,validation or test tokenization change.
Pair presentations30,063,740vsoldMixed29,604,042(+1.55%),roughly half the
Pair-only59,928,480. There were12,743,061odd-start pair occurrences. Therefore
diversity really changed;this is not another whole-history Full/Pair switch.
However both compression-density distribution and pair placements changed,
so the result does not isolate their individual effects or prove that fixed
pairing caused the prior TESTgap.

Independently reconstructed ALL60sampled TRAINepochs and matched every
coverage audit,schedule hash,tokenized-input hash,pair/odd-start/padding count.
44parent pins and10input hashes verified,frozen indexes byte-identical,
initializer/target/case hashes and LR/EMA/updates/rawVALselection verified.
All65prediction files(13new+52old)replayed:1,245,400ranked paths,
2465invalid,0duplicates. Both new best andlast weights downloaded,
finite and epoch/config/scheduler counters checked. Best checkpoint SHA
`474dc2d80189d465ebe70dba0069692badb3bac5632fd0318dea25b30d9d56bf`.
Last full training-state SHA
`da1b6233a5f24ea1fb3c23726a47dc99f2cf122afb381672c16abf6e6bb86071`.

Private raw and derived artifacts:
`/workspace/artifacts/kindle-pilot/random-tokenization-results`.
Includes best/last checkpoints,all predictions,curves/logs/indexes,
`analysis.json`,`model-metrics.csv`,`user-ranks.npz`,and standalone scientific
`random-curves.png`/`.pdf`(VALNDCG/CE,training time,selectedTESTversusFLOPs).
Aggregate evidence:`gr-kindle-random-tokenization-results.json`;
completion:`gr-kindle-random-tokenization-completion.json`. No private raw
data or model weights committed. Reproduce without GPU or model inference:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kindle-pairs-mpl PYTHONPATH=src \
  /tmp/unique4-resume-venv/bin/python scripts/research/analyze_kindle_random_tokenization.py \
  --results /workspace/artifacts/kindle-pilot/random-tokenization-results \
  --parent /workspace/artifacts/kindle-pilot/long-pairs-results \
  --mixed /workspace/artifacts/kindle-pilot/mixed-resume-results \
  --data /workspace/artifacts/kindle-pilot/long-input --plot
```

Full regression1132tests pass,152optional skips,0failures84.895s.
No runs pending. No additional seed,annealing schedule,training,inference
variant or notification launched.
