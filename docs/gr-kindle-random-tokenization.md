# Random old-history pair/single tokenization

Authorized2026-11-15:one bounded pilot of within-history stochastic tokenization.
This differs from the completed Mixed32,which chose either ENTIRE Full32 or
ENTIRE originalPair32 perbatch. One new seed42generator fit;reuse exact frozen
single/pair3+1SID indexes and completed Pair32/Recent32 controls. No encoder or
quantizer fitting,new cache,extra seeds,HPO,notifications or automatic retries.

## Sampling

Keep latest32original events. Older half can consist of adjacent pairs and
single items;recent half always remains original singles. Never cross the
boundary,drop/reorder/duplicate events,join nonadjacent items or pair padding.
Target remains the original single-item3+1SID. TRAINprefixes vary in length
as before;each original141137target occurs exactly once perepoch.

First sample M~Binomial(8,p),p=.5,for each optimizerbatch. All FULL32histories
in that batch use exactlyMpairs,so their native width is4*(32-M)SIDtokens,
plusUSER. For each individual history sample uniformly a nonoverlapping
adjacent matching of that size in its older16events. For nolder nodes and
mpairs,this has C(nolder-m,m)possible placements;choosemsorted distinct
integers in0..nolder-m-1and add0..m-1. Both start parities are possible.
This does not randomly permute SID labels or change the quantizer.

For short contexts,cap=floor(valid_events/4). A shared uniform size-Msubset
of8virtual slots determines how many of the firstcapslots are active. Thus
each short example has marginalBinomial(cap,p)count,rather than being forced
to maximum compression bymin(M,cap). Slots govern counts,not pair positions.
The native32-Mbound remains valid because missing_merges<=8-cap<=32-length.
Only normal short-history padding is needed;full histories are never padded
toFull32. Metadata records counts,actual pairs,odd-start pairs and padding.

Endpoints:M0is EXACT Full32;M8is EXACT originalPair32,including odd older halves
of short histories. Endpoint placements are forced to preserve the original
inference format. AtintermediateM,all valid matchings are eligible. Expected
pair exposure is half of originalPair-only,in expectation also matching the
old50/50batch mixture;actual counts and per-epoch hashes are saved. The new
and old mixtures still differ in density distribution and placements,so this
does not isolate one of those two effects.

Count and tiling RNG streams7401/7402are independent of shuffle/dropout/global
RNG. They are seed/epoch/batch-addressable;repeated epochs change tokenization.
The original seed+epoch permutation is retained. Cached whole-batch layouts
make microbatch splits identical;coverage audit rejects missing/duplicated rows.
The dataset is finite and previous Pair-onlyalready saw varied sliding windows;
this specifically resamples the representation of a fixed training example.

## Protocol and cost

Keep existing wide4+4/d128/FF1024/dkv64/H6,FP16SDPA,train512,AdamLR.003->.001,
warmup2/cosine60,min40/max60,VALevery5,scoreEMA.5/delta.0005/patience15.
RawVALNDCG10selects checkpoint. VAL and finalselectedTEST use ONLY standard
Pair32:8oldpairs+16singles,97positions,FP32beam20/batch64,no seenfilter.
TESTalreadyexamined;followup exploratory. Do not choose a probability or
checkpoint using TEST. No posthoc Full32/shift inference variants authorized.

Nine native training lengths97,101,...129can cause cold compilation;raise
only the isolated worker's Dynamo recompile_limit8->32to accommodate nine
lengths and final-batch shapes. No attention/math/optimizer change;retain
compile mode reduce-overhead/fullgraphfalse/dynamicfalse. Save cold/warm
times,memory,fullbest/last optimizer/scaler/LR/RNG,curves,allVALpredictions,
finalTEST,input/index/initializer hashes,and inference supported-forwardFLOPs.
FinalPair32cost remains.516500848GF;training average cost is higher thanPair-only.

All114727possible TRAINadjacent pairs are already in the existing116401pair
index. No Sentence-T5 encoding,newpairSIDassignment or centroid fitting.
Use oneT4worker only,private/offline`carasik/rhpr-kindle-random-tokenization`,
config`experiments/kindle_random_tokenization.json`. Suite10800seconds,
server180minutes hard lifetime to include bootstrap and cold compilation.
Prelaunchquota37.32GPUhours. Budget roughly1–1.5hours,not a firm runtime estimate.

## Why no probability schedule in this first run

There is no learned softmax distribution needing temperature here;directly
schedulepis clearer. A later clean contrast could increasepfrom.1to.9over60
epochs against constant.5,both averaging.5overa complete60epochbudget. This
would teach more single views early and align to Pair32later. Early stopping
and nonlinear compute mean the realised exposure/cost must still be measured.
Such a schedule may help alignment or remove useful diversity;it is not
assumed better. Onlyconstant.5is implemented for dispatch in this pilot.

CPUpreflight verifies44parent+10inputpins,same native initializer,nine finite
forward/backward shapes,exact Full/Pair endpoints,one real TRAINepoch coverage
and all60epoch schedules. Actual60epoch pair presentations30,063,740(50.166%
ofPair-only59,928,480;oldMixed29,604,042). Firstepoch CPUtokenization4.29s,
215755odd-startpairs. Full1132tests pass,152optional skips,0failures87.093s.
See`gr-kindle-random-tokenization-preflight.json`.
