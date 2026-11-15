# Beauty fixed-tail compression: completed pilot

Collected2026-11-16(clientEurope/Moscow). Private`carasik/rhpr-beauty-pairs/1`,
source7828d110117889667c3a5e5877f7be9bd0222c13. Four completed seed42fits on
2T4s;runtime8179.448s=136.32min after bootstrap,excluding queue/bootstrap.
No new GPU training or inference during collection. All original22363users,
12101items,131413TRAINprefixes;22363targets in each held-out split.

## Selected results

| Model | Best/stop epoch | VAL NDCG10 | TEST NDCG10 | TEST hits10 | Padded forward GF |
|---|---:|---:|---:|---:|---:|
| Recent20 |55/75|.04576327|.03173609|1328|.427088200|
| Recent3 |50/75|.04399019|.02997927|1217|.082200520|
| Random20,Tail3 |85/90|.04730982|.03149949|1317|.257705640|
| Random20,Tail5 |50/75|.04564402|.03162411|1335|.278189884|

Tail3keeps latest3raw events as singles;Tail5keeps latest5. Older events are
randomly tiled with p=.5on TRAIN and deterministically maximally paired on
VAL/TEST. Both pair AND single representations use full3+1SID. Same item
index,pair index,initializer,targets,shuffle,backbone,optimizer and schedule.
Controls retain the unused pair input table for matched parameterization.

AgainstRecent20,Tail3TESTpoint estimate−.75% with−39.66%forwardFLOPs;
Tail5−.35% with−34.86%forwardFLOPs. Tail3VAL+3.38%;Tail5VAL−.26%.
This is an encouraging quality/cost tradeoff in this one pilot,not established
noninferiority or superiority to the best equally expensive truncation.
Recent3is much cheaper and TEST−5.54%relative toRecent20;no Recent12/13models
were trained. Recent12would matchTail3's padded49encoderpositions andRecent13
would matchTail5's53,and are natural future controls before claiming an
advantage from jointly encoding items rather than simply shortening context.
No additional GPU runs were submitted.

Paired5000draw user-bootstrap differences:

| Comparison | VALdelta [95%CI] | TESTdelta [95%CI] |
|---|---|---|
| Tail3−Recent20 |+.001547 [−.000323,+.003364]|−.000237 [−.001891,+.001402]|
| Tail5−Recent20 |−.000119 [−.001863,+.001692]|−.000112 [−.001673,+.001423]|
| Tail3−Recent3 |+.003320 [+.001417,+.005161]|+.001520 [−.000099,+.003176]|
| Tail5−Recent3 |+.001654 [−.000092,+.003350]|+.001645 [+.000151,+.003128]|

Both TESTintervals againstRecent20crosszero and still permit a roughly5–6%
relative degradation. Small point-estimate losses alone do not prove a
predeclared1–2%noninferiority margin. Conditional users only,one pipeline seed,
shared target items across users,no seed/item-clustering uncertainty and no
multiplicity correction. The positive Tail5vsRecent3TESTinterval is an
exploratory comparison,not independent seed replication.

## Actual history lengths

TESTNDCG10by predeclared raw visible-context buckets:

| Context items | Users | Recent20 | Recent3 | Tail3 | Tail5 |
|---|---:|---:|---:|---:|---:|
|1–4|7162|.029430|.026632|.030494|.028543|
|5–6|6901|.029037|.025878|.026246|.029391|
|7–10|4753|.031743|.029702|.031669|.032608|
|11–20|3547|.041635|.045088|.043524|.040871|

Tail3's most evident point-estimate loss is at5–6events(−9.61%againstRecent20),
while at11–20it is+4.54%. ButRecent3is higher than both in the latter group
on VAL AND TEST. Thus longer histories do not uniformly require the compressed
old prefix,and this subgroup does not establish that compression is better
than truncation. All reportedTail3/Tail5vsRecent20TESTintervals for the
predeclared eligible/11–20groups crosszero. Do not promote these groups to a
validated router or causal compression rule.

Actual compression is possible forTail3VAL10980/TEST15201,andTail5VAL6489/
TEST8300. On the common n≥7group(TEST8300),Recent20.035970,Recent3.036277,
Tail3.036735,Tail5.036140. Both compressed variants are slightly aboveRecent20
there,with broad conditional intervals. On allTail3-eligible n≥5TESTexamples,
Tail3.031973vsRecent20.032823(−2.59%). Uncompressed short examples can still
change predictions because TRAINhas changed model weights;do not attribute
all global quality differences to directly merged inputs.

## Learning curves and runtime

All four stopped by the declared EMAearly stopping,not120epoch cap. Best
Recent20at55,Recent3/Tail5at50,Tail3at85. Tail3's best within first60epochs was
.04717074at55;its later peak.04730982is only+.29%relative. Therefore the later
selected epoch does not by itself show a large late quality gain. Minimum
teacher-forced VALCEwas at40for controls/Tail5,and45forTail3;NDCGpeaks later.
The stop rule measures plateau of smoothed VALNDCG,not a global optimum or
absence of overfitting.

| Model | Warm TRAIN epoch,s | Total optimizer TRAIN,s | VAL checks,s | First epoch,s |
|---|---:|---:|---:|---:|
| Recent20 |29.64|2356.99|1178.08|155.51|
| Recent3 |9.16|816.66|607.92|138.15|
| Tail3 |28.30|2623.37|1044.33|89.39|
| Tail5 |30.08|2809.58|922.35|347.19|

Tail3warm epoch is only4.52%faster thanRecent20,Tail5is1.48%slower. Tail3has
more completedepochs and both have TRAINsampling/Python overhead and weaker
average TRAINcompression. Thus forward-FLOPs savings did NOT become a35–40%
training speedup. Cold first epochs are not comparable independent benchmarks:
pair fits are second on each worker and compile-cache effects differ.
Peak allocated memory≈2.18/0.58/2.20/2.20GBrespectively. AMPskips2/2/3/2;
no LR/precision/hardware fallback. Teacher-forced padded B1forward,FMA2,
excludes beam search/backward/text encoding;no serving-latency benchmark was
performed. Original pair-cache preparation is included in overall runtime,
not per-generator TRAINtimes.

## VAL versus TEST

The absolute gap persists onBeauty:Recent20VAL.045763→TEST.031736(−30.65%).
All models have the same direction of gap. This run therefore does not prove
that previous Kindle behavior was a data bug,or that a larger standard dataset
makes VALandTESTequally difficult. Relative method differences are much smaller
here than in the recentKindleprobability pilot;one seed and a changed dataset/
compression policy cannot identify which factor caused that improvement.

Descriptive target support in TRAINnext-item positions:VALmedian13/mean29.15,
TESTmedian10/mean24.76. Cold targets81VALvs196TEST,unique target items8351vs7978.
No held-out target occurs in its own visible input context. These demonstrate
some distribution differences,not a complete explanation of the gap. Split
order is inherited from the normalizedGRID/P5artifact,which lacks timestamps;
we cannot independently verify raw chronological ties from these files.

## Audit and saved evidence

All input catalogue/sequence SHA256pins and all original raw split arrays
verified. Frozen3+1indexes and append-only suffixes verified,max pair suffix51.
CPUcentroid export agrees on12097/12101item prefixes and65076/65080pair prefixes.
Four exceptions in each are numerical:the residual before the first differing
level has norm8.78e-8…4.58e-7. Normalizing a virtually zero residual amplifies
CPU/CUDArounding into later code choices. The original exported CUDAindexes
were retained byte-for-byte for every metric;no index was repaired/reassigned.
This is documented in JSONrather than falsely claiming bit-exact CPUreplay.
Suffix allocation was replayed exactly using the actual exported prefixes.

All67predictionfiles(63VAL+4selectedTEST),29966420candidate paths,independently
replayed;29920invalid paths remain misses at original ranks,zero duplicate
paths. All165random TRAINcount/exposureepochs replayed and both first-epoch
full input tensor hashes matched. Later complete TRAINtensor hashes were not
replayed. Initializers/raw cases/targets/LR/update counters/EMA/selection and
actual selected-checkpoint hashes verified. Eight full best/last checkpoints
finite,epochs/configs checked;last retains optimizer/scaler/scheduler/RNGstate.
No new training,inference,checkpoint search or test-basedmodel selection.

Raw evidence,all weights,vectors,centroids,indexes,CSV,user ranks andPNG/PDF:
`/workspace/artifacts/beauty-pair/results`. AggregateJSON:
`gr-beauty-pairs-results.json`;receipt:`gr-beauty-pairs-completion.json`.
Reproducer:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kindle-pairs-mpl PYTHONPATH=src \
/tmp/unique4-resume-venv/bin/python scripts/research/analyze_beauty_pairs.py \
 --results /workspace/artifacts/beauty-pair/results \
 --data /workspace/artifacts/beauty-pair/input
```

Full regression1160tests passed/152optional skips/0failures,98.581seconds.
