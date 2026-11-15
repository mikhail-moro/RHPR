# Mixed20 and Recent10: completed pilot

Private `carasik/rhpr-kindle-mixed20/1`,source
`4026db0e479881586cad0cdaeae264e0ec4d06ef`,COMPLETE. Exactly two fits,
training/index/augmentation seeds42. No new GPU or inference during collection.
Measured experiment execution2651.821s(44.20min)after initial bootstrap;
queue time is excluded. Both completed60epochs and selected60by rawVALNDCG10.
This reaches the epoch cap,not demonstrated convergence.

## Selected checkpoints and forward compute

Same958users and targets per split,same wideTIGER4+4/d128/FF1024/dkv64/H6,
same initializer,original single-item index and optimization settings.
FP16SDPAcompile512training;VALevery5;EMAstopping .5/delta.0005/patience15,
minimum40/maximum60;rawVALNDCG10selects the checkpoint. TEST is evaluated
only after selection,FP32SDPAbeam20/batch64,no seen-item filter.

|Model|Best epoch|VAL NDCG10 / hits10|TEST NDCG10 / hits10|Forward GFLOPs|
|---|---:|---:|---:|---:|
|Mixed20|60|.014796 / 24|.015643 / 29|.319748|
|Recent10|60|.014931 / 29|.014623 / 26|.217327|
|Reused Recent20|55|.020492 / 38|.014722 / 24|.427088|
|Reused Recent32|55|.016899 / 30|.014447 / 28|.704770|
|Earlier Random32|55|.017308 / 30|.014961 / 26|.516501|
|Original Pair32|55|.015966 / 31|.010651 / 21|.516501|

Mixed20uses exactly the intended deterministic Pair20format on bothVAL/TEST:
5older pair summaries+10latest single items,all3+1codes,61encoder positions
includingUSER. Only TRAINuses random adjacent matchings in the older half.
Recent10uses precisely the same latest10raw events,all single,41positions.
Every original user/target appears once per epoch;the training cases and
catalog are not reduced for either method.

The TEST quality/compute point is promising:Mixed20versusRecent20 gains
6.25%NDCG10at25.13%fewer forward FLOPs. Versus earlierRandom32,it gains
4.56%TESTNDCGat38.09%fewer FLOPs. But VAL is27.80%belowRecent20and14.51%
belowRandom32. Therefore this is not a consistent quality-preserving
compression result across splits or seeds.

Recent10is a strong low-compute control. Its TESTNDCG is only0.67%below
Recent20,with49.11%fewer FLOPs. Mixed20gains6.97%TESTNDCG overRecent10at
47.13%more FLOPs,while VAL is0.91%lower and has fewer hits(24vs29).
Recent10was deliberately a recent-only floor,not a FLOP-matched control.
These point estimates do not prove that compressed older information helps.

All12selected VAL/TEST conditional paired-user95%intervals include0:

|TEST comparison|Delta NDCG10|Conditional95%interval|Gained / lost hits|
|---|---:|---|---:|
|Mixed20−Recent10|+.001020|[−.005751,+.007886]|16 / 13|
|Mixed20−Recent20|+.000920|[−.006148,+.007668]|19 / 14|
|Mixed20−Random32|+.000682|[−.007311,+.008489]|20 / 17|

Mixed20versusRecent20has29vs24hits but4vs7rank1hits;the positive TESTNDCG
margin comes from broader top10membership rather than more first places.
Shared-hit ranks reduce the net gain. The total+.000920margin is less than
one+1discounted-credit contribution divided by958. This describes sensitivity,
not a reason to remove users. The intervals condition on fitted selected
checkpoints and omit seed/index/item-clustering and selection uncertainty;
there is no multiple-comparison correction or noninferiority test.

## Previously defined groups

The earlier low/high recent-popularity split is reused unchanged:meanlog1p
TRAINtarget frequency over latest16raw events,threshold2.30082044. It is a
common diagnostic covariate,not necessarily the exact context ofRecent10.
AgainstRecent10,Mixed20has higher NDCG in the high-popularity context group
on bothVAL(.023227vs.021310)andTEST(.023678vs.020023),but lower in the
low group on both(.006366vs.008553; .008344vs.009718). Sparse descriptive
groups do not establish an adaptive routing rule.

Rare targets are a consistent negative finding in this run:

|Target TRAINsupport1–5|Users|Mixed20 NDCG / hits10|Recent10 NDCG / hits10|Recent20 NDCG / hits10|
|---|---:|---:|---:|---:|
|VAL|402|.000719 / 1|.004199 / 3|.008907 / 7|
|TEST|425|.000838 / 1|.003499 / 3|.005974 / 6|

The rare-target recovery seen with earlierRandom32did not transfer toMixed20.
This is not proof of an exposure mechanism or a general compression defect;
correct predictions in these cells are very few. AtTEST,Mixed20gains in
support6–20(19vs12Recent20hits)and>20(9vs6),offsetting rare-target losses.
VAL's other groups do not show the same benefit. All original groups,including
negative/cold results,are retained in the aggregate JSON rather than selecting
only favorable cells. There is no confirmed history-dependent improvement.

All models report5.108992Mnominal parameters;Recent controls have an unused
131072-parameter pair-input table. Nonembedding Transformer is identical.
Forward cost uses the historical B1CPUmath-SDPA teacher-forced4code/FMA2
definition,not beam search,backward or offline text encoding. It is not a
serving-latency measurement. TEST has been examined before,so later comparisons
on this cohort remain exploratory.

## Training behavior and time

Neither new model stopped by patience;both selected the final60epoch check.
Do not call this a plateau or automatically continue training. Late VAL scores:

|Epoch|Mixed20 NDCG10|Recent10 NDCG10|Recent20 NDCG10|
|---|---:|---:|---:|
|40|.011675|.009085|.013639|
|45|.010485|.011990|.014011|
|50|.012645|.011387|.012036|
|55|.013020|.012490|.020492|
|60|.014796|.014931|.017576|

Mixed20's best VAL teacher-forced CE is9.129648at55,while bestNDCG is at60
withCE9.331905. Recent10's CE also worsens55→60(9.323816→9.408837)while
NDCG rises. This is no evidence that changing to loss-based selection would
improve TEST;no alternative checkpoint is evaluated onTEST here.

Warm optimizer epoch:Mixed20**34.17s**,Recent10**16.31s**.
Total optimizer time2515.22s(41.92min)and1116.82s(18.61min),respectively.
They ran concurrently on separateT4s. First epochs315.20sand143.09sinclude
cold compilation. Mixed20is not faster to train than the historical
Recent20(warm31.25s,total1996.81s),despite lower inference cost;random TRAIN
uses mostly less-compressed sequences and multiple compiled native lengths.
Historical comparisons are not simultaneous benchmarks.

Mixed20peak allocated2,188,569,088bytes,reserved5,114,953,728;
Recent10allocated1,251,176,960,reserved2,390,753,280.
Successful updates16558plus2AMPskips,and16559plus1AMPskip,respectively,
each totaling60×276batches. Each consumed8,468,220TRAINtargets.
Mixed20skip epochs52/59;Recent10skip57. No compile-limit fallback detected.
There was no retraining,automatic precision/LR fallback or additional seed.

## Index and execution provenance

All5011missing pair descriptions were encoded with the frozenSentence-T5
and assigned using the existing normalized RQ-KMeans centroids. The complete
pair index now has121412entries. TRAINadditions receive suffixes before
VAL-only andTEST-only additions;all116401old pair SIDs are unchanged.
Maximum suffix remains31. Nothing was refitted. This is a real runtime cache
extension,not the synthetic missing input codes used for CPUgeometry preflight.
The saved new vectors/codes/keys are retained for independent replay.

The completed audit independently recomputed all5011new codes from their
actual saved vectors using frozen centroids onCPU,reconstructedTRAIN-first
suffix allocation,and verified all old assignments. It verified64parent pins
and10input hashes,unchanged initial weights/raw cases/targets,and every LR,
update,EMA,stopping and rawVALselection decision. ALL60sampled Mixed20TRAIN
epochs were reconstructed with exact input/schedule hashes,coverage and
pair/padding counts:19,665,207pair presentations,7,213,312odd-start pairs.
Both VAL/TESTlayouts were checked asfull61/41positions without padding.
All78new/control prediction files were replayed:1,494,480ranked paths,
2978invalid predictions,0duplicates. Metrics match the saved reports exactly.
The four complete best/last checkpoints were downloaded,finite weights,
epochs,selected-checkpoint hashes and scheduler/update counters verified.
Full regression1138tests PASS,152optional skips,0failures,113.955s.

Raw and derived artifacts,including both full best/last checkpoints:
`/workspace/artifacts/kindle-pilot/mixed20-results`.
Reproduction:`scripts/research/analyze_kindle_mixed20.py`.
Aggregate audit:`gr-kindle-mixed20-results.json`;
completion record:`gr-kindle-mixed20-completion.json`.
Derived`analysis.json`,`model-metrics.csv`,`user-ranks.npz`and standalone
`mixed20-curves.png/.pdf`are alongside raw outputs. The plot includes learning
curves and separate VAL/TESTquality-versus-forward-cost panels. No extra
seed,checkpoint inference,schedule or training continuation was launched.
