# TRAIN sampling probability and curriculum pilot on original Random20

Authorized2026-11-15:try different TRAINsampling ratios and an increasing merge
schedule. Three fresh seed42fits on two isolated T4 workers;private/offline
kernel`carasik/rhpr-kindle-sampling`,config`experiments/kindle_sampling_2t4.json`.
This returns to the completed original Random20(oldest10eligible,recent10single),
not the unsuccessful first16eligible frequency pilot. Only TRAINprobability
changes;input SID indexes,VAL/TEST placement,targets,architecture,optimizer,
selection and evaluation protocol remain the completed Random20control.

## Predeclared comparison

Three new fits:
- `random20_p25_42`:constantp=.25.
- `random20_p75_42`:constantp=.75.
- `random20_ramp01_42`:p(epoch)=(epoch−1)/59,epoch1..60,linear0→1.

Reuse completed`mixed20_wide_42`(user-facingRandom20,p=.5,legacykernel
`carasik/rhpr-kindle-mixed20/1`,source4026db0)andRecent10/Recent20as references;
do not train another duplicatep50control. The comparison remains a single-seed
exploratory screen;existingTESThas already been examined. No TESTselection of
probability,schedule,length,checkpoint or seed. Each predeclared fit gets one
final TESTevaluation of its rawVALNDCG10-selected checkpoint.

At a full20event TRAINwindow M~Binomial(5,p(epoch))per optimizer batch,then
individual histories get uniform nonoverlapping adjacent matchings in old10,
both parities. Thus p is the merge-slot probability,**not**the fraction of
batches fully compressed or a whole-example Full/Pairview mixture. Expected
pairs=5p:1.25/2.5/3.75under constants. Recent10remain single;all raw events stay
chronological exactly once;each item AND pair has complete3+1SID;targets are
always single. Short histories retain original conditional uniform-slot
restriction giving Binomial(floor(valid_length/4),p),not clipping M.

Same index/training/augmentation seeds42;same count/placement streams7401/7402,
same shuffle. The generalized p=.5sampler exactly replays original TRAINtensors
and first-epoch exposure/hash;all60original count/exposure schedules also replay.
No probability lookup mutates RNGor current epoch metadata. Active per-trial
sampling policy is stored in provenance,row,and checkpoint config for future
continuation;no continuation is included in this pilot.

## Inference and compute

ALLnew fits use unchanged original Pair20on VAL/TEST:old10→5pairs plus10recent
singles=15representations×4SIDtokens+USER=61encoderpositions,
**.319748392supported forwardGFLOPs**. This is25.13%belowRecent20(.427087952).
No random pair placement,frequency weighting,new seed or temperature at
inference. SameVAL/TESTraw cases and suffixes as completed Random20.

TRAINnative widths80/76/72/68/64/60SIDpositions follow the chosen per-batch M.
Different p changes TRAINcompute,but not inferencecompute. CPU replay of all
60planned schedules gives the following actual pair presentations and supported
forward-only compute,assuming all60epochs complete:

| TRAINschedule | Mean p | Pair presentations | Mean TRAINforwardGFLOPs/example |
|---|---:|---:|---:|
| constant .25 | 0.25 | 9,839,002 | 0.399761 |
| reused constant .50 | 0.50 | 19,665,207 | 0.372711 |
| constant .75 | 0.75 | 29,499,847 | 0.345905 |
| linear 0→1 | 0.50 | 19,718,546 | 0.372754 |

The ramp has the same expected meanp=.5and nearly the same realized pair
exposure as the historicalcontrol(+0.27%);it is not exactly exposure-matched.
Mean supported TRAINforward cost differs only~.012%. It tests sequencing of
exposure,including interaction with the unchanged LR schedule. It does not
isolate ordering from all optimization effects or ensure a better curriculum.
At epoch1the ramp exactly equals Full20;at60it exactly equals originalPair20,
including short histories. Existing early stopping is preserved;an earlier stop
means the ramp has not reached1. Record actual final p and training budget rather
than claiming the final compressed stage occurred if a fit ends early.

FLOPs use the prior B1CPU mathSDPA supported matmul profiler scope,FMA2,
teacher-forced4code forward;exclude beam/backward/text embedding/CPUplacement.
Do not call these serving latency or actual full-training FLOPs. Log actual
first/warm/total epoch seconds,validation seconds,AMPskips and peak GPUmemory.
Later ramp shapes can cause delayed compile spikes;first-epoch timing alone
does not measure total compile overhead. Two sequential fits onGPU0may reuse
compiler caches;historical timings are not simultaneous benchmarks.

## Fixed data/model/learning protocol

Same958users/16177items/141137TRAINprefix targets per epoch,leave-last-two,
958targets per heldout split. SamewideTIGER4+4/d128/FF1024/dkv64/H6/dropout.1,
5,108,992parameters,originalinitializer. FP16SDPAcompile512/512,Adam.003→.001,
warmup2/cosine60,min40/max60,VAL5/scoreEMA.5,min_delta.0005/pat15epochs,
rawVALNDCG10best. SameFP32SDPAbeam20,batch64,noseenfilter. Full best/last
weights retain optimizer/scaler/LR/RNG. No architecture/HPO/epoch/protocol
changes beyond the declared TRAINsampling probability.

## Frozen cache and verification

Mount completed long-pairs33754d7for64original parentpins and completed
Random20source4026db0for24cache/controlpins,total88. Data10pins unchanged.
Copy its full121412paircache/itemSIDbytes exactly;all119744eligible TRAINpairs
and4739VAL/4734TESTfixedpairs already exist. **Zero new embeddings,zero new
quantizer assignments,zero index modifications.** No Sentence-T5phase needed.
Keep prior source/data/quantizer provenance,including its transductive static
cohort limitation;do not call this a globally chronological deployment study.

CPUpreflight:`scripts/research/preflight_kindle_sampling.py`,evidence
`gr-kindle-sampling-preflight.json`. Verifies88parent/cache+10inputpins,original
p50byte replay,all60schedules/exposurecounts,exact Full20/Pair20ramp endpoints,
unchanged initializer,all6native finite real-model forward/backward passes,
fixed inference61positions/.319748392GF. All codes in this preflight are actual
saved semantic codes;no synthetic missing-pair placeholders are needed.

Focused7tests cover endpoint/mean/invalid probabilities,nested common-RNG
counts,originalp50byte-replay,short-history endpoints,microbatch/epoch replay,
no global RNGmutation,per-epochmetadata after future schedule reads,strict
three-fit contract,and the real runnerhook with fixed VAL/TEST and unchanged
targets/checkpoint policy. Existing Mixed20/Random32/Frequencytests stay intact.
Full repository regression passed:1153tests,152optional skips,0failures,78.177s.

## Execution budget

Quota inspected34.37GPUhours. GPU0runsconstant.25thenconstant.75;GPU1runsramp.
Expected wall70–100min after bootstrap,estimate only. Sameper-fit7200s,
suite10800s/server180minhardtimeout including startup and compile. One private
submission only,notify_telegram=false. No additional seeds,schedules,retries,
checkpoints or evaluation formats. Return after verified submission;collect
when the user reports completion. No training results claimed yet.


## Packaging regression fixed before dispatch

The first full regression found the historical large-manifest recovery bundle
917bytes over Kaggle's1MiBsource limit after adding this module. The new sampling
bundle itself fitted. The builder now falls back from zlib to standard-library
LZMAonly when needed,restoring the exact complete STOREDZIPwithout dropping any
source/config bytes. The legacy bundle now uses737862bytes;the sampling bundle
remains zlib939083bytes. A new test executes the fallback and verifies the same
output contract;existing source-byte integrity/size-limit tests pass. No GPUrun
was submitted before resolving this regression. The initial failed regression
log is retained privately as sampling-regression-initial.log.
