# Mixed20 versus Recent10: requested compute/quality pilot

Authorized 2026-11-15: exactly two fresh seed42 generator fits on the same
long-reader Kindle cohort. Private offline kernel `carasik/rhpr-kindle-mixed20`,
two isolated T4 workers, one fit per worker. No additional seeds, lengths,
schedules, tokenizer refitting or notifications. Reuse completed Recent20,
Recent32 and Pair32 results, rather than retraining those controls.

“Mixed20” here means the latest within-history random-tokenization method,
not the earlier whole-batch Full/Pair mixture. Same constant probability .5:
M~Binomial(5,.5) per optimizer batch, random nonoverlapping adjacent placements
in the older10events independently per history. Recent10events stay single.
For short TRAINhistories,restrict a uniform M-of5virtual-slot subset to the
available floor(length/4) slots;do not clip the full-window merge count.
Every real event appears exactly once, chronological order is preserved,
and every pair AND single retains its complete3+1SID. Target is always single.

VAL and final TEST are deterministic Pair20:5old summaries+10recent singles.
Recent10uses exactly the same latest10raw events, all single. Its purpose is
to check the value of adding compressed older content;it is cheaper, rather
than compute-matched. Recent20is the main full-context quality/compute reference.

|Inference|Encoder positions incl.USER|B1 supported forward GFLOPs|
|---|---:|---:|
|Mixed20, deterministic Pair20|61|.319748392|
|Recent10|41|.217327232|
|Reused Recent20|81|.427087952|
|Reused Recent32|129|.704769968|

Mixed20uses25.13%fewer forward FLOPs thanRecent20. Compare NDCG and compute
jointly;one seed and958targets/split cannot establish noninferiority. Previous
TEST has already been examined,so this is exploratory. Do not select length,
seed,probability or checkpoint using TEST. Forward scope is the same as before:
B1CPU math-SDPA,teacher-forced4code forward,FMA2,excluding beam,backward and
offline Sentence-T5. Actual per-fit warm/total training times and GPU memory
are recorded separately;forward FLOPs are not serving latency.

## Unchanged comparison protocol

Same958users,16177books,141137TRAINprefix targets per epoch,leave-last-two
splits,original cases and catalog. Same wideTIGER4+4/d128/FF1024/dkv64/H6,
original trainable initializer,dropout.1,FP16SDPAcompile512/512.
AdamLR.003→.001,warmup2/cosine60,minimum40/maximum60epochs.
VALevery5,scoreEMA.5/min_delta.0005/patience15epochs;rawVALNDCG10selects best.
Only the VAL-selected checkpoint is evaluated on TEST. FP32SDPAbeam20,batch64,
no seen-item filter. Save all validation rankings,losses,curves,full best/last
weights including optimizer/scaler/scheduler/RNG and per-epoch input audits.

Separate named seed roles: index42,training42,augmentation42. The sampler
now allows an augmentation seed independent of its unchanged training shuffle;
no extra seed experiment is included. Count/placement RNGstreams7401/7402
remain independent of Torch. Original32sampler defaults byte-replay the
previous run's first real TRAINepoch,including input/schedule hashes.

## Frozen index extension

The different window requires5011previously uncached ordered pairs.
TRAINall-adjacent candidate set119744;VALfixed4739;TESTfixed4734.
Encode only missing pair descriptions with the original frozen Sentence-T5,
max2048tokens,labels`First book`/`Second book`;assign3codes using the saved
normalized RQ-KMeans centroids. Add collision suffixes TRAINfirst,then
VAL-only,thenTEST-only. Never renumber any of the116401existing pair SIDs.
Old item SIDs,old pair assignments,code vocabulary and model table sizes stay
fixed. Save new keys,embeddings,codes,indexes,report and provenance.
No text encoder or quantizer is trained.

CPU preflight verifies64parent pins and10input hashes,all60count schedules,
once-per-epoch target coverage,the full20/Pair20endpoints,exact recent10tail,
and unchanged initializer. All6native MixedTRAINwidths(80/76/.../60SIDpositions)
plus Recent10have finite real-model forward/backward passes. Cost atM0matches
the saved Recent20. Missing pair semantics are computed on Kaggle;CPU geometry
tests explicitly use temporary synthetic codes for those missing keys and do
not claim to have validated their actual semantic vectors.

Six targeted tests cover event/order/recent/suffix preservation at every width,
short-history count law,independent augmentation seed with unchanged shuffle,
microbatch replay,TRAIN-first suffix allocation,strict experiment contract,
and both real runner paths with fixed validation/unchanged targets/TESTclosed.
Existing Random32tests are retained.
Full regression:1138tests PASS,152optional skips,0failures,75.973s.
The private/offline bundle was checked byte-for-byte against all214Python
source files and the exact experiment JSON before submission.

## Execution

Config:`experiments/kindle_mixed20_2t4.json`.
Implementation:`src/rhpr/gr/kindle_mixed20.py`.
CPU preflight:`scripts/research/preflight_kindle_mixed20.py`,evidence
`gr-kindle-mixed20-preflight.json`. Quota inspected:36.30GPUh remaining.
Expected wall time roughly45–75minutes including frozen pair encoding and
cold compilation;this is an estimate,not a reserved accelerator guarantee.
Per-fit limit7200s,suite10800s,server hard timeout180minutes including startup.
Submit only once;collection is a later step after completion.

Submitted once as private version1,source`4026db0e479881586cad0cdaeae264e0ec4d06ef`.
Workflow43708134004completed successfully as a submission job;Kaggle status
isQUEUED. Job132793609143,artifact14158704699,actualpush`--timeout10800`
verified. All214deployedPython files,exact config,private/offline settings,
three datasets and completed parent mount match. Receipt:
`gr-kindle-mixed20-submission.json`. No training results claimed yet.
