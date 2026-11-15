# Beauty: fixed recent tail with pair Semantic IDs

Authorized2026-11-15:move the compression pilot to original Amazon2014Beauty,
using the TIGER/P5 dataset and leave-last-two evaluation. Compare protected
recent tails3and5,keeping every original user and all short TRAINprefixes.
No GPU fit has yet been submitted at this preparation stage.

## Declared experiment

Exactly four fresh seed42generator fits on two isolated T4workers:

| Trial | TRAIN | VAL/TEST | Encoder positions including USER |
|---|---|---|---:|
| recent20_42 | latest up to20 singles | same |81|
| recent3_42 | latest up to3 singles | same |13|
| random20_tail3_42 | random adjacent matching before last3 | all fixed prefix pairs |49|
| random20_tail5_42 | random adjacent matching before last5 | all fixed prefix pairs |53|

Assignment:worker0Recent20thenTail3;worker1Recent3thenTail5. Same initializer,
single and pair SIDtables,TRAINtargets,shuffle seed,architecture and optimizer.
Recent3is a truncation reference;it is NOT compute-matched to either pair fit.
This first pilot cannot claim superiority to the best truncation baseline at
identical cost;intermediate recent-window controls would be subsequent work.

For actual raw input length n≤20 and protected tail r,old=max(n−r,0),capacity
floor(old/2). VAL/TESTpair old positions[0,1],[2,3],...;an odd leftover is a
single immediately before the protected recent tail. TRAINuses per-batch
M~Binomial(floor((20−r)/2),.5),conditional uniform virtual-slot restriction for
short histories,then independent uniform size-k matching of each old prefix.
This gives each short history Binomial(its capacity,.5),not clipped global M.
The maximal global endpoint replays the fixed evaluation layout. Sampling
streams7401/7402andaugmentation seed42. Every event occurs once,in order;no
pair crosses the protected tail or consumes the target. All pair AND single
representations carry mandatory3+1SID;targets always single-item SIDs.

## Data and tokenizers

Pinned GRID/P5 Amazon2014Beauty:22363users,12101items,198502events,
153776TRAINevents and131413noninitial TRAINtargets per epoch. Complete
histories have ≥5events;the minimum TRAINinput is ONE event,VAL3,TEST4.
VALhas22363targets;TESThas the same users and observes the VALevent. Context
cap20raw items;no additional positive-rating,metadata,length,core or temporal
filter. Supplied sequence order is inherited from GRID/P5;these normalized
records do not contain timestamps for an independent boundary audit.

Same exact supplied catalogue text for each item,including its original fields.
Frozen cached Sentence-T5-base resources;encode singles max256tokens. Pair
text is `First item: <textA> Second item: <textB>`,max2048tokens,ordered,cached.
These caps are tokenizer caps,not generator history lengths. Fit original
GRID normalized-residual RQ-KMeans3×256 separately on all static catalogue item
vectors and on TRAIN-candidate pair vectors ONLY. Fixed seeds42,10catalogue
passes/level,batch2048,initial buffer3072,torch_cdist. Pair candidates include
all TRAINadjacencies that either policy can sample;held-out candidates include
only actual fixed layouts. Counts:TRAIN51400,VAL28944,TEST36736,union65080.
Held-out pairs get frozen-centroid assignments and append-only suffixes in
TRAIN→VAL→TEST order;no held-out interactions enter pair fitting. All catalogue
metadata is transductive,as in our prior baseline. Fail rather than silently
repair a suffix overflow. Save vectors,centroids,SIDs and complete fit reports.

Compressible actual inputs:

| Split | Examples | Tail3 ≥1pair | Tail5 ≥1pair |
|---|---:|---:|---:|
| TRAIN |131413|60506|45717|
| VAL |22363|10980|6489|
| TEST |22363|15201|8300|

TRAINcounts above mean possible compression,not realized random draws. Report
all users and predeclared actual-length buckets1–4,5–6,7–10,11–20,same users
within each split/comparison,and separately tail-specific eligible inputs.
Changing training changes weights for even uncompressed examples;eligibility
is not a causal attribution of per-user gains to compression.

## Learning and evaluation

Keep previous wide backbone4+4,d128,FF1024,dkv64,H6,dropout.1,user bins2000,
noSEP,SDPA. Identical pair-input table retained even in controls:5108992params.
Initializer CPUhash34e7397ed277c78befd6254cbb736794df4907a011ed032bbd0735d9b7272286
verified in each actual fit. FP16,compiled512/512,Adam.003,wd1e-4,clip1.
Warmup2epochs/cosine to.001at60,then floor.001through120;all variants share
that schedule. min40,max120,VALevery5,EMA.5/min_delta.0005/patience20epochs.
Raw VALNDCG10chooses checkpoint;EMAonlystopping. A120epoch-cap is not proof of
plateau. Save full best weights and last optimizer/scaler/LR/RNGstate for later
continuation. No automatic LR/precision/hardware fallback or fresh retries.

FP32full-catalogue beam20,batch64,Recall/NDCG5/10,no sampled candidates,no seen
filter,invalid sequences remain misses at original ranks. Save VALpredictions
at every check and teacher-forced CE by level;evaluate TESTonce per declared
fit after VALselection. No TESTselection of tail/probability/checkpoint.
This is an exploratory single-seed screen,not a statistical replication or an
exact reproduction of original TIGER:quantizer,optimizer/training duration and
our pair input extension differ.

## Compute and limits

CPUpreflight real backbone,supported mathSDPA teacher-forced4code forward,FMA2:
Recent20.427088GF,Tail3.257706GF(−39.66%),Tail5.278190GF(−34.86%),Recent3.082201GF.
Those are actual FIXED PADDED widths,not means of nonpadding lengths,not beam
serving latency,and exclude backward and text embedding/cache costs. Save
first/warm/total TRAINtime,VALtime,actual per-epoch draws/padding/hash,peakmemory,
AMPskips and each model's profiler result. Pair embedding/cache build is an
additional measured preparation cost. Beauty TRAINhas only7%fewer examples
than our currentKindle;VALis23.3×larger,so shorter histories do not guarantee a
faster whole run with this fixed padded implementation.

Estimate2–4hours after bootstrap depending on convergence,cacheencoding and
validation,not a measured benchmark. Fitwalllimit10000s,suite/hardlimit21600s;
if a fit hits a time limit,keep its checkpoint and mark incomplete. No telemetry
study,no additional seeds/architectures/HPO/notifications.

## Validation and reproduction

Config`experiments/beauty_pairs_2t4.json`;module`rhpr.gr.beauty_pairs`.
CPUpreflight`preflight_beauty_pairs.py`,evidence`gr-beauty-pairs-preflight.json`.
Preflight uses EXPLICITLY SYNTHETIC unique SIDs for geometry and actual backbone
finite forward/backward;it does not pretend a fitted tokenizer exists locally.
Actual GPU RQfits,index reports and suffix counts remain runtime checks.
Input catalogue/sequence SHA256verified against historical Beauty config.
Tests cover lengths0..20,all native budgets,tail/order/suffix preservation,
random adjacency candidate coverage,missing-pair failure,short Binomial counts,
microbatch replay and original one-event TRAINinputs/VAL/TEST alignment.

Full regression:1160tests passed,152optional skips,0failures,102.646seconds.
Seven focused Beauty tests passed. CPUpreflight matched actual dataset digests,
all65080candidate pairs,all native widths,finite real-model forward/backward
and the previous wide-generator initializer. Private offline bundle built.
