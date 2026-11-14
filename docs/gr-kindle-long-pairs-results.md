# Kindle long-pair pilot: completed results

Two new wide Pair32/64 generators and read-only TEST evaluation of existing
wide Recent20/32 controls completed in102.495minutes on2T4. All four models
selected epoch55 by raw VAL NDCG@10 and hit the60epoch cap. Same previously
examined seed42,item SID,targets,initial weights,wide4+4 generator and training
policy. No new item index fit or control training. No followup compute launched.

| Model | VAL NDCG@10 | TEST NDCG@10 | TEST Recall@10 | TEST top10 hits/958 | B1 GFLOPs | Warm epoch,s |
|---|---:|---:|---:|---:|---:|---:|
|recent20|0.020492|0.014722|0.025052|24|0.427088|31.25|
|recent32|0.016899|0.014447|0.029228|28|0.704770|55.84|
|pair32|0.015966|0.010651|0.021921|21|0.516501|37.59|
|pair64|0.018043|0.014054|0.033403|32|1.119082|87.41|

## Interpretation

No observed Pareto improvement. Pair32 vs Recent32 saves26.71% supported
forward FLOPs and32.67% warmed epoch time,but loses5.52% selected VAL NDCG
and26.27% TEST NDCG. Against Recent20 it costs20.94% more FLOPs and loses
27.66% TEST NDCG. Thus the point estimates do not support replacing simple
shorter history with this pair representation.

At evaluation Pair64 adds the compressed older32 items to the SAME uncompressed
recent32 as Recent32;training allocations vary with each prefix length. Selected VAL is+6.77%,TEST is−2.72%,at+58.79% FLOPs and+56.55%
warmed training time. Against Recent20,TEST is−4.54% at2.62times the FLOPs.
Pair64's TEST Recall@10 is higher(32hits vs28/24),but it does not improve NDCG;
this sparse one-seed result does not establish a ranking or recall mechanism.

Pair64 vs Pair32 is+13.01% selected VAL/+31.95% TEST NDCG,but doubles both
raw context and the uncompressed recent tail16→32 while increasing compute.
It does not isolate useful compressed history. No matched wide Recent64 was
trained,so these results do not estimate compression loss against that model.

All selected-VAL and TEST conditional paired-user intervals include0. Across
all30 descriptive comparisons,29include0;fixed40 Pair64−Recent32 is negative
with CI[−.010913,−.001085],without a multiple-comparison correction.
TEST Pair32−Recent32 delta−.003796,
95%CI[−.009507,+.001372];Pair64−Recent32−.000393,
CI[−.007115,+.005916];Pair64−Pair32+.003403,
CI[−.002532,+.009394].5000 paired958-user resamples,conditioned on these four
trained fits/checkpoints;exclude generator/index randomness,item clustering,
selection and multiple comparisons. Lack of significance is not equivalence.
Only21–32TEST top10 hits/model;no seed-robust benefit or harm established.

## Learning and index checks

All4models selected55 and ran60epochs,so unequal observed epochs do not
explain the selected comparison. Train curves are very close:last CE
Recent20/32=8.0422/7.9731,Pair32/64=8.0538/7.9963. Minimum observed teacher-forced
VAL CE is9.1083/9.0768/9.2216/9.1487. Pair32 minimum is at60,the others at55.
No plateau or converged optimum demonstrated;no evidence that extra epochs
would selectively rescue pairs. Four AMP skips in two new fits,zero failures.

The pair tokenizer used all256codes at every semantic level and produced
116401unique3+1 tuples,max suffix31(max collision group32). No overflow or
obvious dead-code collapse. This does not prove preservation of useful semantics.
TRAIN-only fitting and split-priority uniqueness suffixes independently replayed.
All38parent artifact pins,source/config/data/SID/initializer/case and target pins,
LR/EMA/update accounting verified. All52prediction files(48VAL+4TEST) replayed:
996320paths,1866invalid,0duplicates. Original ranks preserved,no seen filter.
Two new best and two new last checkpoints confirmed in private Kaggle output;
control best weights remain in the pinned private parent.

## Practical decision and limits

This pilot does not support the specific claim that this description-pair merge
removes enough SID noise to improve downstream NDCG or yield a better Pareto
point. It also does not prove that attention distance is the cause,that older
history is useless,or that every possible compression method fails. Merging
changes representation semantics,positions,length and active input embeddings
simultaneously. Do not run Control64 merely to seek a desired positive outcome;
first decide which unresolved question would justify its cost.

Same141137train examples and958VAL/TEST targets as parent. TEST opened only
for these four predeclared VAL-selected checkpoints,not for optimization. Later
experiments on this already inspected cohort are exploratory. No additional
seeds,epochs,control64,TEST variants or notifications launched.

FLOPs:CPU math-SDPA,B1 full-context teacher-forced4code forward including output
heads,FMA=2,supported profiler ops;exclude beam/backward/text embedding. Cached
ST5/KMeans preprocessing also costs time/storage. Historical control timings
are not a simultaneous controlled hardware or serving-latency benchmark.

Reproduce with `scripts/research/analyze_kindle_pairs.py --results RESULTS
--parent CAPACITY_RESULTS --data LONG_INPUT --output OUTPUT --plot`.
Private raw evidence:`/workspace/artifacts/kindle-pilot/long-pairs-results`.
Plots:`pair-curves.png`/`.pdf`;analysis.json and model-metrics.csv saved.
