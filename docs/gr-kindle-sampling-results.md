# Random20 TRAIN probabilities/curriculum: completed pilot

Collected2026-11-15,private`carasik/rhpr-kindle-sampling/1`,sourcee239144016df.
Three fresh seed42fits completed all60epochs. Runtime4591.732s=76.53min after
bootstrap;no additional GPU training/inference during collection. Original
oldest10eligible/latest10single;all fixed VAL/TESTinputs and indexes unchanged.

| TRAINschedule | Best epoch | VAL NDCG10 | VAL hits10 | TEST NDCG10 | TEST hits10 | Forward GF |
|---|---:|---:|---:|---:|---:|---:|
| p=.25 |55|.01549107|28|.01000435|19|.319748392|
| p=.50,reused |60|.01479621|24|.01564287|29|.319748392|
| p=.75 |60|.01852654|36|.00906937|21|.319748392|
| linear0→1 |60|.01655969|34|.01076096|22|.319748392|
| Recent10,reused |60|.01493143|29|.01462330|26|.217327232|
| Recent20,reused |55|.02049218|38|.01472239|24|.427087952|

All new point estimates improve VAL relative to previousp50,but lose TEST.
Relative TESTdifferences p25−36.0%,p75−42.0%,ramp−31.2%. No established
schedule/curriculum improvement. Changing p has not resolved VAL/TEST disagreement.
Ramp actually reachedp1at60,so its terminal fully compressed phase occurred.
All models hit the cap rather than satisfying plateau;this does not establish
that longer training would rescue them or that more TRAINrandomness is beneficial.

Paired5000draw user-bootstrap deltas against previousp50:

| Schedule | VALdelta [95%CI] | TESTdelta [95%CI] |
|---|---|---|
| p25 |+.000695 [−.006004,+.007312]|−.005639 [−.012316,+.000730]|
| p75 |+.003730 [−.002811,+.010479]|−.006573 [−.012394,−.001317]|
| ramp |+.001763 [−.003631,+.007178]|−.004882 [−.010818,+.000803]|

Only p75TESTinterval excludeszero in these individual exploratory comparisons.
One seed,previously examined TEST,no multiplicity correction and no seed/item
clustering uncertainty in these CIs;do not make a general causal or replication
claim. Same61position inference cost for all schedules is unchanged by TRAINp.

Warm epoch medians36.33s(p25),31.24s(p75),35.48s(ramp);optimizer totals
2542.28/1880.85/2330.27s,respectively. First epochs336.21/32.97/176.99s;
p75was second on its worker and could reuse compile caches. These cold timings
are not independent latency benchmarks. Each fit16558successful updates plus2
AMPskips;no automatic LR/precision fallback. Teacher-forced VALCEwas lowest at
55for all3and worsened at60,despite late NDCG increases in p75/ramp. Thus even
these final curves do not show every generalization measure improving together.

## Verified evidence

Original parent/cache SHA256pins and10input files verified. Frozen121412pair
cache/itemSIDbytes unchanged;zero new text vectors/refits. Original raw cases,
initializer and targets match all compared models. Independently replayed84
saved prediction files,full catalogue item-level metrics and selected epochs.
All180actual TRAINcount/exposure schedules replayed;all3first-epoch full
input tensor hashes/odd counts/padding replayed. Later full input tensors were
NOT replayed;the180count audits must not be described as180tensor-hash audits.
Six full best/last checkpoints downloaded,finite,selected/last epochs and update
counters checked;optimizer/scaler/LR/RNGcontinuation states retained in last.
Checkpoint SHA256and sizes:`gr-kindle-sampling-completion.json`.

Raw evidence,weights,analysis andPNG/PDFcurves:
`/workspace/artifacts/kindle-pilot/sampling-results`.
Aggregate results:`gr-kindle-sampling-results.json`. CPUreproducer:

```
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kindle-pairs-mpl PYTHONPATH=src \
/tmp/unique4-resume-venv/bin/python scripts/research/analyze_kindle_sampling.py \
 --results /workspace/artifacts/kindle-pilot/sampling-results \
 --parent /workspace/artifacts/kindle-pilot/long-pairs-results \
 --cache-parent /workspace/artifacts/kindle-pilot/mixed20-results \
 --data /workspace/artifacts/kindle-pilot/long-input
```
