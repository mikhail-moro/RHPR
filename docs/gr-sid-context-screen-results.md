# Completed context-ALS screen: validation by history length

Kernel`carasik/rhpr-context-als-tiger-screen`,sourcecommit
`9dcee38f1fe77cff18daf093246c7c81e9dc0751`,completed all four20epoch fits.
Body3640.4seconds,about60.7minutes. All consumed2628260TRAINexamples,
10280optimizer updates,zero AMPskips. Shared initial parameter/case/order hashes
and exact original/swapped input bytes verified. No test evaluation. All16raw
validation predictions were downloaded and their user/target order,NDCG@5/10
and Recall@5/10 independently matched to training curves. No new inference,
training or GPUjob was used for this analysis.

## Primary endpoint: epoch20

Rows are different users grouped by **number of TRAIN events before validation**.
Each user has one validation target. The model observes min(length,20)events,
so20+does not mean processing the complete long history. This is an observational
cohort comparison,not a controlled truncation of identical users/targets.

| TRAIN events | Users | Original | Item ALS | Context20 | Context5 |
|---|---:|---:|---:|---:|---:|
| All |22363|0.036943|0.035149|0.035796|0.034211|
|3|7162|0.037025|0.033146|0.035464|0.034184|
|4|4221|0.035346|0.034760|0.034424|0.032615|
|5–9|7433|0.037023|0.035110|0.034895|0.033955|
|10–19|2528|0.038200|0.037888|0.037906|0.034965|
|20+|1019|0.039276|0.044327|0.045146|0.041013|

NDCG@10. Relative changes against the same concurrent original:

| TRAIN events | Item ALS | Context20 | Context5 |
|---|---:|---:|---:|
|All|−4.86%|−3.10%|−7.39%|
|3|−10.48%|−4.22%|−7.67%|
|4|−1.66%|−2.61%|−7.73%|
|5–9|−5.17%|−5.75%|−8.29%|
|10–19|−0.82%|−0.77%|−8.47%|
|20+|+12.86%|+14.95%|+4.42%|

Thus the positive direction is localized to20+at this endpoint. This cohort is
only4.56%of users. Context20's contribution from it is+0.0002675to the all-user
NDCGmean;losses elsewhere outweigh that contribution,total−0.0011467.

## Uncertainty and controls

Paired2000replicate user bootstraps,seed2740+cohort index;separate target-cluster
sensitivity resamples entire target-item groups while preserving all users of
that target. Intervals are exploratory/unadjusted and conditional on fitted
weights. They do not capture training-seed variation or establish a causal
effect of history length. Checkpoints are not independent replications.

| Comparison on20+ | Absolute delta | Paired user95% interval | Target-cluster95% interval |
|---|---:|---|---|
|Item ALS−original|+0.005050|[−0.001771,+0.011816]|[−0.004579,+0.014822]|
|Context20−original|+0.005870|[−0.002388,+0.013701]|[−0.006056,+0.019209]|
|Context5−original|+0.001736|[−0.005516,+0.008655]|[−0.006901,+0.010559]|
|Context20−Item ALS|+0.000820|[−0.006941,+0.008674]|[−0.009981,+0.011478]|

All these long-cohort intervals include zero. Context20has57better/50worse/
912equal users versus original;ties generally include both missing top10.
Against Item ALS it has54better/55worse/910equal,only+1.85%mean advantage.
Consequently the+14.95%versus original does not isolate a context-signal benefit:
the simpler existing ALSswaps show almost the same endpoint direction.

Overall Context20versus original delta−0.001147has user95%interval
[−0.002554,+0.000230]. Context5's−0.002731has userinterval
[−0.004076,−0.001403]and target-clusterinterval[−0.005242,−0.000203].
The overall Context20versus ItemALS advantage+1.84%also has intervals including0.

## Training dynamics on20+

Relative NDCG@10 against original at the same epoch:

| Variant | Epoch5 | Epoch10 | Epoch15 | Epoch20 |
|---|---:|---:|---:|---:|
|Item ALS|−5.94%|−4.59%|+0.62%|+12.86%|
|Context20|−32.51%|−7.68%|−5.11%|+14.95%|
|Context5|+3.68%|+9.78%|+3.20%|+4.42%|

Context20only turns positive at the final checkpoint. Context5has a positive
sign at all four checks in this cohort,but that is a descriptive trajectory
from the same users and fitted weights,not four successful trials. In the full
population Context5is worse at20despite this subgroup direction.

The CPUcontext graph/count-model diagnostics improved,while overall TIGERNDCG
did not. They remain unvalidated quality proxies. The new graph also changes
confidence policy and semantic distortion. No post-hoc routing threshold,
extended fit or new candidate was selected or launched from these results.

## Retained artifacts and reproduction

`/workspace/artifacts/sid-context-screen/analysis/README.md` contains the full
tables and all30paired contrasts. `history-length.png` shows endpoint quality
and intervals;`epoch-history.png` shows all six cohort trajectories.
`cohorts.json/csv` includes NDCGand Recall@5/10for all4models/4epochs.
`paired-epoch20.json/csv` and `epoch-stability.json` retain all intervals,
contributions and signs. `user-metrics.parquet` and `user-metrics.npz` retain
each user's16ranks,history lengths,target TRAINsupport and target/context path
change flags. Rank999means absent from saved final20beams,not an exact rank.

Small run/config/suite/training-stream metadata and observed download hashes
are retained alongside analysis. All73selected outputs including raw beams are
cached in `/tmp/rhpr-context-screen-completed/`;their authoritative source
remains the private completed Kaggle session. All8best/last checkpoint filenames
are confirmed present there;large checkpoint contents were not downloaded or
independently validated. No new GPUwork is needed for this breakdown.

Reproduce with `PYTHONPATH=src` and the project native Python environment:

```bash
python scripts/analyze_sid_context_screen.py
MPLCONFIGDIR=/tmp/rhpr-mpl-cache python scripts/report_sid_context_screen.py
```

Both scripts accept path arguments;analyze`--help` lists data/assets/completed/
output/repository overrides. The read-only collector is retained at
`/workspace/artifacts/sid-context-screen/collect.py`;it consumes a fresh official
Kaggle output listing,checks the sourcecommit and fetches only numerical/report
files,excluding modelweights. No signed download URLs belong in the repository.
