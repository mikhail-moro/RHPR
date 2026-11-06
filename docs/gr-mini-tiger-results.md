# Mini-TIGER calibration: a useful signal, not a validated proxy

The narrow4+4 generator has promising correlation at epoch10, but the result
does not establish a reliable proxy. Correlation depends on training epoch and
index-seed group, and evaluation-user uncertainty alone is substantial. The
whole48-fit experiment failed its time budget; two architectures have complete
eight-index/two-generator-seed coverage and can be analyzed separately.

Private Kaggle: `carasik/rhpr-ml1m-mini-tiger/1`, executable
`44b84d5dbad6f6cc1815d06038cfcf0f3acedd68`.
Submission workflow42109946957 succeeded; actual training ended ERROR.
No new GPU run or automatic retry was made during collection.

## Completion and verification

- 36/48 fits completed:16shallow,16narrow,4compact. Each finished20epochs,
  5920optimizer updates,4VALchecks,151166examples/epoch,0AMPskips.
- The10000-second internal deadline prevented the next compact fit on each
  worker. Two TimeoutErrors were recorded; ten further fits never started.
  The last kernel-log timestamp is10273.65seconds (171.23minutes).
- Compact covers only SUM/UW seed42,so no eight-index correlation is reported.
- Missing top-level metrics.json/summary.md/run-manifest.json is expected from
  the propagated failure and prevents classifying the whole experiment COMPLETE.
  Completed trial selections,curves,checkpoints and per-user predictions survived.
- Independently replayed144mini VALprediction files,plus8full epoch30prediction
  files. User/target identities,all NDCG/Recall@5/@10 values,invalid paths and
  duplicates matched. Frozen SID/reference hashes,initialization pins,case hashes,
  budgets,worker results and generator shapes were checked.
- TEST was not evaluated. Source audit before submission verified all205embedded
  files,configuration,three datasets,two private parent notebooks,T4 hardware,
  disabled Internet and private visibility.

## Correlation with full epoch30 VAL NDCG@10

Each proxy score below averages two independent generator seeds31415/27182.
The eight indexes are SUM/UW/GradNorm/PCGrad at tokenizer seeds42/101. Full TIGER
has only one generator seed per index,coupled to that index's tokenizer seed.

| Proxy epoch | Shallow1+1 | Narrow4+4 | Original full TIGER,early checkpoint |
|---|---:|---:|---:|
|5|0.548|0.143|0.857|
|10|0.571|0.833|0.667|
|15|−0.048|0.690|0.690|
|20|−0.167|0.690|0.833|

The early-full comparator is from the same trajectories as its epoch30target,
whereas mini models use different generator seeds. Its apparent advantage must
not be read as an independent transfer experiment.

At narrow epoch10,the two mini seeds individually yield rho0.857/0.810.
Their average selects PCGrad seed42,the full-target winner,and preserves both
full-target top2indexes. Against best-of30VAL rather than fixed epoch30,the
average rho is0.857. These are exploratory results across several inspected
architectures/epochs;epoch10 was not an independently validated selection.

The seed-group breakdown limits that result:

| Narrow epoch10 | Eight indexes | Four indexes seed42 | Four indexes seed101 |
|---|---:|---:|---:|
|Spearman of two-seed average|0.833|0.800|−0.200|

Within seed101,the proxy selects SUM,the lowest full-target result in that
four-index group (target NDCG regret0.005474). Pooled correlation partly reflects
separation between the two seed groups. Centering scores within these groups
gives descriptive rho0.405;this is not an adjusted causal estimate. Dropping one
index at a time yields pooled rho0.750–0.964,so no single index alone explains it.

Conversely,shallow epoch20has poor pooled correlation but within-group rho1.0/0.4
and centered rho0.738. A statement that shallow models cannot rank tokenizers
would overinterpret these eight coupled references. Nor is the comparison a
clean depth ablation:width,parameter counts and optimization dynamics also differ.

## Evaluation uncertainty

Shared paired-user bootstrap5000draws,seed20261106,resamples the2786VALusers
jointly across all fixed fitted models and indexes. Target scores and proxy
scores are recomputed on each draw.

| Proxy | Conditional95%interval for pooled rho |
|---|---:|
|Narrow epoch10|[−0.024,0.905]|
|Narrow epoch20|[−0.429,0.810]|
|Shallow epoch20|[−0.619,0.643]|

These intervals describe evaluation-user uncertainty only. They exclude
training randomness,index selection uncertainty,target-item clustering and
post-hoc architecture/epoch choice. They are not intervals for generalization
to new tokenizers. Even conditional on the fitted models,evaluation noise can
substantially change the observed ordering.

## Actual speed

| Model | Parameters | Mean training seconds/epoch | Mean20epoch fit including VAL |
|---|---:|---:|---:|
|Shallow1+1|1,633,664|16.96|5.89minutes|
|Narrow4+4,d64|1,375,040|38.82|13.46minutes|
|Compact2+2,partial|1,175,424|19.99|6.92minutes|
|Full4+4,d128,prior runs|4,977,920|57.91|not rerun|

Per-epoch training speedup is3.42xfor shallow and1.49xfor narrow. The full reference
was measured in earlier sessions,so these are historical comparisons,not a
controlled same-session benchmark. Parameter reduction does not imply equal
runtime reduction. Two-seed proxy averaging also doubles compute per index.

## Interpretation and next decision

There is evidence that a small generator can expose a downstream-related signal.
The narrow model's epoch10result is worth retaining as a candidate. A dependable
ranking proxy and a benefit over early full-model evaluation have not been
established. Global correlations alone hide important group effects here.

Further shrinking or joint tokenization optimization would be premature on this
evidence. A bounded next transfer check could fix narrow epoch10in advance and
evaluate the four already-saved seed202indexes,without retraining tokenizers.
Those would be new candidates for proxy selection,but their full reference still
has only one coupled generator seed. Better full-reference replication is needed
for stronger claims. No such follow-up is authorized or launched by this report.

## Reproduction and retained evidence

`scripts/research/analyze_mini_tiger.py --root <download> --data <clean-input>
--output <analysis>` performs read-only CPU replay. An optional `full-validation`
subdirectory supplies the eight original epoch30VALnpz files for conditional
user resampling. Required data are TRAIN/VALcase files;TEST is not opened.

Retained raw numerical outputs/logs,frozen indexes/full-reference curves,
per-user predictions,replay script,summaryJSON,verification inventory,fit CSV,
calibrationPNG/PDF,and a local evidence ZIP. Generator weights remain in the
private Kaggle output and were not downloaded. Repository regression:
1005tests,91optional skips,0failures (87.557seconds).
