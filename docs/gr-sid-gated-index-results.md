# Completed conditional original/ALS SID experiment

Sourcecommit`45d7dab1d35a55821585e17423f32eb197a5cce3`,private Kaggle session
`carasik/rhpr-gated-sid-tiger-screen`. Both matched models completed20epochs,
2628260examples,10280optimizer updates,zero AMPskips. Both have4978176parameters
and the same SHORT/LONGencoder feature. Control uses original paths in both
branches;gated uses the exact previous ALSpermutation at10+available events.
Body1755.8seconds,about29.3minutes. No new inference,training or test evaluation
was performed to analyze the completed outputs.

## Fixed20epoch endpoint

| Metric | Original+branch feature | Gated original/ALS | Relative change |
|---|---:|---:|---:|
|NDCG@10|0.03587286|0.03603699|+0.46%|
|NDCG@5|0.02729040|0.02769763|+1.49%|
|Recall@10|0.06846130|0.06940035|+1.37%|
|Recall@5|0.04185485|0.04341993|+3.74%|

NDCG@10absolute delta+0.00016414;paired user95%interval
[-0.00111021,+0.00145259],target-cluster sensitivity[-0.00180354,+0.00245069].
847users improve,821worsen,20695tie(often both miss top10). Neither superiority
nor statistical equivalence is established. The other metrics are descriptive
supporting endpoints,not separate confirmed positive results.

| TRAIN events | Users | Control NDCG@10 | Gated NDCG@10 | Change |
|---|---:|---:|---:|---:|
|3|7162|0.034684|0.036093|+4.06%|
|4|4221|0.032800|0.033082|+0.86%|
|5–9|7433|0.036084|0.034655|−3.96%|
|10–19|2528|0.039274|0.041599|+5.92%|
|20+|1019|0.046972|0.044165|−5.98%|

Branch aggregates:<10has18816users,+0.10%;10+has3547users,+2.05%. These
overlap the finer bins. The model observes min(TRAINlength,20)events. This is
a comparison of different users/targets,not an identical-user context truncation.

All cohort user and target-cluster95%intervals include0. For10–19the userinterval
is[-0.001919,+0.006793];for20+[-0.009480,+0.003571]. Bootstrap2000replicates,
seed2748+cohort index,paired across models. Target-cluster resamples entire
target-item groups and uses their variable-size mean. Intervals are conditional
on one fitted pair,unadjusted,and exclude training-seed variability.

## Dynamics and historical references

| Epoch | Control NDCG@10 | Gated NDCG@10 | Change |
|---|---:|---:|---:|
|5|0.025634|0.026237|+2.35%|
|10|0.030124|0.031425|+4.32%|
|15|0.034657|0.033860|−2.30%|
|20|0.035873|0.036037|+0.46%|

The sign changes. Three positive checks share the same fitted trajectories and
validation users,not three independent replications. The primary endpoint is20,
not the earlier epoch with the largest relative gain. On20+the relative sequence
is+3.46%,+14.56%,−0.52%,−5.98%at5/10/15/20.

Earlier unconditioned original/ALS20fits scored0.039276/0.044327on20+;current
tagged control/gated score0.046972/0.044165. The gated point value is close to
the earlier pureALSvalue;the current control is stronger. These historical
references differ in conditioning feature,full initialization/RNGtrajectory and
fitted model. They do not isolate a causal cost/benefit of the branch token or
joint mapping. Overall the previous unconditioned original0.036943is above both
current fits;this screen does not establish a new best baseline.

## What actually changes under the gate?

Only480of12101item-to-SIDassignments differ,about3.97%of catalogue. In validation
the long branch has just116targets from74distinct items whose target SIDdiffers.
At20the control has7top10hits there,the gated model4;NDCG10is0.028001versus
0.014473. These counts are too small for a strong mapping-specific quality claim.

Among long users,1538have a changed visible input or target SID:control/gated
NDCG10=0.039221/0.037175. The remaining2009have identical input/target SIDpaths:
0.043219/0.046288. This shows that the positive long-branch mean is not a direct
local improvement on the changed-path subgroup at this endpoint. Global shared
weights can legitimately alter unchanged-path predictions;this is not evidence
that the gate is ignored or that all differences are noise.

Similarly short-branch input and target mapping remain original in both models,
but shared parameters train on different long-branch labels/inputs. Hence the
gate does not preserve short-user quality identically:3event users improve in
point estimate while5–9decline. The nearly unchanged short mean hides that split.

The implementation works with these close mappings and similar aggregate point
quality. Evidence for a useful learned conditional layout remains limited. Do not
generalize this result to two arbitrary unrelated SIDtrees or select another
threshold from these validation cohorts without treating that as tuning.

## Independent verification and artifacts

`scripts/analyze_sid_gated_index.py` checks exact config/source,shared neural
initialization,raw TRAINcase/order hashes and branch-assignment hash. It compares
the exact downloaded mapping bytes against the pinned private input and checks
the two-matrix hashes in each conditional-index record.

For each of eight saved validation files it recomputes gate IDs from the observed
context length,checks all users/targets/stored branch IDs/lengths and decodes each
row using its active mapping. Full SIDequality independently reproduces all
NDCG/Recall@5/10and branch NDCG10to1e-12. Independently packed4token paths also
reproduce invalid and duplicate counts. Source/case/table mismatch and branch-
unaware scoring are ruled out by these checks,not assumed from logged metrics.

Artifacts:`/workspace/artifacts/sid-gated-index/analysis/`:

- `README.md`,`validation-and-length.png`:main tables,trajectories and intervals.
- `cohorts.json/csv`,`paired-epoch20.json/csv`,`epoch-stability.json`:all cohorts,
  endpoints,intervals,signs and contributions.
- `user-metrics.npz`:all8per-user ranks,branch IDs,history lengths,target TRAIN
  support and whether the input/target SIDchanges. Rank999means absent from
  saved final20beams,not an exact catalogue rank.
- `structural-subsets.json`:changed-path groups and both mappings' SIDcoverage
  on the116changed-target cases. Other-mapping SIDnames a different active item;
  these small counts are not an ignored-gate diagnosis.
- `training.json`,`verification.json`:budgets and numerical/data identity checks.

Small source/suite/download/stream metadata are retained in the parent artifact
directory. All45selected numerical/report outputs(25.98MB)are cached in
`/tmp/rhpr-gated-index-completed`;authoritative outputs and four best/last model
states remain on private Kaggle. Checkpoint filenames were confirmed;large
checkpoint contents were not downloaded or independently validated.

Reproduce with `PYTHONPATH=src`:

```bash
MPLCONFIGDIR=/tmp/rhpr-mpl-cache python scripts/analyze_sid_gated_index.py --plot
```

The script accepts completed/data/assets/output path overrides. Collector:
`/workspace/artifacts/sid-gated-index/collect.py`,using a fresh official output
listing and transient-read retries only. No new GPUjob,retry,threshold change,
test evaluation or external notification is authorized/launched by this analysis.
