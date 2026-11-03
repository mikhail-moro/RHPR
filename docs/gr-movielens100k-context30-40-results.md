# Frozen-index context30/40 pilot: completed results

Collected2026-11-03MSK. Private `carasik/rhpr-ml100k-context30-40 v1`, executable
`bfecf91e658d24415255b480102c9f94e09fcc59`, workflow41662734850.
All4fresh predictors completed30epochs; runner86.08minutes on2T4.
No failures, AMP skips, invalid test predictions or duplicate test predictions.
No new GPU work was started during collection.

## Comparable validation selection through30epochs

Same fixed seed42indices, predictor initial weights, targets, user order,
optimizer schedule, effective256/physical128batch and unmasked beam20FP32eval.
TrainingFP16AMP, original eagerT5attention. No AE/RQ/graph retraining.
Original20window selection below is restricted retrospectively to the same
checks5/10/15/20and21–30; the selected original20graph epoch is24, not31.

| Context items | Control validation NDCG10 | Actor.1 validation NDCG10 |
|---|---:|---:|
|20|.066559|.063452|
|30|.065442|.064609|
|40|.070235|.065990|

Control40vs20+5.52%; graph40vs20+4.00%. All conditional user-bootstrap intervals
for these validation context contrasts cross0. Graph loses to control at all
three validation windows. At40the graph/control difference is-6.04%.
This does not establish a consistent benefit of graph features or longer
context. A single pipeline seed cannot measure training variance.

## Test of the four new equal-budget fits

| Context items | Control test NDCG10 | Actor.1 test NDCG10 | Graph/control |
|---|---:|---:|---:|
|30|.047900|.043613|-8.95%|
|40|.045638|.058873|+29.00%|

For40, absolute graph/control test difference .013235 has conditional
shared-user bootstrap95interval[.00152,.02569];78users improve,60worsen,
805tie. For the graph40vs30, test gain34.99%, interval of absolute difference
[.00465,.02594];77improve/52worsen/814tie. These intervals condition on the
already-fitted models and use10000user resamples; no training variance or
multiple-comparison adjustment. Validation sign reverses, holdouts have been
examined repeatedly, and this is one seed. The large test point estimate is
an exploratory signal, not a validated graph improvement.

Historical20window test control=.045298(selected epoch26), graph=.045935
(selected epoch31). The graph is NOT a30epoch-matched test baseline: its
first30validation budget would select epoch24, whose test predictions were
not evaluated/saved. Do not mix graph24validation and graph31test as if from
one selected model. The control's selected epoch26matches the first30budget,
so its test checkpoint can be compared:30vs20+5.74%,40vs20+0.75%; both
conditional intervals cross0. No additional20window training was launched.

Selected epochs: control30=27,graph30=27,control40=27,graph40=30. The last case
peaks at the cap;30epochs does not demonstrate convergence. No extension is
currently authorized or pending.

## Repetition

Known means all training events plus selected validation item for test,
excluding embargoed same-timestamp-group events. Fractions count top10slots.
The evaluation remains unfiltered; these are diagnostics only.

| Context | Control known-slot fraction/test | Graph known-slot fraction/test |
|---|---:|---:|
|30|43.34%|42.32%|
|40|38.82%|40.25%|

Historical20selected tests:42.84%control/43.89%graph, with the graph budget
mismatch noted above. At selected validation checkpoints, fractions are
43.15/44.67%for20,42.86/42.44%for30,38.84/40.17%for40(control/graph).

At a fixed validation epoch30, control known slots decrease42.10->41.40->41.08%
for20/30/40; graph43.01->42.38->40.17%. Thus direction is not solely an effect
of choosing a different best checkpoint, although selected-checkpoint gaps are
larger. Longer context has not removed repetition: roughly4of10recommended
movies are still already known. It also does not establish a PopBias cause.

## Throughput and memory

| Context/candidate | Mean training seconds/epoch | Peak allocated GiB |
|---|---:|---:|
|20control,first30epochs|51.05|not recovered|
|20graph,first30epochs|51.16|not recovered|
|30control|66.73|1.333|
|30graph|69.44|1.333|
|40control|92.86|1.839|
|40graph|97.83|1.839|

No batch or precision fallback. Relative to matching first30epoch timing,
30is1.31–1.36xslower and40is1.82–1.91xslower. The earlier operation-scaled
projection was conservative for wall time. Peak is torch.cuda.max_memory_allocated
over fitting INCLUDINGvalidation/selection, not total device usage or reserved
memory; CUDA context/library allocations may be outside it. No FlashAttention.
Total runtime includes startup, validation, checkpoint IO and test.

## Scope and mechanism limits

The extra context did not produce a monotonic quality improvement. Graph40has
an interesting test result, whereas validation prefers control40and the graph
advantage changes sign across windows. Save this as a useful screening result,
not a reason to declare an adaptive gate or a general graph benefit.

All4history-length bins(<=20,21–30,31–40,>40)are retained in history-cohorts.json.
There is no clean rule supported on both splits. Graph40also improves some
users whose histories already fit in30; effects therefore cannot be attributed
solely to newly visible events at inference. Training with more context changes
the fitted model for everyone. Also, the existing left-padded T5encoder prepends
a user token before padding: changing maximum width changes relative distances
from that token to active item tokens, even when active history IDs are equal.
That is another reason not to claim a proven long-history mechanism from this
comparison. Relative item-to-item distances in the shared recent suffix remain
unchanged. No positional-layout fix was made in this study.

## Verification and evidence

Independent CPU audit reconstructed64distinct prediction files(68scoringcalls),
all56validation files and4test files of this run plus original reference files.
NDCG matched recorded values to1e-12. Exact config/git revision,943user/target
orders, original reference hashes, frozen SID bytes, matching initial parameter
SHA,30epochs/10020optimizer updates/14validation checks per model, zero AMP
skips and optimizer/scaler/RNG resume proofs were verified.955repository tests
passed(91skipped); no production code changes in this collection.

`docs/gr-movielens100k-context30-40-results.json` preserves metrics, contrasts,
fixed-epoch repetition, subgroup tables and provenance. Workspace evidence is
`artifacts/movielens100k-context30-40/analysis` pointing to the current temporary
cache. `cases.csv.gz` stores11316paired user/model/window/split rows, including
historical20references with budget flags in metrics. Source checkpoints and raw
outputs remain on private Kaggle; temporary ZIP/cache may disappear between
execution environments. Report and numerical summary are committed to GitHub.
Replay source is included in the evidence archive; it expects the recorded raw
run/parent paths and prepared sequence input, without model calls or GPU.
