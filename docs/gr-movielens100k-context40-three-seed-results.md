# Context40 replication: three pipeline seeds

Collected2026-11-04MSK. Replication private `carasik/rhpr-ml100k-context40-replication v1`,
executable`e3308db026871ccec6590ba502b6cce8d65ebd0c`, workflow41682670062,
completed4models in102.45minutes. Reused the completed seed42L40models from
`carasik/rhpr-ml100k-context30-40 v1`, executablebfecf91e…. Seed42was not retrained.
No new GPU work during collection.

## Main result: large seed42test effect did not replicate

Unmasked full-catalogue testNDCG10, same30epoch budget and selection protocol:

| Seed | Control | Actor graph .1 | Graph/control |
|---|---:|---:|---:|
|42|.045638|.058873|+29.00%|
|101|.048617|.047589|-2.11%|
|202|.046563|.048120|+3.34%|
|Mean|.046939|.051527|+9.77%|

The apparent three-seed test advantage is largely driven by seed42. On the two
new seeds alone, control=.047590, graph=.047854, difference**+0.56%**. Thus the
initial29%gain is not a repeatable effect of that magnitude in this replication.
It is also not evidence of equivalence or proof graph features cannot help.

Three-seed test standard deviation:control.001524,graph.006367. These are
pipeline seeds: each has its original frozen seed-specific index and its own
fresh generator initialization, not just generator randomness on a common SID.
The sources of this variance cannot be separated here.

## Validation remains mixed and slightly favors control

| Seed | Control validation | Graph validation | Graph/control |
|---|---:|---:|---:|
|42|.070235|.065990|-6.04%|
|101|.067013|.059357|-11.42%|
|202|.066408|.072542|+9.24%|
|Mean|.067885|.065963|-2.83%|

Graph wins validation only on202. This split disagreement is not explained
solely by best-checkpoint selection: fixed validation epochs20/25/30also have
negative graph/control mean differences(-4.24%/-8.35%/-4.70%). Fixed-check
results and every per-seed trajectory are preserved, without selecting a new
checkpoint or touching test selection.

Conditional shared-user bootstrap95intervals(20000resamples, fixed6fitted models):
- Three-seed graph-control test delta.004588:[-.00217,+.01133].
- Three-seed graph-control validation delta-.001922:[-.00920,+.00540].
- New101/202only test delta.000264:[-.00778,+.00841].
All cross0. These do not account for training-seed uncertainty, repeated
holdout use, selecting window40after exploratory results, or multiplicity.
We have no established overall graph advantage or stable harmful-graph rule.

## A smaller descriptive lead: graph benefits from40vs20on validation

Original20window validation is restricted to the same first30epochs and the
same14check schedule, using the same corresponding frozen index/seed.

| Seed | Graph20validation | Graph40validation | Change |
|---|---:|---:|---:|
|42|.063452|.065990|+4.00%|
|101|.054704|.059357|+8.51%|
|202|.068876|.072542|+5.32%|
|Mean|.062344|.065963|+5.81%|

Direction repeats3/3, but the absolute mean improvement .003619 has conditional
user-bootstrap interval[-.00380,+.01112]. This is a promising descriptive lead,
not a proven context improvement. The control does not show the same pattern:
20->40changes+5.52%/-2.93%/-6.24%, mean.068808->.067885(-1.34%).
Do not compare the three-seed20test mean against40as a matched30epoch effect:
several original20test checkpoints were selected after30epochs. No missing
20test checkpoints were reconstructed or evaluated in this collection.

## Repetition and throughput

Selected-checkpoint mean known-item occupancy of top10at40:

| Split | Control | Graph |
|---|---:|---:|
|Validation|39.95%|40.22%|
|Test|40.23%|40.28%|

At matched20validation selection, corresponding fractions were41.11%/42.99%.
Longer context reduces average repetition somewhat, but it remains around4of10
recommendations and is nearly equal between models at40. All evaluation ranks
remain unfiltered; these statistics do not identify a popularity-bias cause.
Known past excludes embargoed same-timestamp events, adds only selected
validation item for test. Single heldout item and rating-time ambiguity remain.

New model mean training epoch times:control10199.48s/control20299.94s,
graph10195.91s/graph20296.07s. Prior42control92.86s/graph97.83s.
Peak PyTorch allocated fit+validation memory1.839GiB for all six models;
this excludes some CUDA context/library allocations and is not reserved memory.
Effective batch256/physical128unchanged. No memory, precision or hardware fallback.

## Protocol and integrity

Four new predictors start from the corresponding original constructor seeds,
using frozen original SIDindices; no AE/RQ/graph fitting or old predictor weights.
All sixL40models use eagerT5attention, FP16AMPtraining, FP32unmasked beam20eval,
30epochs, first20val5/10/15/20, verified optimizer/scaler/RNGresume, then val
21–30and the original LR schedule. Each has10020updates/14checks/0AMPskips.
Test once at selected validation checkpoint. New selected epochs:control10125,
graph10128,control20226,graph20228; prior42control27/graph30. These are capped
screening runs, not full convergence demonstrations. No new fits pending.

Independent audit rescored102distinct prediction files to1e-12, including all
84validation and6test outputs across the sixL40models, six matched20selected
validation references and six original20epoch30references. Config/gitSHAs,
all943user/target orders, train/validation/test schedule hashes,85470training
targets, shared recent20suffixes, frozen SID bytes, initial parameter hashes,
30epoch curves/updates/check cadence and resume proofs verified. All six test
outputs have0invalid paths/duplicates.956repository tests passed(91skipped).

Caveats: all seeds share previously examined heldouts. More context also changes
fitting for short-history users. With the current fixed left padding and prepended
user token, changing maximum width changes user-to-item relative position biases,
even for unchanged active item histories. We cannot attribute the descriptive
20->40gain solely to useful old events.

A cheap next diagnostic, if requested, would evaluate the frozen40models with
older events masked while retaining the40slot layout and user-token position.
That probes reliance on older events without new training; it would still be
an inference ablation, not an independent holdout. No such inference run was
started in this collection.

## Evidence

Numerical summary:`docs/gr-movielens100k-context40-three-seed-results.json`.
Workspace`artifacts/movielens100k-context40-replication/analysis` links to the
current temporary analysis cache:metrics/aggregates/contrasts/curves/epoch30/
fixed_epochs JSON+CSV, per-user cases.csv.gz, SHA256verification and replay.
Raw large checkpoints remain on the private completed Kaggle runs. Temporary
cache/ZIP may not survive environment replacement; report/summary are committed.
Original seed42study remains unchanged. No Telegram, new dispatch or tuning.
