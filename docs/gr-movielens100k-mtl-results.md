# MovieLens100K: three-modality MTL screen, seed42

Collected2026-11-04 MSK. Private carasik/rhpr-ml100k-mtl v1 COMPLETE;
executable5af82b106c3d2687459dcb9ebc4cc2e4e1547415. Four fresh pipelines,
metadata Sentence-T5 + frozen poster + description-only MPNet, no people.
Runner63.60minutes on2T4. No new training/inference submitted during collection.

## Validation-selected quality

| Method | Val NDCG10 | Test NDCG10 | Test vs sum | Recall10 | Best epoch |
|---|---:|---:|---:|---:|---:|
|sum|0.05998|0.04364|+0.0%|0.09332|30|
|uw|0.06475|0.03927|-10.0%|0.08590|30|
|gradnorm|0.05762|0.04444|+1.8%|0.09862|25|
|pcgrad|0.06001|0.04015|-8.0%|0.08590|20|

UW wins validation(+7.95%) but loses test(-10.01%). GradNorm has only+.000800
absolute test gain, while losing3.93%validation. PCGrad nearly ties validation
(+.05%) and loses7.99%test. No robust benefit established, nor general MTL harm.
All5000draw conditional paired-user95%intervals cross0: UWtest[-.01414,+.00494],
GradNorm[-.00953,+.01117], PCGrad[-.01415,+.00652]. These exclude pipeline-seed
uncertainty, target clustering, multiplicity and previously repeated holdout use.
Do not select GradNorm using its favorable test score; validation would select UW.
Sum and UW select epoch30boundary, so this is a fixed30epoch screen, not convergence.

## What the tokenizer diagnostics show

- All13sampled probes/method have positive pairwise cosines on the LAST shared
  encoder Linear weight. Sum39pair probes:min.708,median.933. Allmethods combined
  minimum.437. This is neither all updates nor all shared layers.
- PCGrad instantaneous projection count is0on all13logged steps (measured over
  its whole all-task-shared coordinate mask). Counts between probes were not
  aggregated, so do not claim PCGrad never projected or conflicts never occurred.
  Earlier differences can propagate into later training despite zero sampled counts.
- Final task weights in text/poster/plot order: GradNorm .952/.962/1.086;
  UW1.218/1.162/1.173. GradNorm modestly favors plot. UW weight sum3.553 vs3control
  (+18.43%): predominantly a global reconstruction-scale change; relative weights
  normalized to mean1 are1.028/.982/.990. UW is not a pure relative-balancing ablation.
- Raw normalized MSE(text/poster/plot): sum .7885/.8132/.8385;
  UW .7810/.8215/.8335; GradNorm .7875/.8266/.8287;
  PCGrad .7848/.8223/.8377. No uniform reconstruction winner or validated NIP proxy.
- Unique semantic triples sum1272, allothers1273; usedL1codes256/255/256/256.
  Full4token SIDs allunique, no dropped films. Occupancy does not rank NIP quality.

The sampled gradients give no evidence of persistent directional opposition in this
three-modality recipe. They do not rule out finite-capacity tradeoffs or objective
mismatch between reconstruction and next-item prediction. Adaptive reconstruction
balancing does not directly learn which modality is useful downstream.

## Cost and verified protocol

Tokenizer(AE+RQ)seconds sum216.6,UW221.2,GradNorm266.9,PCGrad374.9.
TIGER training epochs average51.5–52.9seconds; unchanged generator/input shape.

Every fit: fixed3500AE +3000nativeRQ steps,3x256/latent64, fresh native RQ optimizer,
untouched AE books and verified neural transfer. Same frozen feature bytes and masks,
normalization, per-seed AE and generator initialization pins. Same1273films/943users,
85470strict-past cases/epoch; input timestamp-group split reconstructed and checked
against all3case digests. History20,30epochs,10020updates,0AMPskips, six validations
at5/10/15/20/25/30, constantLR.001, FP16train/FP32eval, unmasked freebeam20.

Independently decoded/rescored all28prediction files (24validation+4test), retaining
invalid/duplicate ranks. Replayed best-VAL checkpoint selection. UWtest has6invalid
paths/18860; others0, all0duplicates. These counts match saved metrics. Every semantic
map was reconstructed from raw codes incl suffix; book counts match assignments.
Bothworkers COMPLETE with no failures. Source SHA/config match the submission.

Logging limitation: the dedicated mtl-learning-curves.json was not exported by the
submitted code. All52MTL events are present in worker stdout and recovered into
analysis/mtl-learning-curves.json, including13points per method. No rerun needed.
No runtime code changed during this collection; CPU replay invariant checks pass.

## Evidence

Replay: scripts/research/analyze_movie_mtl.py INPUT OUTPUT --sequences sequences.json.
Fullsummary docs/gr-movielens100k-mtl-results.json. Local outputs /tmp/mtl-completed;
analysis /tmp/mtl-analysis, including CSV,PNG/PDF, hashes, curves and recovered MTL probes.
Portable numerical ZIP /tmp/movielens100k-mtl-numerical-evidence.zip includes source
replay and canonical sequences; large model weights remain private Kaggle.

## Follow-up: timestamp target-choice sensitivity

The original split was intentionally retained for comparability, including SHA256
selection of one movie from a tied rating-second group. This limits the official
MTL result too. Validation has417/943multi-film groups;test409/943. Other group
members were embargoed rather than put in context. This avoids same-second leakage
but leaves an arbitrary choice of the single scoring target.

Using original selected checkpoints and saved rankings, average ordinary single-target
NDCG over ALL possible single target choices in each group, then over943users:

| Method | Official test NDCG10 | Uniform target expectation | Relative to uniform sum | Singleton test only n534 |
|---|---:|---:|---:|---:|
|Sum|.04364|.04256|—|.04602|
|UW|.03927|.03698|-13.1%|.03445|
|GradNorm|.04444|.04977|+16.9%|.05306|
|PCGrad|.04015|.04082|-4.1%|.04193|

GradNorm's fixed-checkpoint test effect rises from+1.8% to+16.9%;on test singletons
it is+15.3%. This is a larger lead worth recording, not confirmed training benefit.
UW remains adverse on test despite favorable validation. Uniform expected VAL scores:
sum.05887,UW.06567(+11.6%),GradNorm.05788(-1.7%),PCGrad.05970(+1.4%). The val/test
ranking disagreement therefore persists; target choice alone does not remove it.

GradNorm uniform expected TEST delta+.007211,conditional paired-user95%CI
[-.001860,+.016070], still includes0. Allother comparison CIs also cross0. Across
5000shared uniform label draws GradNorm beats sum99.56%of draws, but this is
conditional label sensitivity on these same fitted models/users, NOT a p-value,
training replication or population confidence. Both statistics are saved separately.

Checkpoints were selected on ORIGINAL single-target validation, and test histories
retain ORIGINAL selected validation items. This diagnostic is NOT a replacement
benchmark, new checkpoint-selection protocol or averaged alternative-history inference.
No retraining/inference/GPU work. Singleton results also retain original checkpoints.
Timestamp groups denote rating submissions, not observed watch order.

Replay scripts/research/mtl_timestamp_targets.py; outputs
 docs/gr-movielens100k-mtl-timestamp-targets.json. The full numerical bundle now also
contains embargoed-events.json and timestamp_replay.py for portable CPU replay.
