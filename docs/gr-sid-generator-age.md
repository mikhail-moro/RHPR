# Teacher age screen:1,5,20epochs

Authorized2026-10-28 after the frozen20epoch search accepted only two within-branch
swaps. Test whether earlier frozen generators permit different reassignment.
This run evaluates search behavior, not downstream NDCG or alternating learning.

## Sources and training

One new original-index generator, same named-neural initialization hash
`5f4a24fa8c9b5f7063739fd44822fd0a5f98eff91c7e882563d32529b0c5abff`, seed42,
data, case order, GRID4+4/d128 and FP16 training protocol as previous original
controls. Fit exactly5 full epochs (657065 examples,2570 updates expected),
saving weights after epochs1 and5. Original epoch20 checkpoint is reused from
`carasik/rhpr-sid-corruption-screen`, SHA256
`deadfd084b77f4a68a0f39097a423d187316b94a9b2a4d1ea563639efbf303c9`.
The three ages have the same seed/order but epoch20 is from an earlier execution;
CUDA numerical nondeterminism is not excluded. Local native integration checks
that optional snapshot saving leaves the first5 training losses unchanged.

Early training uses native validation only at5 to retain the existing trainer's
selection contract. Validation does not select teacher age, candidates or moves.
No new embedding/quantizer fitting, test evaluation, downstream20epoch fits,
semantic penalty, extra seeds, retries, automatic continuation or Telegram.

## Shared candidate pool

For each frozen teacher, use the same16384 sampled TRAIN case IDs, seed2758.
Native greedy valid wrong-item predictions become candidates when both items
have >=2TRAIN target positions and exact teacher-forced full-path confidence
margin >=log1.25. Deduplicate unordered pairs and rank by summed gaps capped at5
nats per contribution, just as in the preceding screen. Keep up to2000 per age.

Merge lists round-robin by source rank, deduplicating pairs; keep up to2000 total.
Use source ranks rather than comparing differently calibrated log margins across
ages. Save each pair's ranks in all source lists. Shuffle the resulting pool with
independent seed2759. If an age mines few/no valid pairs, log it without fallback.

Before any search, choose one common prefix of this shuffled list. For each
proposal, compute the complete original64-row minibatch groups touched by either
item in visible TRAIN histories or targets. Reserve twice that row cost: one
original-map static evaluation and one sequential evaluation. Stop before the
next pair would exceed2000000 rows. All three teachers evaluate the exact same
prefix in the same order, with the same row budget. This differs from the previous
20epoch experiment's priority order and single-evaluation budget; compare ages
within this run, not accepted counts across the two different protocols.

## Two comparisons per pair

1. Static:score swapping the pair in the original index, independently of every
   other pair. Record exact likelihood delta and acceptance by original LCP.
   Positive moves are not combined into an index in this branch.
2. Sequential:start from the original index for each age and accept/reject in
   common order using that age's evolving index. This constructs three final
   indices; their greedy trajectories can differ after different acceptances.

Both use exact four-token FP32 teacher forcing, including suffix, full softmax,
and consistently remapped histories AND targets. Aggregate logP deltas inFP64.
Accept delta>max(.01nats,1e-5nats*affected TRAIN rows). No semantic filter/penalty.
All proposals in the shared prefix must be completed. A1200second per-age search
guard fails an incomplete comparison instead of silently comparing different
prefixes. Baseline/mining/final audit are outside the search-row/time counters,
as in the preceding run. No separate accepted-count cap truncates an age.

Static deltas provide the clean paired teacher comparison on the same starting
tree. Sequential outcomes additionally depend on the accepted trajectory. Raw
gain magnitudes need context from each teacher's initial NLL; more accepted moves
alone do not establish useful semantic structure or downstream quality.

## Evidence and execution

Reuse two private datasets and the pinned source kernel; private/offline2T4.
After source training and mining, worker0 searches ages1 then5, worker1 age20.
Expected execution roughly30–45minutes plus queue/startup, based on the previous
measured FP32 scoring and training throughput. No local CPU timing extrapolation.

Save early weights/full last state, training curves/order hashes, mining rows and
predictions, per-age ranked lists, merged pool/source ranks, hashed common prefix,
every static/sequential move delta, final maps/owners, accepted moves, per-TRAIN
initial/final likelihood cache, affected-row counts and changed-prefix depths.
Require exact parent/source/table hashes, shared initial parameter hash,
unchanged teacher weights/RNG, identity tolerance1e-5, final full TRAIN cache audit
within1e-5 per row/.02nats sum, replayable owner permutation and unchanged catalogue
occupancies at every prefix depth. Tests exercise actual early snapshots,
static scoring against independent original-map evaluations, and all three
searches through isolated native CPU workers.

Config:`experiments/amazon_beauty_sid_generator_age_2t4.json`.
Submitted once as private [Kaggle v1](https://www.kaggle.com/code/carasik/rhpr-generator-age-screen).
[Actions40673921485](https://github.com/mikhail-moro/RHPR/actions/runs/40673921485)
succeeded; its job log confirms `Kernel version 1 successfully pushed` from
executable commit`316c1093a0a76ccf408562c5bce0a8640d72f322`.
The cloud executor became unavailable immediately after dispatch, so a direct
runtime-status/embedded-package audit has not been performed. Submission is
confirmed, execution status is not yet checked. Do not resubmit the kernel.
Receipt:`docs/gr-sid-generator-age-submission.json`.

Checks before submission:905 repository tests OK,80 optional skips;four targeted
tests including native snapshots/all three isolated CPU searches passed. Native
first5epoch losses match training without snapshot saving. Source diffs against
remote base verified to contain only the new dispatch and optional save hook.
