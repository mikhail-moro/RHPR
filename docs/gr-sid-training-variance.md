# Fixed SID indices crossed with generator seeds

Authorized GPU follow-up, 24 October 2026. Configuration:
`experiments/amazon_beauty_sid_training_variance.json`.

## Questions and frozen design

The first six-index experiment varied RQ-KMeans seeds at one generator seed.
Its item-level effects mix index structure and the mapping of semantic groups
to randomly initialized embedding/output rows. Some fixed branches also
changed their preferred child during training, including on train contexts.
This follow-up measures reproducibility over new generator seeds before
proposing a different quantizer.

- Import the EXACT original `rq-11` and `rq-22` SID arrays, ordinal suffixes,
  centroids and quantizer configuration. Verify every byte by SHA256 and pin
  the original private notebook's run manifest and source revision. Never refit
  or fall back when an input is unavailable. Original index22 contains the
  previously studied 2166/3500 branch; index11 is a fixed comparison, not a
  newly selected winner.
- Cross each index with three NEW generator RNG seeds: **101,202,303**.
  Six independent models, run IDs `rq-11-ms-101`, etc. Each model seed controls
  initialization and the training/dropout stream. Within each seed, the two
  indices must have the same initial tensor-content hash. Across model seeds,
  initial states differ. Save each initial state separately.
- Data seed4242, raw-item window manifests/order, catalogue, all original
  splits, frozen Sentence-T5 representations and the entire generator/optimizer
  recipe remain unchanged. Twenty epochs, batch512, FP16, Adam5e-4/wd1e-4,
  no early stopping. Validation/checkpoints5/10/15/20, test only at20.
- Preserve raw versus local scoring, full-catalogue evaluation, beam10 for
  all cases and final shared512-case exact/wide-beam panels per split.
  This is a fixed-budget study, not a claim of converged performance.
- Two independent T4 workers. Serpentine assignments distribute both indices
  over both devices: GPU0=(11,101),(22,202),(11,303),
  GPU1=(22,101),(11,202),(22,303). No CPU/single-GPU fallback.

The old seed42 models are archived references, not extra new independent runs.
The observed original training time was approximately44minutes/model and
evaluation50minutes/model. Six new models on two cards should take roughly
five hours plus setup and probe overhead; this is an estimate, not a timeout.
The kernel uses its own private slug, so it cannot overwrite its input study.

## Fixed probes during training

Monitor items2166,3500,822,5355,1107,4165,3294. They were chosen from previous
exploratory findings; do not describe them as a representative catalogue sample.
Record ALL four teacher-forced heads, not only the known failing third head.

The parent prepares one shared `probes/cases.parquet` from raw items:

- Up to32 sampled train windows per selected item from the fixed epoch20
  manifest, preserving source epoch/ordinal, user and actual visible context.
  They are training data; early probes do not assert that each selected window
  has already been visited by that checkpoint.
- Up to16 validation cases per item, sorted by stable case ID. No test probes.
- The same cases are used for every index/model seed. Missing support remains
  missing; no examples or histories are invented.

Probe at step0, every100 ATTEMPTED updates, and every epoch end. Deduplicate
coincident snapshots. Attempted updates determine the common observation
schedule even when AMP skips an effective update. Store both counters.

Probes run FP32/eval/inference mode, with no backward or optimizer operation.
Preserve Python, NumPy and CPU/CUDA Torch RNG and restore model training mode.
The native CPU verification compares parallel probe-enabled training against
serial probe-disabled training: final model tensors and Torch RNG must match
bit for bit, along with examples and effective update counts.

Each `rq-Q-ms-M/probes/step-NNNNNN.parquet` stores run, epoch, counters,
probe ID, split, target item, correct prefix/token, all256logits, raw/local
logprob, token rank, strongest other token and valid-child mass. Snapshots are
atomic and registered by hash in `artifacts`. These sidecars keep the original
26-table evidence schema compatible. DuckDB can query them directly:

```sql
SELECT run_id, epoch, attempted_updates, target_item, level,
       avg(-local_logprob) AS local_nll
FROM read_parquet('rq-*-ms-*/probes/step-*.parquet')
GROUP BY ALL ORDER BY ALL;
```

`probe-summary.json` records observational overhead. The training loop's wall
time includes in-loop probes; subtract their separately logged durations for
throughput comparisons. Scheduled checkpoints still include optimizer/scaler
and RNG; no automatic resume or fallback is introduced. Probe trajectories
localize a failure interval but do not identify a particular causal update.

## Analysis frozen before new outcomes

1. Report every final validation/test metric and all paired index22−index11
   differences at each model seed. Compare between-index differences with
   within-index generator variation; retain index×initialization interaction.
   Three model seeds and two indices support only limited uncertainty claims.
2. On validation, test whether the previous item-level index differences retain
   their direction under independent initializations. Keep the original dense
   item set and previous groups; do not reselect winners using new outcomes.
3. Plot fixed train AND validation probe paths. Does branch2166/3500 switch
   preference under multiple initializations, and does it also happen in11?
   Separate online training loss from eval-mode loss at the observed weights.
4. Inspect first/second decisions as well as late failures. A local improvement
   in third-head likelihood does not establish improved item ranking.

This varies initialization and the training RNG together; it does not
independently isolate initialization from dropout. It also does not separate
history encoding from target factorization. No new representation, quantizer,
token splitting or hyperparameter tuning is part of this run.

## Decision ceilings from the existing run

`scripts/analyze_sid_decision_ceiling.py` uses archived epoch20 full-catalogue
scores and true-path probabilities; CPU only, no new inference/training.
At the TRUE parent of a selected decision, set its true child probability to1,
other children to0. Boost all descendants of that child equally and leave
other parent distributions unchanged. This preserves local-model total mass;
all-four repair yields target rank1. It is an optimistic target-aware ceiling
for local modifications of this frozen model, never a deployable method or an
upper bound on what retraining a different architecture can accomplish.

On the original weighted512-case validation panel, averaged over six indices,
raw exact NDCG@10 is0.03956. Perfect first decision gives0.30234; perfect second
0.35915; both first decisions0.90364; both last decisions0.06222. Late repairs
can matter, but most recoverable loss here involves earlier routing. Raw/local
results remain separate and agree qualitatively. The panel is finite and
previously inspected; these values are descriptive conditional diagnostics.

```bash
PYTHONPATH=src python scripts/analyze_sid_decision_ceiling.py \
  --root /path/to/original-output --output /path/to/ceiling-analysis
```

## Reproducible dispatch

Use the authorized GitHub Actions workflow on the exact pushed commit,
`accelerator=gpu`, private kernel `carasik/rhpr-sid-training-variance`.
Inspect available quota and existing sessions before submitting; submit once.
No external notification is required. Preserve paused TPU work outside this PR.
