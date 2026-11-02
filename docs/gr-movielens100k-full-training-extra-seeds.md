# Frozen-index full-training replication: seeds 101 and 202

Repeat the completed seed-42 pilot for the two remaining existing seeds, with
all four variants per seed: TIGER text+poster+plot without people; TIGER with
DeepWalk people; TIGER with people names in text; item-ID-only SASRec.

Configuration: `experiments/movielens100k_full_training_extra_seeds_2t4.json`.
Private notebook: `carasik/rhpr-ml100k-full-training-extra-seeds`.

## Fixed protocol

Resume each saved epoch-20 `generator-last.pt` (6,680 updates), including model,
Adam/scaler, CPU/CUDA/Python/NumPy RNG, original best checkpoint and history.
Reuse the exact saved encoders, quantizers and ID/SID matrices; no AE, RQ,
encoding or index rebuild. Hash-pin parent manifests, last/best checkpoints,
curves, original validation predictions and index metadata before execution.
Keep original parent notebooks and the completed seed-42 output intact.

Use the pilot's unchanged minimum 30 / cap 100 total epochs, validation each new
epoch, early stopping 20 epochs since the best raw validation NDCG@10, and
ReduceLROnPlateau from epoch 21. Halve LR after five failed checks, absolute
threshold 0.0001, floor 0.00001; initialize the new scheduler from historical
best validation with zero bad checks. Preserve historical validation at epochs
5/10/15/20. Test once after validation checkpoint selection.

Retain 1,273 films, 943 users, strict timestamp-group holdouts, history 20,
85,470 training cases per epoch, effective/physical batch 256/128, Adam initial
LR 0.001 / weight decay 0.0001, FP16 training / FP32 evaluation, full catalogue,
no seen-item mask or sampled negatives. Architectures and per-level validation
CE diagnostics are unchanged. Capacity matching does not imply equal FLOPs.

## Source lineage and execution

Mounted parents remain `carasik/rhpr-ml100k-multimodal-ae` (stages 03/04) and
`carasik/rhpr-ml100k-names-sasrec-controls` (names/SASRec). For each trial, the
declared seed must match its checkpoint, source directory and initialization
group. Compare initialization hashes within predictor and seed, never across
seeds. The suite's top-level seed is 101; the explicit `continuation_seeds`
list and each trial's seed control the two replicated pipelines.

Two T4 workers, each with three TIGER continuations and one SASRec continuation:

- Worker 0: no-people 101, names 101, graph 202, SASRec 202.
- Worker 1: graph 101, SASRec 101, no-people 202, names 202.

The completed pilot took 48.23 minutes; approximately 1.5–2 hours for this suite
is a practical estimate, plus setup/queue. Later best epochs could increase this
substantially. Each model retains its 7,200-second continuation training guard;
no automatic extension, precision fallback, tuning or retries.

## Analysis contract

Combine the new selected results with the existing seed-42 pilot to report all
three seeds individually, mean and sample standard deviation, paired changes
between modalities and selected/stop epochs. Check continuation/source hashes,
cases and scheduler/ES decisions; mark any epoch-cap result budget-censored.
Do not replace the fixed policy with test-selected epochs or the best seed.

This repeats complete saved pipelines, including their different indices; it
does not isolate generator-seed variance. Prior exploratory test results are
not an untouched holdout. The key question is whether the no-people improvement
and graph-versus-text ordering persist across the three existing seeds.

Only one private eight-continuation session is authorized here. No new seeds,
HPO, quantizer retraining, changes to seed 42, TPU or Telegram notifications.
