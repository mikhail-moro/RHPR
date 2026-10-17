# Bounded old-prefix compressor tuning

Configuration: `experiments/yambda_query_prefix_proxy_hpo.json`, seed 42.
The run compares the existing Cross Attention and QSSM compressors with four
summaries each. Their architectures, common SASRec initialization, catalogue,
users, temporal split, anchors and OOV policy remain those of the completed
SASRec prefix pilot (`660c7705ed3e005f35b339313c8c865255e36ede`).

## Frozen length diagnostic

Before tuning, load the pinned best FP32 parent checkpoints. Both FULL inputs
must reproduce the parent's validation support, context hashes and Top100 ranks
exactly. Then evaluate each frozen model with 1024, 2048 and 4096 old events.
Cropping removes only the left side of H: Q=H[-128:] and R128 remain identical,
including their relative positions. This is a validation-only intervention;
it does not estimate the quality of a separately trained short-history model.

## Registered search

Use H4096 + R128 for six independent fresh training runs: three settings for
each compressor. The table gives AdamW learning rates; all other optimization
settings remain fixed.

| Trial | Backbone and item embeddings | Compressor |
|---|---:|---:|
| current_lr | 0.001 | 0.001 |
| lower_lr | 0.0003 | 0.0003 |
| faster_compressor | 0.001 | 0.003 |

Keep physical batch 32, effective batch 128, 512 negatives, weight decay 1e-5,
gradient clipping 1 and the pinned AMP setup. The parent optimizer applies
weight decay to all parameters, including SSM parameters carrying advisory
`_no_weight_decay` attributes; this run preserves its actual behavior. No
architecture widths, summary counts, SSM time scales or extra training seeds
are searched here.

Each proxy receives 20 passes, with validation at 10 and 20. A pass visits
one fixed final window per training user: it is not an exhaustive event epoch.
User ordering and negatives use the parent's original identity-bound schedule.
Only the old input prefix is cropped, before embedding lookup; the original
window endpoint, final 128 labels and negative samples remain unchanged.
The positional table stays sized for 32768, preserving initialization and the
common parameter count. For 4427 users, each proxy has 700 successful updates
and 11,333,120 supervised label presentations if no failure occurs.

Within a trial, select the earliest checkpoint with strictly highest validation
macro-user NDCG@10. Within each compressor family, choose the highest such score;
an exact tie follows the registered trial order. A 20-pass screen favors settings
that learn within this fixed compute budget; it does not establish convergence
or guarantee that their ranking transfers to long histories.

## Full-history finalists

Train the selected configuration of each compressor from scratch on the complete
16K–27K old histories. Neither proxy weights nor optimizer state are reused.
Each finalist receives 30 passes and validation at 10/20/30, selecting the earliest
strictly best NDCG@10 checkpoint. This is 1050 updates and 16,999,680 supervised
label presentations per finalist.

Finish both validation selections before evaluating either selected checkpoint
on test. Test never chooses a trial, history length, checkpoint or stopping time.
Compare the two final models with each other and with the pinned parent saved
ranks, using the same full-catalogue macro-user NDCG/HR@10/@100 protocol and paired
user intervals. The existing test period has already informed exploratory work;
these are exploratory comparisons, not an untouched confirmatory holdout.

## Execution and retained artifacts

A native selective-scan gradient audit runs once. Disposable models profile real
successful Adam updates, including allocated optimizer state, and representative
full-catalogue evaluation for both families at proxy and full lengths. Training
starts only if the complete registered search fits the conservative estimate.
The execution guard is six hours (21600 seconds); the expected cost is roughly
3–4 GPU-hours and must be checked against measured preflight rates. The Actions
timeout includes queueing and is separate from the execution budget. The run
never silently changes its trial count, pass count, catalogue or target set.

Artifacts record source SHA, complete config, schedules, actual versus reference
context hashes, trial learning rates, validation choices, ranks, paired statistics,
training costs and uncached B1/B8/B32 inference latency/FLOPs. Full-history
finalists retain best FP32 weights and final Adam/scaler/RNG state on private
Kaggle storage. Output logs and metrics are collected separately from large
checkpoint files. Progress artifacts remain available after a budget failure.

The current inference benchmark includes recompression on every request. It
cannot establish efficiency in a cached serving system. A later, matched
cache-refresh experiment is specified in `cached-compressor-serving.md`.
