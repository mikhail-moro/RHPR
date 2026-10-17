# 1K compressor learning-rate search

Configuration: `experiments/yambda_query_prefix_1k_search.json`, seed 42.
This is a validation-only follow-up to the completed six 4K screening trials.
It does not repeat their frozen length diagnostics or the completed full-history
Cross Attention training. It starts six fresh models with H1024 + R128.

## Why the search changes

The preceding 4K search selected backbone/compressor LR 0.001/0.001 for CA and
0.001/0.003 for QSSM. Full-history QSSM failed with a nonfinite forward loss during
pass 4. It had completed three passes without AMP gradient retries; there is no
evidence identifying the original failing operation. The larger compressor LR
was the actual selected setting, despite the common config's base LR of 0.001.
CA completed full30 and retained its best and resume checkpoints on private
Kaggle storage. No new test predictions were computed by that failed suite.

All six proxy trials were still improving at their final twentieth pass. In
particular, the lower-LR models were clearly undertrained within that budget.
Frozen checkpoint cropping from full H to H1024 changed validation NDCG@10 by
-0.246% for QSSM and -0.033% for CA, with paired 99% intervals crossing zero.
That makes 1K a useful cheap training proxy; it does not prove that optimizer
choices or learned weights transfer unchanged to full histories.

## Registered protocol

Keep the two architectures, K4, seed, parent backbone initialization, 32768 PE
table, users, full training catalogue, global split, original window endpoints,
last128 targets and identity-bound negatives. Crop only the left side of H on
CPU before embedding lookup, retaining the same Q128 and R128. The parent
physical/effective batch sizes stay 32/128, with 512 negatives, AMP and weight
decay 1e-5. There is no implicit learning-rate reduction or precision fallback.

| Trial | Backbone/item LR | Compressor LR |
|---|---:|---:|
| current_lr | 0.001 | 0.001 |
| lower_lr | 0.0003 | 0.0003 |
| faster_compressor | 0.001 | 0.003 |

Each of six trials receives at least 30 and at most 60 passes, with validation
every ten passes and early-stopping patience of three checks. These are passes
over one fixed final block per user, not exhaustive epochs over all events.
Select the earliest strictly best H1024 validation NDCG@10 checkpoint within a
trial. Save that FP32 state, then evaluate it once on the complete observed H,
using exactly the same validation anchors and full catalogue. This measures
inference-length transfer after 1K training; it is not a full-history retraining.

After all trials finish, select within each family by finite full-history
validation NDCG@10, keeping configuration order on exact ties. Numerically
invalid trials are ineligible and explicitly reported. No test ranks are read
or test predictions evaluated. Checkpoint selection, proxy validation and
full-length validation all reuse an already inspected validation period;
additional seeds and temporal replication remain separate future experiments.

## Failure isolation and retained progress

A numerical failure or CUDA OOM is recorded for its own trial, after which the
remaining registered trials proceed. Unexpected programming/data errors still
abort the suite. If a family has no eligible full-length candidate, save the
complete failure report and fail the suite rather than invent a winner.

On a numerical failure, a diagnostic replay can compare the captured batch
under the original AMP setting and FP32, preserving model weights, gradients
and training RNG. This replay diagnoses a possible precision problem; it never
continues training at a different precision or contributes a selection score.

Retain all available best per-trial FP32 weights. A rolling resume checkpoint
per family contains the actual active trial, last completed validation/save
epoch, Adam/scaler/RNG and source/config metadata. It is overwritten as later
trials progress; the manifest identifies its exact owner. Partial failed passes
are not represented as completed resumable epochs. This limits checkpoint disk
use while preserving completed candidate weights and active search progress.

The native scan audit and disposable real-update/evaluation profiles run before
training. The four-hour execution budget is checked against all six maximum
60-pass costs, validation, full-length transfer evaluation and checkpoint I/O.
Trial count and pass caps do not shrink silently to fit. The Actions timeout
includes queueing separately. On kernel failure, the collector retains available
partial outputs and logs without reporting the experiment as successful.

Outputs include source/config, original schedules, actual/reference input
hashes, both optimizer LRs, pass curves, validation ranks, selected checkpoints,
failure reports and measured costs. This run does not implement cached serving;
the shared CA/QSSM cache-refresh evaluation remains specified in
`cached-compressor-serving.md`.
