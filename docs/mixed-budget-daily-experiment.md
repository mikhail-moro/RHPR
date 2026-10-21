# Mixed budgets and daily evaluation of 16K histories

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This pilot tests whether one shared PairMLP/HSTU model can retain useful quality
at several global summary budgets. It also repeats every previous fixed model
under the new daily evaluation and adds 32 summaries. It does not train a router,
regional merge decisions, or a halting policy.

Configuration: `experiments/yambda_hstu_mixed_budget_daily.json`, seed42.

| Training models | Summary budgets | Evaluated cases |
|---|---|---:|
| HSTU Recent128 | none | 1 |
| Mean, uniform, PairMLP, one-block Q-Former | 4, 8, 16, 32; independently trained | 16 |
| Shared PairMLP + HSTU MixedBudget | 1, 2, 4, 8, 16, 32 | 6 |

There are 18 training models and 23 evaluated cases. Every model receives 20
passes of the same seed/epoch/UID-bound random windows and sampled negatives.
Each pass presents one window per eligible user: 4,427 windows, 35 effective
optimizer updates, and 141,664 next-item labels. This is not an exhaustive epoch
over 89 million training events. Total exposure is 700 updates and 2,833,280
labels per trained model.

Mixed uses one budget per effective update, including all physical microbatches
and any AMP retry. A separate local RNG creates a shuffled balanced schedule;
per-pass counts differ by at most one. The final smaller update means label
exposure is not exactly equal across arms. Reports retain actual updates,
windows, and labels per budget. Each budget sees roughly 117 of the 700 updates,
while the compressor and predictor share all weights across budgets.

The previous HSTU recipe remains d128, two layers, four heads, dropout0.2,
learning rate0.001, AdamW weight decay1e-5, gradient clip1, 512 sampled negatives,
and Time OFF. The full raw window is 16,384. Training preserves 159 raw positions
(127 burn-in + 32 supervised queries); inference preserves128. Summaries contain
only the older prefix, so parallel last32 losses remain causal. All models now
use the same fixed HSTU normalization191 and relative-bias capacity, fitting
159+32 summaries. This normalization changed from the previous pilot's175;
current models must be compared within this suite.

PairMLP shares a four-way merge projection across hierarchy levels. Its regions
are repartitioned for each budget; the six cases are not exact nested early-exit
depths. Prefix work dominates its cost even at one summary. Reported inference
is uncached end-to-end compression, HSTU, and full-catalogue Top10, with B1 and
the selected serving batch, measured stage times, and analytical FLOPs.

## Held-out daily protocol

The public Apache2.0 Yambda source revision and the fixed training cohort,
catalogue, and train timestamps are unchanged. The private Kaggle supplement
adds 4,170,627 observed held-out events once per user. No further training
events are removed. See `docs/daily-evaluation-data.md` for schema and audits.

The first chronological listen per user/day is selected before catalogue
mapping. Validation uses days286–292; test uses293–299. Each anchor receives the
latest16K strictly earlier observations, including earlier held-out days.
Weights are frozen; teacher-forced context updates reflect observed activity.
Recent128 receives the same context cropped to128. Unknown context items remain
occupied zero-ID slots; unknown targets receive zero credit. There is no seen-
item exclusion, matching the previous evaluator.

| Split | Users | Daily anchors | OOV targets |
|---|---:|---:|---:|
| Validation | 4,384 | 26,335 | 1,400 |
| Test | 4,349 | 25,571 | 2,059 |

Full-catalogue ranking computes Top100 once and reports HR/NDCG@10 and @100.
Primary scores average anchors within each user, then weight users equally.
Anchor-micro, warm-only, and first-period-anchor scores are diagnostics. OOV
rates and denominators accompany each score. New daily scores are not directly
comparable with the previous single-anchor scores.

Validation runs every five passes. Fixed models select their checkpoint by
macro-user NDCG@10. Mixed selects **one** checkpoint by the arithmetic mean of
that score across all six predeclared budgets; all six test cases reuse that
same checkpoint. Test outcomes never select a budget, checkpoint, or user rule.

Paired intervals versus Recent128 and adjacent Mixed budgets resample users
after averaging matched daily anchors. Anchor IDs, past-context hashes, and
catalogue support must agree. These intervals are exploratory, conditional on
one training seed, and unadjusted for multiple comparisons. Nonmonotonic point
estimates alone do not establish adaptivity. A subsequent router must learn
from validation, then freeze before test.

## Execution and retained artifacts

GPU preflight measures actual training throughput and daily full-catalogue
evaluation for every model and Mixed budget. Mixed training forecasts use the
slower of budgets1/32. All23 evaluation cases count towards the forecast using
their actual anchor counts. A common20-pass suite must fit the five-hour
execution budget with25% margin; otherwise it stops before quality training.
The workflow timeout is longer to cover Kaggle startup and output collection.

The kernel and dataset stay private; internet is disabled. Best FP32 checkpoints
for all18 training models remain in native Kaggle outputs (about9.7GB).
`checkpoint-manifest.json` records provenance, SHA256, byte sizes, model recipe,
selected pass, and budget metadata. Reconstruct with `build_quality_model`,
attach declared Mixed choices, load the saved state dict with `weights_only`,
set serving tail128, then choose the requested Mixed budget.

GitHub collection uses `.*\.(json|json\.gz|md|log|zip)$`, keeping reports,
compressed per-anchor scores, per-user aggregates, and the exact source bundle.
It skips `.pt` downloads. `output-collection-manifest.json` records the pushed
Kaggle version and which checkpoint files remain there. This avoids losing
weights while uploading gigabytes of checkpoints to GitHub.

Twenty past-only descriptors per daily anchor are already prepared separately
in the private evaluation dataset, using training-only catalogue frequencies.
They include uniqueness, repetition, old/recent overlap, temporal span, and
region statistics. Targets and target timestamps are absent from that table.
Future adaptivity analysis can join by UID/day and full-context hash without
recomputing features or retraining the baseline models.
