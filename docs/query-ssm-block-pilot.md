# Query-conditioned SSM: an 8K history pilot

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

Configuration: `experiments/yambda_query_ssm_block_pilot.json`, seed 42.
This is a bounded comparison of five independently trained models. It tests
whether the recent context is more useful when it changes how a state-space
model accumulates the old history, rather than being used only to read the
resulting memory. It does not test adaptive compression, multiple summary
budgets, user-ID queries, or a learned halting policy.

The old prefix contains 8,000 events. Every compressor returns four summaries;
the predictor also receives the recent raw events. HSTU is held at d128, two
layers, four heads and dropout 0.2. Time features are off. All variants use the
same sampled-negative loss, catalogue and held-out anchors.

| Variant | Old-history representation | Role |
| --- | --- | --- |
| `recent128` | None | Measures the value of adding old history |
| `uniform_m4` | Four evenly spaced region endpoints | Cheap old-history control |
| `query_cross_attention_m4` | Four recent-conditioned queries cross-attend to all 8,000 old events | Direct retrieval competitor |
| `query_ssm_m4` | Four recent-conditioned selective scans | Tests conditioning the accumulation of memory |
| `query_ssm_readout_m4` | Four scans whose recurrence does not depend on the recent context; the output still uses the recent query | Controls for query information available at readout |

The recent-conditioned variants use the same query generator. Learned slots
supply query identity; a function of the available recent embeddings supplies
current context. Queries never use a target item or a user-ID embedding. The
readout control keeps the recent signal at its output, so a comparison with it
can distinguish conditioned accumulation from simply adding another recent-
context pathway. Parameter counts are reported rather than assumed equal.

## One scan and 32 causal targets

A training window has 8,159 observed events: 8,000 old events followed by 159 raw
recent events. Denote the recent events by `r1 ... r159`; the next target after
the window is `r160`.

```text
Observed input: [old1 ... old8000] [r1 ... r128] [r129 ... r159]
Query support:                      r1 ... r128
SSM/attention memory: old1 ... old8000
Scored outputs:                           r128 ... r159
Next-item targets:                        r129 ... r160
```

The query is formed once from the FIRST 128 events of the raw tail, and one
query-conditioned scan processes the old prefix. A single causal HSTU forward
then processes `[summary1 ... summary4, r1 ... r159]`. Loss is computed only at
the final 32 RAW outputs, predicting `r129 ... r160`. All scored targets occur
after every event used to form the query.

Earlier raw representations can indirectly see `r128` through the summaries.
They are burn-in representations and receive no supervised loss. A query built
from all 159 raw events would leak the early targets and is prohibited. Summary
outputs must also be explicitly excluded from supervision: their truthful
source position is the query endpoint, which can equal the source position of
the first scored raw event. A source-position threshold alone is insufficient.

The query is current for the first target and stale by up to 31 events for the
last target. Later raw observations remain directly visible to HSTU. This
retains 32 labels per expensive scan, at the cost of this explicit training
approximation. At validation and test, each anchor gets a freshly computed
query from its actual latest 128 observed events. The serving input contains
8,128 events: the old 8,000 and the latest 128. Queries for one held-out anchor
are never reused for another anchor.

Every variant uses identical training endpoints, labels and sampled negatives.
Recent128 uses the same raw-tail training convention, including its burn-in;
it does not receive a different target schedule. All models share the same
HSTU normalization capacity for 159 raw events plus four summaries. Changing
the normalization and history length means these scores are not a continuation
of the previous 16K pilot's training trajectory.

## Selective scan and gradients

Each query is a learned slot plus cross-attention from that slot to the first
128 raw-tail embeddings with raw-recency positional features. The old memory
is layer-normalized item embeddings plus raw-recency features relative to the
query endpoint. The same query generator and output residual/FFN are used by
the cross-attention and SSM variants.

For each old event and slot, the SSM gate bottleneck is
`r_tj = SiLU(W_x memory_t + W_q write_query_j)`. It generates a positive step
size `delta_tj = softplus(W_delta r_tj + dt_bias)` and input-state coefficients
`B_tj = W_B r_tj`. The shared stable state matrix is
`A = -exp(A_log)`. Input values are projected once per event before expansion
over the four slots. The query-dependent readout `C_j = W_C q_j` is constant
across the old-history scan. The final scan output is projected back to d128
and combined with the same query residual and FFN as the attention competitor.
The readout control substitutes the static learned slot for `write_query_j`,
while retaining the recent-conditioned query in `C_j` and the output pathway.

The configured internal width is 128, state dimension is 16, and gate bottleneck
rank is 32. Gates do not depend nonlinearly on the previous hidden state. This
preserves the affine recurrence needed by the native parallel scan. This is a
custom query-conditioned SSM using the official selective-scan primitive; it
does not include a causal convolution, the Mamba `D` skip, or the Mamba `z`
output gate, and is not an unmodified full Mamba block.

Initial positive steps are log-uniform in [1e-4, 0.1], with inverse-softplus
bias initialization. The slowest initial A=-1 mode can retain information over
roughly 10K events, so an 8K prefix does not initially disappear in every
channel. These are learned recurrence step sizes, not physical timestamps.
Both SSM variants use exactly this initialization. The shared AdamW applies
the configured weight decay to all parameters, including A_log; it does not
interpret the upstream-style `_no_weight_decay` attribute automatically.

The summary uses the final differentiable OUTPUT `y_N`, whose readout depends
on the final hidden state. It does not use the separately returned CUDA
`last_state` as its differentiable training path. The standard scan backward
therefore propagates through the final output into the earlier recurrence.

GPU execution requires the official Mamba selective-scan CUDA extension. A
small native-versus-reference forward AND gradient parity check is mandatory
before profiling or quality training. It must include gradients through the
final output into the input, scan parameters and query-conditioning parameters.
Failure stops the run. There is no Python-loop GPU fallback whose time could be
reported as a native selective-scan result. A CPU reference is retained for
small correctness tests only.

## Training schedule and compute guard

The fixed recipe is AdamW with learning rate 0.001, weight decay 1e-5, gradient
clip 1, 512 sampled negatives, CUDA FP16 AMP with initial loss scale 1024, and an
effective batch of 128. Physical batches 8, 16 and 32 are profiled; accumulation
preserves effective examples and loss weighting, including the final partial
update. The common eligible cohort has 4,427 users. One pass contains one
random full window per user, 35 effective optimizer updates, and 141,664
next-item labels. It is not an exhaustive epoch over all training events.

All five models receive the same number of passes, between 30 and 60. Before
quality results are observed, measured training and full-catalogue evaluation
costs determine a common feasible pass count under a 10,800-second execution
guard with safety factor 1.25. A suite that cannot fit the minimum 30 passes
stops before quality training. Validation runs every ten passes. Scheduled
pass count is not extended or shortened because a model looks promising.

This provides 1,050 to 2,100 effective updates and 4,249,920 to 8,499,840 labels
per model, subject to reporting actual successful updates and any AMP skips.
Each model selects its own checkpoint using validation macro-user NDCG@10.
Test is evaluated only after checkpoint selection. Test scores do not select a
checkpoint, pass count, model configuration, segment or summary budget.

There is no hyperparameter search, and there is one training seed. Reaching the
scheduled last pass does not establish convergence. A winner in this pilot is
a reason for a controlled follow-up, not evidence of an architecture-wide
advantage.

## Data, ranking and paired comparisons

The cohort, catalogue, train timestamps and daily evaluation supplement are
identical to `yambda_hstu_mixed_budget_daily.json`. The public source is
`yandex/yambda`, revision
`dd6f3a19eef5866e346c3270e098baa641a44948`. Immutable cohort, catalogue, mapped
training, train-timestamp and evaluation-content SHA256 values are in the
configuration. The cohort contains 4,427 users and the catalogue contains
1,049,612 mapped items.

Daily anchors remain the first chronological listen per user/day, selected
before catalogue mapping. Validation covers days 286–292 (26,335 anchors,
4,384 users, 1,400 OOV targets); test covers days 293–299 (25,571 anchors,
4,349 users, 2,059 OOV targets). Each context contains only observations strictly
before its target, including earlier held-out observations with frozen model
weights. Unknown context items retain their occupied zero-ID positions;
unknown targets receive zero credit. Seen items are not excluded.

The prepared anchor identity and full 16K reference-context hash are retained
for matching against existing runs. The actual cropped 8,128-event context has
a separate hash; it must not be presented as if the model consumed the full
16K reference. Recent128 uses the final 128 events of that same cropped context.

Full-catalogue Top100 produces HR/NDCG@10 and @100. Primary NDCG@10 first
averages daily anchors within each user and then weights users equally.
NDCG@100 is secondary. Paired uncertainty resamples users, retaining their
matched anchors. The four predeclared primary comparisons are Query SSM versus
Recent128, Uniform4, query cross-attention, and the query-at-readout SSM;
multiple-comparison adjustment uses the Holm family of four NDCG@10 tests.
Intervals are conditional on this training seed and do not measure training-
seed variability. Secondary cutoffs and any discovered subgroups are reported
as exploratory results.

The comparisons answer different questions. A gain over Recent128 establishes
that the extra history representation helps. A gain over cross-attention
supports the selected compressor against its direct competitor. A gain over
the readout control supports using the current query during accumulation.
Post-hoc test segments alone do not establish the last claim.

## Runtime, latency and retained evidence

All models run in the same pinned offline CPython 3.13/PyTorch 2.10.0+cu128/
Mamba 2.3.1 runtime. The private dataset
`carasik/rhpr-cp313-torch210-mamba-runtime` supplies hash-locked wheels. The
runtime configuration is copied unchanged from
`experiments/yambda_long_training_profile.json`; see
`docs/profile-runtime-dependencies.md` for binary hashes and installation.
Runtime installation, checksum checks and import/bootstrap costs are reported
separately from the experiment's training/evaluation compute guard. They still
count toward actual kernel elapsed time and quota.

Benchmark batch sizes are 1, 8 and 32, subject to a profiled safe batch. Use
three warmups and eight measured iterations, report the actual CUDA device and
software versions, and synchronize CUDA around wall-time measurements.
Inference is UNCACHED and includes compression, HSTU and full-catalogue scoring
and TopK. Component and end-to-end scopes must be stated; input preparation and
host/device transfers must not silently disappear into a broader latency
claim. Quality ranking uses chunks of 65,536 items and Top100; a Top10 serving
benchmark is labeled separately when used.

There is no assumed asymptotic advantage over cross-attention with four
queries: both process the 8,000-event prefix, and four query-specific scans
carry four states. Report peak device memory, parameter count, actual timings
and the explicitly scoped analytical operation count. Query-dependent state
cannot in general be reused after the recent query changes. FLOP estimates
and a small persistent state do not by themselves establish lower training
memory, lower latency or a useful cache.

Outputs include finite scalar `metrics.json`, `summary.md`, the richer report,
configuration and source revision, context/schedule identities, preflight and
runtime identities, validation curves, selected checkpoints and manifests,
and matched per-anchor/per-user scores. Datasets, event arrays, credentials
and tokens are not committed to the repository. Kaggle kernels and attached
cohort datasets remain private. Preparing this pilot does not launch it or
spend GPU quota; deployment needs an explicit user run/deploy request under
`AGENTS.md`.

## Related work and attribution

The implementation uses the official Mamba scan API; the proposed experimental
wrapper is not copied from the papers below. These works motivate or precede
query-conditioned sequence processing, and this pilot makes no novelty claim:

- [TM-Mamba](https://arxiv.org/abs/2404.11375) conditions selective parameters on
  an external query, a close precedent for the mechanism.
- [QMambaBSR](https://arxiv.org/abs/2408.08665) uses query-guided state-space
  processing for image restoration, with spatial conditioning distinct from
  four global recommendation summaries.
- [SemaMIL](https://arxiv.org/abs/2509.00442) uses query-guided state-space
  processing in multiple-instance learning; its query construction and task
  differ from recent-context recommendation.
- [DIEN](https://arxiv.org/abs/1809.03672) makes relevance influence recurrent
  interest evolution in recommendation, preceding the general idea of
  query-conditioned memory updates.
- [Official selective-scan interface](https://github.com/state-spaces/mamba/blob/v2.3.1/mamba_ssm/ops/selective_scan_interface.py)
  documents the distinction between differentiable outputs and the separately
  returned last state.
