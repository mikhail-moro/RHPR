# HSTU compressed16K quality pilot

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This private, single-seed pilot asks whether a compressed old prefix adds quality
to a recent-history HSTU. It is not per-architecture HPO or a convergence study.
The earlier HSTU256×4 experiment borrowed a SASRec recipe tuned on50M; it did not
optimize HSTU. This500M pilot instead shares a deliberately cheaper HSTU128×2,
4heads, dropout0.2, AdamWlr0.001, weight decay1e-5, clip1, effective128windows,
512uniform sampled-softmax negatives and seed42 across all variants.

## Models and budgets

13 variants: Recent128; regional mean, uniform regional endpoint sampling,
hierarchical PairMLP extension and learned-query compressor, each with16/8/4
old-prefix summaries. All share the same backbone initialization and raw-recency
PE table16384. Attention normalization and relative-bias bounds are fixed175
for every variant, at both training and inference, rather than accidentally
dividing short compressed attention by16384.

The hierarchical MLP partitions the old prefix into near-equal chronological
regions. Within each region, the same `[4E,E]` GELU residual projection merges
four adjacent children repeatedly until one summary remains. The residual skip
is a mean weighted by represented event counts. Zero-initialized residual
weights reproduce the regional mean, including incomplete groups. No initial
pooling discards information before the learned hierarchy. This is explicitly a
four-way extension of binary PairMLP, chosen for less depth and fewer projections.

The Q-Former-like compressor is randomly initialized native PyTorch:16/8/4
learned queries, one query self-attention, cross-attention to the entire old
prefix and a GELU FFN4×. It uses sinusoidal raw-recency information in prefixK/V;
it is not pretrained BLIP Q-Former. All query slots carry the old-prefix endpoint
as their source position, because they all read the entire old prefix. Query
identity and HSTU relative slot positions distinguish the summaries.

## Matched supervision

The total input budget is16384, not16384old+128recent. This preserves all4427
eligible training users. Each pass samples a new full-window endpoint per user.
Endpoints, user order and negative draws depend only onseed/pass/UID/window,
independently of model RNG and physical batching. These are sampled passes,
not exhaustive epochs over89M events.

Training keeps the last159 events raw:127burn-in events plus32 supervised
next-item states. Each supervised query has128–159 raw past events. The old
prefix contains16225 events and ends strictly before every supervised query.
Recent receives exactly the same159 raw inputs and32 targets as the other
models, cropped before embedding lookup. Only the old-prefix summaries differ.
Last32 encoder states are selected before FP32 loss computation. Partial
effective batches receive the correct window-count gradient weights.

Serving/validation/test keep exactly128 raw events plus16/8/4 summaries of the
old16256-event prefix. This is a small declared train/inference context shift.
Actual old-prefix compression is1016×/2032×/4064×;1024×/2048×/4096× are nominal
16K labels. Every validation/test example predicts one final-state target over
the entire1049612-item train-only catalogue. Repeated items remain eligible.

The [evaluation supplement](long-history-evaluation-data.md) defines temporal
validation/day286–293 and test/day293–300 anchors before catalogue mapping.
Primary metrics include OOV targets as misses; warm-only metrics are secondary.
Occupied unknown-context events retain their zero item embedding and valid
chronological slot; masks come from lengths, never `item_id!=0`.

## Cost and selection

One kernel profiles real successful updates at physical8/16/32 and uncached
full-catalogue inference at batches1/8/32 for all13 variants. OOM shapes are
recorded; no model changes its history budget to fit. CUDA event and synchronized
wall measurements are separate. Compression is included in history latency.
CPU collation and H2D are outside microbenchmark timing; complete evaluation
also records wall time including those costs. FLOPs are estimated executed major
matmuls with multiply-add counted as2; elementwise and packing costs are excluded.

The preflight chooses the fastest successful physical batch and an identical
5–20 training passes for every model before observing quality. A four-hour
Python execution budget includes preflight, training, periodic validation and
test, with25% forecasting margin. Kaggle scheduling/container startup is outside
that measured budget; workflow timeout is six hours. Actual budget overrun fails
the suite rather than silently giving later models fewer passes.

Validation is measured every5passes and at the common final pass. Each model's
best validation-all-anchorNDCG10 checkpoint is selected; test runs once afterward.
Temporary checkpoints are removed and do not enlarge collected kernel output.
Full learning curves, selected epochs, user-matched testNDCG bootstrap intervals,
dataset/schedule signatures and scalar metrics are persisted. Single-seed
bootstrap intervals do not measure training-seed variation or correct selection
among12 comparisons. Failure to improve with4/8/16 summaries alone does not
establish that16K history is useless: compression may be too severe or training
insufficient.

The kernel uses the base Kaggle PyTorch/PyArrow runtime and records its versions;
no Mamba/CUDA extension installation or4.9GB wheel dataset is needed. This saves
runtime preparation but does not pin Kaggle's base image. Internet remains off.

## Additional quality cutoff for future runs

The registered config now evaluates NDCG/HR at both10 and100. All scores still
come from the full training catalogue;100 is a ranking cutoff, not a sampled
candidate count. One catalogue scoring pass keeps top100, from which both
cutoffs are derived. All-anchor/OOV and warm-only denominators are unchanged.
Checkpoint selection remains validation NDCG@10. NDCG@100 is a secondary
diagnostic to reduce censored zero utilities, not a replacement for more targets
per user or independent validation of adaptive routing.

Per-user records preserve primary `rank` (zero outside top10) and add
`rank_at_max_k`, `rank_cutoff`, `hr100` and `ndcg100`. A maximum-rank value of zero
means outside the saved cutoff or an unsupported target, never an exact rank.
The already completed pilot saved only top10 ranks and removed temporary
checkpoints, so its@100 cannot be reconstructed. Its existing@10 report stays
unchanged.

With expanded outputs, repeat the history-budget analysis using
`scripts/analyze_history_budgets.py --metric-k 100` and a separate output
directory. The fixed reference uses each model's original NDCG@10-selected
checkpoint and that checkpoint's validation NDCG@100. Both analysis and figures
record the chosen cutoff; missing@100 inputs fail rather than becoming zeros.
