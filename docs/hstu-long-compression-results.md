# HSTU16K: useful old-prefix summaries, no established adaptive budget yet

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

The completed pilot supports learning a compressed old prefix. PairMLP with eight
summaries improves test NDCG@10 by 38.83% over Recent128 and 16.74% over uniform
sampling at the same summary budget. A user-level follow-up does not yet find a
reliable rule for choosing four, eight or sixteen summaries from past history.

Source: [Kaggle](https://www.kaggle.com/code/carasik/rhpr-hstu-long-compression-quality),
[workflow 38362181309](https://github.com/mikhail-moro/RHPR/actions/runs/38362181309),
code `89723a4369ab0e89f14415814e1ddcc1741f2505`, artifact 11799157536
(SHA256 `5a5177a450102b21574c2a3600430484cbf90236d2ccd3e49fc0af3ccd2fa577`).
Python execution took 84.8 minutes on one T4 using base Kaggle PyTorch 2.11.0+cu128
and Python 3.13.15. All 13 models completed 20 matched passes and 2,833,280 labels
with zero skipped AMP updates. All have best validation checkpoint at pass 20.

## Static quality and cost

The [pilot protocol](hstu-long-compression-pilot.md) uses the same HSTU 128×2,
initialization, window/negative schedule and recipe across variants. Evaluation
has 16,256 old events plus 128 raw recent events and one full-catalogue target.
All 4,349 test anchors remain; 4,024 targets are in the train catalogue, 325 OOV
targets count as misses. Repeat items stay eligible.

| Compressor | Summaries | Test NDCG@10 | vs Recent128 | Train time,20 passes | B32 full inference |
|---|---:|---:|---:|---:|---:|
| Recent128 |0|0.022128|—|2.47min|15.47ms|
| Regional mean |4|0.023473|+6.08%|3.82min|27.08ms|
| Regional mean |8|0.022804|+3.06%|3.82min|26.57ms|
| Regional mean |16|0.022920|+3.58%|3.85min|26.77ms|
| Uniform sample |4|0.024060|+8.73%|3.11min|17.50ms|
| Uniform sample |8|0.026316|+18.93%|3.12min|17.64ms|
| Uniform sample |16|0.024857|+12.33%|3.14min|17.64ms|
| Hierarchical PairMLP extension |4|0.028241|+27.63%|5.60min|39.96ms|
| Hierarchical PairMLP extension |8|0.030720|+38.83%|5.54min|40.05ms|
| Hierarchical PairMLP extension |16|0.028231|+27.58%|5.60min|39.88ms|
| Learned queries/Q-Former-like |4|0.025986|+17.44%|11.13min|61.66ms|
| Learned queries/Q-Former-like |8|0.026712|+20.72%|11.14min|61.92ms|
| Learned queries/Q-Former-like |16|0.027433|+23.97%|11.18min|62.67ms|

Inference is an uncached batch latency, **not milliseconds per user**. Compression,
HSTU, FP32 catalogue scoring and top10 are included; CPU collation/H2D/D2H and
external serving are excluded. These are eight-sample synchronized wall means,
not a precise ranking of tiny timing differences. CUDA event samples and p50/p95
are in the run artifact. All variants use one fixed normalization/bias bound 175.

Pair4/8/16 execute about 1.040/1.043/1.049 GFLOP per user for counted forward
matmuls, including catalogue scoring. They always read and compress the entire
old prefix, so reducing output summaries barely changes their cost. Adaptive
selection among these three models would not yet deliver substantial compute
savings. Uniform8 provides a useful lower-cost history baseline.

## What the differences establish

Paired user bootstrap uses 10,000 draws; sign-flip tests use 50,000 permutations.
Holm correction is applied separately within each explicitly reported comparison
family. These intervals measure user sampling for this one training seed.

| Comparison | Δ NDCG@10 |95% paired CI|Within-family Holm p|
|---|---:|---:|---:|
| Pair8 − Recent128 |+0.008592|[+0.006024,+0.011302]|0.00024,12 history comparisons|
| Pair8 − Uniform8 |+0.004404|[+0.002326,+0.006536]|0.00012,3 same-budget comparisons|
| Q8 − Pair8 |−0.004008|[−0.006465,−0.001697]|0.00270,3 same-budget comparisons|
| Q16 − Pair16 |−0.000798|[−0.002743,+0.001043]|0.41149,3 same-budget comparisons|
| Pair8 − Pair4 |+0.002479|[+0.000385,+0.004614]|0.07176,3 Pair budget comparisons|
| Pair8 − Pair16 |+0.002489|[+0.000119,+0.004906]|0.08524,3 Pair budget comparisons|

The eight-summary peak is visible in point estimates; its superiority over the
other Pair budgets is weaker after correcting the three comparisons. Nonmonotone
aggregate quality alone is not evidence that a useful adaptive policy exists.

Q8 is worse than Pair8 in this recipe. The Q compressor is one global learned-query
block trained from scratch; PairMLP preserves contiguous regional structure and
starts from a mean residual. This is a plausible inductive-bias explanation,
not a measured cause. All validation curves still rise at passes 5/10/15/20;
between 15 and 20, Pair8 validation improved 20.9%, Pair4 improved 36.1%, and Q8
improved 16.7%. Training has not established either architecture's ceiling.

## User history and required budget

Features come only from observed context events before the target. Unknown
item identities are excluded from identity statistics while their event slots
and timestamps remain. Catalogue frequency uses all 89,191,505 eligible training
events before the cutoff. No target item/popularity/time or learned embedding
features enter the router. Item diversity here measures identity repetition,
not semantic diversity.

The five core descriptors were fixed before joining per-user utilities: old-prefix
unique-item fraction, normalized item entropy, recent-event overlap with old
items, old-prefix duration, and total pre-cutoff training-event count. The last
is an activity proxy; visible input length 16,384 is constant for all users.
The old unique fraction has median 19.9%; recent-to-old overlap has median 85.2%;
visible history covers a median 229 days. Fifteen core tests compare each descriptor
with Pair16 − Pair4, Pair16 − Pair8 and Pair8 − Pair4 utility differences.

- Spearman: maximum absolute correlation 0.01822; minimum BH q = 0.793 across the 15
  core tests. No corrected monotone association was found. Warm-only sensitivity
  and a secondary 20-feature analysis do not change that conclusion.
- Secondary quartile omnibus tests allow nonmonotone relationships. The strongest
  hint is old duration vs Pair16 − Pair8: permutation p = 0.00590, BH q = 0.08849, η² = 0.00290.
  Warm-only q = 0.07049. This remains exploratory and does not meet the 0.05 threshold.
- Quartile intervals are descriptive and uncorrected; a subgroup's positive
  difference can reflect the global eight-summary advantage. They cannot be
  treated as independently discovered routing rules.

One future target per user makes the outcome very sparse: 4,131 users (95.0%) have
zero NDCG for all three Pair budgets; 4,234 (97.36%) have tied best budgets. Only 115
users have a uniquely best arm. The future-target oracle reaches 0.037559, 22.26%
above fixed8, but uses the answer to choose among separate trained models. That
upper bound does not show that the choice is predictable from past history.

## A direct routing check

Only test per-user predictions were saved in the original run; validation contains
aggregate metrics. This is therefore an exploratory split of already available
test users, not a new confirmatory temporal test. Stable UID hashing with seed 42
assigns 2,188 development users and 2,161 held-out users.

The development search includes a constant policy and 20 one-threshold policies:
five core features × cuts at 20/40/60/80%. Each leaf chooses its best arm using
development outcomes only, with at least 200 users. The fixed reference is Pair8,
selected from validation aggregates. Prediction uses past features only.

The selected development rule sends histories with old unique fraction ≤ 0.18131
to 16 summaries and others to 8. Its results on held-out users:

| Policy |Held-out NDCG@10|
|---|---:|
| Fixed8 |0.029656|
| Learned rule |0.027872|
| Independent mixture matching the rule's arm shares |0.028820|

Rule−fixed8 Δ=−0.001784,95% paired CI [−0.004316,+0.000693]. Rule−matched mixture
Δ=−0.000948,CI [−0.002704,+0.000814]; mixture shares are recomputed in each user
bootstrap resample. Thus no routing gain was established; the negative estimate
also does not establish that the rule is worse.

A secondary fixed bank of 51 constant/stump/two-cut interval policies selects the
same rule. Fivefold out-of-fold routing gives 0.030469 vs fixed8's 0.030720;
Δ=−0.000251,CI [−0.001415,+0.000847]. That interval is conditional on the recorded
out-of-fold predictions and does not include policy-refitting uncertainty.

## Next informative experiment

Train **one shared MixedBudget PairMLP + HSTU** with budgets 4/8/16. Evaluate the
same weights at each budget, rather than comparing separate models. Use several
future anchors per user, report average user utility and bootstrap users jointly
across budgets. Fit/freeze routing on per-user validation before inspecting test.
Keep one best checkpoint and per-user outputs. The runner now saves per-user
validation rows at the same checkpoint-selection point for future runs.

This checks whether budget effects are predictably heterogeneous. Adapting
compression to different regions inside one history remains a separate hypothesis;
the current global-budget analysis neither establishes nor rules it out.

## Reproduce without GPU

Private histories, user features and prediction rows remain outside the repository.
The extraction and analysis scripts use NumPy/PyArrow; plotting additionally uses
Matplotlib. Aggregate JSON is versioned beside this report.

```bash
python scripts/prepare_history_features.py --cohort COHORT_DIR --evaluation EVAL_DIR \
  --run-output RUN_DIR --output ANALYSIS_DIR
python scripts/analyze_history_budgets.py --run-output RUN_DIR \
  --features ANALYSIS_DIR/history-features.parquet --output ANALYSIS_DIR \
  --permutations 10000 --bootstrap-draws 5000
python scripts/plot_history_budget_results.py --run-output RUN_DIR \
  --analysis ANALYSIS_DIR/history-budget-analysis.json --output ANALYSIS_DIR/results
```
