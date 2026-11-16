# Steam Recent10/20 across two cohorts and three generator seeds: completed

Private kernel `carasik/rhpr-steam-recent10-20/1`, executable `e710aec495fce54dac88979da910e004bb639854`, completed successfully. Nine fresh models plus three hash-verified saved controls produce twelve cells/six paired comparisons. Runtime **7,221.97s (120.37min)** after bootstrap, excluding queue. No further training or inference during collection.

The original+13.3%TEST estimate is not a stable effect size. Recent20 wins five of six TEST pairs, but gains vary from-12.23% to+15.21%. The mean TEST gain is+4.97%on cohortA,+0.67%on cohortB,+2.83%overall; conditional paired-user intervals include zero. This is a weak positive direction, not established large history utility or equivalence.

## All six paired results

| Cohort /data seed | Generator seed | VAL10 | VAL20 | TEST10 | TEST20 | TEST change | Best epoch10/20 | Stop epoch10/20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A /4242 | 42 | 0.023911 | 0.024504 | 0.016610 | 0.018821 | +13.31% | 55/55 | 75/75 |
| A /4242 | 101 | 0.022571 | 0.023228 | 0.020950 | 0.021569 | +2.95% | 30/50 | 70/70 |
| A /4242 | 202 | 0.023193 | 0.026837 | 0.020545 | 0.020604 | +0.29% | 30/45 | 50/65 |
| B /1729 | 42 | 0.026059 | 0.026905 | 0.018990 | 0.019070 | +0.42% | 45/45 | 65/65 |
| B /1729 | 101 | 0.022809 | 0.025382 | 0.020233 | 0.017758 | -12.23% | 50/45 | 60/65 |
| B /1729 | 202 | 0.026070 | 0.024356 | 0.018293 | 0.021076 | +15.21% | 45/30 | 65/50 |

All twelve models stop by the unchanged EMA policy, at50-75epochs; none reaches120. VAL also favors20in five of six pairs, but its losing cell isB/202 while TEST losesB/101. Selecting the seed or model variant on TEST would conceal this variation.

## Means and uncertainty

| Cohort | Mean TEST10 | Mean TEST20 | Relative mean change | Absolute difference | Conditional95%user interval |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0.019368 | 0.020331 | +4.97% | +0.000963 | [-0.002168, +0.004143] |
| B | 0.019172 | 0.019301 | +0.67% | +0.000129 | [-0.002480, +0.002699] |
| Both, equal weight | 0.019270 | 0.019816 | +2.83% | +0.000546 | [-0.001486, +0.002578] |

These are ratios of mean NDCG values, not arithmetic means of percentage changes. Within-cohort means average the three generator seeds. There are **4,000unique users**, each evaluated under three seeds; not12,000independent targets. Bootstrap uses10,000paired-user draws per cohort, preserving all three seed evaluations of each resampled user, with cohorts resampled independently. Seeds and their fitted/checkpoint-selected models are held fixed. No seed or item-cluster uncertainty, selection correction or multiplicity correction is included. Every individual pair interval also crosses zero.

VAL mean change:A+7.02%,B+2.27%,both+4.56%; both absolute delta+.001100,conditional interval[-.001047,+.003257]. Paired TEST delta seed SD:A.001116,B.002629. The six sign outcomes are dependent because both users and generator seeds repeat; do not interpret five wins as six independent statistical trials.

Other TEST metrics: mean NDCG@5 changes+3.90%onA,-2.57%onB; Recall@10 changes+2.60%/+4.64%. The12,000user-seed evaluations have468vs485hits10, but correspond to4,000unique users.

## Curves and computation

Full six-panel VAL ranking and CE plots are saved in `analysis/learning-curves.png/.pdf` and `analysis/validation-loss-curves.png/.pdf`. Paired conditional intervals: `analysis/paired-differences.png/.pdf`. Curves were visually inspected.

Best ranking epochs vary30-55; lowest VAL CE epochs vary25-50. EMA early stopping is an operational criterion, not a proof of convergence. In some Recent10runs the CE minimum is at the stopping epoch even though ranking peaked earlier; do not claim that every fit has fully plateaued in loss or that an extension is guaranteed to help.

Forward FLOPs: Recent10=.217327232G,Recent20=.427087952G (**+96.5%**), fixed padded B1, teacher-forced four-code forward,FMA2. These omit backward,beamsearch,item encoding and serving/caching overhead; they are not online latency. Mean warm train epoch over the six cells per window:14.68s vs28.78s. A small quality point gain here does not establish a Pareto improvement.

## Item concentration and exploratory sensitivity

The mean gain is still concentrated and includes substantial cancellation:

- CohortA: Rainbow Six Siege has23TEST targets and contributes+.002882to the seed-mean total NDCG difference, larger than the full+.000963net gain. Leaving those targets out without retraining gives-.001942on the remainder.
- CohortB: CS:GO has56TEST targets and contributes+.001261, larger than the full+.000129net gain. Leaving them out gives-.001165on the remainder.

These are post-hoc accounting decompositions, not causes or replacement metrics. They do not justify filtering these games or selecting a new sample around them. Unlike the earlier30vs20pilot, GTA V is a negative contribution onA and positive onB for this10vs20comparison. The identity of favorable games is not stable enough to infer a broad rule. Full positive/negative item contributions, user credits and all group slices are saved in analysis JSON/CSVs.

Among users with distinct train/VAL/TEST boundary days, mean TEST10/20:A.019231/.021192,B.020500/.019830. Thus this sensitivity group changes theBmean sign; it is not a new primary split and does not resolve within-day order.

## A concrete architectural hypothesis for later testing

The preserved config uses `num_user_bins=2000`; both cohorts have user IDs1..2000. `pair_unique4._encode4` selects `user_embedding[user_id % num_embeddings]`, so there are exactly2,000distinct USER bins for2,000users: one learned embedding per user. Those embeddings can accumulate static preference information across all TRAINexamples, potentially reducing the marginal need for a longer visible window. This is an allowed personalization channel shared by both baselines, not evidence of target leakage.

This code/data fact has been verified, but its effect has not been ablated. It is a specific hypothesis for a future USER-token ablation; it does not explain these outcomes causally and no such GPU run was submitted.

## Audit and artifacts

All167prediction files replayed (155VAL+12TEST),6,680,000candidate paths,899invalid,0duplicates. Invalid candidates retain ranks. All four input/resource config identities, both cohort data hashes, independently rebuilt prefixes/holdouts, identical original embedding/centroid/SID arrays, per-seed initializers, update counts, LR/EMA histories, rawVALselection and TEST metrics match.

All24best/last checkpoints (18new+6reused) were loaded, checked finite and hashed; last optimizer/scaler/scheduler/RNG/early-stopping/encoded-case states verified. Nine fresh fits total560epochs/5normal AMPskips; all12cells total775epochs/10skips. No numerical/precision/LR fallback.

New artifacts: `/workspace/artifacts/steam-recent10-20-results`; reused controls remain in `/workspace/artifacts/steam-history-results` and `/workspace/artifacts/steam-history-replica-results`. Analysis tables/figures are in the new `analysis/` directory. No raw user records, predictions or weights are committed to Git.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/history-mpl PYTHONPATH=src \
python scripts/research/analyze_steam_recent10_20.py \
  --artifacts /workspace/artifacts \
  --output /workspace/artifacts/steam-recent10-20-results/analysis
```

Full suite:1,167tests,152optional skips,0failures,153.943s. Summary `gr-steam-recent10-20-results.json`; completion `gr-steam-recent10-20-completion.json`. No new experiment has been authorized or submitted during collection.
