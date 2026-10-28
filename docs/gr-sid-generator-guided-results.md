# Frozen-generator SID search: completed screen

Private kernel: [carasik/rhpr-generator-guided-sid](https://www.kaggle.com/code/carasik/rhpr-generator-guided-sid), v1, COMPLETE.
Executable commit: `c3459221dfae7163b46de0d9ec28c4003a679735`.
No improvement established. The search accepted only two leaf-level swaps;
fresh-model validation NDCG@10 ended 3.07% below control, with paired intervals
including zero. This is a limited screen of one frozen-teacher search and one
fresh initialization, not evidence that generator-guided tree learning cannot work.

## Search result

- Existing original-index checkpoint: epoch20, 10280 updates, unchanged weights/RNG.
- All 131413 TRAIN cases scored initially; candidates mined from 16384 TRAIN cases.
- Mining found 13261 distinct pairs and retained 2000 candidates.
- Evaluated 214 candidates, accepted 2, changing 4/12101 items (0.0331%).
- Stopped at the configured scored-row budget: 1990375/2000000 rows;
  the next complete proposal would exceed it. Search took 535.6 seconds, below
  the separate 1200-second cap. It did not exhaust the candidate list.
- TRAIN mean path NLL: 7.452504137 -> 7.451465099, a 0.01394% reduction.
  Total log-likelihood gain: 136.543167 nats. Full TRAIN cache audit drift: exactly zero.
- Both accepted pairs share their original first two tokens; only the third
  token changes, and the suffix remains zero. No upper-branch reassignment survived.

| Original pair LCP | Evaluated | Accepted | Median likelihood change, nats |
|---|---:|---:|---:|
|0|161|0|-596.86|
|1|48|0|-307.75|
|2|4|2|+4.07|
|3|1|0|-29.51|

Every evaluated LCP0/LCP1 move had a negative gain, including the best of each
group (-0.855/-9.918 nats). These were not rejected merely for missing a tiny
positive acceptance threshold. Groups reflect the fixed mined candidates, not
an exhaustive or unbiased sample of all possible item pairs.

Accepted pairs:

| Items | Shared prefix | Changed third codes | TRAIN targets | Description |
|---|---|---|---|---|
|328 / 2126|111,246|125 <-> 181|20 / 99|Bdellium brush / Real Techniques brush starter set|
|4943 / 5058|238,27|226 <-> 148|9 / 30|Proactiv Green Tea Moisturizer 2.5oz / 1oz|

The affected TRAIN union has 873 cases. Visible validation histories or targets
change for 193/22363 users: 157 history cases and 36 target cases, with no overlap.
This is a very small intervention, despite the larger candidate-mining set.

## Matched fresh training

Both new generators used the exact shared neural initialization, seed/order42,
4977920 parameters, 20 full epochs, 2628260 cases, 10280 optimizer updates,
zero AMP skips and four validations. No teacher warm start, encoder/quantizer
refit, semantic penalty, test evaluation or extra seed was used.

| Epoch | Original NDCG@10 | Guided NDCG@10 | Relative change |
|---|---:|---:|---:|
|5|0.0256211|0.0268026|+4.61%|
|10|0.0294257|0.0297161|+0.99%|
|15|0.0341011|0.0334771|-1.83%|
|20|0.0364335|0.0353168|-3.07%|

Primary endpoint is epoch20, specified before the run. Recall@10:
0.0710102 -> 0.0695345 (1588 -> 1555 users hit). Best selected epoch is20 for both.
The endpoint paired-user 95% bootstrap interval for the absolute NDCG difference
is [-0.0023401,+0.0000804]; target-cluster interval is [-0.0029946,+0.0007088].
Both include zero. These 2000-resample intervals, seed2760, condition on fitted
weights and do not cover training/search-seed variance or multiple comparisons.
The apparent epoch5 win also has a target-cluster interval crossing zero.

Fresh validation teacher-forced NLL at20 is slightly lower for the changed index:
8.7928289 -> 8.7871336. Per-level changes are approximately
[-0.00282,+0.00755,-0.01223,+0.00180]. Thus even a small validation likelihood
improvement need not improve top10 ranked retrieval in this run. It does not
identify whether a wider beam would change the comparison.

Only four items changed, but all parameters are retrained jointly. Among the193
users with a changed visible history or target, NDCG goes 0.09755 -> 0.09532;
among the22170 other users, 0.03590 -> 0.03479. The latter account for98.3% of
the net global decline. This arithmetic is not a causal attribution: shared
parameters and optimization trajectories affect users with unchanged inputs,
and the group comparison is conditional on just two fitted models.

## Interpretation and next decision

The frozen teacher has already learned the original code meanings. Evaluating
large reassignment moves without letting it adapt penalizes disruption of those
meanings; evaluating changed history codes adds another source of disruption.
The rejection pattern is consistent with this anchoring, though it does not
prove that adaptation would make rejected moves useful. The accepted changes
mainly repair relative probabilities of nearby leaves, not the upper hierarchy.

Consequently this run provides no useful transfer gain and scarcely tests
reorganization of the tree. Increasing only the search cap would spend compute
on the same frozen acceptance criterion; there is no present evidence that it
would solve the limitation. A subsequent study would need an explicitly different
criterion or alternation that allows the generator to adapt to reassignment,
with an appropriate continuation control. No such run has been launched.

## Saved evidence and reproduction

- Numerical/report outputs: 68 files, 32097788 bytes, at
  `/tmp/rhpr-generator-guided-completed`; full model checkpoints remain on private Kaggle.
- Durable summaries, plots, per-user ranks, source/submission audits and search
  records: `/workspace/artifacts/sid-generator-guided/`.
- All eight saved validation beam files were rescored independently; user/target
  alignment and all eight teacher-forced array aggregates were checked.
- Accepted moves were replayed; complete path multiset, all catalogue prefix
  occupancies, cache gain, original-index hash and training stream were verified.
- Execution body:41.0 minutes, including search and two parallel fresh fits.
  No new training or inference was needed for this analysis.

Reproduce with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/rhpr-mpl-cache \
  /workspace/rhpr-check/bin/python scripts/analyze_sid_generator_guided.py --plot
```
