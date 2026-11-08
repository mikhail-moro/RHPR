# Intermediate proxy architectures: collected results

Private `carasik/rhpr-ml1m-middle-proxy/1` completed all 24 fits without failure
in **86.18 minutes** on 2×T4. Executable source
`2fd232815bc0435f9c54b0f76f5c7964116f5561`; no new runs were submitted while
collecting these results. TEST remains closed.

The fixed primary comparison is epoch10, mean of generator seeds31415/27182,
across the same six frozen SUM/GradNorm indexes ×42/101/202. The target is
full-TIGER selected VAL under min60/max100/patience20, seed161803. All fits
preserve 151166 training windows,2786 VAL users,history20,FP16train/FP32eval,
SDPA,physical/effectivebatch512,Adam LR.001,VAL5 and unmasked beam20.

| Architecture | Shape | Parameters | Own mean VAL NDCG@10 | Spearman vs selected full | Two-seed cost/index |
|---|---|---:|---:|---:|---:|
| Reused narrow |4+4,d64,FF512,H6,kv32|1,375,040|.033584|+.4286|13.36min|
| middle_width |4+4,d96,FF768,H6,kv48|2,897,952|.049051|−.2000|17.20min|
| half_depth |2+2,d128,FF1024,H6,kv64|2,748,416|.050091|−.6000|10.66min|

Cost is the sum of both seed fits' training plus recorded ranking-validation
seconds, averaged across indexes. It excludes loss validation, initialization,
I/O and setup. Narrow was measured in earlier sessions: this is a descriptive
cost comparison, not a controlled GPU throughput benchmark. Mean training
seconds/epoch were38.86,50.00,31.01 respectively. No AMP updates were skipped
in these 24 fits. Each consumed10×151166 examples and2960 optimizer updates.

**Neither new architecture improves the primary proxy.** The larger models
produce better recommendations themselves while ranking tokenizers less like
the selected full model. Standalone prediction quality does not guarantee
ranking transfer. This does not isolate depth as a causal factor: width,FFN
and attention projections also differ between these two designs.

| Proxy | Chosen index | Full VAL score of its choice | Relative regret vs full best | Full best retained in proxy top3? |
|---|---|---:|---:|---|
| narrow10 |sum_seed202|.069456|4.82%|yes (rank2)|
| middle_width10 |gradnorm_seed101|.066399|9.01%|yes|
| half_depth10 |sum_seed42|.069266|5.08%|no|

Full best is `gradnorm_seed202`,NDCG@10=.072976. Regret is descriptive,
relative to this best score among six, not a statistically established TEST
loss. Both new architectures have zero overlap between their top2 and the
full model's top2. Narrow has one of the two.

## Secondary checks and uncertainty

At epoch10,middle_width individual-seed correlations are−.3714/−.0857;
half_depth−.4857/−.6000. Correlation between the two seeds' index rankings
is−.4286 for middle_width,.6000 for half_depth,.8857 for old narrow.
Middle_width epoch5 mean rho=.4857 looks better, but its two seeds give
+.5429 and−.4857. This secondary, unstable result does not justify selecting
epoch5 after seeing the answers or declaring an improved recipe.

Within the three SUM indexes,mean epoch10 rho is−.5 for narrow,−.5 for
middle_width and−1.0 for half_depth. Within the three GradNorm indexes it is
+.5,+.5,−.5. Groups of three are descriptive; none confirms a portable proxy.

5000 shared-user conditional bootstrap draws give rho95% intervals:
narrow[−.5429,.8857],middle_width[−.8857,.6571],half_depth[−.8286,.6571].
They condition on already selected checkpoints and indexes, exclude training
and tokenizer seed uncertainty and selection correction, and do not establish
statistically significant differences between architectures.

These are six related, previously examined candidates on one dataset and the
same VAL cohort, with one full seed. Historical narrow's stronger correlation
on the earlier12-index/fixed30 target does not carry over as a validated result
for this selected long-training target. Early-full best≤15/≤20 correlations
are.6571/.8286, but share the target training trajectory; they are exploratory
controls, not independently validated search methods.

## Verification and retained evidence

Independent CPU replay recomputed NDCG/Recall5/10 from **66 saved files**:
48new predictions(epoch5/10),12old narrow epoch10,and6selected-full predictions.
User/target order,finite scores,valid catalogue paths,zero duplicate/invalid
predictions,case/SID/source/initializer pins,exact fit matrix,worker12+12
assignment,epoch schedules and counters were checked. Saved aggregates match
to absolute tolerance1e−12;raw metrics/summary/manifest are present and valid.
The full-reference and narrow evidence files match their prespecified SHA256s.
Full weights stay in the original private Kaggle kernels;new weights stay in
the completed middle-proxy kernel. The downloadable evidence ZIP omits weights
but includes the predictions,cases,configs,curves,metadata,replay source and
standalone PNG/PDF graphs.

- Results: `docs/gr-middle-proxy-results.json`.
- Replay: `scripts/research/analyze_middle_proxy.py`.
- Protocol/config: `docs/gr-middle-proxy.md`,
  `experiments/ml1m_middle_proxy_2t4.json`.
- Previous full-target analysis: `docs/gr-proxy-plateau-results.md`.

No additional fits,epochs,seeds,architecture search or GPU runs were launched.

Repository regression:1035 tests passed with91 optional dependency skips;
CPU evidence replay passed. No changes to the training implementation were
needed during this collection.
