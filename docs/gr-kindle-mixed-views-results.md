# Full32 / Pair32 mixture: completed results

The50/50 batch mixture improved selected VAL NDCG@10, but did not improve TEST
over the original Pair32 in this pilot. This is not a confirmed benefit or harm:
all six paired-user confidence intervals for selected VAL/TEST comparisons
include zero. There is one previously examined generator/index seed and only958
targets per split. Random mixing of individual pair/single SIDs within a history
was not tested or launched.

Private kernel `carasik/rhpr-kindle-mixed-resume/2` COMPLETE, source
`88722e9081aae120e9af261f550d50fe74864da6`. The same fit resumed from epoch30/
8280updates, completed60epochs and selected55 by raw VAL NDCG@10. The original
30curves and six prediction files are preserved exactly. Optimizer, LR,
scaler, EMA, cases, indexes, initializer and RNG restoration were verified.
Two AMP updates were skipped in the continuation, counted separately from
16558successful updates:16558+2=60×276batches. No numerical failure.

| Model | VAL NDCG@10 | TEST NDCG@10 | TEST hits/958 | B1 forward GFLOPs | Warm epoch, s |
|---|---:|---:|---:|---:|---:|
| Recent20 |0.020492|0.014722|24|0.4271|31.25|
| Recent32 |0.016899|0.014447|28|0.7048|55.84|
| Pair32 |0.015966|0.010651|21|0.5165|37.59|
| Mixed32 |0.018937|0.009874|20|0.5165|49.48|

Every model selected epoch55 and completed60. Controls are reused; none were
retrained or re-evaluated on GPU. The mixture received the original targets
exactly once per epoch,4,285,310Full32 and4,182,910Pair32 examples in total
(50.60%Full). All evaluation uses Pair32:8 old pair SIDs+16 recent singles,
each3+1, plus USER=97encoder positions. Both tables and the Transformer are
unchanged; each model has5,108,992nominal parameters. FLOPs are supported B1
teacher-forced forward operations including output heads, FMA=2; no beam,
backward or offline pair/text preprocessing. Equal forward cost is not a
measured serving latency result.

Compared with Pair32, selected VAL improves18.61%; TEST declines7.30% in point
estimate. The TEST difference is−0.000777, conditional95%shared-user bootstrap
CI[−0.005460,+0.004115]. Mixed gains12 former misses but loses13 original hits;
15users improve NDCG,16worsen,927tie. VAL gains20hits and loses16, net+4;
its NDCG difference+0.002972 has CI[−0.002626,+0.008626]. Mixed beats Pair32 on
5of7late VAL checks30–60, including55and60, but these are correlated checks of
the same trajectory, not independent confirmations. Fixed40 favors Pair32.

TEST versus Recent32 is−31.65%, CI for absolute difference
[−0.010313,+0.000965]. Recent20 still has better point quality and lower forward
cost than Mixed32. There is no observed Pareto improvement.

The warm epoch median rises31.62% relative to Pair32. Total recorded optimizer
training is3247seconds(54.12minutes), versus2357seconds(39.29minutes):+37.75%.
The latter includes a second cold compilation after resuming; warm timings also
come from separate sessions and are not a simultaneous benchmark. The recovery
session lasted1680.15seconds(28.00minutes). None of the first30epochs were
retrained. Both complete best/last checkpoints were downloaded and verified;
the best checkpoint SHA matches the one used for final TEST.

The two prior exploratory groups, using the unchanged recent-context popularity
cutoff2.300820444821813, do not show a general rescue:

| TEST group | Users | Pair32 NDCG@10 / hits | Mixed32 NDCG@10 / hits | Recent32 NDCG@10 / hits |
|---|---:|---:|---:|---:|
| Less-popular recent16 |502|0.004209 /5|0.004994 /7|0.011788 /11|
| More-popular recent16 |456|0.017742 /16|0.015246 /13|0.017374 /17|
| Target has1–5 TRAIN target positions |425|0.000784 /1|0.003217 /4|0.006773 /5|

The rare-target result is a small lead consistent with the original motivation,
but only three extra hits. It does not identify a cause: the mixture changes
single-input exposure, Pair-format exposure and regularization together. Of
the16Recent32 TEST hits originally missed by Pair32, Mixed recovers4; on VAL it
recovers3of14. Other original Pair hits are lost, so this is not a general fix.

Independent CPU replay verified52prediction files(12VAL+1TEST per model),
996320candidate paths, all NDCG/Recall5/10 values,958same users/targets,
44completed-parent pins,15resume-parent pins,10input hashes and the private
checkpoint archive. There are1921invalid paths across all files and zero
duplicates. The first30epoch records/predictions, all60format schedules,
epoch/update/LR/EMA accounting and selected checkpoint identities match.
Full regression:1119tests,152optional skips,0failures. The scientific plot
was regenerated after the final layout adjustment.

Evidence and replay:

- Aggregate: `docs/gr-kindle-mixed-views-results.json`.
- CPU CLI: `scripts/research/analyze_kindle_mixed_views.py`.
- Private raw artifacts, best/last weights, ranks, CSV and PNG/PDF curves:
  `/workspace/artifacts/kindle-pilot/mixed-resume-results`.
- Original interrupted partial fit and failure evidence are retained separately;
  submission receipts remain historical and completion is recorded separately.

Intervals condition on the fitted checkpoints and users, omit seed/index/item
clustering and selection uncertainty, and are not corrected for multiple
comparisons. TEST had already been examined before this follow-up, making the
result exploratory. Epoch60is a cap, not demonstrated convergence. CUDA resume
does not guarantee bitwise equivalence with uninterrupted GPU training.
No extra seeds, training, inference variants or within-history mixtures launched.
