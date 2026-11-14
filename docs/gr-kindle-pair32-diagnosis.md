# Pair32 vs Recent32: exploratory error diagnosis

Saved predictions and cached embeddings only. No fitting,new model inference,
Kaggle launch or quota use. Same already examined seed42. All four original
models selected epoch55;this analysis does not reselect checkpoints by TEST.
Full aggregate diagnostics and all tested groups are in the JSON;private user
features and examples stay in local CSV/JSON rather than Git.

## The large relative TEST loss is not a uniform learning collapse

Selected VAL NDCG .015966 vs .016899(−5.52%),TEST .010651 vs .014447(−26.27%).
Pair32 beats Recent32 VAL NDCG at10of12saved checks,including5of7at30–60.
Mean paired VAL delta across30–60checks is approximately+.000012.
These are correlated observations on the SAME users,not independent repeats.

On VAL,Pair32 loses14control top10hits and gains15,31vs30total;ranking of
shared hits accounts for the net loss. On TEST,it loses16control hits and
adds9,21vs28total. Of16lost TEST hits,15targets leave the saved beam20 and
one falls to rank17. Across the12shared TEST hits,Pair32's rank contribution
actually improves by+.4387summed credit. Lost membership dominates the loss;
final-beam absence alone cannot identify the decoding stage of failure.

The net TEST difference is−3.6363summed discounted credits across958users.
Two losses of control rank1→outside beam20 contribute−2credits(55.0%ofnet
loss,23.8%ofgross negative credit). This demonstrates sensitivity to a few
high-rank outcomes,not a reason to remove cases or claim equality. The gap
remains negative after omitting those two as a sensitivity calculation.

## Strongest exploratory lead: less popular recent context

Popularity here is the number of TRAIN next-item target positions for each
book. For each user take mean log(1+frequency) over the latest16observed books,
which remain UNSQUEEZED in Pair32. Cut at the VAL-covariate median2.30082044
and reuse that threshold on TEST;the threshold is not optimized on NDCG.
Choosing this feature after inspecting many groups remains exploratory.

| Split/context group | Users | Recent32 NDCG@10 | Pair32 NDCG@10 | Delta |
|---|---:|---:|---:|---:|
|validation: less popular recent context|479|0.016051|0.009888|-0.006163|
|validation: more popular recent context|479|0.017747|0.022043|+0.004296|
|test: less popular recent context|502|0.011788|0.004209|-0.007578|
|test: more popular recent context|456|0.017374|0.017742|+0.000369|

The less-popular context group loses on all five late VAL checks40/45/50/55/60;
the more-popular group gains at fourofthese five. Same users/trajectory,not
replication. Low-context group loss is also visible among targets with>5train
positions:VAL .02073vs.03371(n200),TEST .00942vs.02109(n189),so the observed
pattern is not confined to the rarest-target bucket. These cells still have
only4–13correct top10targets/model and do not identify a cause.

In the low-popularity context TEST group,the conditional user-bootstrap delta
CI is[−.01596,−.00053];VAL interval crosses0. These posthoc subgroup intervals
exclude seed,index,item clustering and selection uncertainty and have NO
multiple-comparison correction. Do not present a confirmed subgroup effect.

Pair32 recommends somewhat more popular items:among valid top10candidates,
share with>20TRAIN target positions is23.57%→26.31%onTEST and23.46%→27.60%
onVAL(Recent32→Pair32). Mean candidate frequency15.78→16.47onTEST.
This is a small consistent shift in outputs,compatible with a popularity bias
hypothesis;it does not establish that popularity causes the quality loss.

## Rare targets and concrete examples

On TEST,targets with1–5TRAIN positions:425users,Pair32 .000784vsRecent32
.006773,1vs5top10hits. This bucket accounts for70.0%ofnet TEST credit loss,
but only four net hits and a few top ranks drive that percentage. At6–20
positions,415users:15vs19hits. At>20positions,79users:5vs4hits.

Two largest TEST losses have a related book already in the uncompressed tail:

* `Renewing the Soul: Souls of Chicago series`(8TRAIN positions):controlrank1,
  Pair outside beam20;`Protecting the Soul: Souls of Chicago Series`is among
  the latest16observed books. Pair's leading predictions go to other series.
* `Dark Hunter’s Query (... Book56)`(5TRAIN positions):controlrank1,Pair outside
  beam20;Book49ofthatseries is in the latest16. Pair instead ranks Book53/54
  first/second. Target SID[84,203,91,0],those predictions[84,203,190,0/1]:
  the first TWO index levels match,but the final item is different. This is
  evidence about the final candidates,not the step at which target was pruned.

The examples show why "the only useful clue was merged away" is insufficient:
the recent clue survives in both. They do not prove that its representation,
attention use or learned associations are equally good after separate training.
The two examples also differ:one is a wrong-volume error within the same SID
branch,the other leaves the target branch. There is no single demonstrated
failure mechanism.

## Explanations not established by these diagnostics

* Newly observed pairs occur for830/958VAL and847/958TEST users. Their quality
  association changes across splits and does not consistently isolate the loss.
* History-length bucket effects change across VAL/TEST. No stable critical user
  history length found on this run.
* More dissimilar pair constituents do not cleanly separate outcomes across
  splits. A target-conditioned embedding attenuation split has worse point
  estimates in its upper half on both splits,but is weak,posthoc and depends
  on the unavailable future target;not a causal semantic-loss diagnosis/router.
* Pair text embeddings are slightly closer to the first constituent on average
  (mean cosine difference.0167VAL/.0175TEST),but its subgroup quality association
  reverses across splits. No established first/second-position mechanism.
* Seen-item echoes do not increase:Pair32 top10seen fraction24.05%vs24.58%
  Recent32onTEST,and24.50%vs24.75%onVAL. No filtering applied.
* Earlier checks already ruled out suffix overflow/obvious dead-code collapse;
  preserved unique3+1IDs do not guarantee useful recommendation semantics.

## Why prior small-loss pilots are not directly comparable

| Cohort | Books | TRAIN positions | TEST targets | TEST target<=5TRAIN positions | Median target frequency |
|---|---:|---:|---:|---:|---:|
|old_dense|4969|92133|2625|292(11.1%)|19|
|long_core|16177|141137|958|464(48.4%)|6|

Cohort/catalogue selection,window,model widths and training policy differ
between pilots. This is a support/sparsity comparison,not an isolated causal
explanation of the Pair penalty. The current catalogue is3.26times larger,
TRAIN supervision only1.53times larger,and TEST2.74times smaller. Both the
learning problem and the sensitivity of relative NDCG gaps have changed.

Working hypothesis:under this sparse task,pair inputs may shift the generator
toward broader,more popular patterns and hurt some less-supported preferences.
The strongest next mechanism check would keep trained weights and recent
positions fixed while masking/replacing only old summaries;that would test
whether old summaries interfere with the still-present recent clue. It has
NOT been run. This analysis does not authorize more GPU runs or TEST tuning.

Reproduce:`scripts/research/diagnose_kindle_pair32.py --results RESULTS --data
LONG_INPUT --item-embeddings LONG_WINDOW_ITEM_CACHE --old-data OLD_DENSE_INPUT
--output OUTPUT --plot`. Private evidence directory:
`/workspace/artifacts/kindle-pilot/pair32-diagnosis`. Saved diagnosis.json,
user-features.csv,groups.csv,extreme-examples.json,andPNG/PDF.1109regression
tests passed with152optional skips;all analysis executes on saved arrays.
