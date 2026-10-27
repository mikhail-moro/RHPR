# Frozen-model context truncation: completed

Run: `carasik/rhpr-context-length-replay`, code source
`32b7ce0089b3aab3c70483c07a319b28170f64cb`, parent source
`10e716dfe69a7afef766e7647c876a07478c2714`. Both frozen checkpoints from epoch 60,
eight validation-only passes in FP32, batch 32, no fitting/test. Body: 227 seconds;
whole notebook about 290 seconds. All eight raw beam files independently verified.

The same 1,019 users with at least 20 TRAIN events are compared at every length:

| Visible events | Original NDCG@10 | ALS-swapped | Relative swap change |
|---|---:|---:|---:|
|3|0.03776843|0.03407376|-9.78%|
|5|0.05699833|0.05785161|+1.50%|
|9|0.06759191|0.07381307|+9.20%|
|20|0.05611497|0.06650280|+18.51%|

Primary prespecified interaction, `(swap20-original20)-(swap3-original3)`,
is +0.01408250 NDCG@10; paired-user bootstrap 95% interval
[+0.00376442,+0.02362841]. Target-item cluster resampling sensitivity also
stays positive: [+0.00238178,+0.02613200],694 target-item blocks. There are 108
positive / 75 negative user interactions.71 items contribute positively and 49
negatively. The largest positive item is10624 SHANY Nail Art Set, 6 users,
3.31 summed points,10.2% of gross positives; the remaining net stays positive.

Secondary interactions have the same sign but intervals include zero:

- 4,242 users with history >=9, context 3 to 9: +0.00221189[-0.00223314,+0.00667018].
- 10,980 users with history >=5, context 3 to 5: +0.00180736[-0.00031394,+0.00409753].

Full 20-event context reproduces the parent predictions exactly for both models.
Unchanged-input rows retain identical predictions at every length; no model
state or SID changes. Full-user order/batch composition and padded width 20
are fixed across passes. Thus the primary result is a context-length interaction
within this one fitted pair, rather than a change in user/target composition.

Do not generalize to all users, training seeds or epochs, or to pure semantic
versus CF recommenders. Both models are NIP generators with different frozen
SID maps. Learned user-bin information and training exposure stay fixed, so this
does not simulate new users. Secondary metrics/cohorts remain exploratory.
Both models have their highest absolute score at 9 visible events; 20 hurts both,
but hurts the swapped index less. Relative swap benefit is not monotone absolute
benefit from adding history.

Artifacts: `/workspace/artifacts/sid-context-ablation/analysis/README.md`,
`context-length-effects.png`,full-user Parquet,paired metric/interaction JSON,
target contribution sensitivity and hash verification. Raw beams and inputs
remain on the private Kaggle outputs. No further experiment was submitted.
