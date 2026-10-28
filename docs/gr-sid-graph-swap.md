# Fixed-tree SID graph placement

The Field RQ-VAE branch was closed by the user on 2026-10-27. Completed curves,
indices and downstream results remain under the existing Field experiment docs
and `/workspace/artifacts/`; no further Field fits are pending. Those tests did
not establish a mechanism explaining the T5 baseline advantage.

## CPU experiment

`experiments/amazon_beauty_sid_graph_swap_cpu.json` and
`scripts/probe_sid_graph_swap.py` implement the next experiment. Two completed
Sentence-T5 indices are reused: native RQ-VAE and the earlier normalized-residual
RQ-KMeans seed42. These are different quantizers, not matched reconstruction
objectives. All 12,101 complete four-token paths include a uniqueness suffix.

Moves swap whole paths, preserving every catalogue prefix occupancy, per-level
code frequency, semantic collision count and full-path uniqueness. They change
the item-to-path lookup; centroid nearest-neighbour export and suffix regeneration
must not replace the saved full paths in a downstream experiment.

The semantic graph is exact top-k positive cosine on the frozen full-metadata
Sentence-T5 embeddings. `max(cosine,0)` drops nonpositive edges; no arbitrary
negative-value shift introduces attraction. A zero-weight row contributes no
edges. A separate contrastive experiment provides relative repulsion.

The CF graph is binary item-user-incidence cosine using TRAIN only, minimum two
users per endpoint. One fixed random 75/25 user split separates graph optimization
from graph and count-predictor evaluation. Original recommender validation/test
events are not used for fitting, selecting or scoring this experiment. Catalogue
embeddings are transductive and frozen. Top-k graphs are directed; every affected
incoming/outgoing edge is included in a swap's exact delta.

Pair score is weighted mean LCP/3, ignoring suffix. Maximize CF + lambda*semantic.
Default k20,lambda0/.3/1/3,semantic-only; k5/k50 sensitivity changes both k values
together, not a full independently crossed k_sem/k_cf study. All trials share
20,000 uniform nonself pair proposals, seed2702. Save snapshots at0/1k/5k/10k/20k.
A random control matches lambda1's accepted-swap count at every snapshot, rather
than forcing20k destructive random swaps.

One InfoNCE experiment samples up to two weighted positives per CF anchor and16
distinct uniform catalogue negatives, excluding self and all its graph positives.
Candidates stay fixed throughout search. Logits are LCP/3 divided by temperature
0.2; maximize `(log(17)-CE)/log(17) + semantic`. Contrastive skill is not bounded
below by zero, and its lambda is not directly calibrated to pair-score lambda.
Uniform candidate negatives may include unobserved true neighbours. This is a
discrete contrastive graph-placement objective, not an implementation or
reproduction of PLUM, and does not require differentiability or a neural head.

## Diagnostics and interpretation

Primary transfer diagnostics: CF LCP and contrastive CE on unseen TRAIN users;
previous-original-item -> next-SID-prefix gain at depths1/2/3, count predictor fit
only on optimization users, evaluated only on held-out users. Also record rare,
middle,popular and CF-cold target cohorts. User split is fixed, not fresh per
snapshot; results are exploratory, with no downstream NDCG claim.

Legacy metrics keep their old definitions: all-TRAIN three-user-fold gain,
unigram/conditional NLL, co-user cosine, trimmed price R²; catalogue feature
variance explained, code usage/perplexity, unique triples and maximum collision.
Legacy crossfit is **adaptive** here: graph placement already saw some of those
users. It is a descriptive connection to old plots, not an independent test.

PCA99 and all catalogue code counts are fixed by construction; TRAIN-weighted
prefix entropy can change. Frozen-book latent relative MSE is recorded for
RQ-VAE. Normalized incoming-residual MSE is recorded separately per RQ-KMeans
level and must not be plotted as the same reconstruction metric. No decoder
reconstruction loss, learned field losses or generator NDCG is fabricated.

Exact delta caches are audited against full scans at every checkpoint. Whole-path
and all-prefix occupancy conservation are verified; save complete paths, source
owner permutation and every accepted move. Inputs, graph arrays, negative panels,
user split, proposals and executed-source hashes allow replay without retraining.

Run:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python scripts/probe_sid_graph_swap.py --config experiments/amazon_beauty_sid_graph_swap_cpu.json
PYTHONPATH=src python scripts/report_sid_graph_swap.py
```

No Kaggle dispatch, GPU usage or Telegram is part of this experiment.
