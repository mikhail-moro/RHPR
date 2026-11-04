# Three-seed MTL comparison with unambiguous earlier holdouts

User authorized2026-11-04MSK: fresh sum/UW/GradNorm/PCGrad pipelines on seeds42,
101,202 (12fits), metadata+poster+plot, window20,30epochs,valevery5. No people,
new loss/HPO/window variants or Colab work. Original four-model MTL run preserved.

## Data protocol fixed before training

Reconstruct all91469retained canonical interactions from pinned original943-user
sequences plus2042embargoed events. For EACH user find the last TWO timestamp
groups containing exactly ONE retained catalogue film. The earlier is validation,
the later test. Keep a user only with at least two training timestamp groups
strictly earlier than validation (at least one training prediction).

903users remain,82234training interactions and80268strict-past training targets.
One val/test target each903. Excluded26users lack two singleton groups;14have
insufficient earlier training groups. Among retained users discard3098events
AFTER test;3173events BETWEEN validation and test remain TEST-CONTEXT ONLY.
Test context = train + validation + all intervening events, then last20items.
Validation context = train, then last20. No intervening event trains the model.
Order within historical ties uses timestamp/item ID; training cannot see any
same-timestamp event as context. This is still next-RATING prediction, not views.
Singleton groups are defined in the frozen metadata-filtered canonical catalogue.

All1273films remain in the catalogue and tokenizer fits, incl films absent from
new TRAIN: static catalogue metadata is not interaction supervision. Frozen
Sentence-T5/poster/description-onlyMPNet bytes reused; no people block active.
Original SHA256-manifest-verified Kaggle dataset remains the only ratings source;
the new split is derived deterministically at runtime under singleton-input/,
with source hashes, exclusions/per-user boundaries and complete case digests.
Preflight digests are pinned in config and checked after derivation.

## Matched model/training protocol

Fresh3500step continuous AE -> native residual KMeans initialization ->3000RQ-VAE
steps per model, same3x256/latent64; fresh RQ Adagrad accumulators, learned MTL
weights transfer, GradNorm reference resets. Same algorithm conventions as MTLv1
(see gr-movielens100k-mtl.md), incl unconstrained UW total reconstruction scale.
PCGrad projection totals now aggregate ALL optimizer calls; per500gradient/weight
probes are explicitly exported as mtl-learning-curves.json as well as stdout.
These additions do not alter optimization equations. No saved model reuse.

TIGER4+4/d128, FP16training/FP32eval, batch256/physical128, constantLR.001,
30complete epochs,9420attempted updates, val5/10/15/20/25/30, no early stop/scheduler.
Best original single-target validation NDCG10 checkpoint; one final full-catalogue
unmasked freebeam20test. Same input cases for ALL methods/seeds. Pipeline seeds
vary AE/RQ and generator initialization/order; not independent holdout repeats.
All seed-specific original full-size neural initialization hashes must match.

2T4 workers: sum+GradNorm on0, UW+PCGrad on1, sequential seeds42->101->202.
Estimated3–3.5hours session based on measured63.6min/4fit MTL pilot and smaller
new train; not measured. GPU quota15.35h at preparation; watchdog300min, per-fit
training guard2400s. No automatic retries, HPO, extensions, fallback or Telegram.
New private kernel carasik/rhpr-ml100k-singleton-mtl, preserve older notebooks.

The new endpoints remove arbitrary target selection INSIDE tied groups. This is
an evaluation-population change selected after earlier exploratory research, not
an untouched independent dataset. Old/new absolute metrics are not comparable.
Report all methods and all seeds, no test-selected checkpoints or methods.
