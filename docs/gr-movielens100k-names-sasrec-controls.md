# Names-in-text TIGER and ID-only SASRec controls

Two controls, each with seeds42/101/202: (1) text+poster+description, with cast and
director names moved into Sentence-T5 text and DeepWalk disabled; (2) SASRec using
item IDs alone. Config: `experiments/movielens100k_names_sasrec_controls_2t4.json`.
One new private kernel: `carasik/rhpr-ml100k-names-sasrec-controls`.

## Frozen movie features

Reuse existing notebook outputs only, checking immutable manifests, executable
SHAs, vector SHA256, source-film order and encoder provenance. Base poster and
MPNET-description vectors/masks come from completed multimodal v2. Replace the
text array with the exact prior full-metadata Sentence-T5 vectors from the
original ML100K text ladder stage07. They contain title with year, genres,
explicit year, runtime, TMDB keywords, all cast and directors. Same pinned model,
max2048tokens; observed maximum1587, zero truncated films. No new encoding,
feature upload or Dataset, and no change to the selected1273films.

Only text/poster/plot are active in the new quantizer. The old people array and
unused heads remain in the frozen schema solely to preserve neural constructor
and seed-matched initialization; their values cannot enter the encoder or loss.
Compare the new fit with both the existing no-people stage3 and graph-people
stage4 under the same AE recipe. Replacing the representation also changes its
fixed-normalization statistics; these are recalculated from the new text array.

AE up to5000steps (same spectral criterion: min2000/probe500/patience4), native
KMeans initialization, retained neural/BN state, fresh RQ Adagrad3000steps. Same
3×256books/latent64, per-head sum of normalized reconstruction MSE, no UW.

## Parameter-capacity match

Use the existing `rhpr.sasrec.model.SASRecModel` with identity compressor and no
semantic, time, user-ID or graph inputs. Architecture: d_model256,7layers,
4attention heads,FFN768(multiplier3),dropout0.1; tied item input/output matrix.
ID0is PAD; internal catalogue IDs0..1272 map to embedding IDs1..1273. Left-padded
strict-past cases are packed to the right-padding convention expected by the
existing SASRec model without changing event order or the last valid state.

A real loss backward determines which parameters are reached. Exclude every
`nn.Embedding` weight on both models, including position/user/relative-bias
tables, deduplicating tied parameters. Both have no inactive nonembedding weights:

| Predictor | Total parameters | Active nonembedding parameters |
|---|---:|---:|
| TIGER4+4,d128 | 4,977,920 | 4,590,464 |
| SASRec7×256,FFN768 | 4,940,800 | 4,609,536 |

Difference: +0.415%; preflight must stay within1% before training. Initial weights
are pinned separately for each predictor and seed. This chooses architecture by
capacity, not validation performance; no hyperparameter-search advantage is used.

This is **parameter and data-budget matching, not equal executed FLOPs**. For B1
and20valid history items, profiler-supported training-forward FLOPs(FMA=2,
including token/item output heads, math SDPA) are approximately0.42737GFLOPs for
TIGER and0.18737GFLOPs for SASRec. SASRec analytic matmul estimate is0.18702G.
TIGER expands20items into80SID tokens plus user prefix and has a decoder. The
native CPU profiler report records supported operations and exclusions; it is
not a backward or beam-search FLOP estimate. Offline Sentence-T5/AE/RQ costs are
also distinct from predictor capacity. No claim of equal wall-clock compute.

## Shared training and evaluation

Same943users/1273items, timestamp-group holdouts, strict-past contexts≤20items,
85,470cases per complete epoch and identical seed+epoch permutations. Same
20epochs, effective/physical batch256/128, Adam0.001/weight_decay0.0001,
gradient clip1,FP16train/FP32eval. Validation5/10/15/20, one selected-checkpoint
test. The common trainer/evaluator supports a narrow item-model loss hook and
writes SASRec predictions as `item_ids`, not `semantic_ids`.

SASRec uses exact full-catalogue cross-entropy, one next-item target per case,
no sampled negatives, PAD logit, popularity/seen-item mask, or semantic features.
Score all1273items and take top20. TIGER keeps SID-token CE and free beam20.
Both report full-catalogue item Recall/NDCG@5/10 on the exact same users/targets.
Their training loss scales and retrieval algorithms differ. TIGER's existing
hashed-user embedding is unchanged; ID-only SASRec does not take user identities.

Save selected/last model checkpoints, optimizer/scaler/RNG state, all generator
curves, validation/test predictions, input provenance, quantizer diagnostics and
capacity preflight. Additionally measure FP32 resident-model recommendation
latency at B1/B32,10warmups/50samples: all-catalogue SASRec top20 vs TIGER free
beam20, the same real training contexts with20valid events. Exclude input transfer,
CPU decoding and offline index construction; record median/p95/throughput and
benchmark case digest. This complements full epoch/evaluation times.

The prior seed42 pilot reproduced its AE/index but not generator optimization
bitwise. This experiment preserves the existing numeric settings for comparison;
seed matching is not a guarantee of deterministic generator trajectories.

## Execution

Six fresh fits across2×T4, balanced as two TIGER+one SASRec on worker0 and one
TIGER+two SASRec on worker1. Roughly40–60minutes plus setup/queue; SASRec speed is
not yet measured on the target T4. No extra HPO, seeds, retries, TPU or Telegram.
Old notebooks remain intact. New results must be compared by predictor, seed
and case schedule; mixed architectures are not required to share neural hashes.
