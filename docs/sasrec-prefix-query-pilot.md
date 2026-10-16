# SASRec with an old-prefix-conditioned query

Configuration: `experiments/yambda_sasrec_prefix_query_pilot.json`, seed 42.
This pilot trains Recent128, four-summary cross-attention and four-summary
Query SSM from scratch with the same two-layer SASRec predictor. It replaces
the previous HSTU/stale-recent-query training protocol; scores from that
experiment are not a controlled comparison with this one.

For each user's complete train-only sequence `T`:

```python
X, Y = T[:-1], T[1:]
H, R = X[:-128], X[-128:]
Q = H[-128:]
Z = compressor(H, Q)                     # four summaries
states = SASRec([Z, R])                  # causal attention
loss = next_item_loss(states[-128:], Y[-128:])
```

Recent128 uses `SASRec(R)` with exactly the same labels and negative samples.
One forward supplies 128 supervised positions. The query reads only the old
prefix, so summaries cannot disclose any of the 128 raw-tail events. SSM uses
the differentiable final scan output, not the native API's detached final state.
All of `H` is retained, about 16K–27K events in this cohort; a 32,768-event
safety ceiling raises an error rather than silently trimming longer inputs.
The query generator uses four learned slots to attend over the last 128 old
events. Cross-attention reads the full old prefix with those queries. QSSM
conditions its selective scan on those queries and emits four summaries.

Every pass repeats the same final 128 labels per user. Only user order,
negative samples and dropout vary. This uses the full old history as context
but does not train on every next-item transition. There are 4,427 users and
566,656 supervised positions per pass. The first raw state sees one raw event;
the last sees 128. Evaluation uses the last state with all 128 raw events.
This is a fixed-window SASRec-style pilot, not a claim of exact equivalence
between all training states and separate production prefix forwards. The
existing recency positional encoding remains relative to the full window.
Short-history generalization and incremental caching are outside this pilot.

All models use d128, two layers, four heads, FFN multiplier four, dropout 0.2,
Time OFF, AdamW at 0.001, weight decay 1e-5, clipping at 1.0 and 512 sampled
negatives per user/window, shared across that window's targets. Negatives are
keyed to user and pass independently of physical batch size or model RNG.
Backbone and item-table initialization are identical. The compressor is
attached after SASRec initialization so its special SSM step-size bias is
preserved. Native CUDA output and gradient checks, including zero summary
gradient to the entire raw tail, are mandatory before quality training.

Physical batches 8/16/32 are profiled with real full-history updates and
effective batch 128. Disposable profiling models do not alter training
initialization. A common maximum pass count is selected from measured costs
before validation scores are inspected, between 30 and 100, within a
10,800-second budget with safety factor 1.25. Validation runs every ten passes;
early stopping has patience three scheduled checks and cannot stop before
pass 30. The earliest strictly best validation macro-user NDCG@10 selects each
checkpoint. Ties count as non-improvements. A budget-limited result does not
establish convergence. Cost outputs include successful updates, AMP retries,
windows/second, supervised positions/second and seconds/pass.

The existing global-time split, train-visible catalogue, daily anchors and OOV
policy remain fixed. Validation/test rank the full 1,049,612-item catalogue,
with NDCG@10 primary and NDCG@100 secondary. Evaluation batches independent
anchors and takes only each example's final state. Its history includes all
events observed before that anchor, up to the same ceiling. Actual full-history
input hashes are recorded separately from the evaluation dataset's reference
16K hashes. Test is evaluated once per selected model; paired user bootstrap
intervals describe uncertainty conditional on this training seed. The held-out
period has already informed earlier research, so results remain exploratory.

Inference timings use batches 1/8/32 and include uncached compression. History
encoding and full-catalogue ranking are measured separately as well as together.
Reported operation counts are explicit major-operation estimates, not a claim
to count every native scan instruction. Best FP32 weights and the final model,
optimizer, scaler and RNG state remain private Kaggle outputs. Standard scalar
metrics, summary, manifests and richer diagnostics are downloaded by Actions;
large checkpoints are excluded from automatic artifact download.
