# MovieLens100K: context 50, seed 42

Authorized 2026-11-04: try a 50-item context with the ordinary T5 implementation.
Two fresh predictors: no-people control and actor-only graph (reconstruction
weight 0.1), in parallel on the two T4 GPUs of one private Kaggle session.

Config: `experiments/movielens100k_context50_seed42_30epochs.json`.
New private kernel: `carasik/rhpr-ml100k-context50`.

Only visible context changes relative to the completed seed-42 context-40 pilot.
Reuse the exact corresponding frozen original SID indices from
`carasik/rhpr-ml100k-actor-full-training v1`; no encoder, graph or quantizer fitting.
Initialize generators from their original constructor seed 42, without loading
old generator weights. Keep TIGER 4+4 layers, d_model 128, d_ff 1024, six heads
with d_kv 64, ReLU, T5 learned relative bias and eager attention. Each film has
four SID tokens, so the encoder width is 201 including the prepended user token.

Keep FP16 AMP training with FP32 parameters/GradScaler, FP32 evaluation,
physical batch 128/effective 256, LR 0.001, weight decay 0.0001. Keep the original
30-epoch protocol: validation at 5/10/15/20, verified optimizer/scaler/RNG resume,
then validation every epoch 21-30 and the existing plateau scheduler. Select by
validation NDCG@10 and evaluate test once. No new sampling, known-item filtering,
beam width change (20), FlashAttention, TPU, tuning, extra seeds or auto retries.
The original 7200-second per-model safety cap remains; no hardware/batch fallback.

Targets, user order and data splits remain unchanged: 85,470 training examples,
943 validation/test users, 1,273 catalogue items. Verify that the last 20 visible
items exactly match the reference schedule and pin every parent index/lineage
file by SHA256. Save per-epoch curves and validation predictions/losses,
selected test predictions, repeat occupancy, epoch durations and peak allocated
memory, plus all original runtime/index/initialization/resume provenance.

This is a one-seed screening run, not independent confirmation. Window 50 was
chosen after exploring the same holdouts; report validation and test separately
and retain unfavorable outcomes. Maximum width also changes the prepended user
token's relative position to items, so context effects cannot be attributed only
to added old events. Compare with already completed 20/30/40 results without
retraining them; restrict reference validation to the same first 30 epochs.
Historical 20-item test checkpoints may have different selection budgets.

Before submission, GPU quota was 4.69 hours. Based on measured context-40 epochs
of 93-100 seconds, provisionally expect about 1-1.5 hours elapsed with two models
in parallel; this is an estimate, not a measured context-50 runtime. No other
Kaggle run is authorized by this configuration.

Preflight: full repository suite passed (957 tests, 91 skipped). Five native
context-runner tests also passed, including the full 30-epoch resume path.
Two full-size constructor/backward probes matched the original seed42 weights
and had finite gradients. Parent fingerprints and all 20/40 suffix checks passed.
