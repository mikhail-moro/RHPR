# Steam: is additional history useful to TIGER?

Authorized 2026-11-16: a small history-length pilot before returning to pair-SID compression. The user approved proceeding within the available ~30 GPU hours, then explicitly chose **Steam only; defer Kindle** after the Kindle-2014 title-coverage audit. Exactly three fresh fits, one pipeline seed 42, windows 10/20/30. No compression, architecture search, extra seeds, retries or notifications.

## Frozen data

Source: SASRec authors' `Steam.txt`, raw Steam reviews and game metadata. `scripts/prepare_steam_history_cohort.py` pins all three source SHA256s. It reconstructs encounter-order user/game IDs from raw reviews and proves equality of **all 3,686,172 unique user/game incidences** with the authors' prepared file. This is not a claim of bit-exact reconstruction of their within-day ordering. The provided order is preserved and verified nondecreasing under the earliest raw review date for each user/game pair. The raw single-pass filtered set has 526,945 duplicate rows.

Apply iterative global user/item 5-core, then full history length >=32, then uniformly sample 2,000 complete users with NumPy `default_rng(4242)` from canonically sorted eligible source IDs. No within-sample item pruning, rating filtering, metadata filtering, or selection on model quality. Preserve the full 11,961-game global-core catalogue, including unobserved games. This is a static transductive catalogue and a long-user cohort, not the population-wide original Steam benchmark.

- Events: 133,023; training events: 129,023; all noninitial training positions: **127,023 per epoch**.
- Leave-last-two-out: 2,000 validation and 2,000 test targets; test observes validation. All users have a full 30-item context at both evaluations.
- Every window uses the same user IDs, item SIDs, training targets, shuffle schedule and raw held-out cases. Only raw context truncation changes.
- Cold targets absent from selected users' training histories: validation 67/2,000 (3.35%), test 75/2,000 (3.75%). Retained in the full-catalogue evaluation.
- Timestamp resolution is one day. Train/validation share a day for 307 users; validation/test share a day for 261. Preserve provided within-day order; this does not establish the true temporal order of tied events. Report as a limitation and retain user audit for sensitivity analysis.
- All 11,961 games have a title; 173 lack genres and remain with title only.

The private input dataset `carasik/rhpr-steam-history-input` stores normalized catalogue/sequences, source provenance, counts and per-user date audit. Every input file is pinned in `experiments/steam_history_windows_2t4.json`. Raw downloaded reviews and user-ID maps remain private workspace artifacts outside Git.

## Fixed tokenizer and generator

Sentence-T5-base, FP32 encoding, title plus genres, max256 tokens, one cached full-catalogue embedding matrix. Native GRID normalized residual RQ-KMeans, seed42, 3x256, 10 passes/level, batch2048, initialization3072, torch cdist. Append the ordinary collision ordinal to **every** item (3+1 SID); never omit or widen suffixes. Fit once; all three generators use identical frozen SIDs. No behavior objective enters the tokenizer.

Current wide TIGER: encoder4 + decoder4, d128, FF1024, d_kv64, heads6, dropout.1, user bins2000, no separator, SDPA. Keep the previous input adapter for continuity; total 5,108,992 parameters includes the unused pair input table. Parameter-initialization SHA256 must equal `34e7397ed277c78befd6254cbb736794df4907a011ed032bbd0735d9b7272286` for all three models.

FP16 compiled training, physical/effective batch512; Adam lr.003/weight_decay1e-4/clip1. Warmup2, cosine to.001 at epoch60, then fixed.001 through the cap. Minimum40/maximum120 epochs, validation every5, EMA(alpha.5,min_delta.0005), patience20 epochs. Select the best **raw validation NDCG@10** checkpoint; EMA controls stopping only. A cap or early stop is not proof of an absolute plateau. Existing trainer saves full continuation state after every completed epoch; best is saved separately. Save train/validation losses and all validation predictions, and evaluate selected test once. FP32 beam20, batch64, full catalogue, no seen-item filtering or sampled negatives.

CPU shape preflight uses explicitly synthetic unique SIDs, not fitted-index quality evidence. It independently replays every source training prefix and both holdouts, then checks finite real-backbone forwards/backwards at all three widths. Supported B1 teacher-forced forward FLOPs (FMA2): Recent10 **0.217327232G**, Recent20 **0.427087952G**, Recent30 **0.656522272G**. Encoder positions41/81/121 include USER. These exclude beam search, backward and item-encoder/index construction; they are not measured serving latency.

## Execution and interpretation

One private offline Kaggle session on 2xT4. GPU0: Recent30 then Recent10; GPU1: Recent20. No hardware fallback. Per-fit training wall ceiling9,000s; server session ceiling21,600s (six hours, including startup). Expected ~1.5-3 hours after startup based on earlier similar-sized fits, not a measured Steam timing; conservative device-time ceiling is twelve GPU-hours if counting both T4s separately. Do not resubmit a queued session. Preserve partial checkpoints on timeout/failure; no automatic restarts.

Primary question: does additional history improve validation ranking at a sufficiently trained checkpoint, consistently enough to merit more seeds? Inspect curves by epochs, examples and measured time. Final test is a secondary confirmation, not a selection criterion. One pipeline seed cannot establish reproducibility; dataset, tokenizer and generator axes remain separate. Do not select user subgroups based on which window wins.

Kindle is deferred: official2014 metadata has titles for only8/61,934 items; the2018 crosswalk recovers31,808 titles. No Kindle model is part of this run.
