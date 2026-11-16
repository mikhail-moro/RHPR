# Steam Recent10 vs Recent20: fixed cohorts crossed with generator seeds

Authorized2026-11-16: compare Recent10 and Recent20 on multiple generator seeds and user cohorts. Use the two existing, outcome-independent2,000-user cohorts (data seeds4242 and1729; source-user overlap0), generator seeds42/101/202, and the same frozen item index seed42. This yields12comparison cells; reuse3completed controls and fit exactly9missing models. No new user sampling, tokenizer seeds, Recent30, compression, HPO or automatic retries.

## Registered cells and reuse

| Cohort | Generator seed | Recent10 | Recent20 |
| --- | ---: | --- | --- |
| steam_a,data4242 | 42 | reuse original pilot | reuse original pilot |
| steam_a,data4242 | 101 | new | new |
| steam_a,data4242 | 202 | new | new |
| steam_b,data1729 | 42 | new | reuse disjoint replication |
| steam_b,data1729 | 101 | new | new |
| steam_b,data1729 | 202 | new | new |

Original controls are from `carasik/rhpr-steam-history/2`, executable72ceff63e025cb1cb4731672b665594c6f97db3b; the replication control is from `carasik/rhpr-steam-history-replica/1`, executablea16145d4aa6e3ee526198482712d0282d7a8bc59. Config `experiments/steam_recent10_20_crossed_2t4.json` records the exact result-file/best-checkpoint hashes and full metric/protocol references. Local CPU preflight verifies them against all original artifacts. No controls are re-trained or re-evaluated. Final collection combines their saved predictions with the new results; the new session reports9fresh models, not12fresh fits.

## Fixed data, model and training

Cohort A:133,023events/127,023TRAINprefix examples. Cohort B:131,305events/125,305TRAINprefix examples.2,000VAL and2,000TEST targets each, identical targets for both window sizes. Every noninitial TRAIN target appears once per epoch, with the latest10or20 observed events. TEST seesVAL as an observed event. Global source5core/full-history>=32/whole-user sampling/static11,961game catalogue remain unchanged. Same-day ordering and selected-long-user limitations remain; do not filter targets post hoc.

Use existing private input datasets, not a new upload. Recompute title+genres Sentence-T5 embeddings and normalized residual RQ-KMeans3x256+suffix from the same resources. Tokenizer seed42 is fixed, independently of generator seed. Both regenerated SID matrices MUST match original file SHA256 `3dd4d4d272286f0403acf51e4d3240f6ea0971814f113e134ed97d6c1b5a4262` before any generator fit. No cached-vector/checkpoint transfer to Kaggle.

Same T5 4+4/d128/FF1024/dkv64/H6,USER2000,noSEP,SDPA,5,108,992parameters. FP16,torch.compile reduce-overhead,physical train512,FP32eval64,beam20,full catalogue,no seen-item mask. Adam.003->.001 with2epoch warmup/cosine60,wd1e-4,clip1,min40/max120epochs,VAL every5,EMAalpha.5/delta.0005/patience20epochs. RawVALNDCG@10 selects the best checkpoint; TEST is evaluated once after selection. All training settings equal the completed controls.

Generator seeds change parameter initialization, Python/NumPy/Torch/CUDA RNG and epoch permutation (`seed+epoch`); the trial seed is passed into the real training configuration as well as the model constructor. Per-seed initial parameter hashes are pinned in config and CPU-preflight. Both lengths and both cohorts share initialization WITHIN each seed. Do not require initialization equality between different seeds.

Two isolated T4 workers,9trials interleaved for similar predicted load. Existing6hour server/suite and9,000s per-fit caps. Expected2-3hours after bootstrap, excluding queue, depending on stopping epochs. No hardware, precision or LR fallback. All VAL predictions/losses and full best/last optimizer/scaler/RNG/scheduler checkpoints retained.

## Analysis fixed before new results

Primary question: does increasing context10->20 improve selected TESTNDCG@10 consistently across the two cohorts and generator seeds? Report all six paired differences, both cohort means, seed spread, rawVAL results, learning curves, best/stop epochs and forward costs. Do not select cohorts/seeds on TEST or hide unfavorable cells. The original+13.3%TEST estimate is a pilot observation, not a promised replication effect.

Use paired-user uncertainty within each fixed fitted-model pair and separately show variation across generator seeds. Do not treat12models as12independent replications: windows share targets, generator seeds repeat across cohorts, and items are shared. Two cohorts and three seeds are limited evidence; they do not cover index seed or dataset changes. Inspect item-level gain concentration descriptively without filtering popular games. More context has different compute; a positive result is history utility, not a compression/Pareto result.

## CPU preflight

`preflight_steam_recent10_20.py` verifies all10input hashes, disjoint source-user sets, all TRAINprefix/holdout positions, frozen trueSIDhash, all3reused result/checkpoint references and identical old/new training configs. Real model forward/backward is finite for all3seeds x2cohorts x2windows. A12-cell matching check combines references with fresh-trial metadata and checks cases within cohort and initialization within seed. Report `gr-steam-recent10-20-preflight.json`.

## Pre-submission checks

CPU preflight passed using the real frozen SIDs. Focused7tests passed, including real two-process event collection and partial-failure preservation for the9-fit suite. Full suite:1,167tests,152optional skips,0failures,131.010s. Private offline two-T4 bundle built. Quota observed before submission:GPU28.85hours remaining. New kernel slug not found in the account; no prior run to duplicate.
