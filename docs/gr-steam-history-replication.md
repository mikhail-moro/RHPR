# Steam history utility: disjoint-user replication

Authorized 2026-11-16: exactly two new generator fits, Recent20 and Recent30, seed42, one isolated T4 each. This checks dependence on the pilot user sample before spending on the full million-event long-user cohort. No Recent10, compression, HPO, additional seeds or automatic retries.

## Cohort fixed before outcomes

Use the same pinned Steam sources, original ordering and global iterative5core (281,210users,11,961games). Among15,239 users with full length>=32, reconstruct the original2,000 users with data seed4242 and exclude them. Uniformly sample2,000 from the remaining13,239 with data seed1729. The original and new source-user sets are disjoint. Do not replace this sample based on results.

New cohort:131,305events,127,305TRAINevents,125,305 noninitial TRAIN next-item examples per epoch;2,000VAL and2,000TEST targets. Median full length47,p90=116,max715. VAL/TEST have30 actual context events. Cold targets87VAL/83TEST (4.35%/4.15%). Equal boundary days303 train/VAL,237VAL/TEST; retained as in the first pilot. All within-day ordering limitations remain.

Private input `carasik/rhpr-steam-history-replica-input` contains only the five original-format JSON files. No cached vectors or checkpoint transfers. Catalogue is byte-identical to the first pilot; no within-sample item pruning. Selected and excluded source-user IDs are recorded in private provenance.

## Fixed model and training

Config `experiments/steam_history_replication_2t4.json`. Same Sentence-T5 title+genres embeddings and normalized residual RQ-KMeans3x256+suffix, tokenizer seed42, full static catalogue. Recompute from the already used sources and require original SID file SHA256 `3dd4d4d272286f0403acf51e4d3240f6ea0971814f113e134ed97d6c1b5a4262` before any generator training. Never silently accept a changed index.

Same 4+4/d128/FF1024/H6/dkv64 T5, USER2000,noSEP,SDPA,5,108,992parameters. Initial parameter SHA256 `34e7397ed277c78befd6254cbb736794df4907a011ed032bbd0735d9b7272286`. FP16 compile reduce-overhead, physical batch512. Adam .003->.001,2epoch warmup/cosine60,weight decay1e-4,clip1. Min40/max120epochs,VAL every5,EMA alpha.5/min delta.0005/patience20epochs. Best checkpoint chosen by raw VAL NDCG@10; TEST evaluated once afterward with FP32 beam20/batch64,full catalogue,no seen-item mask.

GPU0 Recent30,GPU1 Recent20. Existing9,000s per-fit and21,600s server limits remain. Expected45-60minutes excluding queue if convergence resembles the pilot; caps are safeguards, not predicted runtime. Save all VAL predictions/losses, TEST predictions and full best/last checkpoints with optimizer/scaler/RNG/scheduler state.

## Interpretation

Primary question: does Recent30 again beat Recent20 on the predeclared new cohort? Report both VAL/TEST differences, conditional paired-user bootstrap intervals, learning curves and stopping epochs regardless of sign. Do not select seeds, checkpoints or samples on TEST. Changing training users changes item support, not just the evaluation sample; retain and report cold-target counts. This is one disjoint-cohort replication conditional on a fixed index and generator seed, not proof across generator/index seeds or datasets.

CPU preflight `gr-steam-history-replication-preflight.json` independently replays every TRAIN target and temporal holdout, verifies real generator finite forward/backward and the initializer, and profiles .427088/.656522 forward GFLOPs. It uses synthetic SIDs only for shape checks; actual CUDA index equality is checked at runtime. The worker regression covers two real OS processes and partial-failure preservation for both original and replication configurations.

## Pre-submission verification

Private dataset is ready; all five downloaded JSON SHA256 values match the pinned config. Local private offline GPU bundle built successfully. Full suite:1,165 tests,152 optional skips,0 failures,126.094s. CPU preflight passed with no GPU use.

## Submission

Submitted once: private `carasik/rhpr-steam-history-replica/1`, executable source `a16145d4aa6e3ee526198482712d0282d7a8bc59`, workflow43924489379/job133500204440. Push2026-11-16T18:01:55Z, timeout21,600s. Kaggle first statusQUEUED; no training outcomes observed. All218 deployed Python files, exact config, private/offline GPU metadata and three dataset sources verified. Receipt `gr-steam-history-replication-submission.json`. Do not dispatch a duplicate.
