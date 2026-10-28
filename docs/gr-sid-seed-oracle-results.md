# Identical SID index, different neural seeds: oracle control

Completed private Kaggle session `carasik/rhpr-sid-seed-oracle`, executable
commit `178883fe87351ee81ea70830688782129370584d`. Both models use the exact same
original Sentence-T5 / RQ-KMeans complete SID mapping. Neural initialization and
dropout seeds are 42 and 43; data and epoch-order seed remain 42. No branch
token, swaps, quantizer refit, or test evaluation. Same 22,363 validation users
and targets as the earlier original/ALS comparison.

## Primary endpoint: 20 epochs

| Comparison | Model A NDCG@10 | Model B NDCG@10 | Oracle NDCG@10 | Above best fixed model |
|---|---:|---:|---:|---:|
| Same original index, seeds 42/43 | 0.03644111 | 0.03621833 | **0.04785565** | **+31.32%** |
| Historical original / ALS indices | 0.03694259 | 0.03514883 | 0.04729654 | +28.03% |

For each user, oracle chooses the complete predicted list with the better
target rank. NDCG@10 is zero outside top 10 and otherwise 1/log2(rank+1).
There is one relevant target per user. Lists are never merged or reordered.

The same-index pair has 404 hits unique to seed 42, 382 unique to seed 43,
1,196 shared hits, and 20,381 shared misses. Oracle hits are 1,982: Recall@10
8.8629%, versus 7.1547% for the better individual model. The historical
different-index pair has 1,984 oracle hits, almost the same descriptive count.

Large oracle headroom is reproducible without changing the SID hierarchy.
Consequently the earlier +28% oracle alone does not establish index-specific
complementarity. These pairs provide no demonstrated additional oracle benefit
from different indices. This is evidence for ordinary training diversity as a
sufficient explanation of a large oracle, rather than an estimate of its exact
causal share. One pair per condition cannot establish equivalence or rule out
other index effects. A deployable router still requires winner predictability
from information available before the target, which has not been tested here.

## Training dynamics and conditional uncertainty

| Epoch | Seed 42 | Seed 43 | Same-index oracle | Oracle headroom | Historical different-index headroom |
|---|---:|---:|---:|---:|---:|
| 5 | 0.026915 | 0.025736 | 0.034870 | +29.55% | +20.97% |
| 10 | 0.029618 | 0.029624 | 0.039266 | +32.55% | +32.37% |
| 15 | 0.033463 | 0.033838 | 0.044090 | +30.30% | +27.80% |
| 20 | 0.036441 | 0.036218 | 0.047856 | +31.32% | +28.03% |

Epochs share fitted trajectories and validation users; they are not independent
replications. The fresh seed-42 fit differs from the historical seed-42 fit
despite matching initial parameters, cases, order, and declared protocol.
GPU training nondeterminism remains another source of prediction diversity.

At epoch 20, absolute same-index oracle headroom is 0.01141454; conditional
paired-user 95% interval [0.01056810, 0.01231526], target-cluster sensitivity
[0.00963978, 0.01344644]. Historical different-index headroom is 0.01035395.
Their difference, same minus different, is 0.00106059; paired-user interval
[-0.00004407, 0.00217890], target-cluster [-0.00044921, 0.00268799]. These are
2,000 paired resamples, seed 2752, conditional on fixed fitted weights and
unadjusted. They exclude training-seed uncertainty. Positive oracle headroom
is not a significance test for a learnable router, and overlap with zero in
the contrast is not proof of equivalence.

## Verification and retained outputs

Both fits completed 20 full epochs, 2,628,260 examples, 10,280 updates, zero
AMP skips, and four validation checks. Each has 4,977,920 parameters. FP16
training, FP32 validation, batch 32, free beam 20, full catalogue 12,101.
Execution body 1,756.4 seconds, about 29.3 minutes on two T4s.

`scripts/analyze_sid_seed_oracle.py` independently checks source/config,
byte-identical original SID files and identity ownership, distinct pinned
neural initialization hashes, shared case/order hashes, budgets, users and
targets. It packs four-token paths into integers, rescoring all eight raw beam
files and matching NDCG/Recall@5/10, invalid/duplicate counts, cached per-user
ranks, and all oracle values. Historical users, targets, history lengths, and
declared model/training protocol also match. No new training, inference, or
test evaluation was used for this analysis.

Artifacts: `/workspace/artifacts/sid-seed-oracle/analysis/` contains `README.md`,
`epochs.json`, `cohorts.json`, `intervals.json`, `training.json`,
`verification.json`, and all eight rank arrays in `user-metrics.npz`.
Rank 999 means outside the saved 20 beams, not exact full-catalogue rank.
The parent directory retains small run/suite/oracle/download metadata.
All 46 numerical/report files, 25.88 MB, are cached at
`/tmp/rhpr-seed-oracle-completed`; authoritative outputs remain on private
Kaggle. Four best/last checkpoint filenames were confirmed, but checkpoint
contents were not downloaded or independently validated.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_sid_seed_oracle.py
```

No new run, seed extension, routing fit, test evaluation, or notification is
authorized or launched by this completed-output analysis.
