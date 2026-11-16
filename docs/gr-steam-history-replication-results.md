# Steam disjoint-cohort replication: completed

Private kernel `carasik/rhpr-steam-history-replica/1`, executable source `a16145d4aa6e3ee526198482712d0282d7a8bc59`, completed successfully. Runtime **2,853.85s (47.56min)** after bootstrap, excluding queue. Exactly two fits; no new GPU fits or inference during collection.

The large gain from the first cohort did not repeat. Recent30 remains slightly ahead on selected TEST NDCG@10, but behind on VAL and the paired user interval includes zero. This does not establish a reliable30vs20 history benefit, nor prove that extra history is useless.

| Window | Selected VAL NDCG@10 | Selected TEST NDCG@10 | TEST hits /2,000 | Best /stop epoch | Warm epoch | Forward GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent20 | **0.026905** | 0.019070 | 80 | 45 /65 | 29.68s | 0.427088 |
| Recent30 | 0.026187 | **0.019764** | 87 | 45 /65 | 38.64s | 0.656522 |

| Cohort | Data seed | Recent30vs20 VAL | Recent30vs20 TEST |
| --- | ---: | ---: | ---: |
| Original2,000users | 4242 | +8.77% | +23.77% |
| New disjoint2,000users | 1729 | **-2.67%** | **+3.64%** |

Both cohorts share the exact item embeddings, centroids,3+1SID matrix, generator initializer and training protocol. Changing users changes the training signal and target support as well as evaluation users; it is not simply evaluation of one fixed model on a larger sample. Generator and index seeds remain42. Two cohorts are insufficient to estimate between-cohort variation reliably.

## Uncertainty and alternative ranking metrics

10,000 paired user bootstrap resamples, fixed seed4242, conditional on trained models and selected checkpoints:

| Split | Absolute NDCG@10 difference30minus20 | Conditional95% interval |
| --- | ---: | --- |
| VAL | -0.000718 | [-0.005242, +0.003826] |
| TEST | +0.000694 | [-0.003411, +0.004715] |

On TEST59user contributions improve,54 worsen,1,887 tie. The interval allows roughly-17.9% to+24.7% of the observed Recent20 NDCG; it is not evidence of equivalence. Intervals omit generator/index seed variation, item clustering, selected-checkpoint uncertainty and multiplicity correction.

TEST Recall@10 rises from.0400 to.0435 (80to87hits); TEST NDCG@5 falls from.013969 to.012331. There is no uniform improvement across cutoffs. No TEST metric was used to choose checkpoints.

## Learning curves

Both raw VAL maxima are epoch45 and both EMA rules stop at65. Both VAL teacher-forced CE minima are epoch40, almost identical:8.446970 for20 and8.446163 for30. Early on Recent30 often has lower CE; after45 it is higher. Training CE continues decreasing while VAL CE increases. No clear sign that simply extending these fits resolves the comparison. EMA stopping is not proof of global convergence.

At common epochs, VAL NDCG@10:

| Epoch | Recent20 | Recent30 |
| --- | ---: | ---: |
| 30 | .023255 | .023426 |
| 40 | .023405 | .025862 |
| 45 | .026905 | .026187 |
| 50 | .024962 | .023609 |
| 55 | .021328 | .020853 |
| 60 | .019923 | .020886 |
| 65 | .022887 | .025197 |

Curves alternate in rank. Do not pick epoch40 or65 retrospectively to claim an advantage. TEST uses the same selected epoch45 for both fits. Forward FLOPs are fixed padded B1 teacher-forced costs, not beam-serving latency; Recent30 uses53.72%more counted FLOPs. End-to-end fitting takes36.0min for20 and46.1min for30 on separate T4s.

## Exploratory checks

- Source-user overlap between cohorts is exactly0;2,000new VAL and2,000new TEST targets. Catalogue bytes match. All embeddings, centroids and SIDs are byte-identical, including original SID file hash `3dd4d4d272286f0403acf51e4d3240f6ea0971814f113e134ed97d6c1b5a4262`.
- Among1,572users with distinct train/VAL/TEST boundary days, VAL20/30=.026254/.027760, TEST=.020465/.020692. TEST difference remains small; this descriptive group does not remove arbitrary within-day history order.
- For1,159users with pre-VAL history30-49, both splits favor30; groups50-99 and100+ favor20 on both splits in this cohort. These post-hoc groups are not validated routing rules and do not establish history-length monotonicity.
- All87VAL/83TEST targets with zero TRAIN target support miss under both models. Full exploratory groups and per-user contributions are retained in analysis artifacts; no subgroup was used to filter the fitted cohort.

## Artifact audit and reproduction

All28prediction files replayed:26VAL checks plus2selected TEST files;1,120,000candidate paths,271invalid,0duplicates. Invalid paths retain their ranks. All input hashes, independently rebuilt raw prefix examples/holdouts, case targets, initialization, config, update/LR/EMA history and checkpoint selection match. Both65epoch fits complete,one normal AMP skipped update total, no fallback.

All four full best/last checkpoints are downloaded, finite and hash-verified. Last states retain optimizer/scaler/scheduler/RNG/early-stopping and encoded-case hashes. Frozen index, prepared cases, logs, predictions, metrics and plots are preserved outside Git in `/workspace/artifacts/steam-history-replica-results`. Input is `/workspace/artifacts/steam-history-replica-dataset`. No user-level raw records or weights are committed.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/history-mpl PYTHONPATH=src \
python scripts/research/analyze_steam_history.py \
  --results /workspace/artifacts/steam-history-replica-results \
  --data /workspace/artifacts/steam-history-replica-dataset \
  --output /workspace/artifacts/steam-history-replica-results/analysis
```

Analyzer supports the original and disjoint configurations with separate exact executable-source checks. Summary `gr-steam-history-replication-results.json`; completion receipt `gr-steam-history-replication-completion.json`.

A next confirmation, if authorized, can cross another generator seed with both existing cohorts rather than immediately scaling to the full million events. No such run has been submitted.

Validation: full suite1,165tests,152optional skips,0failures,142.742s. The updated analyzer also replayed the original cohort; original metrics, intervals, candidate counts and checkpoint hashes are unchanged.
