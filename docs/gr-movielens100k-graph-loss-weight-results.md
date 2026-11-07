# Graph reconstruction weighting: completed paired pilot

Collected 2026-11-02. Private `carasik/rhpr-ml100k-graph-loss-weight` v1
completed both seed-42 fits without failures in 46.97 minutes on two T4s.
Both stopped by early stopping, below the cap. No new training was submitted.

## Recommendation quality

| Graph reconstruction weight | Selected val NDCG@10 | Test NDCG@10 | Test Recall@10 | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|---:|
| 1, fresh control | 0.06786 | **0.04554** | **0.09226** | 27 | 47 |
| 0.1 | 0.06601 | **0.04202** | **0.08590** | 20 | 40 |

Both use their best validation checkpoint, one final test, exactly 3,500 AE
updates and 3,000 native RQ updates. The graph coefficient applies in both AE
and RQ reconstruction; other coefficients remain one. Features, initialization,
codebook capacity and the generator policy are matched. Neither run hits the
epoch/time cap and neither has an AMP-skipped update.

The observed test NDCG difference is -0.00351 absolute / **-7.7% relative**.
The weaker arm improves per-user NDCG for 63 users, worsens it for 64 and leaves
816 unchanged. The conditional paired-user bootstrap 95% interval is
[-0.01534, +0.00757] absolute and includes zero. It conditions on two fitted
models, excluding training/runtime/seed uncertainty and multiplicity. This is
one paired pipeline seed, not evidence of a stable benefit or stable harm.

## Representation tradeoff did move

Raw reconstruction MSE remains on the same per-channel scale:

| Channel | Weight-1 control | Graph weight 0.1 |
|---|---:|---:|
| Text | 0.81263 | **0.79893** |
| Poster | 0.84231 | **0.83421** |
| Plot | 0.86001 | **0.85220** |
| People graph | **0.79349** | 0.88231 |

The graph term matters: weakening it improves the other channels' reconstruction
and worsens graph reconstruction. Weighted total losses across arms are different
objectives and should not be used as a common quality score.

For each film, take ten nearest centered-cosine neighbors in frozen features,
excluding itself. The fraction sharing the first semantic token changes as follows:

| Neighbor set | Weight 1 | Weight 0.1 |
|---|---:|---:|
| Text | 4.10% | **4.74%** |
| People graph | **3.86%** | 1.75% |
| Plot | 3.06% | 3.32% |

Random distinct-pair prefix agreement is 0.449% / 0.496%. Graph-neighbor lift
falls from 8.61x to 3.53x; text-neighbor lift rises from 9.13x to 9.54x. These
diagnostics indicate a changed representation balance, not a validated proxy for
next-item quality. Both indices have all 1,273 distinct semantic triples, so
catalogue collisions do not explain the difference. The weak arm generated one
invalid SID across 18,860 test beam positions; the control generated none.

This pilot supports the existence of a reconstruction/index tradeoff, but does
not show that recovering text geometry improves recommendation quality. It does
not establish an optimal coefficient or invalidate graph representations.

## Historical repeat and numerical uncertainty

The fresh weight-1 control reproduces the old seed-42 graph AE neural-state hash,
raw reconstruction metrics, semantic codes and full SID matrix exactly. Its
generator initial neural-parameter hash also matches. However, its generator
training trajectory differs from the previous run already at epoch 1: train CE
9.2839230765 previously versus 9.2836567678 now. The exact numerical/runtime
cause has not been identified.

Its selected test NDCG is now 0.04554 versus the historical 0.04075, despite the
same index. This repeat difference is larger than the present treatment gap.
Use the fresh paired control as the primary comparison; do not substitute the
historical control, attribute its change to fixed warm-up, or count it as another
independent seed. This is another reason to avoid a strong conclusion from the
single-seed coefficient comparison.

## Training dynamics and verification

- Independently decoded and rescored all 57 saved validation/test prediction
  files, including invalid and duplicate beam slots. Reported NDCG/Recall at
  5/10 and invalid/duplicate counts match; same 943 users/targets per split.
- Verified exact executable/config, downloaded-file hashes, frozen feature hash,
  catalogue order, initial quantizer/generator pins and reconstructed SID maps.
- Both AE runs complete 3,500 fixed updates with untouched books, verified
  neural transfer, fresh RQ optimizer and zero initialization-latent drift.
  Native RQ completes 3,000 reconstruction updates plus one initialization batch.
- Verified raw/weighted AE and RQ diagnostic totals. Copied prefix curves equal
  the first 20 full-history rows, with validation at 5/10/15/20. Resume reports
  verify epoch 20 / 6,680 updates, Adam/scaler/RNG and case/map digests.
- Independently replayed LR and early-stopping decisions. Control LR reductions
  occur after epochs 32/37/42/47; weak-arm reductions after 25/30/35/40. Both
  selected checkpoints precede their first LR reduction. Training CE declines
  and validation CE rises after selection in both arms.
- All complete epochs contain 85,470 cases / 334 updates, zero AMP skips.
  Control ends at 47 / 15,698 updates; weaker arm at 40 / 13,360 updates.
- Same prior split, history 20, full catalogue, FP16 generator training / FP32
  evaluation, no seen-item mask or sampled negatives. No test at epoch 20.

All 941 repository tests pass (87 skipped). These results remain exploratory
MovieLens next-rating prediction with one target per user in each holdout, not
an untouched test set or observed VOD viewing chronology.

## Reproducibility

Executable: `35b75d22360eada2053f59177acd18f928fbd867`.
Configuration: `experiments/movielens100k_graph_loss_weight_2t4.json`.
Local analysis, plots, metrics, per-user ranks and replay script:
`/workspace/artifacts/movielens100k-graph-loss-weight/analysis/`.
Numerical evidence without large checkpoints:
`/tmp/movielens100k-graph-loss-weight-numerical-evidence.zip`.
Temporary files may disappear on environment replacement; large selected/last
checkpoints and original outputs remain in the private Kaggle notebook.
