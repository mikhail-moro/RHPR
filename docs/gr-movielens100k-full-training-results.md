# MovieLens 100K: full-training pilot, seed 42

Collected 2026-11-02. All four continuations completed successfully in private
[Kaggle notebook v1](https://www.kaggle.com/code/carasik/rhpr-ml100k-full-training-seed42).
The notebook body took 48.23 minutes on two T4s. All models stopped by early
stopping, below the 100-epoch cap. No further GPU work was submitted.

## Selected test results

| Predictor | Previous test NDCG@10 | Continued test NDCG@10 | Relative change | Best epoch | Stop epoch | Test Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| TIGER: text + poster + plot, no people | 0.03940 | **0.05315** | +34.9% | 27 | 47 | 0.10710 |
| TIGER: same + people graph | 0.03605 | 0.04075 | +13.0% | 27 | 47 | 0.08165 |
| TIGER: names in text, poster + plot | 0.03891 | 0.04520 | +16.2% | 27 | 47 | 0.08696 |
| SASRec: item IDs | 0.05411 | **0.05411** | 0.0% | 10 | 30 | 0.11559 |

Previous means the best validation checkpoint within the original 20-epoch
budget, not the last checkpoint at epoch 20. Continued means the best validation
checkpoint across retained history and continuation. Test is evaluated after
checkpoint selection. These are four fitted models at one seed.

The no-people TIGER is now 1.8% below SASRec on test NDCG@10. This shows that the
short budget understated this TIGER configuration; it does not establish
equivalence to SASRec. Graph and names remain below no-people on seed 42, which
already favored no-people in the earlier experiment. This does not settle the
three-seed average or show that graph representations are generally harmful.

## Training dynamics

All three TIGER models select epoch 27, at LR 0.001. Their first LR reduction is
after epoch 32. Thus the selected gains precede LR reduction. Longer training at
the original LR and denser checkpoint selection changed together; this run does
not isolate their effects.

The common validation peak at epoch 27 is narrow. Selecting only recorded
multiples of five would choose epochs 40/45/35 for no-people/graph/names. This is
a post-hoc subsample of the observed trajectory, not a counterfactual training
run: a different validation cadence would also change scheduler decisions.

After epoch 27, training CE falls while validation CE rises. LR reductions do
not recover a new NDCG maximum before the stopping rule fires. SASRec retains
its original epoch-10 checkpoint and its validation item CE worsens during
continuation. These trends suggest overfitting under this recipe rather than a
need to increase the epoch cap. Compare CE trends within each index; token CE
across different SID assignments is not a common item-likelihood scale.

| Predictor | Old best val NDCG@10 | New best val NDCG@10 | Last val NDCG@10 | Val CE at best | Last val CE |
|---|---:|---:|---:|---:|---:|
| No people | 0.05988 | 0.06595 | 0.06031 | 6.2255 | 6.4733 |
| People graph | 0.05885 | 0.06772 | 0.05448 | 6.1302 | 6.3961 |
| Names in text | 0.05924 | 0.06781 | 0.05653 | 6.1892 | 6.4419 |
| SASRec | 0.06438 | 0.06438 | 0.05807 | Not recorded at epoch 10 | 6.7614 |

Validation favors names/graph at their selected checkpoints, whereas test orders
them oppositely. With 943 targets per holdout and one training seed, model
ranking remains uncertain. The next confirmation would apply the same fixed
policy to the remaining seeds; it has not been launched.

## Verification and protocol

- Independently rescored 115 prediction files: 111 current validation/test
  files and four original selected tests. Same 943 users and targets per split.
- Frozen ID/SID matrices match parent file hashes and checkpoint fingerprints.
  The first 20 curve rows reproduce the exact source-file SHA256; copied prefix
  predictions retain their original byte hashes.
- Resume reports verify model transfer, Adam/scaler and CPU/CUDA/Python/NumPy
  RNG restoration. This does not guarantee bitwise CUDA determinism.
- Independently replayed all scheduler and early-stopping decisions. Every
  epoch has 334 updates and 85,470 targets; zero AMP-skipped updates.
- All TIGER models add 27 epochs / 9,018 updates, ending at 47 / 15,698.
  SASRec adds 10 epochs / 3,340 updates, ending at 30 / 10,020.
- Retained minimum 30 / cap 100 epochs, ES patience 20 since best raw validation
  NDCG@10, validation each new epoch. ReduceLR halves after five failed checks
  from epoch 21, absolute threshold 0.0001, minimum LR 0.00001.
- Encoders, AE, quantizers and indices were not retrained. Same timestamp-group
  holdouts, history 20, FP16 training / FP32 evaluation, full-catalogue metrics,
  no seen-item mask or sampled negatives. Architectures remain TIGER 4+4/d128
  and capacity-matched SASRec; matching parameter count does not match FLOPs.

The conditional paired-user bootstrap intervals for absolute test NDCG@10
differences are graph minus no-people [-0.02236, -0.00254], names minus no-people
[-0.01909, +0.00288], and no-people minus SASRec [-0.01356, +0.01127]. They condition
on these fitted models, exclude training-seed/runtime variation and multiplicity,
and do not establish equivalence or noninferiority.

This is exploratory next-rating prediction on MovieLens, not observed VOD
viewing chronology or an untouched test holdout.

## Reproducibility and artifacts

Executable commit: `ce0efd6d6816a24d5710d5baeda706abc3d8e8cb`.
Configuration: `experiments/movielens100k_full_training_seed42_2t4.json`.
Submission provenance is preserved in the existing submission receipt.

Local evidence: `/workspace/artifacts/movielens100k-full-training-seed42/analysis/`.
Includes metrics, curves, per-user ranks, verification, paired comparisons,
PNG/PDF plots and the replayable analysis script. Full numerical bundle without
large weights: `/tmp/movielens100k-full-training-seed42-numerical-evidence.zip`.
The temporary bundle can disappear when the environment is replaced; original
outputs and large selected/last checkpoints remain in the private Kaggle run.
