# MovieLens 100K: full training across three pipeline seeds

Collected 2026-11-02. The extra-seed notebook completed all eight continuations
without failed trials in 109.43 minutes on two T4s. Combined with the retained
seed-42 pilot, all twelve models stopped by early stopping, below the epoch cap.
No new GPU work was submitted during collection.

## Selected test NDCG@10

| Predictor | Seed 42 | Seed 101 | Seed 202 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| TIGER: text + poster + plot, no people | 0.05315 | 0.04054 | 0.04649 | **0.04672 ± 0.00631** |
| TIGER: same + DeepWalk people graph | 0.04075 | 0.04008 | 0.04697 | **0.04260 ± 0.00380** |
| TIGER: names in text, poster + plot | 0.04520 | 0.04431 | 0.04455 | **0.04469 ± 0.00046** |
| SASRec: item IDs | 0.05411 | 0.05412 | 0.04817 | **0.05214 ± 0.00343** |

Each test result uses its model's best validation checkpoint. The seeds repeat
the saved full pipelines, including different indices; they are not three
generator-only repeats with a common index. Sample SD is not a confidence
interval. Per-seed numerical results are saved in
`gr-movielens100k-full-training-three-seed-metrics.csv`.

SASRec exceeds every TIGER variant on all three seeds. The mean no-people TIGER
is 10.4% below SASRec; graph and names are respectively 18.3% and 14.3% below.
The near tie on seed 42 did not persist. These results do not establish a
universal dense-versus-generative ranking or a statistically resolved gap for
every comparison.

Mean test Recall@10 is 0.09862 / 0.08943 / 0.09579 / 0.10887 for no-people,
graph, names and SASRec respectively.

## What longer training changed

| Predictor | Old mean test NDCG@10, 20-epoch budget | Continued mean | Relative change | Seeds with test improvement |
|---|---:|---:|---:|---:|
| No people | 0.04292 | 0.04672 | +8.9% | 2/3 |
| People graph | 0.04326 | 0.04260 | -1.5% | 1/3 |
| Names in text | 0.03772 | 0.04469 | +18.5% | 3/3 |
| SASRec | 0.05261 | 0.05214 | -0.9% | 0/3, two unchanged |

Old means best validation checkpoint within the old 20-epoch budget, not the
last weights used for continuation. The extended protocol changes training
duration, LR scheduling and validation cadence together.

The large +34.9% no-people improvement on seed 42 overstates the three-seed
average improvement. Names benefit on all three seeds and have the smallest
observed test spread; three repeats cannot establish that they are inherently
more robust. There is no basis for claiming that longer training rescues the
current graph representation.

Graph minus no-people is -23.3% / -1.1% / +1.1% on seeds 42/101/202. Thus the
large negative mean is driven mostly by seed 42. Graph minus names is -9.9% /
-9.5% / +5.4%. Neither comparison has a consistent sign across seeds; this
does not show that graph representations are intrinsically harmful or useless.

## Training and validation

| Predictor | Best epochs, seeds 42/101/202 | Stop epochs, seeds 42/101/202 |
|---|---|---|
| No people | 27 / 37 / 41 | 47 / 57 / 61 |
| People graph | 27 / 29 / 32 | 47 / 49 / 52 |
| Names in text | 27 / 34 / 25 | 47 / 54 / 45 |
| SASRec | 10 / 5 / 25 | 30 / 30 / 45 |

The common epoch-27 peak from seed 42 is not reproduced on the other seeds.
Four of the six new TIGER fits select after their first LR decrease: no-people
101/202, graph 202 and names 101. The pilot's observation that all selected
gains preceded LR reduction therefore does not generalize to every seed.
This is still not a causal LR-scheduler ablation.

After the selected point, training CE decreases while validation CE increases
in every new TIGER fit. No model reaches the cap. The results support using a
stopping rule rather than assuming that another increase in epoch count will
improve generalization.

Graph has higher selected validation NDCG@10 than no-people on all three seeds,
despite lower mean test quality. Mean selected validation is 0.07020 for graph,
0.06673 for no-people, 0.06731 for names and 0.06511 for SASRec. This disagreement
limits conclusions from validation ranking alone. Each holdout has only 943
targets. Denser validation offers more selection opportunities; it does not
guarantee a better test checkpoint. No test-selected epochs are substituted.
The disagreement is not evidence by itself of an incorrect split.

## Independent audit

- Rescored 396 saved prediction files across all twelve fits, including both
  old selected tests and new selected tests, matching reported NDCG/Recall at
  5/10 and invalid/duplicate counts. Decode SID paths using each frozen map;
  preserve invalid and duplicate beam positions.
- Same 943 users and targets per split across all predictions. Parent manifests,
  copied prefix predictions, first-20-epoch curves and frozen index hashes match.
- Resume reports verify model transfer, Adam/scaler and all RNG streams. This
  does not enforce bitwise deterministic CUDA execution.
- Replayed all scheduler and early-stopping decisions. All epochs contain
  85,470 cases and 334 attempted updates. Actual AMP-skipped updates are one for
  no-people seed 101 and three for no-people seed 202, zero for the other ten
  fits. Counts reconcile with optimizer updates; no precision/LR fallback.
- Same history 20, timestamp-group holdouts, full catalogue, no seen-item mask
  or sampled negatives, FP16 training / FP32 evaluation. Frozen encoders, AE,
  quantizers, indices and model architectures are unchanged. Parameter-capacity
  matching does not imply equal FLOPs.

Conditional paired-user bootstrap intervals for mean absolute NDCG differences
are graph minus no-people [-0.01059, +0.00221], names minus no-people
[-0.00813, +0.00395], and no-people minus SASRec [-0.01381, +0.00300]. Users are
resampled jointly after averaging the same three fitted seeds. These intervals
exclude training-seed/runtime uncertainty and multiplicity; they cannot certify
equivalence, noninferiority or a resolved modality ranking.

This remains exploratory next-rating prediction on MovieLens, not observed VOD
viewing chronology or an untouched test holdout. The present evidence supports
neither a stable graph advantage nor a general claim that names are unsuitable
for the text encoder. No additional training has been queued.

## Provenance and artifacts

- Extra-seed executable: `8e1819a6137aaae30e43a4115cb6a4c2aae1b819`.
- Seed-42 executable: `ce0efd6d6816a24d5710d5baeda706abc3d8e8cb`.
- Private Kaggle: `carasik/rhpr-ml100k-full-training-extra-seeds` v1 and retained
  `carasik/rhpr-ml100k-full-training-seed42` v1.
- Local report data, plots, verification and replay script:
  `/workspace/artifacts/movielens100k-full-training-extra-seeds/analysis/`.
- Raw numerical evidence, including per-user ranks, without large checkpoints:
  `/tmp/movielens100k-full-training-three-seeds-numerical-evidence.zip`.
  Temporary files may disappear if the environment is replaced; original
  outputs and large selected/last checkpoints remain in the private notebooks.
