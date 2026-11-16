# Steam history utility pilot: completed

Private kernel `carasik/rhpr-steam-history/2`, executable source `72ceff63e025cb1cb4731672b665594c6f97db3b`. All three fits completed successfully. Runtime **3,906.34s (65.1min)** after runtime bootstrap, excluding queue time; two isolated T4 workers. No further fits or inference were run during collection.

The longer context has higher validation and test point estimates. This is a promising setting for confirming history utility before returning to compression, not yet a replicated result.

| Window | Selected val NDCG@10 | Selected test NDCG@10 | Test hits /2,000 | Best /stop epoch | Warm train epoch | Forward GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent10 | 0.023911 | 0.016610 | 66 | 55 /75 | 15.09s | 0.217327 |
| Recent20 | 0.024504 | 0.018821 | 74 | 55 /75 | 27.89s | 0.427088 |
| Recent30 | **0.026653** | **0.023294** | **89** | 40 /60 | 38.37s | 0.656522 |

Recent30 vs Recent20: **+8.77% validation**, **+23.77% test**. Vs Recent10: +11.47% validation, +40.24% test. Train examples, targets, item SIDs, model size, initialization, LR schedule and selection protocol match. All validation/test users have30 available events. The tokenization/index was fixed across the three fits; the regenerated embeddings, centroids and SID arrays are also byte-identical to version1.

These comparisons vary computation with context length. They establish neither compression efficiency nor a Pareto improvement. The supported forward FLOPs count padded B1 teacher-forced model computation, FMA2; they exclude beam search, item encoding and backward. Warm epoch times exclude validation, compilation spikes and checkpoint writing.

## Curves and stopping

All models early-stopped under the declared EMA rule, without reaching the120epoch cap. Recent30's raw validation peak is epoch40; the two shorter models peak at55. The lowest teacher-forced validation CE is also at40 for all three. Subsequent training loss declines while validation loss rises: this run gives no clear reason to extend the fits just to hit120.

At matched validation epochs:

| Epoch | Recent10 | Recent20 | Recent30 |
| --- | ---: | ---: | ---: |
| 30 | 0.020313 | 0.018411 | 0.022671 |
| 40 | 0.022433 | 0.019133 | 0.026653 |
| 50 | 0.022943 | 0.024374 | 0.024672 |
| 55 | 0.023911 | 0.024504 | 0.024585 |
| 60 | 0.020180 | 0.021835 | 0.022000 |

The separation narrows at50-60; it is not a large uniform ranking advantage at every epoch. Recent30 has lower teacher-forced validation loss than Recent20 at all12 shared evaluation checkpoints. The test comparison uses each model's best raw-val checkpoint and never selects on test.

Curves: `/workspace/artifacts/steam-history-results/analysis/learning-curves.png` and `.pdf`. The four panels show validation ranking by epochs, validation CE, ranking by optimizer-training time and training CE.

## Uncertainty and sensitivity

Paired user bootstrap,10,000 resamples, data seed4242; absolute NDCG differences:

| Comparison | Split | Delta | Conditional95% user interval |
| --- | --- | ---: | --- |
| 30 minus20 | Validation | +0.002150 | [-0.002972, +0.007233] |
| 30 minus20 | Test | +0.004473 | [+0.000056, +0.009067] |
| 30 minus10 | Validation | +0.002742 | [-0.002592, +0.008097] |
| 30 minus10 | Test | +0.006684 | [+0.001499, +0.011938] |

The val intervals include zero. The test lower bound for30vs20 is very near zero. These intervals condition on the fitted models, selected checkpoints and this one sample; they exclude generator/index seed variance, item clustering and selection uncertainty, and have no multiplicity correction. Do not call this a stable or formally confirmed gain. On test,30vs20 improves61 user NDCG contributions, worsens45 and ties1,894; most users remain misses under both models.

A descriptive sensitivity group retains1,552 users whose train/validation/test **boundary days differ**. Selected validation NDCG: .022430/.023572/.025176 for10/20/30; test: .016541/.019202/.023727. Thus the direction persists after removing boundary-day ties. This is not proof of true ordering inside history-day groups, nor an independently chosen evaluation protocol.

All67 val and75 test targets without training target support are misses under all three models. The test gain is concentrated among well-supported targets: for1,256 test targets with>20 training next-item positions, NDCG is .025452/.028124/.036013. These are exploratory observations, not a new user-selection rule or router.

The next focused confirmation would hold this index and cohort fixed and repeat Recent20/30 for two additional generator seeds. A new cohort/index replication remains a subsequent question. No new runs were authorized or submitted during this collection.

## Artifact audit

- Replayed all45 prediction files:42 validation checkpoints plus3 selected tests, **1,800,000 candidate paths**,158 invalid paths,0 duplicate paths. Invalid paths retain original ranks and receive no credit.
- Input file pins, raw case hashes, initialization, frozen SID identity, epoch/update counts, LR, EMA state, raw-val selection and saved item metrics verified.
- Six full checkpoints (best/last for each window) downloaded and checked finite. Best-checkpoint hashes match the test reports; last optimizer/scaler/scheduler/RNG and encoded-case hashes verified. No re-training was performed for this audit.
- Full suite:1,164 tests,152 optional skips,0 failures,115.488s.
- Original ordering/long-user/transductive-catalogue limitations remain as documented in `gr-steam-history.md`.

Reproduce after downloading all predictions and six checkpoint files:

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/history-mpl PYTHONPATH=src \
python scripts/research/analyze_steam_history.py \
  --results /workspace/artifacts/steam-history-results \
  --data /workspace/artifacts/steam-history-dataset \
  --output /workspace/artifacts/steam-history-results/analysis
```

Summary `gr-steam-history-results.json`; completion audit `gr-steam-history-completion.json`. Raw artifacts, ranks/credits CSVs, frozen index/embeddings and all checkpoints remain outside Git under `/workspace/artifacts/steam-history-results`.
