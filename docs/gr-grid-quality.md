# First GRID Beauty quality diagnostic

Registered configuration:
[`amazon_beauty_grid_quality_fp16_b512.json`](../experiments/amazon_beauty_grid_quality_fp16_b512.json).
The user authorized one full Kaggle run on 2026-10-23, then selected FP16 and a
larger batch. No search or second control run is included.

The target remains GRID Beauty Flan-T5-XL → normalized-residual RK-Means 3×256
→ encoder-decoder 4+4. The published five-seed mean is Recall@10 .0639 /
NDCG@10 .0347. This is a single-seed diagnostic with explicit execution changes,
not an exact reproduction claim.

| Component | This run |
| --- | --- |
| Data and text | Same verified 12,101-item / 22,363-user author Beauty assets |
| Encoder | Pinned Flan-T5-XL, FP32, batch8, length128, masked mean including EOS |
| Quantizer | GRID source losses and SGD(.5), eight sequential logical ranks, batch2,048/rank, 1,000 total steps/level including two initialization-buffer steps |
| Distance | Native non-MM `torch_cdist`; upstream K-means++ distance retained |
| Generator | Same source-default 4+4/d128/FF1024/heads6 configuration as the paper profile |
| Training | FP16 AMP + GradScaler, effective and physical batch512, Adam LR5e-4 / WD1e-6; no scheduler or clipping |
| Validation | Full catalogue, FP32, beam10, every100 successful updates; latest20items |
| Selection | Early stopping: ten stale NDCG@10 checks; checkpoint: best Recall@10 |
| Test | Once, after loading the Recall-selected checkpoint; Recall/NDCG@5/10 |
| Budget | At most10,800s generator loop, 13,800s cooperative whole-run budget with180s reserved for final evaluation; Kaggle server timeout240min |

FP16 and effective batch512 differ from the reference FP32/batch256. Keeping
the LR and100-update validation cadence means51200 rather than25600 examples
between checks. Quality can improve or decline; no fixed expected drop is
assumed. Report the stop reason and whether early stopping was reached.

The full fitter preserves the tested initialized DDP gradient arithmetic and
rank-local counts. Each level's1,000 steps includes `ceil(3072/2048)=2` buffer
steps, followed by998 shared SGD updates. Rank0's original K-means++ initializer
sets common centroids directly, equivalent to the source's initialization
loss/world-size rescaling. Independent seeded catalogue cycles replace TF
file/worker shuffling; the original paper's full data order is unavailable.
Frozen previous-level residuals are cached; later predictions that do not enter
the active layer loss are omitted. Final assignments are computed using the
completed codebooks, in original catalogue order.

Before whole-catalogue encoding, the run checks the actual larger AMP batch,
compares source/native distances and assignments on device, and repeats the
same eight-logical-rank operator proxy as the completed T4 source profile.
`rk-distance-profile.json` compares its time with the earlier2.0870529487s/update
measurement. This is a comparison between runs, not simultaneous controlled
benchmarking or complete-fit speedup. Actual complete encoding, fitting,
training, validation and test times are also saved separately.

Saved artifacts include embeddings, centroids, unique SIDs/catalogue mapping,
code diagnostics, generator provenance, validation curves, best and last
checkpoints, selection/stop reason and test predictions. The last checkpoint
saves optimizer/scaler state but does not automatically resume the augmentation
stream. `metrics.json`, `summary.md` and `run-manifest.json` are written by the
standard runner. Inputs and outputs stay on private Kaggle.
