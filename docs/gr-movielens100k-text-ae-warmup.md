# Text-only AE → native RQ-VAE, seed 42

One fresh text-control fit to test continuous warm-up against the completed
headed-text control. Same architecture, initial quantizer parameters, frozen
input vectors, fixed catalogue normalization and TIGER protocol. This tests the
staged recipe; it does not isolate warm-up from its extra compute budget.

Config: `experiments/movielens100k_text_ae_warmup.json`.

## Stages

1. Train existing encoder heads/shared encoder and shared decoder/decoder heads
   as a continuous AE, bypassing all quantization and quantization loss. Only
   text is active; other inputs remain zero/masked. Native Adagrad 0.001, FP32,
   fixed catalogue centering/scalar RMS, dimension-mean text MSE, no UW.
2. Probe full-catalogue eval latent PCA every 500 updates. Reuse Field-Aware
   water-filling plateau: minimum 2,000 updates, four consecutive eligible probes
   with identical active-mode count; earliest stop 3,500, cap 5,000. Nominal
   24-bit residual rate is a diagnostic heuristic, not a recommendation guarantee.
3. Retain every neural weight and BatchNorm buffer. Initialize untouched books
   using pinned native GRID residual KMeans on trained eval latents. Keep neural
   modules in eval during initialization and first reconstruction batch, matching
   prior Field-Aware handling; then jointly train all active neural modules and
   quantizer with fresh native Adagrad accumulators. Keep **3,000 native
   Adagrad optimizer steps**, exactly as in the control. Native initialization
   uses separate SGD and does not advance Lightning's Adagrad global step;
   initialization batches and reconstruction batches are recorded separately.
4. Fresh TIGER 4+4 layers/d_model 128, FP16 train/FP32 eval, 20 complete epochs,
   validation 5/10/15/20, one test at best validation NDCG@10 checkpoint. Identical
   seed42 generator initialization, training cases, epoch order, batch sizes,
   history 20, free beam 20 and no seen-item mask.

Assert books stay byte-identical/uninitialized during AE, retained neural hashes,
zero fresh RQ accumulators and eval latent equality before native initialization.
AE Torch RNG is isolated from native initialization randomness. Save AE weights,
optimizer, latents, PCA/loss curves and stop reason; RQ checkpoint, curves, final
books/SIDs; generator checkpoints and all validation/test predictions.

## Inputs and comparison

Use the existing completed **private** notebook output
`carasik/rhpr-ml100k-multimodal` v2 through Kaggle `kernel_sources`. No feature
upload or new Dataset. Verify the exact parent executable/manifest, frozen
`shared-modalities/features.npz` and item-order hashes before training. Reject
later or changed parent outputs. Modern and legacy Kaggle notebook mount paths
are supported explicitly. Features SHA256:
`eeccc0bf7e11f0e2d97acce14ce4973b96d0b4816c6e35ffef679681802dabda`.

Text is the exact previous Sentence-T5 base metadata vector (title with year,
genres, explicit year, runtime, TMDB keywords), without names or plot. Inactive
arrays/heads are retained solely to preserve old architecture/initialization.
No embedding recomputation. Same 1,273 films/order/masks, 943 users, strict-past
timestamp-group holdouts, 85,470 targets per full training epoch.

Main reference: completed headed-text seed42 `seed42_stage_01`. Plain RQ-VAE on
same T5 vectors is secondary, with a different architecture. Compare validation
and test separately, loss/occupancy curves, collisions before suffix and extra
AE cost. One seed cannot establish a stable gain or prove parameter count caused
the previous decline.

## Execution

One private `carasik/rhpr-ml100k-text-ae-warmup`, one cuda:0 worker and one trial.
Standard T4 allocation can expose a second GPU; no second fit is scheduled.
No hardware fallback, retries, extra seeds, TPU, tuning or Telegram. Expected
body roughly 20–25 minutes plus setup/queue, based on prior ~51–53 seconds per
TIGER epoch and ~100 seconds for direct RQ training.
