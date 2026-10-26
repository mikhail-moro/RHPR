# Continuous AE warm-up before native RQ-VAE

User-approved diagnostic,26 October2026. Configuration:
`experiments/amazon_beauty_field_ae_warmup_rq_2t4.json`.
One private session,two T4s,one independent pipeline seed per GPU(42/43).
No generator or recommendation evaluation. This replaces the tentative
four-way plan containing standalone full-catalogue RQ-KMeans variants.

The AE is the existing native RQ neural path with quantization removed:
learned brand/category embeddings and three field projections plus frozen
Sentence-T5 title768D → native BatchNorm/L2 → MLP encoder → continuous64D
latent → native MLP decoder → four field heads. There is no Gaussian sampling
or KL term. Input960D,hidden widths768/256/128 and reverse decoder remain
unchanged. Reconstruction is the existing fixed-normalized CE/BCE/Huber/title
MSE objective,calibrated once from catalogue constant predictors. FP32,
batch2048,native Adagrad0.001;not another loss/depth/dimension sweep.

Each seed trains one AE,then transfers its exact parameters and buffers into
three fresh native RQ models. This includes field input tables/projections,
normalization,encoder,decoder,heads and uncertainty module state(its weights
are unused/frozen with the fixed-normalized objective). Each branch starts
from the same AE checkpoint and the same fresh initial RQ state within seed.
The two seeds are full pipeline repetitions,not merely KMeans seeds.

|Branch|Frozen after warm-up|Trainable in RQ phase|
|---|---|---|
|joint|Frozen title targets only|Feature encoder,normalization,MLP encoder/decoder,heads,books|
|ae|Entire AE including BN statistics and decoder/heads|Books only|
|embeddings|Complete feature_encoder:tables and per-field projections|Normalization,MLP encoder/decoder,heads,books|

The embeddings label therefore covers the full field input representation
module,not just categorical lookup tables. All branches have the same frozen
title targets. Gradients and eval mode are both enforced for frozen modules;
before/after hashes include buffers. A mismatch fails the run.

## Warm-up stopping

Reference:[Continuous First,Discrete Later:VQ-VAEs Without Dimensional Collapse](https://arxiv.org/html/2605.06870),
section4 and appendixC.1. At step0 and every
500updates,compute centred full-catalogue latent covariance eigenvalues λ.
Solve Gaussian reverse water-filling:

`sum(max(log2(lambda_i / D),0))/2 = R`.

The count of λ>D estimates the directions that survive at rate R. Record
the entire spectrum,water level,active count and PCA99%-variance dimension.
The article's effective dimension(PCA99%) and predicted active-mode count
are separate measurements;the stopping implementation monitors the latter.

For a single256-way book R=8bits. For our three residual books the stopping
rate is their aggregate nominal24bits;8bit diagnostics are also retained.
Treating this sum as one Gaussian rate constraint is an explicit heuristic
adaptation. The article studies a different quantization setting,so its bound
is not established for this nonlinear residual catalogue quantizer.

Our bounded stopping rule:at least2000updates,then four consecutive catalogue
probes with exactly equal24bit active-mode count. Probes before2000do not
count towards patience. Earliest stop3500updates;hard cap5000. These patience
and budget settings are ours,not hyperparameters copied from the article.
Do not call hitting the cap convergence. Even a spectral plateau does not
guarantee that all field reconstruction losses have converged;save those
curves separately. No extra GPU tuning is implied if the criterion fails.

## Transition and comparison

Native GRID already initializes3x256books with mini-batch residual KMeans:
buffer3072,1000internal iterations,unchanged TorchCdist squaredL2. We retain
that initializer rather than introducing another full-catalogue algorithm.
Native STE,summed codebook+beta0.25commitment loss at weight1 and residual
normalization policy are unchanged. No dead-code resets or suffix repairs.

During initialization and the first reconstruction batch,all AE modules run
in eval mode in every branch. This fixes the starting latent coordinates and
BN state across freeze policies. Subsequent joint/embeddings branches resume
train mode for their unfrozen modules. Freeze does not disable books.
Native reconstruction STE sends gradients into the AE,not the centroids;
with the entire AE frozen,book adaptation is driven by the native codebook
loss. Its reported field losses are diagnostic,not additional centroid
reconstruction gradients. This is a fixed-representation control,not joint
decoder learning under another name.

Each RQ branch receives5000main updates plus separately counted native
initialization batches. A fresh native Adagrad is used at transition;AE
optimizer accumulators are retained in its checkpoint but not transferred.
Data order/initialization seed matches across branches within a pipeline.
Check initial common/full-state hashes per seed,not across seeds.

Warm-up is additional training. Compare the three branches as matched
pretraining contrasts;do not call their comparison with old cold-start5k
runs equal-total-compute. Report AE and RQ update counts,exposures and wall
times separately. The completed64D cold runs remain historical references;
no duplicate cold run is authorized in this suite.

## Saved evidence

- `warmup/seed{seed}/`:AE checkpoint and optimizer/RNG,exact catalogue targets
  and schema,feature and latent arrays,field curves,spectral curves,stop reason.
- `trials/{mode}_s{seed}/`:RQ checkpoint/optimizer/RNG,semantic codes,input
  feature arrays,field and codebook curves,per-level occupancy/perplexity,
  quantization geometry,collisions,complete native initialization/main counts.
- `warmup-transfer.json`:source checkpoint/neural-state hashes,all AE module
  hashes before/after,freeze verification and optimizer transfer policy.
- Each catalogue RQ probe adds MSE drift of full features/latents relative to
  its AE checkpoint,latent PCA/water-filling,book and actual quantized-output
  PCA99%dimensions. Frozen AE must have zero drift;embedding freeze must have
  zero input-feature drift. Arrays/snapshots preserve item order for follow-up.
- `workers/`, `suite-results.json`, flat finite `metrics.json`,run provenance
  and config. Warm-up failure skips its dependent branches without repeating
  it;another worker may still complete. No automatic retries/fallbacks.

Interpret reconstruction,relative quantization error,usage and collision
measurements together. More occupied codes alone does not establish better
NIP or a causal explanation of the previous collapse.

Tests exercise the analytic water-filling equation,rotation/translation
invariance,minimum/patience,strict hardware/config checks,and the real two
CPU-worker path with two AEs and six native RQ branches. They verify shared
weights/first-update books,exact frozen BN/parameter invariance,zero frozen
drift,and continued encoder/decoder learning in unfrozen branches.
