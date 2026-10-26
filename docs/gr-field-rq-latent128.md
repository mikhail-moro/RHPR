# RQ-VAE latent64 to latent128

User approved two fresh latent128 fits after discussing codebook dimension.
Configuration:`experiments/amazon_beauty_field_rq_latent128_2t4.json`.
One private two-T4 session:fixed-normalized reconstruction on worker0,
GradNorm on worker1,5000 main optimizer updates each. Native initialization
consumes separately recorded extra batches. No new64D controls or AE fits.

Change `quantizer.latent_dim` to128. The native recipe references encoder output
dimension from decoder input,book vectors and the nested initializer's feature
dimension;the builder resolves all of these together. Keep the960D input,
three learned64D structured projections,frozen768D title,MLP hidden widths,
three256-code books,STE,beta0.25,outer loss weights1,neural/codebook Adagrad
LR0.001,GradNorm alpha1.5/weight LR0.025,batch2048,FP32 and seed42. No automatic
MSE/beta compensation or codebook-LR change. Input/output field heads stay the
same shapes. The optional dimension override validates a positive integer;
omitting it preserves the original native64D recipe exactly.

Completed native64D references in `rq64_controls` record kernel/source revision,
report hash,fields,parameters,steps,exposures,usage and collision measurements.
They are copied to `rq64-controls.json` in run outputs. Both came from
`carasik/rhpr-field-rq-balancing`,source
`3dc5c122d42744ab4599aad9a77885f5ed516cfa`:

|64D RQ control|Distinct semantic tuples|Used codes per level|Largest group|
|---|---:|---|---:|
|Fixed normalized|188|7/10/16|907|
|GradNorm|144|6/10/16|1312|

Byte-verified title arrays,static catalogue hashes,field calibration and trial
loss policies remain identical. Seed42 is preserved. Architecture changes
parameter shapes and constructor RNG consumption,so do not claim identical
weights or trajectories across64D and128D. The previous expected initial64D
AE hash is intentionally absent. Both new128D arms must match common-neural
and full-RQ initial hashes within their architecture before comparing objectives.
No custom cross-dimensional weight copying or warm-start is introduced.

Every500 updates retain existing per-field reconstruction,gradient/weight
diagnostics,codebook occupancy/perplexity,geometry and semantic collision
snapshots. Additional shared geometry fields record latent dimension,mean
encoder/quantized norms divided by sqrt(dim),and residual quantization MSE
relative to zero and catalogue-mean latent predictors. A degenerate zero
denominator is null,not infinity. Raw norms normally grow with dimension;
per-coordinate/relative metrics are required for this comparison.

Larger code-vector dimension increases geometric flexibility and the size of
the adjoining MLP matrices. It does not change the number of index sequences:
three256-way choices still have the same discrete capacity. Averaging MSE
over dimensions also changes individual coordinate gradients when dimension
changes. Thus this is an operational latent-size ablation,not a proof that
capacity alone caused an improvement. One seed per objective is diagnostic.

Do not pick a winner from distinct-code count alone. Compare usage/collisions
with brand accuracy,category F1,price MAE,title MSE versus its mean-vector
baseline,relative quantization errors and learning curves. Save all checkpoints,
optimizer/RNG,schemas and initialization/main-update counts. Suffix overflow
is measured without repairs. No generator,validation/test recommendation
evaluation or downstream quality claim.

Tests cover propagation into encoder,decoder,all book vectors and nested
initializer;both128D objectives train with native quantization and GradNorm
updates;checkpoint reload reproduces exact semantic codes. Independent numeric
fixtures verify dimensionless geometry and degenerate baselines.
