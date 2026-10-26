# AE architecture, reconstruction and MTL curve probes

User requested AE-parameter/reconstruction changes,then explicitly added UW and
other MTL methods. One private two-T4 diagnostic session;no TIGER or collaborative
objective. Configuration:`experiments/amazon_beauty_field_ae_ablation_probes_2t4.json`.
Nine interventions,each with seeds42/43,5000updates and checkpoints at
0/100/250/500/1000/2000/3500/5000. No adaptive stopping or additional tuning.

|Variant|Single intervention relative to completed fixed-normalized AE|
|---|---|
|latent128|Latent/code-vector dimension64→128;native hidden widths unchanged|
|slower_lr|Neural Adagrad LR0.001→0.0003|
|frozen_bn|Freeze all BatchNorm running statistics at initialization;affine weights train|
|no_price|Multiply normalized price reconstruction by0;price input remains|
|cosine_title|Use PyTorch mean1−cosine for title,normalized by mean-title predictor error|
|title_only|Only normalized frozen title MSE;all inputs remain,original title scale retained|
|uw_lr003|Existing LibMTL uncertainty weighting on raw field losses,weight LR0.03;neural LR0.001|
|dwa|Existing LibMTL DWA,T2,updated on complete catalogue passes|
|gradnorm|Existing LibMTL GradNorm,alpha1.5,weight Adagrad LR0.025,last decoder weight|

The original CE(brand),BCE(categories),Huber(price) and title MSE remain the
field targets except the declared title-cosine variant. Every evaluation logs
actual title MSE AND cosine similarity,regardless of its optimized loss. Fixed
normalizers are constant-predictor errors,calculated once from catalogue inputs.
UW learns task weights;DWA and GradNorm retain the existing implementation and
catalogue normalization. These are controlled methods at declared settings,
not an exhaustive search or independent evidence of recommendation quality.

Reuse the exact pinned title arrays from`rhpr-field-ae-title-5k`. Each worker
owns one seed and evaluates nine independent models;no cross-model warm start.
For64Darms verify initial common neural state against completed seed42/43
controls. Group initialization equality by(seed,latent dimension),so changing
model shape is not incorrectly treated as a failed seed match. The128Darm has
a different random initial model;compare its own step0as well as the control.
Snapshots preserve full neural/optimizer/RNG state and original eval latents.

Each snapshot gets identical sequential raw-latent sklearn residual Lloyd
KMeans3×256,seed+level,n_init1,max300,tol1e-4,CPU thread1,with no PCA projection,
L2orresidual normalization. The code-vector dimension follows AE64/128. Thus
128Dchanges storage and distance compute too;it is not a fixed-compute test.
Keep identical catalogue order and TRAIN-only price/co-user/count-model proxies.
Retain per-step codebooks,codes,hashes,usage,collisions,relative error and losses.

Reuse the completed fixed-normalized AE curves from`rhpr-ae-warmup-probes`
source2f26ea92de59aa5570efb5b041b474190db45ace. Pin its manifest,curves and each
nested KMeans report. Overlay the seed range and existing frozen-T5/RQ-VAE
reference on one panel per intervention. T5 uses full-metadata inputs and a
different quantizer/training budget;its dashed line provides scale rather than
a matched architecture-only control. Plot first-prefix proxies and full-triple
collision/error measures together;better reconstruction is not established NIP.

No generator,validation/test scoring,collaborative loss,extra baseline fits,
automatic retries or Telegram. The lower LR and larger latent are screened at
equal updates,not guaranteed convergence. UW0.03,DWAT2andGradNorm settings are
reused from earlier tested experiments;they are not selected using test.
