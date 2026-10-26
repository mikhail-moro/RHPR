# Separately trained continuous autoencoder control

Authorized2026-10-26 after the fields+title RQ5000-step diagnostic exported
1263semantic tuples with a maximum1343-item group. Configuration:
`experiments/amazon_beauty_field_title_ae_5k_control.json`.

Keep catalogue/preprocessing,frozen Sentence-T5 title,three learned64D field
projections,960D input,GRID BatchNorm+L2,MLP768/256/128→latent64,
decoder64→128/256/768→960,and all four field heads unchanged. Use the same
CE/BCE/Huber/title-MSE and LibMTL uncertainty parameters. Construct the native
model with seed42 first,then extract its neural modules without reinitializing
them;tests compare every common initial state tensor exactly with the RQ arm.
Codebooks are absent from the trained model,optimizer and checkpoint.

Forward is field encoder→normalization→MLP encoder→continuous latent64→MLP
decoder→field heads. No nearest-code lookup,STE,codebook/commitment loss or
initialization procedure is called. Adagrad comes from the same native GRID
recipe(lr0.001,no scheduler or clipping),FP32,batch2048,seeded catalogue
shuffle,incomplete final batches retained,5000actual optimizer updates.
Uncertainty log variances initialize identically. Depth/width/loss scales are
not tuned in this control.

RQ's initialization consumes extra batches before normal optimization;this
control has no initialization warm-up. Match optimizer updates and the seeded
loader policy,but record item exposures rather than claiming identical total
data exposure or RNG trajectory through RQ-specific initialization. This removes
the entire RQ block and its auxiliary loss together;it does not isolate hard
assignment from commitment regularization.

Evaluate all catalogue fields at step0,every500updates and finally. Log raw
field losses,observation counts,uncertainty weights/log variances,gradient norms,
brand accuracy,category microF1,price MAE,title cosine/MSE and zero/mean-vector
predictor baselines,plus latent norm. Evaluation preserves RNG,module modes
and BatchNorm buffers. There is no generator,validation selection,test scoring,
SID or collision metric;zero collisions would be a misleading report here.

Save `autoencoder.pt`(all neural modules,uncertainty,optimizer,RNG,schema and
recipe),`autoencoder.json`,`field-learning-curves.json`,
`reconstruction-learning-curves.json`,`catalogue-fields.npz`,
`feature-schema.json`,`latent-embeddings.npy`,`item-embeddings.npy`,title
encoder provenance,and catalogue row mapping. Checkpoint reload must reproduce
latents and reconstruction metrics. Automatic resume is not implemented.

The earlier continuous-input probe used a decoder trained with RQ and could
not answer this question. This control trains the continuous path separately.
If reconstruction improves substantially,it motivates inspecting the RQ block;
if a field remains poorly reconstructed,investigate objective/optimization and
capacity for that field. Neither outcome alone identifies a unique cause.
