# Frozen field AE -> residual KMeans -> matched TIGER

User approved this follow-up on26October2026. Configuration:
`experiments/amazon_beauty_field_ae_rqkmeans_tiger_2t4.json`.
One private two-T4 session,two new generator fits,pipeline seeds42/43.
The completed AE-freeze->native-RQ5k generator results are reused as controls.
No additional generator controls,3000-step comparison or price-loss ablation.

## Question and inputs

Compare two ways of discretizing the same fixed AE representation:
native GRID initialization plus5000Adagrad codebook updates versus sequential
full-catalogue residual Lloyd KMeans. Neither arm changes AE weights. This
tests the operational quantization methods,not an equal-update/compute budget:
initialization sample,optimizer and residual-stage scheduling also differ.

Parent `carasik/rhpr-field-ae-warmup-rq`,source
`2fb587ce2acf1f2ec29b71ba82bbd1890ff27383`,contains the two continuous
3500-update AE catalogue latent matrices. Pin the parent manifest,input
provenance,suite result,and each latent array/AE report by hash and size.
Check catalogue/order,seed,continuous-AE identity,3500steps,latent64 and source
checkpoint hash before fitting. There is no encoder/decoder execution or
training in this run;the original saved latent arrays are the inputs.

The preceding CPU prototype used CPU-reexported AE latents. Small numerical
differences from the saved GPU-evaluated arrays can alter KMeans assignments,
so the original byte-pinned parent arrays are definitive for this experiment.
Preflight on those originals produced12030/12013distinct triples,max group3;
the prototype produced12019/12024. Do not demand identical code counts across
different numerical inputs or sklearn runtimes. Save actual inputs' hashes,
runtime versions,books and assignments in the run output.

## Residual quantizer

Use popular sklearn `KMeans` directly:three sequential256-centroid books,
64coordinates,kmeans++ initialization,n_init1,Lloyd,max_iter300,tol1e-4,
random_state=AEseed+zero-based level,one CPU thread. Each level fits all12101
current residual vectors,assigns their nearest centroids,and subtracts those
centroids before the next level. Preserve raw AE latent scale. No L2,PCA,
standardization,residual normalization,gradient refinement or dead-code reset.

Record iterations,inertia,iteration-cap flag,wall time and package versions.
An iteration cap is reported rather than silently extended. Diagnostics retain
book occupancy/perplexity,collisions and relative quantization error. Append
the unchanged catalogue-order ordinal suffix to every semantic triple using
the existing `SidIndex`,including0for singleton groups;no suffix repair.

This stage executes once in the parent process on CPU before GPU generator
workers start. It is a new KMeans fit,so `quantizer_retrained=true`;
`autoencoder_retrained=false`. It is not a continuation of the native RQ-VAE.

## Generator controls

Keep generator/training/data/evaluation/execution dictionaries identical to
`amazon_beauty_field_warmup_tiger_2t4.json`:

- TIGER4+4/d128,ordinaryFF1024,d_kv64,6heads,dropout0.1,userbins2000,noSEP,
  tied weights,4977920parameters.
- Adam1e-3,wd1e-4,clip1,effective batch256/physical128,FP16 training,
  FP32 evaluation and free beam20 with unchanged invalid/duplicate accounting.
- All131413noninitial train positions per epoch,latest20items,shared generator/
  data seed42 and seed+epoch permutation. Twenty complete epochs,validation
  only10/20,validation item NDCG@10 selection,one selected-checkpoint test.
- One worker per T4,one CPU thread each,no extra telemetry/profiling.

Assert the CPU initial neural-parameter hash and training-case hash match
the completed control before workers start. Workers also check initialization
before each fit. The fixed index lookup differs by design. A truncated epoch
budget is a failed suite rather than a shorter successful comparison.

Completed reference kernel:`carasik/rhpr-field-warmup-tiger`,source
`10b53b8a3c7d9bf17f66909d2d4a819faf96e622`. Both AE-freeze controls selected
epoch20. Test NDCG@10 was0.0202812914(seed42)and0.0202405417(seed43),Recall@10
0.0398425971/0.0404239145. Copy their reported scores/source hashes to
`controls.json`;do not spend quota training them again. Compare on the same
split and selected-checkpoint rule. Two AE/index seeds with fixed generator
seed do not establish generator-seed robustness.

## Evidence and boundaries

Outputs retain per-index codebooks,semantic/resolved codes,KMeans reports,
AE source provenance and diagnostics. Generator output keeps the existing
best/last checkpoints,optimizer/scaler/RNG,training/selection curves,
per-user test predictions,shared stream,worker and run manifests.

Earlier codebook audit found healthy Field occupancy but a price-oriented
first partition and weaker TRAIN collaborative coherence. CPU KMeans improved
quantization MSE about8%without improving those behavioural proxies. These
are exploratory observations,not a known NDCG outcome or proof of a sole
cause. The separate3000-vs5000hypothesis remains open:existing SID snapshots
can support later generator fits without quantizer retraining,but are not
part of this authorization.

Tests cover raw-scale preservation,unmodified inputs/global RNG,deterministic
replay,greedy assignments,finite/shape/identity guards,and the actual two-worker
path from frozen latent arrays through KMeans to two tiny native TIGER fits.
The full repository suite is required before publication.
