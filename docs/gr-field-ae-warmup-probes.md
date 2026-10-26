# Fixed AE stopping probes, without TIGER

User clarification on26October2026:compare different AE warm-up stopping points,
but do not train TIGER yet. Treat index diagnostics as exploratory quality
proxies. Configuration:`experiments/amazon_beauty_field_ae_warmup_probes_2t4.json`.
One private two-T4 session,two independent AE trajectories,seeds42/43.

Each AE uses the existing fields+frozen-title inputs,native GRID neural modules,
latent64,Adagrad0.001,FP32,batch2048 and fixed-normalized CE/BCE/Huber/titleMSE.
Run exactly5000updates,without the earlier water-filling plateau stop. Save full
AE/optimizer/RNG states and original eval latents at0,100,250,500,1000,2000,3500,
5000updates. These are predetermined probes,not PCA-selected winners. PCA is
measured on the learned64Dlatent;no PCA projection or whitening is applied.
Checkpoint saving/evaluation must not change the optimization trajectory.
Snapshots are evidence,not an automatic mid-DataLoader resume implementation.

Fit full-catalogue sequential sklearn Lloyd RQ-KMeans3×256 independently at each
snapshot. Preserve raw latent/residual scale;kmeans++,n_init1,max_iter300,tol1e-4,
CPU thread1,random_state=AEseed+level. Hold this seed fixed across checkpoints
within each AE trajectory. Save actual package versions,iterations,inertia,
books,codes and source hashes. Run seed42 and43 on separate T4 workers;no
hardware fallback,automatic retries,extra seeds or hyperparameter search.

Record catalogue field reconstruction and latent spectrum (PCA99 and nominal
24bit active count),book occupancy/perplexity,collisions and relative error.
Recompute the same first-prefix proxies as the previous audit:

- Price R² on the catalogue,also excluding the highest1% observed prices.
- Degree-normalized binary TRAIN item-user cosine within first-code groups.
- Previous-item count-model predictive gain versus its own prefix unigram,
  smoothing alpha10 and0.5/count prior,user folds by original row modulo3.

No validation/test events enter the probes. The count model is refit on the
other user folds,and evaluated on the held-out TRAIN-user fold. These proxies
are not generator NDCG;price R²/coherence also depend on partition granularity.
Two paired pipeline seeds are exploratory evidence,not a downstream causal
claim. Save raw arrays so other metrics can be recomputed later.

There is no RQ-VAE gradient phase,STE,suffix construction,generator,beam search,
validation/test evaluation or collaborative loss. The previous plan for four
TIGER fits was superseded before submission;it must not be dispatched.

Required outputs:root metrics/summary/run-manifest,input provenance,suite
results,CSV/JSON probe curves and `warmup-proxies.png`;each trial retains AE
curves/final checkpoint plus per-step full AE checkpoint,latent array,
reconstruction/spectrum report,codebook array,semantic codes and KMeans report.

## Collaborative objective: separate follow-up, not implemented

A TRAIN-only collaborative objective could move AE geometry toward behavioural
similarity that metadata reconstruction does not require. It could improve SID
learnability,but may instead emphasize popularity/co-exposure or sacrifice
useful content information. A first isolated comparison could add a modest
contrastive latent loss using TRAIN-only positive pairs with explicit
popularity/support control,while holding the chosen AE budget,architecture,
KMeans and seeds fixed. A loss on an unrelated projection head need not improve
the actual quantized latent. Do not choose its weight using test or assume
better coherence establishes better NDCG. This is a future hypothesis,not part
of the authorized run.

Validation covers unchanged weights/optimizer/RNG with extra snapshots,
TRAIN-only label-permutation-invariant proxies,missing/degenerate groups,and
two complete native CPU AE workers through saved checkpoints,KMeans and plots.
GPU timing and numerical behavior remain properties of the actual Kaggle run.
