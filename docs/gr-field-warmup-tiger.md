# TIGER on completed AE-warm-start RQ indices

Authorized26October2026. Configuration:
`experiments/amazon_beauty_field_warmup_tiger_2t4.json`.
One private two-T4 session,four fits,two sequential fits per worker:

|Worker|Completed quantizer seed|First generator|Second generator|
|---|---:|---|---|
|0|42|Entire AE frozen index|Joint refinement index|
|1|43|Entire AE frozen index|Joint refinement index|

All four generators/data streams use seed42. These are two realizations of
each index recipe,not four independent generator seeds. No extra cold control,
embedding-freeze generator,encoder or quantizer training is included.

## What led to this test

Parent:`carasik/rhpr-field-ae-warmup-rq`,version1,source
`2fb587ce2acf1f2ec29b71ba82bbd1890ff27383`. Six quantizer branches completed
successfully in753.72seconds. Both continuous AE warm-ups stopped at3500,
then each native RQ branch received5000main updates plus4initialization-only
batches. Metadata reconstruction used fixed normalization,latent64,3x256,
batch2048,FP32,native initializer/Adagrad/STE and beta0.25.

|Completed index|Distinct semantic triples|Largest collision group|Category F1|
|---|---:|---:|---:|
|Entire AE freeze42|12004|3|65.86%|
|Entire AE freeze43|12018|3|65.66%|
|Joint42|11542|15|75.01%|
|Joint43|11258|17|72.79%|
|Historical cold fixed42|188|907|49.45%|

Full-freeze latent/feature drift was zero and every frozen parameter/BN buffer
hash matched. Relative quantization MSE versus catalogue-mean latent predictor
was0.00156–0.00210,versus historical cold0.90156. Joint results were0.00729–
0.00753 with better brand/category/title reconstruction. No recommendation
metric has been measured for these new indices yet.

PCA99%dimension was5 and nominal24bit active count8 at the AE stopping point.
The warm-up rule did not establish reconstruction convergence or the article's
dimension-expansion mechanism. Pretraining adds3500updates and changes the
starting geometry,so comparisons to cold5k are not equal-total-compute.

GPU audit:common AE and initial pre-KMeans states matched within seed,but
first saved post-update book snapshots did not match bitwise. Book1 matched,
book2 differed around1e-6,and book3 differed more. Pre-first-update books were
not saved;initialization versus update contributions cannot be separated.
For a future strictly controlled freeze ablation,initialize books once per
seed and clone that initialized state. This run evaluates the four actual
saved indices;do not describe it as that stricter causal ablation.

## Reuse and training controls

The parent root manifest,source revision,input provenance and suite report are
byte-pinned. Each of the four semantic-code arrays and quantizer reports has
its own SHA256 and size in the configuration. Verify catalogue hashes and
original item order against the generator input before any fit. Verify report
seed,freeze policy,book shape,collision statistics and frozen-state check.
Nested trial paths reject traversal,symlinks,missing files and byte changes.
An updated mutable parent slug therefore cannot silently replace the indices.

Reconstruct the unchanged `SidIndex` with3semantic codes plus mandatory
catalogue-order ordinal suffix,including0for singleton groups. Width256,
four total SID tokens per item. No suffix remapping,repair or index fitting.
Save exact resolved SIDs,their hash,original item IDs and per-index provenance.

Keep generator,training,data and evaluation dictionaries identical to
`amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json`:

- Early GRID TIGER4encoder+4decoder layers,d128,FF1024,d_kv64,6heads,
  ordinary FF,dropout0.1,user bins2000,noSEP,weight tying,beam20.
- Adam1e-3,weight decay1e-4,clip1,effective batch256,physical128,
  FP16 AMP training and FP32 evaluation/beam probabilities.
- All131413noninitial train positions exactly once per epoch,latest20items
  context,NumPy permutation(seed42+epoch),incomplete batches retained.
- Twenty complete epochs,validation only10and20,no epoch1check. Select
  validation item NDCG@10,then test once. Preserve the same per-user leave-last-
  two split,test context including observed validation item,full-catalogue
  candidates and original invalid/duplicate free-beam accounting. No seen-item
  exclusion or sampled negatives. A time-budget-truncated trial fails the suite
  rather than silently becoming a shorter comparison.

Build a shared CPU neural-parameter hash before workers start. Every worker
checks its model against this hash before training;index lookup data differ
by design. Record shared train-case and per-epoch order hashes. Parent source,
index codes and raw example schedule remain fixed throughout each fit.

## Outputs and interpretation

Save per-index semantic/resolved codes,diagnostics and provenance. Per trial
save generator config/parameter count/initial hash,best and last checkpoints
(including optimizer/scaler/RNG in last),training curves/selection/stream,
test metrics and per-user test predictions with scores/targets. Root worker
logs/status,suite results,config and manifest retain run identity. No broad SID
study telemetry,full-path logits,exact-search panels or repeated test selection.

The main comparison is validation-selected test NDCG@10 with Recall@5/10 and
NDCG@5 alongside it. Distinct-code count and reconstruction were candidate
selection diagnostics,not known retrieval objectives. Compare paired index
seeds descriptively;two index seeds with one fixed generator seed do not
establish generator-seed robustness. No automatic follow-up run is authorized.

CPU tests execute the real parent-to-two-worker path and four tiny native
TIGER fits without quantizer training. They check byte/order guards,mandatory
suffixes,matching initial weights,complete target exposures,validation cadence,
one test per model,prediction identity and checkpoint artifacts. The existing
full repository suite is required before publication.
