# RHPR agent instructions

## Research status

- The history-compression / Query SSM track was closed on 2026-10-21. Read
  `docs/research/closed-history-compression/README.md` for its final evidence.
- Earlier experiment plans are historical and have no pending GPU launches.
- The active direction is comparing item encoders for SID-based generative
  recommendation. Read `docs/gr-research-brief.md`, `docs/gr-baseline.md`, and
  `docs/research/item-encoders/README.md`.
- Latest user authorization,2026-10-26:continuous AE warm-up then native RQ-VAE
  in three modes:joint,entire AE frozen,feature-encoder-only frozen. This
  supersedes the earlier four-arm plan with separate full RQ-KMeans variants.
  Use `experiments/amazon_beauty_field_ae_warmup_rq_2t4.json`;see
  `docs/gr-field-ae-warmup-rq.md`. One private two-T4 session,two pipeline
  seeds42/43,one shared AE per seed,then three5000-main-update RQ branches.
  Transfer embeddings/projections,BN buffers,MLP encoder/decoder,heads and loss
  state. Native mini-batch residual KMeans initializer remains unchanged.
  AE max5000,min2000,500-step catalogue PCA;stop after four equal active-mode
  counts using reverse water-filling at aggregate nominal24bits. This is a
  heuristic adaptation of2605.06870 to3x256 residual books,not a proven bound
  or guarantee of reconstruction convergence. Also log8bit count and PCA99pct.
  Keep fixed-normalized four-field objective,latent64,Adagrad0.001,FP32 and
  batch2048. Reset optimizer accumulators at phase transition,freeze eval modes
  and verify frozen buffers/weights. Save both phases,drift,field losses,usage
  and collisions. No KL,generator,suffix repair,automatic retries or tuning.
- Latest user authorization,2026-10-26:increase RQ latent/code-vector dim64
  to128 with two5000-update runs,fixed normalization and GradNorm,on two T4s.
  Use `experiments/amazon_beauty_field_rq_latent128_2t4.json`;see
  `docs/gr-field-rq-latent128.md`. Keep3x256,input960,hidden widths,LRs,beta,
  initializer,seed42,batch2048 and FP32. Reuse completed64D RQ controls.
  Initial128D hashes must match BETWEEN new arms;do not require the old64D
  hash across different shapes. Log norms divided by sqrt(dim),relative
  quantization MSE,all field metrics,book usage and collisions. No generator,
  codebook-LR change,extra AE fits,suffix repair or automatic tuning/retries.
- Latest user authorization,2026-10-26:restore native RQ-VAE with the completed
  fixed-normalized and GradNorm objectives,one private two-T4 session,two
  independent5000-update quantizer-only trials. Use
  `experiments/amazon_beauty_field_rq_balancing_2t4.json`;see
  `docs/gr-field-rq-balancing.md`. Keep3x256/latent64/native initializer/STE,
  summed codebook+beta0.25commitment at weight1,and neural Adagrad0.001/FP32.
  GradNorm balances only four field losses,not the quantization auxiliary.
  Reuse byte-pinned frozen titles and completed AE controls;verify the same
  initial neural hash before training. Record actual initialization batches
  separately from main reconstruction updates. No generator or suffix repair.
- Latest follow-up:the user requested alternatives to UW and asked about
  available compute. Run one private two-T4 continuous-AE suite using
  `experiments/amazon_beauty_field_ae_nonuw_2t4.json`:equal raw sum,square-root
  constant normalization,DWA(T2) and GradNorm(alpha1.5,weight Adagrad LR0.025),
  5000updates each,same seed42/neural initialization/Adagrad0.001/FP32.
  DWA/GradNorm use the completed fixed-normalized AE as control. Preserve all
  per-field losses/checkpoints/weight diagnostics;no RQ,generator or SID yet.
  See `docs/gr-field-ae-nonuw.md`;do not automatically tune or retry trials.
- Latest user authorization,2026-10-26:check multiple fast continuous-AE loss
  hypotheses,parallel on two T4s. Use
  `experiments/amazon_beauty_field_ae_loss_suite_2t4.json`;four independent
  5000-step trials:UW log-variance LR0.03/0.1 (neural Adagrad LR0.001),fixed
  constant-predictor-normalized losses,and title-only reconstruction with all
  input fields retained. Same seed42,initial neural weights,FP32,latent64 and
  batch2048. Reuse byte-verified frozen title targets and the completed original
  UW control;no baseline rerun,RQ,SID,generator or recommendation claims. See
  `docs/gr-field-ae-loss-suite.md`. This authorizes one private two-GPU session,
  not arbitrary follow-up tuning or automatic retries.
- Latest user authorization,2026-10-26:run the separately trained continuous
  fields+frozen-title autoencoder control,5000updates,seed42,same native neural
  initialization/latent64/Adagrad1e-3/uncertainty objective. Use
  `experiments/amazon_beauty_field_title_ae_5k_control.json`;see
  `docs/gr-field-autoencoder-control.md`. Remove both quantized forward and
  codebook/commitment loss;no SID,codebook initialization or generator. Compare
  every field at500-step catalogue evaluations,not only weighted total loss.
- Latest follow-up,2026-10-26:the user requested a5000-step extension of the
  fields+frozen-title QUANTIZER-ONLY arm. Use
  `experiments/amazon_beauty_field_title_rqvae_5k_diagnostic.json` with identical
  architecture,optimizer,seed42,objective and500-step diagnostics. Fresh total
  5000updates,not5000extra and not automatic checkpoint resume. Depth remains
  unchanged;the user's deeper-network suggestion is a future separate ablation.
- Latest user decision,2026-10-26:two QUANTIZER-ONLY3000-step GPU runs to compare
  ordinary Sentence-T5(full metadata)RQ-VAE with the joint structured encoder
  plus frozen Sentence-T5(title only). Use
  `experiments/amazon_beauty_rqvae_plain_3k_diagnostic.json` and
  `experiments/amazon_beauty_field_title_rqvae_3k_diagnostic.json`. Title768D
  concatenates directly with learned192D fields;fourth reconstruction head MSE
  joins uncertainty weighting. Same seed42,3x256,latent64,native optimizer and
  catalogue diagnostics every500steps. Explicitly no generator/validation/test.
  Suffix overflow is a recorded measurement in this diagnostic stage,not an
  execution failure;never repair/widen it. This supersedes initial title exclusion
  for the new arm only. Read `docs/gr-rqvae-title-comparison.md`.
- Both3kdiagnostic runs completed:plain11991tuples/max group4/used143,256,256;
  fields+title845tuples/max1162/used10,42,157,still suffix overflow. The plain
  run has13stdout events and7full probe snapshots;its console summary was
  shorter,not absent. Future plain logs now expose matching summary fields.
  Do not retrain to recover these artifacts or call title a complete collapse fix.
- The user specified joint feature-encoder/RQ-VAE reconstruction on2026-10-26:
  brand embedding+Linear/categories MeanPooling embeddings+Linear/standardized
  price+Linear,concatenate into native GRID RQ-VAE;CE/BCE/Huber combined by
  LibMTL uncertainty weighting. Title is explicitly excluded. Read
  `docs/gr-field-rqvae.md`;config
  `experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json`. Preserve the
  fast generator/training/evaluation control. This metadata objective is not
  joint NIP training and does not establish that text encoders are inferior.
- The first field-RQ-VAE run failed at SID suffix overflow after3000updates:
  68unique semantic IDs for12101items,largest group1796,used codes5/8/13.
  The user requested5000updates and codebook diagnostics on2026-10-26. Use
  `experiments/amazon_beauty_field_rqvae_tiger_5k.json`:fresh same-seed fit,
  full-catalogue eval probes after initialization/every500steps/final,counts,
  entropy/perplexity,residual/centroid geometry,raw loss components and snapshots.
  Preserve architecture,optimizer,uncertainty,fast generator and suffix policy;
  do not silently widen the suffix or reinitialize dead codes.
- The user authorized a fast Sentence-T5 + native GRID RQ-VAE3x256 + previous
  GRID TIGER4+4/d128 baseline on2026-10-25:3000 quantizer updates,20 generator
  epochs,FP16 generator,validation every10,no study telemetry. Read
  `docs/gr-rqvae-baseline.md`. This is separate from joint encoder training;
  do not resume the paused index-study kernels.
- The user authorized restoring the faster early prefix protocol on2026-10-25:
  sampling is not the item-encoder research axis. Use
  `experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json`.
  Restore the early4+4/d128 ordinary-FF generator: noSEP,user bins2000,
  4,977,920parameters,beam20,Adam1e-3/wd1e-4/clip1,batch256/physical128.
  Every noninitial train position appears once per epoch, latest20 context,
  131,413examples. Preserve RQ-VAE3x256/3000updates, frozen Sentence-T5,
  FP16 generator,20epochs, validation only at10and20, NDCG@10 selection.
  The expanded-FF and plain-FF sampled-window configs are historical;
  do not launch them as the active baseline or revive the paused studies.
- The SID index-structure investigation was paused on 2026-10-25 at the user's
  request. Read `docs/research/paused-sid-index/README.md` and its results ledger
  before reopening it. It did not establish a causal semantic-collision rule
  or prove text encoders are defective; SHANY primarily exposed missing brand
  metadata. The historical study (2026-10-24) used six RQ-KMeans seeds,
  frozen Sentence-T5 embeddings, identical generator initialization and shared
  TIGER/GRID window manifests. Two T4s in one session, one independent seed
  worker per GPU (three seeds each), FP16, 3+3 layers at d128, 20 epochs,
  validation every5, no early stopping, final-epoch comparison. Read
  `docs/gr-sid-index-study.md`. No UI: Parquet evidence with optional DuckDB.
- Raw autoregressive scores and valid-child locally renormalized scores are
  separate objectives. Compare exact search and beams under MATCHING scoring.
  Do not interpret six index seeds as six independent generator seeds.
- The completed follow-up crossed frozen original rq11/rq22 indices with
  generator seeds101/202/303, preserving data seed4242 and20epochs. See
  `docs/gr-sid-training-variance.md`. Fixed train/validation probes preserve
  RNG/weights; their Parquet sidecars are distinct from checkpoint evaluations.
  Initialization hashes must match WITHIN each model seed across indices.
- The user explicitly wants GPU, not TPU; implementation does not authorize
  dispatch. Preserve the paused TPU changes; do not restart previous kernels.
- Historical reproduction work is in `docs/gr-genpas-tiger-epochs.md`,
  `docs/gr-genpas-tpu.md`, `docs/gr-grid-quality.md` and
  `docs/gr-grid-reproduction.md`. Do not silently revive those protocols.
- Do not implement unrequested tokenizers or assert a representation bottleneck
  before the baseline and diagnostics support that claim.

## Working agreement

- Keep experiments reproducible: configuration, seed, code revision, and final
  scalar metrics belong in the run output.
- Never commit credentials, `.env` files, downloaded private data, or Kaggle
  API tokens.
- Work on a branch and prefer a pull request over writing directly to `main`.
- Run `python3 -m unittest discover -s tests -v` before publishing changes.
- Do not spend Kaggle quota unless the user explicitly asks to run or deploy an
  experiment. Code changes alone are not authorization to start compute.

## Experiment contract

- Configurations live under `experiments/` as JSON.
- `kaggle/run.py` is the stable kernel entry point.
- Successful runs write `/kaggle/working/metrics.json` and
  `/kaggle/working/summary.md`.
- `metrics.json` must be a flat JSON object of finite scalar values. Add richer
  outputs as separate artifact files.
- Keep Kaggle kernels private unless the user explicitly requests publication.

## Integrations

- Use the GitHub connector/MCP for repository, PR, workflow, and run status
  operations. Use `gh` only when the connector does not expose the needed
  operation.
- Use Kaggle MCP for quota/status/output inspection and interactive control.
  Use `.github/workflows/kaggle-experiment.yml` for reproducible execution.
- Telegram is an outbound notification channel only. Send lifecycle summaries
  and workflow links; never send secrets, datasets, stack traces, or full logs.
- When MCP and CLI report different state, treat Kaggle/GitHub server state as
  authoritative and record the discrepancy.

The repo-local `rhpr-ops` plugin contains detailed GitHub, Kaggle, and Telegram
skills. Use the relevant skill whenever a request touches those integrations.
