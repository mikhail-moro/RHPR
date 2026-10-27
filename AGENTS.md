# RHPR agent instructions

## Research status

- Completed context ablation,2026-10-27:no pending GPUjobs. Source32b7ce00,
  kernel`carasik/rhpr-context-length-replay`. See
  `docs/gr-sid-context-ablation-results.md` and local artifacts
  `/workspace/artifacts/sid-context-ablation/analysis/README.md`. Eight inference
  passes,zero training/test,227secbody. Primary same1019history>=20users:
  swap NDCG10change at L3/5/9/20=-9.78%/+1.50%/+9.20%/+18.51%.
  Prespecified3to20interaction+0.0140825,userCI[0.003764,0.023628];
  target-block sensitivityCI[0.002382,0.026132]. Secondary history>=9and>=5
  interactions positive butCIoverlap0. Both models peak in absolute quality at9,
  then decline at20. Exact parent replay,unchanged-input controls and weight
  hashes all pass;all8rawbeamfiles independently verified in memory. One fitted
  epoch60pair only,not seed/epochreplication or a pure semantic/CFcomparison.
  Data/Parquet/contributions/figures retained;no next experiment authorized.

- Latest authorization,2026-10-27:inference-only fixed-user context truncation.
  See `docs/gr-sid-context-ablation.md` and
  `experiments/amazon_beauty_sid_context_ablation_2t4.json`. Exactly the two
  frozen epoch60checkpoints from `carasik/rhpr-rqkmeans-als-long`,source10e716df;
  pin source/checkpoints/SIDs/selection/referencepredictions. No fitting,optimizer,
  checkpointselection,indexupdates,testevaluation or upload. One model per T4,
  FP32batch32/freebeam20. Mask old events toPAD in the original20positions,
  retaining identical full-user batch layout across lengths3/5/9/20. Primary
  cohort1019users history>=20usesall4lengths;secondary4242history>=9uses3/5/9
  and10980history>=5uses3/5. Primary interaction is change in swap advantage
  from3to20within the same users/targets. Save allpredictions and compactuser
  metrics,verify modelunchanged and exact outputs on unchanged-input rows;
  report parent replaydrift. No otherGPUjobs,seeds,retries or Telegram.

- Completed long-run analysis,2026-10-27:the two60epoch fits finished;no pending
  launches. Artifacts `/workspace/artifacts/sid-als-long/analysis/README.md`.
  Epoch60validation NDCG10 original0.03985706 vs ALSswap0.04075540(+2.25%),
  paired-user interval for delta[-0.000694,+0.002469];all6checkpoint intervals
  overlap0. Late30/40/50/60mean gain+0.375%.987better/961worse at60;
  descriptive>=3/4late checks with no opposite signs gives281better/288worse,
  1126mixed-sign users. Full user/item Parquet,complete stable-group CSVs,
  extrema,histories,paths and all12validation predictions saved. Both fits
  consumed7884780examples;7AMPskips each,30833actual updates out of30840.
  Earlier rarebrush10906anecdote is not replicated:both hit26/26at60;rare
  TRAINtarget-count1–4cohort has41vs37hits. History5–9has consistent late
  descriptive gain;multiple post-hoc cohorts/one fitted pair prevent a claim.
  TRAIN-only exact-ID context-profile probe also saved:ALSalready improves
  first-branch last5contextcoherence~0.97%,but low-support profiles have poor
  disjoint-user reproducibility. `context-hypothesis.md` formulates a frozen
  prefix encoder/context-distribution graph with confidence shrinkage;it is
  a proposal,not a launched experiment. No new training,swaps,test evaluation,
  GPU sessions or Telegram were used for this analysis.

- Latest authorization,2026-10-27:two longer fits,original RQ-KMeans versus
  the latest screen's ALSuniform candidate. User chose60epochs and then
  explicitly corrected validation to every10epochs. See
  `docs/gr-sid-als-long.md` and `experiments/amazon_beauty_sid_als_long_2t4.json`.
  Two fresh matched fits,one per T4,shared seed42,GRID4+4/d128,FP16,all-prefix
  sampling. Frozen indices reuse the existing private ALSscreen dataset;no
  graph/quantizer/encoder refit or upload. Explicit original-path control keeps
  manifest-pinned original bytes and identity ownership. Primary endpoint is
  epoch60validation,also retain10/20/30/40/50trajectory and best separately.
  Save all per-user validation predictions and full last state. No test access,
  extra candidates,seeds,tuning,automatic retries or Telegram. Earlier screen
  restrictions are historical for its completed run,not a block on these fits.

- Latest authorization,2026-10-27:screen the next graph hypotheses with10TIGER
  epochs,validation5/10. See `docs/gr-sid-als-screen.md` and
  `experiments/amazon_beauty_sid_als_screen_2t4.json`. Exactly six new fits:
  two frozen T5index families x ALSuniform/ALSpriority/random-on-priority-itemset.
  Reuse original/legacycooc epoch10controls. ALSimplicit0.7.3/64factors/.1reg/
  alpha20/20iterations on fixed75%TRAINusers;no encoder/quantizer updates.
  Max100kchecks/cap256accepted;all guided trajectories reached256. Preserve
  exact full paths and suffixes. Shared seed42,4+4/d128,FP16,all-prefix sampling.
  Primary score is epoch10validation,even if best5/10differs. Save per-user
  validation at5/10 and full last checkpoint with optimizer/scaler/allRNG for
  later continuation. No test access/evaluation or automatic20epochfinalists.
  Private derived public-Amazon inputs:`carasik/rhpr-sid-als-screen`. No raw
  reviews,text,names,email,credentials;transfer audit under local artifacts.
  Do not launch otherCFmethods,tuning,seeds,retries or Telegram. Field stays closed.

- Latest request,2026-10-27:inspect user-level validation differences and explain
  graph proposal selection/alternative signals. Training saved val aggregates
  only. Prepare one short inference-only replay on the four frozen epoch20
  checkpoints;see `docs/gr-sid-validation-replay.md` and
  `experiments/amazon_beauty_sid_validation_replay_2t4.json`. CPUfull replay
  costs about3hours;two-T4 replay expected5–8minutes. No optimizer/training,
  model selection or test pass. Pin every checkpoint/SID/report;FP32batch32,
  beam20,preserve complete paths and record any aggregate numerical drift.
  Save per-user val predictions/ranks/metrics for all22363users. No other
  training,CF-method experiments,retries or Telegram. Current graph is sparse:
  88.3%top20edges have one shared TRAIN user;this is a hypothesis,not a cause.

- Latest authorization,2026-10-27:run TIGER20epochs on the completed graph
  candidates atlambda0.3,k20,20kproposals. Use
  `experiments/amazon_beauty_sid_graph_tiger_2t4.json` and
  `docs/gr-sid-graph-tiger.md`. Exactly three new fits in one private two-T4
  session:original/swapped RQ-KMeans on the same worker,swapped native RQ-VAE
  on the other. Reuse completed original RQ-VAE source6e37388a,whose seed/data/
  generator/training configs match;old RQ-KMeans100epoch run is not a control.
  Frozen index dataset:`carasik/rhpr-sid-graph-indices`,manifest3df39c67….
  Preserve exact complete SID paths INCLUDING suffix viafrom_full_sids;no
  centroid assignment,suffix regeneration,encoder/quantizer/graph fitting.
  Shared seed42,4+4/d128,FP16,batch256/physical128,all131413prefix cases,
  20complete epochs,val10/20,NDCG10selection,one FP32freebeam20test. Check
  initial neural5f4a24fa…and case6f7ba06c…hashes. Save full checkpoints,
  curves and per-user test predictions. No otherlambda,k,InfoNCE,extra seeds,
  tuning,automatic retries or Telegram. Field branch remains closed.

- Latest request,2026-10-27:the Field RQ-VAE branch is closed. The active
  experiment is CPU-only fixed-tree SID graph placement;see
  `docs/gr-sid-graph-swap.md` and
  `experiments/amazon_beauty_sid_graph_swap_cpu.json`. Reuse frozen full-metadata
  Sentence-T5 embeddings and saved native RQ-VAE/normalized-residual RQ-KMeans
  indices. Swap complete paths including suffix;preserve all prefix occupancies.
  Fixed semantic/CF graphs,75/25 TRAIN-user split,weighted-LCP and one sampled
  InfoNCE objective;save legacy RQ diagnostics and independent-user metrics.
  Current local probe completed18trials/90snapshots in203.7seconds;no generator
  trained or GPU/Kaggle submission. Artifacts:`/workspace/artifacts/sid-graph-swap`.
  Small positive gain changes do not establish a downstream improvement;paired
  user-bootstrap intervals overlap zero. All earlier Field fits below are
  historical,no pending launches. Preserve paused TPU edits;no Telegram.

- Latest request,2026-10-26:run TIGER on saved GradNorm field-encoder indices.
  Use `experiments/amazon_beauty_field_gradnorm_tiger_2t4.json`;see
  `docs/gr-field-gradnorm-tiger.md`. One private two-T4 session,two generator
  fits on ablation source d13dd8aada6c1f60ce36409147c261af5a09b608,
  gradnorm_s42/43 at AE5000. Reuse exact saved RQ-KMeans codes/books/reports;
  no encoder forward,AE/quantizer fit or nearest-centroid re-export. Preserve
  shared generator/data seed42,4+4/d128,FP16,20epochs,all-prefix protocol,
  validation10/20,NDCG10selection and one FP32beam20test;verify original neural
  and case hashes. Reuse completed AE3500→KMeans TIGER controls rather than
  rerunning them;different AE budgets prevent a pure objective-only claim.
  Field-level encoders are the research focus;title-only is a text-semantics
  control,not the main method and not launched here. No extra controls,tuning,
  automatic retries or Telegram.

- Latest request,2026-10-26:run AE-parameter/reconstruction/MTL curve probes,
  still WITHOUT TIGER. Use `experiments/amazon_beauty_field_ae_ablation_probes_2t4.json`;
  see `docs/gr-field-ae-ablation-probes.md`. One private two-T4 session,nine
  interventions x seeds42/43,each5000fixed updates,same eight checkpoints.
  Variants:latent128,neural LR0.0003,frozen BN running stats(affine trains),
  no-price reconstruction(input retained),cosine title loss,title-only,UW
  weight LR0.03,DWA T2,GradNorm alpha1.5/weightLR0.025. Hold other settings
  fixed;reuse pinned original title inputs,completed fixed-normalized AE curves
  and frozen-T5/RQ-VAE reference. Keep raw-latent3x256KMeans,dimension follows
  AE64/128;initial neural equality is checked within(seed,shape). Save complete
  checkpoints/latents/books/codes,per-field MSE/cosine,TRAIN-only proxies and
  comparison plots. No generator,validation/test,collaborative objective,
  automatic extra tuning/retries or Telegram.

- Latest user clarification,2026-10-26:run fixed AE warm-up stopping probes
  WITHOUT TIGER. Use `experiments/amazon_beauty_field_ae_warmup_probes_2t4.json`;
  see `docs/gr-field-ae-warmup-probes.md`. One private two-T4 session,two
  continuous AE trajectories seeds42/43,5000fixed updates,same field/title
  inputs,latent64,fixed-normalized objective and native Adagrad0.001/FP32.
  Save complete checkpoints and GPU-eval latents at0/100/250/500/1000/2000/
  3500/5000;no adaptive stopping or PCA projection. Fit identical raw-latent
  RQ-KMeans3x256 at each checkpoint and record geometry,reconstruction,usage,
  collisions and catalogue/TRAIN-only first-prefix behavioural proxies.
  Preserve raw arrays,curves,provenance and plots. No generator,TIGER,
  validation/test metrics,collaborative loss,automatic tuning/retries or
  Telegram. The preliminary four-TIGER plan was cancelled before submission.

- The history-compression / Query SSM track was closed on 2026-10-21. Read
  `docs/research/closed-history-compression/README.md` for its final evidence.
- Earlier experiment plans are historical and have no pending GPU launches.
- The active direction is comparing item encoders for SID-based generative
  recommendation. Read `docs/gr-research-brief.md`, `docs/gr-baseline.md`, and
  `docs/research/item-encoders/README.md`.
- Latest user authorization,2026-10-26:run two TIGER fits on AE->RQ-KMeans,
  pipeline seeds42/43,one private two-T4 session. Use
  `experiments/amazon_beauty_field_ae_rqkmeans_tiger_2t4.json`;see
  `docs/gr-field-ae-rqkmeans-tiger.md`. Use byte-pinned original3500-step AE
  catalogue latent arrays from the completed warm-up kernel;no AE/encoder/
  decoder/embedding update or re-export. Fit sequential full-catalogue sklearn
  Lloyd KMeans3x256 on raw64D latents,CPU thread1,n_init1,max_iter300,tol1e-4,
  random_state=AEseed+level,without latent/residual normalization. Then freeze
  the resulting SID index and train only TIGER. Preserve the completed
  AE-freeze generator control's4+4/d128,FP16,20epochs,all131413prefix cases,
  validation10/20,NDCG@10selection and final FP32free-beam20test. Generator/
  data seed42 is shared;match initial neural and case hashes against completed
  control `carasik/rhpr-field-warmup-tiger` source10b53b8. Copy completed
  controls rather than rerunning them. No3k-budget comparison,price ablation,
  automatic tuning/retries or Telegram. Save books,codes,fit diagnostics,
  source AE identity,generator checkpoints/curves/predictions and run metadata.
- Latest user authorization,2026-10-26:compare the completed entire-AE-freeze
  and joint-refinement semantic indices using the same TIGER. Use
  `experiments/amazon_beauty_field_warmup_tiger_2t4.json`;see
  `docs/gr-field-warmup-tiger.md`. One private two-T4 session,four generator
  fits on saved indices ae/joint x quantizer seeds42/43,with generator/data
  seed42 fixed across all four. Byte-pin the completed warm-up parent's source,
  manifest,catalogue order,semantic-code arrays and reports. No encoder,
  quantizer,book initialization or embedding reruns. Preserve early4+4/d128,
  FP16,all131413train-prefix positions per epoch,20epochs,validation10/20,
  NDCG@10 selection and one final full-catalogue free-beam20test per model.
  Keep ordinal suffix/invalid beam accounting unchanged. Check initial neural
  parameter hash against shared CPU control before training;immutable SID
  lookup data intentionally differ. Save best/last,curves,predictions and
  training/index provenance;no study telemetry or automatic retries/tuning.
  Two index seeds are not independent generator-seed repeats. Earlier native
  GPU initializations did not produce bitwise matched third books;this tests
  the saved candidate indices,not a perfectly isolated causal freeze ablation.
- Completed warm-up suite:all six fits successful in12.56minutes. At3x256/
  latent64,full AE freeze produced12004/12018distinct tuples,max group3;
  joint produced11542/11258,max15/17;input-feature freeze11674/11296,max8/14.
  Earlier cold fixed-normalized64D control produced188,max907. Warm-ups both
  stopped3500steps;PCA99%dimension5,nominal24bit active count8. Joint improves
  brand/category/title reconstruction relative to full freeze. Warm-up adds
  compute;this does not isolate codebook lag or establish better NIP yet.
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
