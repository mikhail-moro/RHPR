# RHPR agent instructions

## Research status

- The history-compression / Query SSM track was closed on 2026-10-21. Read
  `docs/research/closed-history-compression/README.md` for its final evidence.
- Earlier experiment plans are historical and have no pending GPU launches.
- The active direction is comparing item encoders for SID-based generative
  recommendation. Read `docs/gr-research-brief.md`, `docs/gr-baseline.md`, and
  `docs/research/item-encoders/README.md`.
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
