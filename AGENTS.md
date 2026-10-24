# RHPR agent instructions

## Research status

- The history-compression / Query SSM track was closed on 2026-10-21. Read
  `docs/research/closed-history-compression/README.md` for its final evidence.
- Earlier experiment plans are historical and have no pending GPU launches.
- The active direction is item representations for SID-based generative
  recommendation. Read `docs/gr-research-brief.md` and `docs/gr-baseline.md`.
- The user cancelled the TPU launch and paused reproduction work. The active
  study (2026-10-24) investigates SID index structure using six RQ-KMeans seeds,
  frozen Sentence-T5 embeddings, identical generator initialization and shared
  TIGER/GRID window manifests. Two T4s in one session, one independent seed
  worker per GPU (three seeds each), FP16, 3+3 layers at d128, 20 epochs,
  validation every5, no early stopping, final-epoch comparison. Read
  `docs/gr-sid-index-study.md`. No UI: Parquet evidence with optional DuckDB.
- Raw autoregressive scores and valid-child locally renormalized scores are
  separate objectives. Compare exact search and beams under MATCHING scoring.
  Do not interpret six index seeds as six independent generator seeds.
- The authorized follow-up crosses frozen original rq11/rq22 indices with
  generator seeds101/202/303, preserving data seed4242 and20epochs. See
  `docs/gr-sid-training-variance.md`. Fixed train/validation probes preserve
  RNG/weights; their Parquet sidecars are distinct from checkpoint evaluations.
  Initialization hashes must match WITHIN each model seed across indices.
- The user explicitly wants GPU, not TPU; implementation does not authorize
  dispatch. Preserve the paused TPU changes; do not restart previous kernels.
- Historical reproduction work is in `docs/gr-genpas-tiger-epochs.md`,
  `docs/gr-genpas-tpu.md`, `docs/gr-grid-quality.md` and
  `docs/gr-grid-reproduction.md`. Do not silently revive those protocols.
- Do not implement a new tokenizer or assert a representation bottleneck before
  the baseline and diagnostics support that claim.

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
