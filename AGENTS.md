# RHPR agent instructions

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
