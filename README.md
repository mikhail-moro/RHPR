# RHPR experiment runner

Reproducible Codex -> GitHub -> Kaggle experiment loop with Telegram-only
notifications.

## Flow

1. Ask Codex to implement an experiment from a chat or a written brief.
2. Codex changes code and opens or updates a GitHub branch/PR.
3. After an explicit request to spend Kaggle quota, Codex dispatches
   `Kaggle experiment` for that branch.
4. GitHub Actions builds a self-contained Kaggle kernel, pushes it, waits for
   completion, downloads outputs, and publishes them as a workflow artifact.
5. The workflow writes metrics to the GitHub job summary and sends a compact
   success/failure notification to Telegram.

The GitHub Actions workflow is the deterministic control plane. Kaggle MCP is
also bundled for interactive inspection and control; it is not the only path
used to publish notebooks because CI needs repeatable, auditable execution.

## Repository layout

```text
experiments/                  versioned experiment configurations
kaggle/run.py                 Kaggle entry point
src/rhpr/                     runner, metric contract, Telegram client
scripts/                      bundle, submit, collect, notify utilities
.github/workflows/            validation and Kaggle execution
plugins/rhpr-ops/             repo-local Codex MCP + skills plugin
.agents/plugins/              repo-local plugin marketplace
```

Every experiment must produce `metrics.json` and `summary.md`. The included
`experiments/baseline.json` is a deterministic, dependency-free smoke test;
replace its implementation with the project model while preserving the output
contract.

## One-time setup

### GitHub repository secrets

Add these under **Settings -> Secrets and variables -> Actions -> Secrets**:

| Name | Required | Purpose |
| --- | --- | --- |
| `KAGGLE_API_TOKEN` | yes | Kaggle API token (KGAT), used only by Actions |
| `TELEGRAM_BOT_TOKEN` | yes | token from `@BotFather` |
| `TELEGRAM_CHAT_ID` | yes | destination user/group/channel ID |
| `TELEGRAM_THREAD_ID` | no | topic ID for a forum-enabled group |

Add this under **Actions -> Variables** (it is not a secret):

| Name | Example | Purpose |
| --- | --- | --- |
| `KAGGLE_KERNEL_ID` | `mikhailmoro/rhpr-experiments` | stable private kernel slug |

`GITHUB_TOKEN` is created automatically by GitHub Actions. Do not add a PAT for
the workflow. If Telegram targets a channel, add the bot as a channel admin.
To discover a chat ID, message the bot once and inspect `getUpdates` locally;
do not paste the bot token into an issue, PR, chat, or committed file.

### Codex plugin and MCP

From the repository root:

```bash
codex plugin marketplace add .
codex plugin add rhpr-ops@rhpr
codex mcp login kaggle
codex mcp login github
```

Restart Codex or start a new task after installation. GitHub may already be
available through the built-in GitHub plugin; if so, keep one GitHub connection
enabled to avoid duplicate tools.

For local Telegram MCP calls, export (do not commit) the same variables:

```bash
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
# export TELEGRAM_THREAD_ID='...'
```

Kaggle and GitHub remote MCP use OAuth. The CI workflow still needs the Kaggle
secret because an unattended GitHub runner cannot reuse your interactive OAuth
session.

### Codex cloud environment

Set the repository to `mikhail-moro/RHPR` and use this setup command:

```bash
bash scripts/codex_setup.sh
```

No deploy credential is required in the Codex cloud environment. Cloud secrets
are setup-only and are intentionally not part of the runtime design; GitHub
Actions owns the Kaggle and Telegram credentials.

## Local development

```bash
bash scripts/codex_setup.sh
python3 -m unittest discover -s tests -v
python3 scripts/run_local.py --experiment experiments/baseline.json
```

Build the exact directory uploaded to Kaggle:

```bash
python3 scripts/build_kaggle_bundle.py \
  --experiment experiments/baseline.json \
  --kernel-id mikhailmoro/rhpr-experiments \
  --accelerator cpu \
  --output build/kaggle
```

## Running an experiment

In GitHub, open **Actions -> Kaggle experiment -> Run workflow**, select the
branch, experiment JSON, accelerator, and timeout. An agent can dispatch the
same workflow through the GitHub connector or `gh workflow run`, but should do
so only when the prompt clearly authorizes a Kaggle run.

Outputs are retained as the `kaggle-results-*` workflow artifact. The Telegram
message contains the result and a link to the workflow; full logs and metrics
stay in GitHub.

## References

- [Kaggle MCP documentation](https://www.kaggle.com/docs/mcp)
- [Official Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
- [Official GitHub MCP server](https://github.com/github/github-mcp-server)
- [OpenAI plugin packaging](https://developers.openai.com/plugins/build/plugins)
