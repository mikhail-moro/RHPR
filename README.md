# RHPR experiment runner

Reproducible Codex -> GitHub -> Kaggle experiment loop with Telegram-only
notifications.

The history-compression / Query SSM research track was closed on **2026-10-21**.
Its code, experiment configurations and tests are retained for reproducibility.
Start with the [research archive](docs/research/closed-history-compression/README.md)
for final results, data audits, literature and limitations, or the
[documentation index](docs/README.md) for the earlier stages.

The active research direction is **item representations for SID-based generative
recommendation (GR)**. Start with the [research brief](docs/gr-research-brief.md),
[framework audit](docs/gr-framework-audit.md), and
[Amazon Beauty baseline](docs/gr-baseline.md). Historical compression plans
remain closed.

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
docs/research/                closed research reports and aggregate evidence
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

Kaggle's current MCP endpoint can use a KGAT token instead of OAuth. Store it in
the user environment, never in this repository or an MCP JSON file. When Codex
Desktop runs the project in WSL, put the token in the Windows user environment
and explicitly pass that variable through `WSLENV`. Run this in PowerShell and
then fully exit Codex, run `wsl --shutdown`, and reopen Codex:

```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_your_token", "User")
$wslEnv = [Environment]::GetEnvironmentVariable("WSLENV", "User")
$entries = @($wslEnv -split ":" | Where-Object { $_ })
if ($entries -notcontains "KAGGLE_API_TOKEN") { $entries += "KAGGLE_API_TOKEN" }
[Environment]::SetEnvironmentVariable("WSLENV", ($entries -join ":"), "User")
wsl --shutdown
```

Register the endpoint from the repository root:

```bash
codex plugin marketplace add .
codex plugin add rhpr-ops@rhpr
codex mcp add --url https://www.kaggle.com/mcp \
  --bearer-token-env-var KAGGLE_API_TOKEN kaggle
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

The GitHub remote MCP uses OAuth. The CI workflow still needs the Kaggle secret
because an unattended GitHub runner cannot reuse the desktop environment
variable.

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

## Historical Yambda-50M SASRec baseline

This section describes the initial baseline. The later 500M long-history,
daily-evaluation and rotating-target protocols are documented in the
[research archive](docs/research/closed-history-compression/README.md).

The `yambda_sasrec_*.json` configurations train a PyTorch SASRec whose item
embedding is followed by an explicit sequence-compressor interface and then the
causal Transformer. The baseline compressor is the identity operation. A future
compressor returns the shortened embeddings, their validity mask, and the
source end-position represented by every compressed token; this keeps shifted
next-item targets aligned without constraining the compressor internals.

The data contract is deliberately narrow:

- only Yambda-50M `listens.parquet` is used; both official sequential and flat
  schemas are accepted;
- every listen is positive, repeated items remain, and users are ordered by
  ascending timestamp;
- each user is split `n-2 / 1 / 1`; training uses shifted targets from the last
  `max_seq_len + 1` train events;
- validation runs after every epoch and early stopping monitors full-catalogue
  NDCG@10; the test target is evaluated once from the best validation epoch;
- seen items are not removed because a repeated listen is a valid target.

The four initial loss configurations are BCE with one uniform negative and
sampled softmax with 128, 512, or 1028 catalogue-uniform negatives. Full
softmax is intentionally absent. They attach the official `ploshkin/yambda`
Kaggle mirror and read only
`/kaggle/input/yambda/sequential/50m/listens.parquet`.

Run `yambda_sasrec_bce_calibration.json` first: it processes the full data for
one epoch, reports separate training/validation durations, and deliberately
does not inspect the test target. Use that measurement to decide whether the
four loss runs fit the remaining GPU budget.

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
