---
name: github-ops
description: Manage RHPR branches, pull requests, GitHub Actions experiment dispatches, run status, and artifacts. Use for any GitHub change or CI operation in this repository.
---

# GitHub Ops

Use the connected GitHub tools first. Fall back to `gh` only when the connector
does not expose the operation.

1. Inspect the current branch and existing user changes before editing.
2. Make a focused branch/commit and run the repository tests.
3. Prefer a pull request; never force-push or rewrite shared history.
4. Before dispatching `kaggle-experiment.yml`, confirm the prompt explicitly
   requests an actual Kaggle run. A request to edit or review code is not
   permission to spend quota.
5. Dispatch against the exact pushed branch/ref and record the workflow URL.
6. Read the job summary and artifact metrics before reporting success.

Do not request a GitHub PAT for Actions: the workflow receives `GITHUB_TOKEN`
automatically. Read [the runbook](references/runbook.md) for commands and state
transitions when GitHub MCP cannot perform a step.
