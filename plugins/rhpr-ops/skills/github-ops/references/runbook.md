# GitHub runbook

## Validation and PR

```bash
python3 -m unittest discover -s tests -v
git status --short
gh pr create --fill
```

Do not include `.env`, Kaggle outputs, downloaded datasets, or credentials in a
commit. Check workflow permissions and untrusted input boundaries when editing
files under `.github/workflows/`.

## Dispatch an authorized run

```bash
gh workflow run kaggle-experiment.yml \
  --ref <pushed-branch> \
  -f experiment=experiments/<name>.json \
  -f accelerator=cpu \
  -f timeout_minutes=360
```

Then obtain the run URL with `gh run list --workflow kaggle-experiment.yml` and
watch it with `gh run watch <run-id>`. Prefer the GitHub MCP equivalents when
available. Never dispatch a second run merely because the first is queued.
