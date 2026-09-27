---
name: kaggle-experiments
description: Build, launch, monitor, cancel, and analyze RHPR experiments on Kaggle. Use whenever work touches Kaggle data sources, kernels, quotas, outputs, or metrics.
---

# Kaggle Experiments

Use the official Kaggle MCP server for live notebook status, quota, metadata,
outputs, datasets, and interactive control. Use the GitHub Actions workflow for
reproducible execution of repository code.

Before a run:

1. Confirm explicit authorization to spend Kaggle quota.
2. Validate the chosen JSON under `experiments/` locally.
3. Ensure data/competition/kernel/model sources are declared in its `kaggle`
   object and the kernel remains private.
4. Inspect quota; if the requested accelerator is unavailable, report that and
   do not silently switch hardware.

During a run, monitor the existing kernel instead of starting duplicates. On
completion, retrieve `metrics.json`, `summary.md`, and `run-manifest.json`.
Treat a missing or invalid metrics file as a failed experiment even if Kaggle
reports `complete`.

Use [the experiment contract](references/experiment-contract.md) when creating
or reviewing configurations.
