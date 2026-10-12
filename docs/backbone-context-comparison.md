# Matched SASRec and HSTU context screen

This experiment asks whether the recommendation backbone changes the benefit
of retaining the old history. It trains four independent models: SASRec128,
SASRec1028, HSTU128 and HSTU1028. The two context budgets for each architecture
run sequentially in one private Kaggle kernel. HSTU1028 runs first to expose the
largest memory case before spending compute on HSTU128.

Configurations:

- `experiments/yambda_backbone_sasrec_15ep.json`
- `experiments/yambda_backbone_hstu_15ep.json`

Both use `kind: yambda_backbone_context` and the same common recipe. Each pair
has a 14,400-second deadline. The wrapper loads and maps the dataset once, then
starts each model from a fresh seed42 initialization. Checkpoints are separate
under `runs/context_128/` and `runs/context_1028/`; no weights transfer between
contexts or architectures.

## Shared protocol

- Yambda50m chronological leave-two-out split, with the existing mapped item
  catalogue and user set. No catalogue-frequency filtering is introduced.
- Raw input-window limit1028 for all models. The128 model removes the old
  prefix inside a parameter-free tail-only compressor. The1028 model keeps
  the full raw window with an identity compressor. Neither creates summaries.
- Original recency positions are retained after cropping. Events outside the
  final raw1028 window are not visible to any model in this screen.
- Parallel shifted next-item training, with loss only on the last128 raw
  source positions. This preserves the same supervised targets across contexts;
  the attention context available to those positions differs.
- Sampled softmax with512 catalogue negatives, Time OFF, seed42, physical
  batch8, effective batch128,15 full epochs, and the same optimizer recipe.
- Full-catalog NDCG@10 and Recall@10 at validation epochs5/10/15. Each model
  selects its own best checkpoint by its own validation NDCG@10. Patience16
  prevents early stopping during this screen. Test is reserved.
- The existing benchmark measures B1 history encoding and complete catalogue
  scoring, B32 throughput, GPU memory and implemented-operation FLOPs.
  Warmup30,200 B1 timing iterations,20 throughput iterations and B32 are
  retained. OOM fallback is disabled; batch sizes cannot silently change.

HSTU uses the official native PyTorch attention implementation integrated by
`rhpr.sasrec.hstu`. Its implementation provenance and selected architecture
appear in saved model/benchmark metadata. The common screen uses the SASRec
starting recipe without a separate HSTU hyperparameter search. Equal epochs and
batch sizes do not imply equal parameter counts or training FLOPs; these are
reported rather than assumed.

The [vendor notes](../src/rhpr/vendor/hstu/README.md) document the exact upstream
pin, PyTorch packing shim, position-only bias, and fixed attention divisor1028.
The native block matches the unmodified official reference in forward/backward
tests; its input, objective and retrieval head follow this repository's recipe.

`experiments/yambda_catalog_profile.json` is a separate CPU-only diagnostic.
It ranks items by frequency in the chronological training split and estimates
the catalogue needed to cover99% of its events, then checks coverage of the
original validation targets and old-prefix replay group. It does not filter
items, alter histories, move held-out targets, or inspect test targets. A later
filtered-catalogue screen would need the same filter for all compared models;
its absolute metrics would not be directly comparable with full-catalogue runs.

## Interpretation

This is a fixed-window, single-seed15epoch screen. It does not establish either
architecture's converged optimum or reproduce its paper's independently tuned
training system. HSTU-specific hyperparameter tuning and random training-window
sampling are separate possible experiments.

The primary within-architecture contrast is the paired change
`long1028_minus_short128`. A useful cross-architecture contrast is
`(HSTU1028-HSTU128)-(SASRec1028-SASRec128)`. A positive difference indicates that
the longer context helps the tested HSTU recipe more than the tested SASRec
recipe. It does not by itself prove why the benefit differs.

Every prediction preserves the full raw context hash and held-out target. A
separate input hash and input length identify the actual retained context. Raw
history features use only item IDs and observed lengths, so they are common
across independently trained embedding tables. No learned cosine quartiles are
used to compare these models.

Within each pair, diagnostics report all users, old-prefix-present users,
length groups, old-item-uniqueness quartiles and targets seen only in the old
prefix. That last group depends on the held-out target and is an offline
diagnostic, not an input to a production router. Bootstrap95% intervals resample
paired validation users and condition on the selected checkpoints and seed.
They omit training-seed and checkpoint-selection uncertainty, and multiple
cohort comparisons remain exploratory.

After both kernels finish, compute the four-model interaction without retraining:

```bash
PYTHONPATH=src python -m rhpr.sasrec.backbone_interaction \
  --sasrec /path/to/sasrec-output \
  --hstu /path/to/hstu-output \
  --output /path/to/interaction-output
```

## Artifacts and completion

Each context directory contains its exact training/model configuration, history,
selected weights and SHA256, scalar metrics, validation predictions, immutable
raw-history features, diagnostic lineage and benchmark artifacts. The pair root
contains:

- `metrics.json`: finite scalar metrics with a prefix for each context.
- `run-manifest.json`, `summary.md`, `comparison.md`: status and recipe.
- `history-features.json/parquet`, `predictions.json/parquet`: strict common
  identities and independently selected model outcomes.
- `paired-predictions.parquet`, `cohort-results.parquet`,
  `backbone-cohorts.json`: per-user effects and paired confidence intervals.
- `backbone-results.json`: context/checkpoint selection summary.

`backbone_context_complete` remains false until both models finish their full
epoch budget, evaluate all validation users, verify their checkpoint checksums,
pass the identity join and write final diagnostics. Partial results are labelled
incomplete.
