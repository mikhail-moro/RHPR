# SID index study: paused research record

Paused by the user on 2026-10-25. This is a preservation checkpoint, not a
declaration that SID research is exhausted or that a mechanism was disproved.
The next direction is [item encoders](../item-encoders/README.md).

## Research question and design

We sought observable properties of a Semantic-ID index that explain downstream
generative recommendation quality, including item-level differences hidden by
global averages. The controlled reference was Amazon 2014 Beauty, frozen
Sentence-T5-base embeddings, normalized-residual RQ-KMeans 3×256 plus an ordinal
suffix, GenPAS-derived Transformer 3 encoder + 3 decoder layers, d128, FP16,
Adam5e-4, 20 epochs and validation every five epochs. This is a TIGER-style
hybrid, not a reproduced published TIGER/GRID score.

There are 12,101 catalogue items and 22,363 users. Train is L[:-2], validation
target L[-2], test target L[-1]; evaluation contexts contain up to 20 items.
Each user has one target per split. Histories are deduplicated user-item data,
with no real timestamps. Repeated listening/purchase claims cannot be studied
with this prepared dataset. Training uses the saved TIGER/GRID window manifests,
not one evaluation target per user.

Original experiment: quantizer seeds 11/22/33/44/55/66, common generator seed42
and data seed4242; identical initialization and 5,826,688 sampled examples per
model. Follow-up: frozen original indices11/22 crossed with generator seeds
101/202/303, same data manifests, configuration and 20-epoch budget. Generators
are not proven converged. See `gr-sid-index-study.md` and
`gr-sid-training-variance.md` two directories above.

## What survived the investigation

- Index-dependent item differences partly persist across generator seeds,
  but the overall winning index changed. Index22−11 test NDCG@10 was +0.001720
  at seed42 and −0.001323/−0.000616/−0.001175 at the three new seeds.
- Early decisions dominate observed beam10 misses. Loss of the true prefix is
  a location of error, not by itself proof of a beam-search defect. Exact
  ranking and search errors must remain separate.
- On original validation, local conditional NLL by semantic level is about
  4.643/3.253/0.692 versus train-frequency control 5.461/3.309/0.429.
  Late frequency substitution improved some likelihoods but did not establish
  a held-out NDCG gain.
- The model uses history. On the weighted 512-case validation panel,
  full-history shuffling reduced exact raw NDCG@10 from 0.03956 to 0.03430.
  Zero encoder memory is an OOD sensitivity probe, not a trained baseline.
- About 50.4% of validation model-cases have a singleton first-two-code pair;
  low supervision is widespread. An independent quality benefit of increasing
  pair size at comparable support was not established.
- Semantic token associations and extreme cases are inspectable, but no
  general causal rule for semantic conflicts, purity, code depth or density
  was established. Catalogue associations are not measured neural reliance.
- Repeated rare-branch switching tracks recent training examples, but its
  mechanism is unresolved and the canonical example is a small fraction of
  evaluation. More epochs are not a demonstrated universal fix.
- SHANY primarily exposed missing brand metadata. Wrong top1 item categories
  also need not mean wrong marginal category decisions. Neither observation
  proves Sentence-T5 itself failed.

The detailed [results ledger](results-ledger.md) records each analysis and its
limits. All stages, including inconclusive and superseded interpretations,
are retained in the archive.

## Archive and resumption

Workspace entry point: `/workspace/artifacts/sid-research-archive-2026-10-25/`.
The preservation package has one companion ZIP of analyses/source/notes plus
the full original and follow-up output ZIPs. Its manifest gives exact file
coverage and integrity checks. The companion ZIP alone is not the raw dataset.
See the archive README for actual completion status; never infer completeness
from this planning description.

The source revision for the first run is
`e955381a2f0f24416085ab29ae11d0ae77bdf723`; for the follow-up it is
`cbe4965f3f3d75a245da45cb6d89f9679a3acdda`. The archive includes source snapshots,
analysis scripts, manifests, configuration, item metadata, full raw outputs,
checkpoints, exact-panel catalogue scores and derived semantic atlas, subject
to the explicit coverage report. Existing paused TPU edits are preserved
separately; there is no queued TPU/GPU work.

To resume: read the archive's context and ledger, verify its checksums, unpack
the required data, and inspect the named scripts' original path assumptions.
Do not reuse exploratory validation A/B subsets as a fresh blind holdout.
Keep raw/local scoring separate, and model-case repetitions distinct from
independent users or training seeds. A new causal experiment must isolate a
factor; this archive is not authorization to dispatch it.
