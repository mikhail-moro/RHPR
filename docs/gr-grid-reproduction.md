# GRID Beauty reproduction target

Decision, 2026-10-22: reproduce an identifiable published baseline before new
representation experiments. The completed Sentence-T5 / RK-Means hybrid is an
infrastructure check, not the reproduction target. The planned hybrid follow-up
and GPU profile were not submitted.

## Published reference

Target **GRID Beauty, Flan-T5-XL + RK-Means 3×256 + 4+4 encoder-decoder**.
[GRID Tables 1 and 3](https://arxiv.org/html/2507.22224v1#S4) report
Recall@10 **0.0639** and NDCG@10 **0.0347**, averaged over five seeds. A single
new seed cannot establish reproduction of that mean. First establish the recipe
and one working run; subsequent seed comparisons must not select by test.

Keep quantizer width 256 for this reference. Width 128, smaller generators and
alternative representations are later ablations.

## Alignment required

| Stage | Completed hybrid | Required reference |
| --- | --- | --- |
| Item encoder | Sentence-T5-base, 768D, Dense + normalization | Flan-T5-XL encoder, 2048D, masked mean |
| Text length | 256 tokens | Official embedding configuration: 128 tokens |
| Tokenizer | GRID normalized-residual RK-Means, 3×256, ten catalogue passes per level | Same algorithm; align the published clustering budget, not only codebook dimensions |
| Generator | 4+4, d128, ordinary FF, dropout .1, user bins 2000, no separator | Preserve 4+4 and align remaining settings with the declared GRID source recipe |
| Training examples | Full causal prefixes, once per finite epoch | Author contiguous-subsequence augmentation from full training rows |
| Optimizer | Adam LR 1e-3 / WD 1e-4 | Paper: LR 5e-4 / WD 1e-6 / batch 256 |
| Selection | NDCG@10 every five epochs | Resolve paper/code metric and cadence differences explicitly |

Use author components from
[`2fe3475b2d369580234093f35d52b1a2f54d0472`](https://github.com/snap-research/GRID/tree/2fe3475b2d369580234093f35d52b1a2f54d0472).
The public generator YAML has `mlp_layers=2`, dropout .15, no user bins and a
separator enabled by the constructor default. Its collator enumerates contiguous
subsequences of full training histories, samples among them, and then truncates
to 120 SID codes including the target. Do not replace this with the current
finite-prefix protocol while claiming the same training recipe.

## Embedding assets

Use the existing lossless author text and item order: `catalog.json` SHA256
`0336747ab3d50741fde2bb17f734b9e393c418bc54f818ab714977bacd2ae767`.
No new metadata serialization or item filtering is needed.

The inspected author data archive contains only undocumented 768D vectors, not
an identifiable Flan-T5-XL embedding artifact. Regenerate the 12,101×2048 matrix
with the official
[embedding configuration](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/sem_embeds_inference_flat.yaml):

- `google/flan-t5-xl`, encoder only; selected immutable model revision
  `7d6315df2c2fb742f0f5b556879d730926ca9001`.
- Maximum length 128, right padding, truncation and special tokens enabled.
- Mean of non-padding encoder states, including EOS; no Dense head or L2
  normalization. Official inference uses FP32 and batch 8.
- Row index equals original item ID. Save encoder revision, text/tokenizer
  settings and embedding hashes independently from quantization.

This revision is a reproducible pin selected now; the authors did not publish
a checkpoint revision for their individual runs. Do not assert that it proves
the exact historical weight bytes.

## Resolve before submitting compute

Paper and released example YAML are not one identical configuration. The YAML
uses different LR/weight decay, monitors Recall@5, and accumulates variable
augmented batches; the paper specifies batch 256, early stopping by NDCG@10 and
checkpoint selection by Recall@10. Its clustering budget is also larger than
the example YAML. Record which source governs each setting, rather than calling
an unmodified YAML an exact paper recipe.

The paper specifies 1,000 quantizer steps per level on eight GPUs with 2,048
items per device; the example YAML has 30 total steps. A single-GPU substitute
must account for sampling and running cluster counts before claiming equivalent
updates. The generator YAML takes 32 source users, expands to at most 512
windows and accumulates 16 such batches; this is not the paper's batch 256.
The paper also leaves exact seeds and Table 1 beam width unspecified. Source
defaults, such as free-form beam 10, are declared fill-ins rather than proof of
the exact historical experiment.

The user preference for validation every ten epochs remains recorded. A faithful
step-based reference and the unbounded author training stream cannot silently
inherit that setting: first define the update/exposure budget and selection
cadence. Runtime optimizations must preserve the declared effective batch,
targets, loss and selection behavior, or be recorded as deviations.

The next authorized step is a bounded runtime profile of the aligned recipe;
see the [paper-priority execution contract](gr-grid-paper-recipe.md). It loads
the actual Flan-T5-XL encoder and measures a fresh aligned generator with
explicitly labelled runtime-only SID assignments. It reports no quality score.
The older saved-checkpoint profiler remains separate from this configuration.
