# Next direction: item representations for SID

Update2026-10-26:the user chose a joint structured-field encoder and RQ-VAE,
reconstructing original brand/categories/price with CE/BCE/Huber and LibMTL
uncertainty weighting. Title is excluded. See
[the implementation protocol](../../gr-field-rqvae.md). The fast generator,
prefix sampling and evaluation remain controlled; the metadata encoder and
quantizer are trained jointly, with next-item training still a separate stage.

User decision, 2026-10-25: pause the index-diagnostics branch and return to
comparing item encoders. The user subsequently authorized the
[Sentence-T5 / RQ-VAE / TIGER baseline](../../gr-rqvae-baseline.md), retaining
the old GRID4+4/d128 generator with FP16,20epochs and validation every10.
The user subsequently chose the faster early prefix protocol, since sampling
is not the research axis. The active config is
`experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json`:
every train target once per epoch, latest20context, early ordinary-FF4+4/d128,
noSEP,user bins2000,FP16,20epochs and validation only at10and20.
The RQ-VAE uses the native GRID3x256 recipe and3000updates; this first run
does not jointly train the text encoder. No encoder sweep is queued.
See the original [research brief](../../gr-research-brief.md)
and the [paused SID record](../paused-sid-index/README.md).

The question is whether representation architecture, available metadata and
representation objective can provide better downstream Semantic IDs at a
matched budget, and whether simpler representations suffice. Objective mismatch
is a testable motivation, not proof that pretrained text embeddings fail.

The SHANY case is a useful warning about missing source brand fields: most
apparently wrong brand recommendations still contain SHANY in their titles.
It does not isolate a Sentence-T5 embedding error. Compare actual embedding
neighborhoods or change representations under control before making that claim.

## What can be reused

- Prepared Amazon2014 Beauty item order, split and train-window manifests.
- Item metadata and actual text passed to the frozen encoder.
- Item-ordered embedding/tokenizer/generator boundaries in the existing code.
- RQ-KMeans, injective SID suffix handling, full-catalogue item metrics.
- Teacher-forced diagnostics, exact-score panel, per-item support and the
  semantic atlas. Use semantic diagnostics as explanations, not surrogate
  optimization targets already known to improve retrieval.
- Generator-seed evidence: one index/model seed can be misleading, so record
  quantizer variation separately from model variation.

## Controls for the first future comparison

Decide the minimal methods later. Keep data, metadata access, item order,
normalization/dimension policy, quantizer budget, generator, actual sampled
examples, evaluation and validation selection explicit. Architecture-only and
objective-only contrasts are different experiments; do not silently vary both.
Learned non-text encoders need a stated training objective. Collaborative
signals must come only from train. Include preparation/tokenizer cost as well
as final generator training and serving cost.

Do not automatically improve/impute metadata in one arm only. Separate raw
missingness, text that contains the same information, feature encoding, encoder
architecture and task-aware representation learning. Metadata corrections can
be a distinct ablation shared across arms.

The paused study's many validation analyses mean its A/B subdivisions are not
fresh blind evaluations. New confirmatory claims need a declared selection
and held-out assessment. Do not assume20epochs are sufficient for every arm,
but do not start a large hyperparameter sweep before a bounded comparison.

This note preserves intent and constraints; it is not an approved experiment
specification or launch request.
