# ML1M clean contiguous holdouts: one-seed MTL comparison

Authorized2026-11-05: four fresh Sum/UW/GradNorm/PCGrad pipelines, seed42,
30epochs, validation every5epochs, FP16 generator, native SDPA, effective and
physical batch512/512. Private kernel `carasik/rhpr-ml1m-clean-mtl`; config
`experiments/ml1m_clean_mtl_2t4.json`. Two T4 workers, two fits each; no other
seeds, tuning, retries, hardware fallback, Telegram or extra experiments.

## Data and objective

Fixed2874film catalogue with all metadata/poster/description source coverage.
2786users,151166TRAINcases,2786VALand2786TESTcases. One literal uninterrupted
20raw-context+val+test ending per user, latest qualifying original timeline
ending. Targets are singleton timestamps BEFORE catalogue filtering. Every
partially touched context timestamp group must be fully covered. Any unknown
item remains a barrier; contexts are never spliced through omitted events.

Train on all earlier admitted raw up-to20windows, strictly before chosen val;
short contexts only at natural original-history starts with wholly covered
past. At least one training case per user. Within a covered tied TRAIN group,
all item targets share the strictly earlier context; no intra-second causal
order is invented. Val/test are adjacent raw events: TESTcontext equals VAL's
last19context events followed by the observed val item. This is enforced by
array assertions. All rating values retained; times are rating-submission times,
not observed viewing order. Static catalogue features may include heldout items
but consume no user interactions.

Input preparation was independently replayed byte-for-byte, matching the prior
per-user CPU audit endpoints AND all151166training cases. Cases are immutable
NPZ arrays in catalogue-index order; input manifest SHA256
`023d2b52dd8a12553d1dcf24a8a3e9a59f22983ad3cd0b768c7da6318aa98c16`.
TR/VAL/TESTcase SHA256 in config and preflight, checked before model work.
One shared schedule/cohort/catalogue for every method.

## Frozen channels and tokenizer

- Metadata Sentence-T5-base, pinned revision
  `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`, same title WITH year, genres,
  explicit year, runtime and TMDBkeywords. No people names. Max2048tokens;
  actual max298,p95=103. Tokenizer's actual longest-input mask must match raw
  token count. No truncation. Shared encode once, batch32, FP32.
- Poster: keyed768D Zenodo15403972 `json/vit_cls.json`, same source as ML100K.
- Description ONLY: pinned `all-mpnet-base-v2`, revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`, max384tokens. Reuse existing
  tested token-count weighted pooling of nonoverlapping chunks, then L2.
  No genres/tags/reviews mixed into plot, and no long-description truncation.

All2874feature vectors prepared once in the legacy4.47encoder runtime on GPU0
and shared byte-identically across methods. Output features/provenance/hashes
are saved. Modern5.22.1 runtime installs offline into a SEPARATE site and is
used only by fresh workers; Torch2.11/KaggleCUDA remain unchanged.

Preserve historical tokenizer capacity: four head slots with head_dim128,
head_hidden256, shared input512, latent64 and three256codebooks. The historical
INACTIVEpeople128D slot stays zero and unobserved; it never contributes to
encoder input values or any reconstruction/MTL loss. New optional
`zero_inactive_slots=['people']` skips undefined variance normalization for this
unused constant slot. Other configurations retain the old normalization path.
The full quantizer neural initialization pin remains exactly the old seed42
`54253fbb9f6837546a0c29291db31a12cb0be65f75609a6d8f7c3d89aeccc69b`.
No graph data, extra active channel, or tokenizer capacity change is introduced.

Each method independently trains AE3500fixed steps → native RQ initialization
buffer2874 → RQ-VAE3000steps. Quantizer batch2048, so the new catalogue spans
TWO batches; this path was separately tested. AE/RQ use FP32 and the same native
GRID Adagrad recipe as earlier MTL experiments; no precision ablation or extra
hyperparameter tuning. MTL applies in both AE and reconstruction phases; fixed
quantization/commitment weights stay unchanged. Sum/UW/GradNorm/PCGrad equations,
phase reset and diagnostics follow existing implementations. UW also changes
global reconstruction scale; results cannot be attributed solely to balancing.

## Fresh generator and evaluation

Same TIGER4encoder+4decoder,d_model128,d_ff1024,6heads,d_kv64,dropout.1,
2000remainder-hashed user bins,3SIDlevels+uniqueness suffix,no separator,
freebeam20. Native Transformers5.22.1SDPA with T5scale1.0; inherited tested
cache-constructor/T5Stack compatibility, no custom attention implementation.
This is a NEW dataset/protocol: no previously trained generator weights loaded.
Full modern seed42initializer pin
`d57fc2e783c0a04bfc567c0082541f757e241d1c53d7e79b3e8f281bf2eb9337`.
All four share initial predictor and tokenizer weights and example order.

AdamLR.001,wd.0001,clip1,GradScaler1024,FP16,physical/effective512. No LR scaling
or scheduler. Exactly30full epochs, validation5/10/15/20/25/30; min_epochs30
prevents early stopping.296scheduled updates per epoch,8880over30epochs before
any recorded AMP skips. Save curves, quantizer/MTL diagnostics, checkpoint and
user-level predictions at every validation. Choose checkpoint by VALNDCG@10;
one final TESTevaluation. EvaluationFP32, full catalogue decoding, no seen mask,
no wider beam or training/test objective changes. Save actual epochs/update
counts and skipped updates; incomplete30epoch/6validation fit is a failure.

Historical user hash behavior is retained. ML1M user IDs hash to2000bins and
can collide; model comparisons are within this common protocol, not direct
score comparisons to ML100K. Larger batch halves updates per example versus
old effective256; user accepted this change and all four methods share it.

## Resources and verification

Existing private inputs: phase0GRID/runtime/Sentence-T5 assets, original
ML100Kmodal input for the pinned MPNet assets ONLY, public-package SDPA runtime.
New private input `carasik/rhpr-ml1m-clean-mtl-input` is12.9MB, containing ONLY
prepared public MovieLens/TMDB/Zenodo cases/texts/poster vectors, no private
checkpoint/model weights. Source checksums and all identities pinned.

Before submission:982regression tests pass(13skipped),9new protocol tests pass.
All4tiny native end-to-end pipelines complete AE/RQ/fresh SDPA predictor,
30epochs/6validationchecks/test. Two additional native multi-batch catalogue
checks (Sum/PCGrad,32items,batch16,initbuffer32) pass. Full-sized initializer
pins verified separately; Sentence-T5real CPU forward and longest token count
checked. These CPU smoke predictions are not quality results or CUDA claims.

Quota observed11.83of45GPUhours available. Estimate1.5–2hours on2T4 including
shared feature preparation, with uncertainty until real epoch timings. Generator
per-fit7200s guard is not intended to replace the30epoch budget; incomplete fits
must be reported. No automatic shorter runs, retries, extensions or new seeds.
Submission receipt records actual workflow/kernel version/source audit.
