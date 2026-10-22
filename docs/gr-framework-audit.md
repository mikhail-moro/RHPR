# Generative recommendation framework audit

Audit date: 2026-10-22. Scope: choose infrastructure for Phase 0 of
[item representation learning for Semantic IDs](gr-research-brief.md).
This audit inspected source code; it did not train or benchmark the alternatives.

## Decision

Use **Snap Research GRID**, pinned to
`2fe3475b2d369580234093f35d52b1a2f54d0472`, through a small RHPR adapter.
It already provides the three relevant stages: embedding inference, residual
quantization, and an encoder-decoder SID generator. Representations can be
prepared independently, which is the important boundary for later tabular item
encoders. Replacing the whole training infrastructure is unnecessary for the
first run. [GRID paper](https://arxiv.org/abs/2507.22224),
[pinned repository](https://github.com/snap-research/GRID/tree/2fe3475b2d369580234093f35d52b1a2f54d0472).

The first configuration is a **TIGER-style functional baseline**: full frozen
Sentence-T5 embeddings, three-level RK-Means, a collision suffix, and the actual
GRID generator. It deliberately combines components from different published
recipes. It must not be reported as an exact reproduction of TIGER or of a GRID
table. A functioning pipeline and a matched paper-number reproduction are
different acceptance criteria.

## Alternatives inspected

| Framework and audited revision | Useful components | Practical limitation for this task |
| --- | --- | --- |
| [GRID](https://github.com/snap-research/GRID/tree/2fe3475b2d369580234093f35d52b1a2f54d0472), `2fe3475b2d369580234093f35d52b1a2f54d0472` | Separate embedding stage; RK-Means, RVQ and RQ-VAE; Hugging Face T5-based SID generator; Hydra/Lightning composition | Heavy default environment, TFRecord input coupling, released defaults differ from paper recipes; small audited adapter required |
| [phonism/genrec](https://github.com/phonism/genrec/tree/45acaa29fc8cadd1f29dc9c58c8f715035c8ecee), `45acaa29fc8cadd1f29dc9c58c8f715035c8ecee` | TIGER, RQ-VAE, RQ-KMeans, Amazon adapters, item-level SID resolution and several other recommenders | Useful fallback, but an independent reimplementation; its TIGER imports a local T5 implementation, despite comments describing direct Hugging Face usage; RK-Means defaults target OneRec, not TIGER |
| [Tiny-Snow/GenRec](https://github.com/Tiny-Snow/GenRec/tree/e69d13371a5042c4dcb1bb3259cd7298cdd8a15b), `e69d13371a5042c4dcb1bb3259cd7298cdd8a15b` | Typed component interfaces, separate datasets/models/trainers, RQ-VAE and TIGER model code | Python 3.13 requirement; authors explicitly mark quantizer and GR modules as under development; inspected tree has no GR trainer directory; custom TIGER blocks differ from GRID; no documented RK-Means path |
| [RecTokens](https://github.com/EdoardoBotta/RecTokens/tree/e8359bc6aa17de349272eb0c654fd199e2c77044), `e8359bc6aa17de349272eb0c654fd199e2c77044` | Reusable RK-Means/RQ-VAE tokenizers, GPU trie decoding, Hugging Face integration | Primarily a tokenizer/decoding library; end-to-end examples fine-tune causal Qwen models, rather than reproduce the requested small encoder-decoder; introduces Triton dependency |

These projects share names with unrelated “GenRec” papers. Use the exact owner
and revision above when recording dependencies. The recommendation is based on
component fit and provenance, not on unverified claims of higher accuracy or
speed.

### Licenses

- GRID's [LICENSE](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/LICENSE)
  permits noncommercial research and requires retaining its license and
  third-party notices upon redistribution. It is not a general commercial-use
  license. Keep upstream attribution separate from RHPR's own adapter.
- phonism/genrec is
  [MIT licensed](https://github.com/phonism/genrec/blob/45acaa29fc8cadd1f29dc9c58c8f715035c8ecee/LICENSE).
- Tiny-Snow/GenRec declares
  [GPL-3.0](https://github.com/Tiny-Snow/GenRec/blob/e69d13371a5042c4dcb1bb3259cd7298cdd8a15b/pyproject.toml).
- RecTokens' actual
  [LICENSE is Apache-2.0](https://github.com/EdoardoBotta/RecTokens/blob/e8359bc6aa17de349272eb0c654fd199e2c77044/LICENSE),
  although its package classifier still says MIT. Record the license file,
  rather than copying the inconsistent classifier.

## GRID integration boundaries

### 1. Item representations

Store a stable item-ordered embedding matrix with model revision, metadata
template, preprocessing, dimension, dtype and content hashes. The quantizer
should consume this artifact without knowing whether its source was a language
model, a frozen feature transform, or a trained tabular encoder. Record which
catalogue items were available when fitting the tokenizer.

The released [embedding configuration](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/sem_embeds_inference_flat.yaml)
uses Flan-T5-XL encoder output with mean pooling. Overriding only its model name
to Sentence-T5 is insufficient. The audited
[Sentence-T5-base checkpoint](https://huggingface.co/sentence-transformers/sentence-t5-base/tree/fc5d4628481afbbaaacd7af6bb07cf9d3865f781)
has a T5 encoder, masked mean pooling, learned dense projection and normalization.
Use the complete SentenceTransformer pipeline so the projection is retained.
The checkpoint choice is an explicitly selected realization: TIGER describes
768-dimensional Sentence-T5 embeddings but does not identify an exact public
checkpoint or serialization template.

### 2. Quantization

GRID's [ResidualQuantization module](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/modules/clustering/residual_quantization.py)
accepts separate normalization, encoder, decoder and quantization modules.
Its [RK-Means configuration](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/rkmeans_train_flat.yaml)
uses layerwise residual mini-batch K-Means with normalized residuals. This is a
natural frozen-representation baseline; it does not train an upstream encoder
through next-item loss.

Fit/encode stages should save centroids, SID assignments and item ordering.
Log code usage, entropy, residual distortion, collision groups and maximum
suffix. A complete SID must map to exactly one item. Do not count a match to an
ambiguous three-code tuple as a correct item prediction.

### 3. Generator

Use the actual [GRID SID generation classes](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/models/modules/semantic_id/tiger_generation_model.py)
with an explicit configuration. Keep training targets and evaluation independent
of representation preparation. Later representation comparisons can then share
the same target schedule, generator, optimizer budget and evaluation.

The [released generator YAML](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/tiger_train_flat.yaml)
uses 4 encoder and 4 decoder layers, hidden size 128, six attention heads with
head dimension 64, and FF dimension 1024. Its extra FF hidden layer increases
parameters substantially relative to ordinary HF T5 with the same listed
dimensions. Report the actual instantiated parameter count rather than assuming
that “T5 128” specifies compute.

## Practical issues found in the released source

1. **Default installation is too broad for this baseline.** The pinned
   [requirements](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/requirements.txt)
   include a CUDA-specific Torch stack, TensorFlow, TorchRec/FBGEMM, DeepSpeed,
   DVC and cloud tooling. Install only the dependencies exercised by the adapter;
   preserve the runtime's working CUDA Torch installation.
2. **An annotation imports the TFRecord stack.**
   [data interfaces](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/data/loading/components/interfaces.py)
   import `RawDataIterator` for annotations, causing
   [iterators.py](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/data/loading/components/iterators.py)
   to import TensorFlow even when only model classes are wanted. Postponed
   annotations plus a `TYPE_CHECKING` import is a small I/O dependency patch;
   record its exact bytes and upstream hash. It does not replace model or
   quantization algorithms.
3. **Paper and example YAML differ.** Default encoder, dropout, training budget,
   checkpoint metric and history length are not a complete published recipe.
   The RK example has 30 total steps; the GRID paper discusses a larger
   multi-GPU clustering budget. An adapter must label its own exact recipe.
4. **Training “epochs” are not interchangeable.** Upstream uses unbounded
   iterables and a collator that expands contiguous subsequences. Record source
   users, supervised targets and optimizer updates. Using a finite prefix
   dataset is a declared protocol change, even if the generator is unchanged.
5. **Constrained beam mode has a first-step indexing bug.** In
   `_beam_search_one_step`, an unconditional second mask assignment indexes
   the batch dimension with a vocabulary mask. A CPU reproduction with batch 2
   and width 4 raises `IndexError`. Free-form mode avoids that branch. Any
   constrained-mode patch needs explicit tests; constrained renormalization also
   changes scoring relative to free-form probabilities followed by pruning.
6. **SID-level metrics need an injectivity proof.** Upstream evaluator compares
   complete SID tuples. Independently decode to item IDs, check duplicate and
   invalid predictions, and document whether invalid positions retain their
   ranks. Keep all suffixes inside the configured output vocabulary.

The framework is usable after these bounded adaptations. It should be treated
as research source with an audited experiment contract, rather than a turnkey
guarantee of a published score.

## Reproduction labels and Phase 0 acceptance

[Canonical TIGER](https://arxiv.org/abs/2305.05065) uses Sentence-T5, RQ-VAE and
an encoder-decoder trained on next-item prediction. The requested RK-Means
substitution changes the tokenizer. GRID's own published baseline instead uses
Flan-T5 and normalized residual clustering. Keep the two reference recipes
separate from the initial hybrid.

For Phase 0, require a hashed Beauty artifact, chronological held-out-prefix
checks, complete finite embeddings, an injective item/SID mapping, finite
training gradients, independent item-level ranking metrics, and a bounded real
GPU run selected using validation only. A tiny synthetic fit demonstrates that
the code works; one real seed establishes an experimental starting point.
Neither establishes a new scientific advantage. Further seeds, domains and
matched controls come after this pipeline passes.
