# Research brief: item representations for Semantic IDs

Date: 2026-10-22. Source: the user's research brief supplied in this task.
Update 2026-10-24: reproduction launches are paused. The active controlled
GPU study is [SID index sensitivity](gr-sid-index-study.md), with frozen
representations and quantizer-seed variation before further encoder research.
Status: a research program and set of hypotheses, not an established novel
architecture. The earlier history-compression/QSSM study remains closed.

## Question and scope

For Semantic ID-based generative recommendation, how much pretrained text
representation learning is necessary? Can compact encoders of structured item
features produce equally useful or better IDs at lower total cost? How much
comes from the representation architecture, its objective, and its interaction
with quantization?

The [TIGER pipeline](https://arxiv.org/abs/2305.05065) combines separately trained
components:

```text
Item metadata -> Sentence-T5 -> RQ-VAE -> item Semantic IDs
User history as Semantic IDs -> encoder-decoder -> next-item Semantic ID
```

Different component objectives are a reason to investigate this pipeline, not
evidence that it is broken. Text semantics may be useful, collaborative signals
may complement them, and improving reconstruction need not improve retrieval.
The immediate aim is to find a reproducible limitation before choosing a new
method to address it.

Representation-first and direct feature-token construction are separate
families. The initial infrastructure supports the former. Direct/compositional
tokenization remains a possible later comparison; Phase 0 does not implement a
new tokenizer, a tabular model zoo, or end-to-end joint optimization.

## Working hypotheses

- **Sufficiency:** simple structured-feature representations may suffice for
  good SIDs on some catalogues, reducing dependence on a large text encoder.
- **Objective:** the representation's training objective may matter more than
  a change of encoder architecture.
- **Interaction:** representation geometry, quantization distortion and
  downstream next-item quality may rank candidate methods differently.
- **Stability:** representations may differ in sensitivity to missing fields,
  text formatting and changes that preserve item meaning.
- **Cost:** quality comparisons should include representation preparation,
  tokenizer fitting and generation, rather than only the last stage.

None is yet a finding. “Tabular encoder for SID” is a broad topic description,
not a novelty claim. A negative result showing that a cheap representation is
sufficient, or that a proposed improvement disappears under a matched
quantizer, can still answer a useful question.

## Phase 0: establish the experimental instrument

Use [GRID](https://arxiv.org/abs/2507.22224) as the component framework, with the
source revision and integration issues recorded in the
[framework audit](gr-framework-audit.md).

The first run is Amazon **2014 Beauty**, a complete frozen Sentence-T5 pipeline,
three RK-Means levels of width 256, a collision-disambiguation suffix, and a
GRID encoder-decoder. Its exact configuration and measured counts are the
contract. It is a TIGER-style hybrid: RK-Means deliberately replaces TIGER's
RQ-VAE, and Sentence-T5 differs from GRID's Flan-T5 example. It cannot establish
that a published TIGER or GRID score has been reproduced exactly.

Keep the interfaces explicit:

1. Prepared item features and chronological interactions, with immutable item
   order and held-out target manifests.
2. Item-ordered representations, with encoder revision, preprocessing and hash.
3. Fitted tokenizer and item-to-SID assignments, with collision diagnostics.
4. Generator configuration, target schedule, checkpoints and item-level metrics.

First verify a small fit and a bounded real Beauty run. Test finite losses and
gradients, train/validation/test boundaries, exact item lookup, invalid beams,
duplicate candidates and metric recomputation. Record source users, target
examples, optimizer updates, runtime and peak memory. Select checkpoints on
validation; use test only after selection.

The user's brief also requests reproduction of a published GRID baseline. The
initial hybrid is the first functional milestone, not completion of that
stricter reproduction milestone. A separate declared paper-aligned recipe and
matched preprocessing are required before making that claim.

## Minimal next experiment matrix

After Phase 0 passes, freeze data, metadata availability, SID budget, generator
and selection protocol. Change one source of variation at a time.

| Arm | Item representation | Representation objective | Tokenizer | Question |
| --- | --- | --- | --- | --- |
| A | Complete frozen Sentence-T5 | Original pretrained checkpoint | RK-Means | Reference |
| B | Fixed structured feature representation | None | Same RK-Means | How much learned representation is necessary? |
| C | Field embeddings plus a small MLP | Masked field prediction or reconstruction | Same RK-Means | Does learning feature interactions help? |
| D | Same architecture as C | Train-only collaborative objective | Same RK-Means | Does the objective change SID utility? |
| E, optional | Same representation as A | Same as A | RQ-VAE | Does the quantizer change the conclusion? |

The exact raw-feature encoding, missing-field policy, normalization, dimension
and objective must be specified before these arms run. This is a proposed
minimal matrix, not an authorized queued sweep. Add FT-Transformer or another
larger tabular encoder only after a simpler control reveals a limitation.
An RK-Means stage does not transmit recommendation gradients to an upstream
encoder; every learned encoder needs an explicit training objective.

Metadata access must be matched. A method using title, brand, category and price
should not be compared as an architecture-only change against one given extra
description, item ID or future collaborative statistics. Report item-ID access
separately because it can change both memorization and cold-item behavior.

## Related work to verify before claiming a gap

The following are research leads from the supplied brief. Except for TIGER and
GRID protocol/source checks in the framework audit, this task has **not**
independently reproduced their results or verified all of the brief's detailed
interpretations. Preserve that distinction when writing a paper.

- [ReSID](https://arxiv.org/abs/2602.02338),
  [source](https://github.com/FuCongResearchSquad/ReSID): the closest identified
  lead for recommendation-native structured-feature representations and
  specialized quantization. Verify FAMAE inputs, item-ID access, training-only
  use of histories, GAOQ controls and complete preprocessing cost. Do not assume
  its aggregate gains isolate its encoder.
- [ActionPiece](https://arxiv.org/abs/2502.13581),
  [source](https://github.com/google-deepmind/action_piece): a lead for feature
  merging and context-dependent action tokenization. Verify which features are
  raw and which derive from semantic embeddings/OPQ before interpreting it as
  evidence that text representations are unnecessary.
- [PLUM](https://arxiv.org/abs/2510.07784) and
  [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478):
  leads for task-aware representation geometry and semantic/collaborative
  complementarity.
- [3MToken](https://www.amazon.science/publications/multimodal-music-tokenization-with-residual-quantization-for-generative-retrieval):
  a lead for quantizing heterogeneous music features; using heterogeneous
  fields alone is not a defensible novelty claim.
- [Faithful Evaluation of Semantic-ID Tokenizers](https://arxiv.org/abs/2605.25330):
  a methodological lead for collision-aware item-level evaluation. Our own
  evaluator should already require injective IDs and exact item ranks,
  independently of this paper's claims.

The proposed architectural tools—FT-Transformer, TabTransformer, DCN-V2,
TabNet and SCARF-style objectives—are candidates, not an obligatory checklist.
Changing several at once would make an improvement harder to explain.

## Evaluation and interpretation

Use exact item-level Recall/NDCG at declared cutoffs, with an explicit policy
for invalid and duplicate generations and no credit for ambiguous SID matches.
Keep Amazon 2014 and Amazon 2023, preprocessing versions and split protocols
separate. A transductive metadata catalogue is a declared setting; a later
cold-item claim requires a different controlled test.

For later scientific comparisons, report multiple seeds and domains, quality
by frequency/cold-item/metadata-availability strata, tokenizer usage and
collisions, distortion, preprocessing cost, training cost and serving cost.
Fit any collaborative representation using training interactions only.
Tune comparable arms using validation under a documented budget. Reusing one
generator configuration initially isolates representation effects but does not
prove every tokenizer has reached its best possible quality.

Proceed to a new method only if a stable difference or limitation survives
these controls. If strong text embeddings remain competitive and cheap, or
simple features close the gap entirely, that result should guide the scope
instead of prompting an automatic increase in architectural complexity.
