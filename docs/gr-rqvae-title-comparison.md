# Plain RQ-VAE versus structured fields with frozen T5 title

User request,2026-10-26:two GPU runs,3000quantizer updates each,diagnostics
only. No generator fitting,validation or test evaluation. The previous
3000/5000field-only runs remain historical controls.

| Arm | RQ input | Reconstruction |
| --- | --- | --- |
| `amazon_beauty_rqvae_plain_3k_diagnostic.json` | Original cached768D Sentence-T5 over the complete serialized metadata | Native normalized-input embedding MSE |
| `amazon_beauty_field_title_rqvae_3k_diagnostic.json` | Learned192D brand/categories/price concatenated directly with frozen768D title-only Sentence-T5;960D total | CE/BCE/Huber plus MSE to the original frozen title vector,combined with four LibMTL uncertainty scales |

Use the same complete SentenceTransformer pipeline and pinned Sentence-T5-base
revision `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`. The plain arm verifies
the original cached full-metadata embedding hash/catalogue order. The new arm
extracts only title from the conservative published-text parser,encodes it once
offline from the prepared local model,then keeps those vectors fixed. No brand,
category or price strings enter that title encoder. Missing titles receive
zero input and are masked out of title MSE and uncertainty regularization.

There is no learned projection on the768D title before concatenation. Retain
the previous three64D field projections. The native GRID BatchNorm+L2 and
MLP768/256/128→latent64 remain;input/output widths change to960D. A separate
Linear head reconstructs the768D title target,which is detached/fixed. Title
MSE averages over its dimensions and observed items;it is not rescaled to match
CE magnitude. Save title cosine similarity and zero/mean predictor MSE so a
small numerical MSE is not mistaken for good reconstruction.

Both use seed42,native3x256 residual books,STE,beta0.25,Adagrad1e-3,
batch2048,3072initialization buffer,1000initializer iterations and FP32
quantizer training. A3000-update budget matches optimization steps,not FLOPs,
parameter count,information availability or objective magnitude. In particular,
the plain arm still has full metadata text,while the new text component has
title only. The earlier field-only3000-step arm is the useful title-addition
reference;plain-versus-fields changes more than one factor.

## Existing evidence before the new runs

Completed original Sentence-T5/RQ-VAE3k run exported11991semantic tuples for
12101items. There were100collision groups containing210items(1.7354%of
the catalogue),110excess assignments(0.9090%),and a maximum group size4.
Used codes were143/256/256. This was ordinary occasional collision,not the
severe assignment collapse of the structured-only variant.

Structured-only3000steps:68tuples,max group1796,used codes5/8/13.
Structured-only5000steps:132tuples,max group2132,used codes6/12/23.
More distinct tuples did not prevent an even larger dominant collision group.
These observations motivate the comparison;they do not establish why collapse
occurs or prove that missing title is the sole cause.

## Outputs and failure semantics

`pipeline.stage=quantizer_only` returns after tokenizer fitting and diagnostic
export,even if the suffix cannot represent all collisions. `metrics.json` and
`run-manifest.json` then record a completed diagnostic experiment,including
`suffix_representable=false`,with no fabricated recommendation metrics.
`tokenizer-status.json` still records `collision_overflow` explicitly. Full
generator runs retain the strict suffix-overflow error. No SID is repaired.

Both arms save full-catalogue codebook probes after initialization,every500
updates,and finally. Occupancy,entropy/perplexity,residual/centroid geometry,
collision histograms and codebook/per-item assignment snapshots use the same
diagnostic code. Probes preserve RNG,TrainerFn,module modes and BatchNorm;
paired native CPU fits verify identical final weights/codes with and without
probes. Codebooks,MLPs,optimizer,RNG and schemas remain in `quantizer.pt`.

The title arm additionally saves `title-encoder/item-embeddings.npy` and its
model/revision/text-hash provenance,`catalogue-fields.npz` with fixed title
targets,learned concatenations,per-field losses/weights,and all trainable field
components. The title encoder itself stays the resource-pinned pretrained model.

## Completed3000-step diagnostics

Both new kernels completed at commit `20dc54faaf2cac5c8fa7fd14cb37165154e01a2a`.

| Arm | Unique semantic tuples | Max group | Used codes | Suffix fits |
| --- | ---: | ---: | --- | --- |
| Plain Sentence-T5/RQ-VAE |11991|4|143/256/256|yes|
| Structured fields + frozen title |845|1162|10/42/157|no|

Title improves tuple diversity relative to the prior field-only3k run(68tuples,
max group1796),but severe assignment collapse and suffix overflow remain.
The diagnostic stage deliberately completes with this measurement;COMPLETE
does not mean that the index can be used by the unchanged generator.

Title reconstruction MSE is0.00255964 versus zero-predictor0.00130208 and
catalogue-mean-predictor0.000287248. Cosine is0.53673. Thus the small-looking
raw MSE is not evidence of successful title reconstruction. Brand CE4.5305,
category BCE0.04653 and price Huber0.25711 are separate objectives;the native
beta quantization loss is about30.216,versus0.0001205in the plain arm.
These results are one seed;they do not isolate the training failure's cause.

The plain run DID log diagnostics:13stdout progress/codebook events,seven
catalogue probes at1/500/1000/1500/2000/2500/3000,full JSON curves,and seven
NPZ codebook snapshots. Its stdout summary was shorter than the field arm.
Future plain-run console summaries now include perplexity,suffix status,
quantization MSE,embedding reconstruction MSE and native beta loss. Explicit
scope labels distinguish full-catalogue eval from last-training-batch loss.
No retraining is needed to recover the completed run's already saved diagnostics.
