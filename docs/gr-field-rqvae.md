# Joint structured-field encoder and RQ-VAE

User decision,2026-10-26:learn feature projections and the quantizer together
through field reconstruction. Title is explicitly excluded. Configuration:
`experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json`.

Active follow-up: `experiments/amazon_beauty_field_rqvae_tiger_5k.json`.
The user requested5000joint updates and fuller codebook logging after the first
run failed at collision resolution. The original3000-step configuration is
retained for historical replay; generator/data/optimizer/objective are unchanged.

| Field | Input | Reconstruction target | Loss |
| --- | --- | --- | --- |
| brand | Trainable64D embedding, observed flag, Linear65→64 | Original brand class | Cross entropy over2075classes |
| categories | Mean of selected trainable64D embeddings, observed flag, Linear65→64 | Original234D multi-hot vector | BCEWithLogits, mean over labels and observed items |
| price | Catalogue z-score and observed flag, Linear2→64 | Original standardized scalar | Huber,delta1 |

Concatenate192D, then reuse native GRID's BatchNorm+L2 normalization and
MLP768/256/128→64. Three256-code residual books, native STE, beta0.25,
initialization and Adagrad1e-3 remain intact. The native decoder
64→128/256/768→192 feeds separate Linear heads for the original fields.
There is no reconstruction MSE against the trainable input embeddings and no
skip path around the quantizer. All learned parts are in the same optimizer.

The added adapter overrides only GRID's model-step inputs and reconstruction
objective. GRID still owns residual recursion, quantization loss, STE,
initialization-only optimization and subsequent joint updates.

## Field preparation and objective

Reuse the conservative parser of the published item text; do not infer brand
from title, repair metadata or use interactions. Vocabulary and price mean/std
use the entire static catalogue, as in the transductive baseline. Standard
deviation is population(ddof0); a constant numeric field uses scale1. Apply
saved statistics without refitting on reload. Unknown nonmissing labels in a
new catalogue fail explicitly. Duplicated category labels are removed; category
order is intentionally discarded by mean pooling.

Missing brand gets ID0; its embedding is padding. Empty/unparsed categories
pool to zero, and absent price has standardized input0. Every field also has
an observed flag. Missing targets contribute neither reconstruction loss nor
the corresponding uncertainty regularizer. Empty category lists are treated
as missing metadata, not as known negative labels for every category.

LibMTL's MIT-licensed uncertainty implementation is adapted as a differentiable
forward module instead of importing its complete training framework:

`L = L_GRID_quantization + sum_observed_fields [L_field/(2 exp(s_field)) + s_field/2]`.

The three trainable log variances start at−0.5, matching LibMTL. Quantization
loss is not uncertainty-weighted. Huber is a robust surrogate here; this is
not an exact mixed-data likelihood. The loss may become negative due to the
log-variance terms; this alone is not numerical failure. Raw errors, observed
counts, learned weights and log variances are saved separately.

The source URL, retrieved source hash and MIT license are in
`src/rhpr/gr/vendor/libmtl/`. Native GRID licensing/provenance is unchanged.

The verified Beauty catalogue has12101items,2075brands,234category labels,
2118missing brands and585missing prices. There are11173distinct combinations
of these inputs; the largest identical-input group has23items. Such items
are indistinguishable to this deterministic encoder. Keep the existing
collision suffix and explicit256-suffix overflow check; do not impute fields
or inject item IDs to resolve this silently.

## Bounded experiment and saved state

Use the previous3000FP32 quantizer updates,batch2048,3072initialization buffer,
1000native initializer steps and cdist squared L2. This is joint training of
the feature encoder and quantizer on metadata reconstruction. It does not
backpropagate next-item prediction through discrete index changes.

After export,freeze the index and run the unchanged fast TIGER baseline:
4+4/d128,ordinary FF,noSEP,user bins2000,FP16,effective batch256/physical128,
20epochs,validation only at10/20,NDCG@10 selection,test once. All131413train
next-item positions appear once per epoch with latest20context. No SID-study
telemetry or hyperparameter sweep is included.

Save `feature-schema.json`(vocabularies,scaling,provenance),
`catalogue-fields.npz`(exact prepared field values/masks in catalogue order),
`quantizer.pt`(feature encoder,heads,uncertainty parameters,native
MLPs/codebooks/normalization,optimizer,RNG,initialization flags,resolved recipe),
`semantic-codes.npy` and suffix-extended `semantic-ids.npy`.
`item-embeddings.npy` contains the learned192D concatenation before native
normalization; it is not a Sentence-T5 embedding. Raw field losses and weights
are logged every500updates; final catalogue brand accuracy,category microF1
at0.5,and price MAE accompany reconstruction errors. These are fit diagnostics,
not held-out recommendation quality.

Rebuild through `build_rqvae(...,feature_schema=saved_schema,field_dim=64)`,load
the state_dict and initialization flags,attach a Lightning Trainer and export
with `export_field_codes`. The test verifies exact code/embedding replay and
finite nonzero gradients to every component, including reconstruction-only
gradients reaching the feature encoder through native STE. Automatic training
resume is not implemented.

This comparison removes title and changes the representation/objective together.
It tests a complete alternative recipe; a gain or loss cannot isolate an effect
of architecture alone or prove objective mismatch. Later objectives and title
ablations require explicit matched controls.

## First-run failure and diagnostic follow-up

Private kernel `carasik/rhpr-field-rqvae-tiger-beauty`,version1,commit
`094b716cb8c73660dade1ab9cc5ec1ae269c2659`,completed all3000joint updates
in about104seconds,then failed before generator training. Only68semantic tuples
remained for12101items. Books used5/8/13codes; the largest tuple had1796items.
Thirteen groups exceeded256suffix values,containing11419items. This is severe
assignment collapse,not merely occasional duplicate tuples or a beam-search error.

Saved checkpoint CPU replay reproduced the68tuples. Catalogue-native beta
quantization loss was about28.995(codebook23.196+commitment5.799). Brand
CE4.740/accuracy0.1086,category BCE0.05339/microF10.4072,and price
Huber0.1141/MAE6.198in original units are distinct diagnostics. The final training
batch's total32.55 includes quantization and uncertainty-weighted field errors;
it must not be presented as reconstruction error alone. Reconstruction without
quantization uses the same trained decoder and is only an observational probe,
not a separately trained continuous baseline or a causal bottleneck test.

The5000-step follow-up is a fresh same-seed fit,not an additional5000steps or an
automatic resume. A full-catalogue eval probe runs after initialization,at every
500completed updates and finally. Probes preserve CPU/CUDA RNG,all module
training flags,TrainerFn and BatchNorm state. A real paired CPU fit test verifies
identical final state_dict and codes with probes enabled/disabled.

- `field-learning-curves.json`:deduplicated positive step numbers,total loss,
  native quantization/weighted reconstruction terms,raw field errors,weights,
  observation counts and gradient norms for encoder/decoder/heads/books.
- `codebook-learning-curves.json` and latest `codebook-diagnostics.json`:full
  occupancy arrays including dead codes,entropy/perplexity,dominant-code share,
  latent/quantized variance,centroid norms/separation and residual MSE per level.
  Include quantized and continuous-input field errors and beta-loss decomposition.
- `codebook-snapshots/step-NNNNN.npz`:all codebook vectors,per-item semantic
  codes and occupancy counts in immutable catalogue order.
- `semantic-code-diagnostics.json`:all prefix collision size histograms,largest
  groups,full-tuple overflow groups/items and suffix representability.
- `tokenizer-status.json`:explicit stage,code SHA,seed,largest group and suffix
  capacity; written before attempting generator construction.

Intermediate-prefix collisions are expected and are not called complete-SID
overflow. Only the entire semantic tuple determines whether the suffix fits.
No reset,EMA,LR/precision fallback or suffix widening is introduced. More steps
may not cure assignment collapse; the diagnostic run tests its trajectory.
