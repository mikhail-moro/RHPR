# Joint structured-field encoder and RQ-VAE

User decision,2026-10-26:learn feature projections and the quantizer together
through field reconstruction. Title is explicitly excluded. Configuration:
`experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json`.

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
