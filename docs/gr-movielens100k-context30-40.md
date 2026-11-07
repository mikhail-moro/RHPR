# MovieLens context30/40 pilot, frozen indices

Authorized2026-11-03: user explicitly requested training on30and40items without
FlashAttention, then selected **seed42and up to30epochs**. Four fresh predictors:
control(no people) and actor graph.1, each with30or40visible events. Original
TIGER4+4/d128/FF1024, eager attention, user token, FP16AMP+GradScaler; FP32
validation/test, unmasked full catalogue, beam20. No seen mask or wider search.

Reuse the exact seed42indices of private
`carasik/rhpr-ml100k-actor-full-training v1` (source70a0c16). No text encoding,
AE/RQ-VAE fitting, graph changes or trained predictor weights are reused.
Every SID/lineage file and parent manifest is hash-pinned. Fresh predictor
initial parameter hash must equal the original constructor hash5f4a24fa…,
for both modalities and windows. This avoids fresh quantizer variation.

Budget: exactly30epochs; first20validation at5/10/15/20, then validation each
new epoch21–30, as in the original schedule. Verified optimizer/scaler/RNG
resume from new run's own20epoch checkpoint. Original LR.001, plateau factor.5/
patience5from21, effective batch256/physical128, weight decay.0001, gradclip1.
Best validation NDCG10 checkpoint selected; test once afterward. The30epoch cap
is a screening budget, not evidence of convergence or a full early-stop result.
No extensions, extra seeds, hardware/precision/batch fallbacks or HPO.

Input windows contain30/40films =121/161encoder tokens including user token.
Same85470strictly-past training targets,334attempted updates/epoch,943validation
and943test cases. CPU audit verifies identical users/targets and exact last20
context suffix for all splits and windows. More history does not add targets.

Compare window20 validation on its recorded first30epochs: identical check
schedule and target budget. Historical window20test selected after longer
training is an unmatched reference; the actor's original best epoch31exceeds
this pilot cap. No new20window controls are trained. Window30versus40and the
paired graph/control comparisons within this run share budget and initialization.

Two T4workers, one control and one graph, each processing30then40. Private NEW
kernel `carasik/rhpr-ml100k-context30-40`; immutable parent. Quota before launch
7.86GPU-hours. Compute estimate~1.5–2hours elapsed total; actual timing recorded
per epoch and peak allocated GPU memory per trial. No automatic time extension.

Outputs include all validation predictions/losses, curves, selected and last
checkpoints, resume proof, frozen indices, test predictions, original reference
metadata, and unfiltered known-item top10occupancy as diagnostic only.

Implementation: `rhpr.gr.movie_context_training`;
config `experiments/movielens100k_context30_40_30epochs.json`.
Validation includes a native tiny-GRID30epoch fit with verified20epoch resume
and14validation checks, plus actual-architecture CPU forward/backward and
initialization checks for all four configurations. GPU performance remains
unmeasured until the authorized training run.

Checks before submission:955repository tests passed(91skipped);3native GRID
checks passed separately. All4real-schema forward/backward tests passed,
matching original initializer SHA and finite gradients. No GPU calibration or
trial has been launched locally.
