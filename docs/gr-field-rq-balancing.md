# RQ-VAE with fixed-normalized reconstruction and GradNorm

Authorized2026-10-26 after the continuous-AE non-UW suite. Configuration:
`experiments/amazon_beauty_field_rq_balancing_2t4.json`. Two fresh independent
fits,one per T4,5000 main native optimizer updates each;no new AE fits.

The completed controls have the same seed42,neural initialization,static
12101-item Beauty catalogue,frozen title vectors,preprocessing,960D input,
latent64,batch2048,FP32 and native neural Adagrad0.001:

|AE control|Brand accuracy|Category F1|Price MAE|Title MSE|
|---|---:|---:|---:|---:|
|Fixed normalized|0.11019|0.70496|0.64697|0.000240686|
|GradNorm|0.33707|0.80291|3.32778|0.000519457|

Control kernel IDs,source revisions,report hashes and reconstruction metrics
are recorded in the configuration and copied to `ae-controls.json`. Title
targets reuse the separately byte-verified original AE output parent. There is
no text-encoder rerun. Before fitting,hash the common neural state and require
`56da423e243262fbeda4a444959429f9482f7b695b31f4299d392e446ab5b63a`,the
completed AE controls' initial hash. Also compare full initial RQ states across
the two new trials. Common initialization does not imply identical trajectories.

Restore the native pinned GRID initializer,three256-code residual books,STE,
encoder/decoder and codebook+beta0.25 commitment loss. The total training loss
is native quantization loss plus weighted raw-field reconstruction,both outer
coefficients1. There is no rescaling or adaptive weighting of the quantization
auxiliary in this experiment. Log its magnitude separately;an apparent failure
can still involve imbalance between that term and reconstruction.

Both reconstruction policies use the exact shared objective implementation
as their AE controls:fixed catalogue-constant-predictor normalization,or that
normalization followed by GradNorm(alpha1.5,Adagrad weight LR0.025). GradNorm
measures the last shared decoder Linear weight and balances only the four
CE/BCE/Huber/title-MSE losses. No fifth task for codebook/commitment losses.
Unused UW parameters are frozen. Neural/codebook parameter group stays0.001.

Keep GRID's native manual optimization and initialization-only SGD path.
Its temporary SGD initialization steps consume extra batches but do not advance
the wrapped main optimizer's global step. Thus record physical initialization
batches,main reconstruction updates and total item exposures separately.
The5000-update budget matches the AE's main updates;RQ has additional initial
exposure. Do not describe it as the same number/order of physical batches.

Only complete catalogue passes with reconstruction available from their start
contribute to GradNorm's reference. Exclude the partly initialized first pass;
the first subsequent full pass uses equal multipliers and establishes reference
means. This preserves the native initializer instead of adapting weights using
zero/missing reconstruction losses. The warm-up differs from continuous AE
because RQ needs initialization. Partial final-pass accumulators are checkpointed.

GradNorm's auxiliary backward runs in Lightning's `on_before_backward`,after
the native loop clears gradients and before the total-loss backward. It updates
only the weight logits. Effective weights in the reconstruction objective are
detached. A native two-process CPU integration test verifies that these gradients
survive manual optimization,the logits actually change,and restored checkpoints
reproduce semantic codes and field metrics exactly. Probes preserve RNG/buffers.

Every500 main steps and after initialization,save the existing full-catalogue
quantized and continuous-path reconstruction probes,code usage/counts,
entropy/perplexity,residual geometry,codebook/commitment losses,gradients and
semantic-prefix collision statistics. Save semantic codes,embeddings,checkpoints,
optimizer/RNG,schemas,normalizers,GradNorm state,initial hashes and source recipe.
The continuous path inside the RQ probe uses the RQ-trained decoder;it is distinct
from the separately trained AE control.

Suffix overflow is a measured outcome,not a failed run:do not append or widen
an ordinal suffix,repair assignments or reinitialize dead books. No generator,
recommendation validation/test or downstream quality claim. Two same-seed arms
are objective controls,not independent replications. Compare all fields and
collisions together;more distinct codes alone is not a winner criterion.

The shared `ae_suite` process runner now also supports `gr_field_rq_suite`.
Each subprocess sees exactly one assigned CUDA device;requesting two GPUs has
no CPU/single-GPU fallback. Preserve successful sibling artifacts if a trial
fails;the overall suite then fails. No automatic retries or tuning.
