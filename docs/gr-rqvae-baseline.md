# Sentence-T5 + native GRID RQ-VAE + TIGER,20epochs

## Active fast prefix baseline (2026-10-25)

The user chose the early fast protocol: sampling is not an axis of the
item-encoder research. Use
`experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json`.
The sampled-window configurations below are retained as history.

Restore the exact generator and optimizer recipe from
`amazon_beauty_gr_baseline.json`, verified against completed run
`2975942b9a2baa92f9b46c20a0755ff870f80165`:4+4/d128,ordinary FF1024,
dropout0.1,user bins2000,noSEP,beam20,4,977,920parameters. Adam1e-3,
weight decay1e-4,clip1,effective batch256/physical128,FP16; FP32 evaluation.
Keep the current frozen Sentence-T5 inputs and native GRID RQ-VAE3x256,
3000updates and saved differentiable state unchanged.

Each epoch enumerates all131,413noninitial train positions once, with at most
20previous items and seed+epoch shuffle, retaining incomplete final batches.
Validation input is the latest20train items; test adds the observed validation
item. No held-out target enters its own context. Four SID codes per item give
80encoder SID positions plus one user token:81positions, verified by actual
CPU forward. The prior stream fed120raw positions plus30SEP tokens:150total,
and its first epoch produced289,689examples, rather than131,413.

Run20complete target-position epochs, validation only at10and20(no epoch1
check), select by validation item NDCG@10, then test once. This is not an
equal-exposure comparison with20sampled-window epochs. Keep best and last
checkpoints, scalar curves and sampling/count metadata; no study telemetry
or profiling. Old configs still select the original sampled-window trainer.
Timing separates optimizer training, validation and complete training-loop
elapsed time. The old early run averaged about78seconds per epoch on T4;
this is a reference, not a guaranteed timing for the new quantizer/run.

## Historical sampled-window configurations

Authorized by the user on2026-10-25. One private GPU run, seed42. Configuration:
`experiments/amazon_beauty_rqvae_tiger_fp16_20epochs.json`.

Update:the user stopped the expanded-FF run and authorized the faster
`experiments/amazon_beauty_rqvae_tiger_plain_ff_fp16_20epochs.json`.
The original configuration below is preserved as historical provenance.

## Active ordinary-FF configuration

Only `generator.mlp_layers` changes from2to null. Four encoder and four decoder
layers,d128,d_ff1024,dropout0.15,separator,batch512,FP16 training,20source-user
epochs and validation every10 remain the same. RQ-VAE3x256,Sentence-T5 inputs,
sampling,LR,selection and evaluation also remain the same.

The native GRID constructor now retains ordinary T5 feed-forward blocks instead
of replacing each one with `128→1024→1024→128`. Parameter count falls from
13,128,064to4,722,048. At identical batch1/seq120,CPU-profiler recognized forward
matmul FLOPs fell from2,134,824,373to834,590,133. This is a compute comparison,
not a measured CUDA training speedup. The previous heavy recipe took about
0.530seconds/update on T4, consistent with about300seconds for566updates.

This is an intentional architecture change for faster experiments, not a claim
that the expanded released GRID generator is reproduced. Both configurations
retain the standalone native GRID RQ-VAE and its saved differentiable components.

## Requested generator and protocol

Reuse the generator fields from `amazon_beauty_grid_epochs_fp16_b512.json`,
the earlier GRID reproduction attempt:4encoder+4decoder layers,d128,d_ff1024,
d_kv64,6heads,dropout0.15,extra MLP2,separator enabled,no user bins. FP16
training, FP32 evaluation/beam probability arithmetic, batch512,Adam5e-4,
weight decay1e-6. Full-catalogue free beam10, original invalid/duplicate SID
accounting, no sampled negatives or seen-item masking.

Twenty finite source-user epochs with the existing sampled contiguous GRID
window stream; validation at10and20, selection by validation Recall@10 and
one final test pass. Patience20cannot stop this20-epoch run before its final
epoch. A declared3-hour training ceiling remains a failure/budget safeguard,
not evidence of convergence. Save best and last generator states, short scalar
curves and aggregate metrics. No SID-study Parquet telemetry, target-path logits,
exact panels, wide beams, training probes or runtime profiling.

## Native differentiable quantizer

Use the pinned GRID `rqvae_train_flat.yaml` with the unchanged native
ResidualQuantization,VectorQuantization,MLP,STE,BetaQuantizationLoss and
initialization loop. Config references are resolved against768D cached
Sentence-T5-base inputs in their original catalogue order and verified hashes.

- Input:BatchNorm1d then L2 normalization, as in the GRID recipe.
- Encoder:hidden768/256/128,latent64; mirrored decoder128/256/768→input dim.
- Three256-code books, jointly trained, residual normalization disabled.
- Loss:sum of native codebook/commitment losses,beta0.25,plus normalized-input
  reconstruction MSE. Native Adagrad1e-3,no scheduler.
- Initializer:3072-item buffer, native KMeans++ and1000inner KMeans steps.
- Training:3000optimizer steps on shuffled catalogue batches2048, one GPU.
  The20-epoch request refers to the generator, not this distinct fitting stage.
- Quantizer FP32 for native initialization/STE stability; generator FP16.
- Optional direct `torch.cdist` squared-L2 replaces the broadcast distance
  calculation, including initializer references; floating rounding may differ.
  No changes to centroid-update equations or gradients, and no hidden LR,
  precision or code-collapse fallback.

This uses the GRID recipe rather than asserting exact canonical TIGER
reproduction:Sentence-T5 replaces GRID's Flan encoder, and single-GPU scheduling
and generator FP16/batch512 are declared choices. The tiny native test verifies
exact exported-code restoration and nonzero finite gradients to encoder,
decoder and codebooks. It does not establish future end-to-end NIP training.

## Artifacts and later experiments

`quantizer.pt` stores native state_dict(including BatchNorm statistics and all
MLPs/codebooks), optimizer, resolved model configuration, initialization flags,
input dimension, recipe hash and RNG. `semantic-codes.npy` is the three-code
export; `semantic-ids.npy` additionally has the catalogue-order suffix making
complete IDs unique. `quantizer.json` includes usage counts and reconstruction
MSE; it is not a per-item telemetry database.

Restore using `build_rqvae` with the saved spec/source recipe, load state_dict,
restore each layer's `is_initialized` and set `is_initial_step=False`. Native
forward requires a Lightning trainer:attach a Trainer and set its state to
PREDICTING when exporting (`export_codes` does this); a normal new Trainer can
subsequently own training. Automatic training resume is not implemented.

### First-run export fix and precision

Version1 completed3000quantizer updates in about109seconds on T4, then failed
before SID export:Lightning's fit teardown had moved the model to CPU, while
the external exporter built CUDA inputs. Explicitly move the whole module,
including BatchNorm buffers, to the export device. Save `quantizer.pt` before
export so a later failure does not discard completed training. This is an
adapter device-lifecycle error, not a numerical training failure.

The pinned GRID recipe requests BF16-mixed on its original hardware. Our T4
run uses FP32 for the quantizer and FP16 for the generator. FP16 AMP for the
quantizer would need explicit treatment:its native manual initialization uses
raw SGD steps, so enabling Lightning's gradient scaler alone would change
initialization unless scaling/unscaling is handled. Keep nearest-code distance,
residual/loss arithmetic and initialization FP32 while experimenting with AMP
MLPs, and measure actual benefit before changing this short preprocessing step.
No quantizer precision change is included in the export fix.

Changing quantizer/encoder weights later can change SID assignments; joint
training needs an explicit objective and code-update policy. This first run
provides a saved differentiable baseline, not an implicit joint-training method.
