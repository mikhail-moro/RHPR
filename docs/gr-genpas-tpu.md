# Native TPU launch for the TIGER window baseline

Authorized on2026-10-23 after GPU quota fell below40 minutes. The earlier GRID
TPU probe was cancelled; it supplied no compatibility result. Use the actual
TPU image's matched torch/torch_xla/libtpu stack. Do not install GPU Torch wheels.

The first bounded run is `gr_genpas_tpu_probe.json`. It checks actual TPU identity,
the released full13,127,936-parameter GenPAS generator's CPU/TPU FP32 loss,
three effective BF16 Adam updates, then FP32 constrained beam generation against
the same post-update CPU weights. Missing compatibility fails the run, with the
stage and error in `pipeline-status.json`. No CPU fallback counts as a TPU pass.

## Input preparation

Private input `carasik/rhpr-genpas-beauty-resources` contains verified GenPAS source,
the lossless author Beauty data conversion and auxiliary wheels. It has22,363
users and12,101 items. TensorFlow is not installed: previously verified TFRecord
conversion preserves all training events and the supplied validation/test
contexts (last20 observations, one target). `prepare_genpas_resources.py` records
hashes and retains Snap's original attribution/license.

The paper's public Drive archive contains neither the named Flan-T5-XXL SID file
nor a tokenizer trainer. Its embedded vectors are768-dimensional and are not
used as claimed XXL representations. The new preparation explicitly computes
Flan-T5-XXL4096-dimensional item vectors and a fresh frozen RK-Means tokenizer.

`prepare_genpas_xxl_encoder.py` derives only the encoder/shared weights from
`google/flan-t5-xxl@ae7c9136adc7555eeccc78cdd960dfd60fb346ce`. It downloads pinned
HTTP ranges and converts FP32 to BF16 with round-to-nearest. The encoder has
4,762,310,656 parameters (9,524,621,312 tensor bytes). Source header/range hashes,
publisher full-shard hashes and derived artifact hashes are recorded. Full source
shard hashes are **not** claimed verified by the subset download. Decoder tensors
are not downloaded. Preserve the model card and Apache license.

Encoding uses fixed padded token batches, maximum128 tokens, masked mean pooling
including EOS, and FP32 accumulation/output. BF16 weights/inference are an explicit
numerical departure. Output `item-embeddings.npy` is catalogue aligned and retained
for subsequent experiments.

Because GenPAS supplies no quantizer trainer or downloadable named SID artifact,
declare the new tokenizer implementation: CPU sklearn Lloyd KMeans, k-means++,
seed42+level, normalized incoming residuals, three256-way levels and a stable
collision suffix. The exact n_init/iteration/thread budget belongs in the run
config and output. This differs from GRID's online distributed quantizer; the run
is a reproducible baseline, **not** a claimed exact reproduction of paper metrics.

## TPU portability changes

The runner initially uses one actual TPU device. Other devices are not presented
as utilized. It preserves the released4+4 generator, loss summed over four SID
codes, Adam/weight decay and the source's epoch-based LR schedule.

- Source users are visited once per seeded epoch. Select precisely the released
  window sampler's random indices, retaining sorted unique membership and source
  order; invert its enumeration instead of looping over every unselected window.
  The CPU equivalence test compares actual masked inputs/labels for multiple caps
  and seeds. Alpha/beta/gamma and GenPAS augmentation remain disabled.
- Buffer windows into fixed physical batches and pad the final batch. Exclude
  padding from loss and metrics. Weight accumulated gradients by the actual
  number of unpadded examples, including the final accumulation group.
- Replace two dynamic beam-mask assignments with equivalent `masked_fill`, and
  replace catalogue prefix broadcasting with exact fixed-iteration lower_bound
  on precomputed sorted integer prefix keys. No candidate or ranking change.
- Keep model/Adam weights FP32; use XLA BF16 autocast for training. Beam arithmetic
  remains FP32. Metrics run on host CPU after the TPU generates candidates.
- Set CPU and XLA RNG seeds explicitly. Validate every10 completed epochs,
  select by validation Recall@5 and evaluate test once from that checkpoint.
- Save the best and last checkpoint at validation boundaries. Automatic resume
  is not implemented. A time limit stops at the next scheduled validation;
  the server timeout remains the hard limit.

CPU tests cover window equality, prefix membership, identical native/adapted
beams, padding-gradient equality, partial batches,20 epochs with checks at10/20,
checkpoint selection and the declared RK tokenizer. These tests establish
algorithmic equivalence on CPU, not TPU compatibility or throughput.
