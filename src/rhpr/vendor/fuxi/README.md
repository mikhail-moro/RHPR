# Official FuXi-Linear native blocks

Source: [USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear),
revision `5a704061a7ebf0b81afb465f12f93f2747ba4667`. Exact upstream source
fixtures, Apache-2.0 license and attribution are shipped alongside the subset.
`provenance.json` records source paths and Git/SHA-256 identities.

The transduction block, three attention channels, multistage feed-forward and
chunkwise state scan are reused from the official implementation. Scoped
packing replaces FBGEMM; unused framework imports/wrappers are excluded. Runtime
positional queries use the prefix of the original fixed-size parameter table.
The chunkwise path receives no global sequence-square attention mask because the
official channels use their own chunk-local causal masks. Small-input tests check
forward and backward parity against the unmodified source with a full mask.

The custom state scan has a local AMP correction: FP16/BF16 KV products and
FP32 learned decay use a promoted accumulator of at least FP32. Saved states
retain that precision for decay gradients, while returned states retain the
original KV dtype. Backward explicitly uses the accumulator dtype and keeps
broadcast axes when reducing decay gradients. This avoids mixed-dtype failures
when backward runs outside forward autocast, without downcasting learned decay.
Float32/double behavior is preserved; mixed-precision gradients are checked
against an independent differentiable recurrence. Exact upstream fixtures are
unchanged.

The RHPR profile uses d128, two layers, four semantic heads, eight temporal heads,
positional dimension 32 and chunk size 128. Time OFF retains temporal channel
compute with an all-zero timestamp payload, so queries contain no actual future
timestamp information. Real-time features and the paper's training recipe are
not reproduced. Python state scans, native einsums and the AMP-corrected custom
autograd are benchmarked; no fused Triton implementation or recurrent serving cache is
claimed. The adapter uses shared catalog lookup, input normalization/dropout and
sampled-softmax loss, without adding a second positional embedding to the native
learned positional channel.

`major_matmul_flops` counts multiply-adds in the native three-channel projections,
local chunk attention, KV chunk summaries/cross-chunk reads and gated FFN, with
two FLOPs per multiply-add. It assumes full valid input rows, rounds channel
width to chunk size and counts positional QK once per shared batch map. Tests
compare this formula to actual native `mm`/`bmm`/`addmm` profiler events. It excludes
normalization, nonlinearities, decay/gates, packing, elementwise state scans,
catalog lookup and loss; it is not a total training-FLOP estimate.

Dependencies: PyTorch, NumPy and `einops==0.8.1`.
