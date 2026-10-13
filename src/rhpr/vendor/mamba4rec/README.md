# Official Mamba4Rec blocks

Source: [chengkai-liu/Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec),
commit `44a0c8432b34e3b89d46035b10e8525b1bd368dc`, MIT license in `LICENSE`.
`native.py` contains exact `MambaLayer` and `FeedForward` class slices. Only the
Mamba factory imports the external dependency lazily. The exact original source
and config are kept as fixtures; hashes and integration changes are recorded in
`provenance.json`.

The RHPR adapter uses the official no-absolute-PE input policy, dropout before
input LayerNorm, Mamba1 then dropout/LayerNorm, residual when there is more than
one layer, and the official four-times-width GELU feed-forward block. The profile
uses `d=128`, two layers, state size **32**, convolution width **4**, expansion
**2**, and LayerNorm epsilon `1e-12`. Published config uses d64 and one layer;
profile dimensions are a resource screen rather than tuned paper settings.

Native Mamba1 initialization is intentionally preserved in full. The original
recommendation wrapper recursively reinitializes all Linear weights and biases;
that overwrites the SSM's specially initialized `dt_proj.bias`, even though its
parameter has `_no_reinit=True`. Here only the item embedding and recommendation
FFN projections receive normal initialization. This is an explicit integration
change, not an exact reproduction of the original RecBole training recipe.

The GPU model requires installed **mamba-ssm 2.3.1** with a compatible
`selective_scan_cuda` extension. `causal-conv1d` enables the native fused Mamba
inner kernel; without it the native selective scan still runs on CUDA alongside
a PyTorch depthwise convolution. Actual package versions, native source hash,
extension availability, and selected path are exposed in model metadata.
There is no production reference-scan fallback.

State-spaces Mamba source fixtures are pinned to
`c5afbdf3bda1a09d68f65181ae3a43ec71079820` (tag v2.3.1), Apache-2.0 under
`STATE_SPACES_LICENSE.txt`. Mamba1 and selective scan source files are byte-identical
to v2.2.6.post3. The CPU tests execute the exact official Mamba1 class and
`selective_scan_ref` from these fixtures using a test-scoped layout helper; the
profiling path imports the installed CUDA implementation instead.

Right padding follows real events and cannot alter earlier causal states.
The wrapper masks PAD outputs after each layer and returns original raw source
positions so shared loss code selects the same final 128 targets as other
architectures. Tests cover pinned-source identity, block forward/backward parity,
native initializer preservation, future-token isolation, batch-versus-solo
padding invariance, and finite loss gradients into the old prefix.
