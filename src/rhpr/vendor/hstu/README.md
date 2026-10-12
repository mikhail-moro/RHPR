# Pinned official native PyTorch HSTU

The transduction block comes from Meta's public
[generative-recommenders repository](https://github.com/meta-recsys/generative-recommenders),
commit [`876dcefb40cea14e87660adfd7b965869e6cd403`](https://github.com/meta-recsys/generative-recommenders/commit/876dcefb40cea14e87660adfd7b965869e6cd403).
The upstream source is
[`generative_recommenders/research/modeling/sequential/hstu.py`](https://github.com/meta-recsys/generative-recommenders/blob/876dcefb40cea14e87660adfd7b965869e6cd403/generative_recommenders/research/modeling/sequential/hstu.py),
licensed under Apache 2.0. The complete unmodified file is preserved as
`upstream_hstu.py.txt`; its Git blob identity is
`78329791d6c87d7826bef07fc89f7021ad197a37`. The upstream license is included in
`LICENSE`, and upstream copyright notices remain in `native.py`.

`native.py` vendors the source through `HSTUJagged`, excluding the full upstream
`HSTU` model/training wrapper. It retains the official UVQK projection, SiLU
activation, SiLU attention aggregation, pre-normalization, multiplicative gating,
residual addition and block parameter initialization. This is the official
reference implementation, not its optimized production kernels.

Local changes to the vendored subset are limited to:

- Remove six imports used only by the excluded full upstream wrapper.
- Replace FBGEMM packing calls with scoped `packing.py` equivalents. The shim
  implements one ragged axis with ordinary PyTorch indexing and contains no HSTU
  attention or projection mathematics. It does not register global torch ops.
- Add an optional `normalization_length` to the native SiLU attention path. Its
  default `None` preserves upstream `1 / padded_width` behavior. Our wrapper sets
  it to the fixed raw maximum length (1028 for both Full and Recent128), so changing
  input budget, padding or batch membership does not alter this normalization.
  Dense attention still uses the actual retained padded width.

The experiment adapter lives in `rhpr.sasrec.hstu`. It applies the same item-table,
original-recency positional embeddings, input normalization/dropout and source
position policy as the SASRec comparison. It uses the official *position-only*
relative bias with a fixed 2055-parameter table in both input budgets. A runtime
indexing adapter materializes the exact crop of the official fixed-size bias
matrix; it supplies a zero shape carrier rather than timestamps. Time features,
cached incremental inference and region compression are absent from this screen.
The small encoder adapter runs native blocks under the caller's AMP policy instead
of using `HSTUJagged`'s internal CUDA autocast context.

The default dimensions match the upstream ML-20M reference configuration at
embedding dimension 256 and four heads: `dv=64`, `dqk=64`, four blocks. The adapted
experiment keeps our loss, scoring, input and optimizer protocol; it does not claim
to reproduce the upstream paper's full training recipe, which additionally uses
different sampling/normalization/temperature settings. `ffn_multiplier` is retained
as common configuration metadata and does not create an FFN in HSTU.

`tests/test_hstu_model.py` executes the unmodified pinned upstream block with an
independent slow packing reference. It verifies exact forward equality and
backward numerical parity, including dropout; the fixed-divisor runtime crop also
matches the official fully padded execution. Other tests cover causal isolation,
padding invariance, 128 original targets/positions, common initialization/RNG and
four-block finite gradients.

Latency measurements apply to this native PyTorch adapter and its pure PyTorch
packing. They cannot establish the speed of Meta's fused production kernels.
