# Independent Mamba mechanisms in Query SSM

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This validation-only screen adds one Mamba-inspired mechanism at a time to the
selected Query SSM compressor on H1024+R128. It is not a cumulative staircase:
each fresh arm differs from the same base by one mechanism. No full-history
training, full-history transfer, test evaluation or extra learning-rate search
is included.

| Trial | Change from the base | Training in this run |
|---|---|---|
| `base` | Original K32/slow-dt Query SSM | Reuse verified historical result |
| `conv4` | Causal depthwise convolution and SiLU on shared history features | Fresh |
| `output_gate` | Event-conditioned output gate before the output projection | Fresh |
| `skip_d` | Channel-wise direct input skip before the output projection | Fresh |

All arms retain 32 summaries, inner width64, state size16, query gate rank32,
initial dt range `[0.0001, 0.01]`, query context128 from the old-history suffix,
and the same two-layer d128 SASRec predictor. The original full32768-position
table, item initialization, predictor initialization and shared compressor
parameter initialization are preserved. Extra modules are initialized without
changing existing initial tensors or the subsequent training RNG.

## Exact feature definitions

The base passes normalized, position-enriched old-history features `m_t` through
shared value and input-gate projections, then runs a separate query-conditioned
selective recurrence for each summary slot. Only its final differentiable output
`y_Nj = C(q_j) h_Nj` supplies the summary readout.

`conv4` replaces the shared history features used by **both** the value and
input-gate projections with
`SiLU(DWConv1d(m, kernel_size=4, groups=d_model, bias=True))`. Three zeros are
padded on the left, with no right padding, so feature position t uses only
positions t-3 through t. The query generator reads its original old suffix and
is unchanged. This ablation tests a causal local convolution **with its SiLU**;
it does not isolate the convolution kernel from the activation and does not
claim to reproduce the entire Mamba block.

`output_gate` uses `z_N = Linear(m_N)` with output width64 and computes
`y_Nj * SiLU(z_N)` before the original output projection. The gate is shared
across query slots and depends on the final old-history event, as an event gate
in Mamba does. It is not a new query-conditioned gate.

`skip_d` computes `y_Nj + D * u_N` before the original output projection. `D` is
a trainable FP32 vector of width64 initialized to ones. This is a direct local
input path and could improve recency modeling without improving long memory.

For `output_gate` and `skip_d`, applying the operation only to the gathered
final valid output is exactly equivalent to applying it at every scan output
and then gathering. Neither operation feeds back into the recurrence. This
avoids materializing unnecessary K-by-history gate tensors. Each trial enables
only its named feature; the existing query residual, normalization and summary
FFN remain intact.

## Comparable training and control reuse

Each fresh model starts from the original **untrained** common initializer,
not the trained control checkpoint. Preserve seed42, the fixed parent catalogue,
global daily splits, original user endpoints, last128 shifted targets, and512
identity-bound sampled negatives. A pass visits each user's final training
window once; it is not an epoch over every chronological next-item target.
Crop only H to1024 before embedding. All128 recent losses remain parallel and
causal, with the old-history summaries shared across them.

Use physical batch32, effective batch128, backbone/compressor LR0.001,
AdamW decay1e-5 on **all** parameters, predictor dropout0.2 and compressor
dropout0. Preserve the historical convention even for parameters carrying
Mamba no-weight-decay annotations. AMP and gradient clipping remain unchanged.
Train minimum20 and maximum60 passes, validate every10, stop after two checks
without strict improvement, and select the earliest best macro-user NDCG@10.
NDCG@100 is secondary.

The control is the selected `query_ssm_m32/dt_slow` trial from
`carasik/rhpr-query-qssm-search`, source revision
`da2ed97f8d2af078ecf535e9c730e7cb8f6c60b3`. Its best pass is20 and its saved
NDCG@10 is0.06528778720374186. The complete source search and the immutable
history-source registration are embedded in the new configuration. The loader
verifies source documents, policy replay, selected checkpoint bytes, validation
rank support/metrics, query/raw contexts and strict finite FP32 model tensors
on CPU. It copies only portable weights and H1024 validation artifacts. It
neither retrains the base nor reads historical full-history or test ranks.
Aliasing the trial name as `base` does not rewrite the checkpoint payload;
the original trial and exact source remain in provenance.

## Cost gate and interpretation

The fixed ceiling is10800 seconds. Each fresh shape is profiled separately with
real effective Adam updates and full-catalogue H1024 validation. Before training,
price all three recipes at60 passes and all six validation checks, checkpoint
overhead and safety factor1.25. Early stopping can save compute; it cannot fund
unregistered trials. There is no automatic batch, LR, precision or pass fallback.
Numerical/OOM failures remain explicit, and successful other arms are retained.

Previous K32 measurements suggest roughly one to one-and-a-half hours when the
arms stop around40 passes, and about100 minutes of training/validation if all
reach60. These are historical estimates; fresh measurements determine whether
the registered three-hour ceiling is feasible. The ceiling is not a promise of
three hours of GPU use.

Report each arm's best short-context quality, parameter counts, scoped FLOPs,
uncached throughput and memory. Compare every successful fresh arm with the
same historical base using paired user bootstrap5000 on identical daily anchors.
Timing ratios across separate runs are descriptive. A single training seed,
repeated validation selection and three architecture comparisons make this an
exploratory screen, not a final claim of superiority or a test of cached serving.
