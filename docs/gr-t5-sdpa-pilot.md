# Native T5 SDPA pilot on Kaggle T4

Authorized 2026-11-05: bounded technical pilot, not a quality experiment or full
ML1M training. Private kernel `carasik/rhpr-t5-sdpa-pilot`, config
`experiments/t5_sdpa_t4_pilot.json`. No Telegram, extra seeds, retries, fallback
hardware, dataset changes, or training extension.

Three sequential paths on the same physical T4:

1. Historical Transformers4.47.0 eager.
2. Transformers5.22.1 eager, isolating library migration.
3. Transformers5.22.1 native SDPA, isolating the backend change.

All load the byte-pinned, trained singleton sum/seed42 checkpoint, identical
frozen SIDs and strict-past ML100K examples. Architecture remains4+4,d128,
d_ff1024,6heads,d_kv64,dropout.1. Both encoder and decoder configurations request
SDPA; T5's native implementation sets attention scaling1.0.

Compatibility changes only: newer T5Stack owns its embedding constructor;
GRID's keyword EncoderDecoderCache constructor maps to the newer positional
constructor through a subclass scoped to the loaded GRID module. The original
GRID checkout stays byte-verified and untouched. Without an explicit backend,
the historical4.47model construction and random draw order remain unchanged.

Check dropout-disabled FP32 hidden states, loss, ALL95 parameter gradients
(including relative-position bias), and cached beam20 on8 fixed validation
examples. Compare old vs modern eager and SDPA. Require exact beam paths and
close scores; fail if checkpoint names/shapes change. Numerical tolerances are
fixed in source before GPU execution. No test-set quality is measured.

Benchmark windows20and50, fixed256real training examples, physical128,
Adam.001/wd.0001,GradScaler1024,clip1,FP16,dropout.1. Each path/width gets10warmup
and50timed optimizer updates, plus10end-to-end measurements including SID
batch construction/H2D, and one profiled update. All restart from the SAME
trained weights with fresh optimizers and seeds. This repeated-batch pilot is
not a convergence or NDCG comparison. Save every timed loss, skipped-update
counts, peak allocated/reserved memory, actual SDPA operator names and profile.
Runtime dispatch may use the math backend; requesting SDPA alone is not evidence
of fused execution or speedup. No automatic benchmark configuration search.

Child execution budget1200seconds total, plus bounded installation and data
verification. The old runtime remains intact; modern packages install offline
into a separate site and only appear in modern child processes. The new private
`carasik/rhpr-t5-sdpa-runtime` dataset contains PUBLIC PyPI wheels only (43.3MB,
28manifest files), hashes locked; no Torch/CUDA/Triton replacement. Existing
model weights mount directly from the completed private Kaggle notebook.
Automatic approval review rejected re-uploading private checkpoint artifacts
into a new dataset; that upload was not performed. Public wheels and direct
same-account notebook inputs avoid the rejected transfer.

CPU preflight on the actual trained checkpoint (PyTorch2.10CPU): old→modern
eager hidden/loss exact, max gradient delta7.45e-9; old→SDPA max hidden/loss
delta1.19e-6, gradient delta2.09e-7. All8×20beam paths exactly equal, beam-score
max delta4.10e-8. These checks do not establish CUDA speed or FP16 stability.
CPU evidence: docs/gr-t5-sdpa-cpu-preflight.json. GPU results/submission recorded
separately; do not claim execution or acceleration from CPU checks.

Quota observed before submission:12.07of45GPUhours remaining. Only this short
pilot is authorized. Collect metrics.json, summary.md, run-manifest.json and
pilot-results.json; no full ML1M experiment is queued by this change.
