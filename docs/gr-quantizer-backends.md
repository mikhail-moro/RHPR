# GRID quantizer implementation and extension boundary

Decision, 2026-10-23: retain GRID RK-Means and replace only its repeated
point-to-centroid distance calculation with optional native PyTorch `cdist`.
No Faiss/Triton dependency or framework migration is needed.

## Distance backend

Set this field inside the existing `quantizer` configuration:

```json
"distance_backend": "torch_cdist"
```

Omitting the field, or setting it to `"upstream"`, retains the original GRID
distance implementation. Existing registered configurations and completed
results keep that default. Both `fit_grid_rkmeans` and the paper runtime
profiler use the option and record it in their reports.

The new backend computes squared Euclidean distances with
`torch.cdist(x, centroids, p=2,
compute_mode="donot_use_mm_for_euclid_dist").square()`, in chunks of 256 input
rows. PyTorch reduces each pair without materializing GRID's
`[256, 256, 2048]` difference tensor (512 MiB in FP32). It works on CPU or CUDA
and keeps clustering in FP32; FP64 is also supported for reference checks.

The direct reduction is deliberate: the alternative GEMM identity
`||x||² + ||c||² - 2x·c` can lose small distances through cancellation. K-means++
always retains GRID's original distance and sampling, independently of the
assignment backend. Residual normalization, catalogue order, cumulative cluster
counts, losses, and centroid updates stay in the upstream components. The
paper profiler also retains separate counts per logical rank.

Floating-point reduction order and the square-root/square round trip differ
from GRID, so bitwise identity and identical assignments near ties are not
guaranteed. Tests compare distances, assignments, manual updates, source
centroid gradients and complete tiny fits. CUDA throughput and complete-run
quality must be measured before claiming a speedup or reproduction score.

The earlier profile's 104-minute RK estimate applies to the upstream backend,
not to this candidate. This change does not turn the finite-catalogue-pass
baseline runner into the paper's distributed step schedule.

## RQ-VAE and VQ-VAE

The generator consumes catalogue-aligned integer codes, not the quantizer
class: `codes[items, levels]` → `SidIndex` collision suffix → generator. That
boundary remains usable with RQ-VAE, residual VQ, or single-level VQ-VAE.
Changing the code vocabulary requires training a generator for the new SIDs.

The pinned GRID source already includes an MLP encoder/decoder RQ-VAE recipe
(`configs/experiment/rqvae_train_flat.yaml`) and identity-encoder residual VQ
(`rvq_train_flat.yaml`). Its residual quantization module can use one level for
VQ-VAE; there is no separate single-level recipe in the inspected source.

RHPR currently implements the RK-Means training adapter only. Learned
quantizers will need their own training and checkpoint adapters that save the
encoder/decoder and codebooks, then export codes in the same catalogue order.
This distance option neither implements those adapters nor enables joint NIP
training through the frozen embedding/tokenization stages.
