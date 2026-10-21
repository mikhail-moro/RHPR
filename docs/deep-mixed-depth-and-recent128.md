# Deeper MixedDepth and a trained Recent128 control

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

These two fresh 15-epoch runs distinguish stronger compression from removing
the old prefix. They retain raw windows of 1028, the latest 128 supervised
positions, sampled softmax with 512 negatives, Time OFF, the seed 42 backbone
recipe, physical batch 8 / effective 128, and full-catalog validation at B32.
The test split remains reserved. Neither run loads pretrained weights.

Configurations:

- `experiments/yambda_mixed_depth_64_15ep.json`
- `experiments/yambda_recent128_15ep.json`

The deeper model samples one depth per physical batch from 0 through 6.
All passes reuse one PairMLP projection; odd old singletons carry through,
raw endpoints compose, and the last 128 original events are retained.
At raw 1028, encoder lengths are 1028/578/353/241/185/157/143.
New depths participate in training, rather than being forced only at evaluation.

| Depth | Old-prefix reduction | Training probability |
| ---: | ---: | ---: |
| 0 | ×1 | 0.25 |
| 1 | ×2 | 0.125 |
| 2 | ×4 | 0.125 |
| 3 | ×8 | 0.25 |
| 4 | ×16 | 1/12 |
| 5 | ×32 | 1/12 |
| 6 | ×64 | 1/12 |

Depth 0 and depth 3 retain their prior 25% probabilities. Depths 1/2 receive half
their prior exposure; the released 25% goes to the three deeper modes. This
choice preserves two useful anchors but changes the joint training objective.
It is not a continuation of the old checkpoint or a controlled comparison of
training schedules. Record actual per-depth batch/user exposures and curves.

At epochs 5/10/15, forced validation at all seven depths selects one common
checkpoint by the unweighted mean of the seven NDCG@10 values. Each final
prediction and benchmark uses that checkpoint and records its SHA256.
No separate checkpoints per depth are selected. Saved cohort comparisons
include every depth against 0, new depths against 3, and adjacent deep budgets.
Representation diagnostics retain raw spans, norms and spread before PE and
input LayerNorm, so scale changes under recursion remain inspectable.

Recent128 is trained from scratch to use only the latest 128 raw item embeddings.
It uses `compressor_type=tail_only`, which has no parameters and preserves the
original source positions. The raw window and position table remain 1028, so PE,
target selection and full-context validation hashes can be matched to MixedDepth.
The model never attends to discarded old events. This differs from the earlier
evaluation-only deletion on frozen MixedDepth weights, and from the existing
`recent` compressor, which keeps a larger paired-length budget.

Recent128 validates at epochs 5/10/15 and selects its own best checkpoint by
validation NDCG@10. Patience 16 ensures it completes all 15 epochs. Both runs use
the same epoch cap, data and hyperparameters; their training and inference cost
need not be equal. Neither run establishes convergence if its best epoch reaches
the cap. The mixed model's checkpoint objective aggregates seven modes, whereas
Recent128 has one mode; record this difference when comparing their optima.

Each run writes the standard metrics, manifest and summary, per-user predictions,
trained-checkpoint history features and benchmarks. Compare models offline only
after strict user/context/target joins. Use the deep model's frozen history bins
as a common basis for embedding-derived slices; independently fitted cosine
quartiles from Recent128 describe different users. Raw length and item uniqueness
can be matched directly. User bootstrap remains conditional on selected weights
and seed, and overlapping slices are exploratory.

B1 and physical B32 timings include selected compression/selection, encoder and
full-catalog scoring/top-k. They exclude CPU collate, transfers, checkpoint
loading and external serving. Inputs and weights are already on GPU; summaries
are not cached. FLOPs are nominal dense matmul counts from actual executed
shapes, including every compressor pass. Allocated GPU memory is not a minimum
VRAM requirement. Recent128's raw embedding lookup occurs before tail selection
and is included in time even though only 128 tokens enter the encoder.

The private kernels use Tesla T4 and the Yambda dataset with internet disabled.
Workflow concurrency is keyed by kernel ID: different authorized experiments
may run concurrently, while runs updating the same kernel remain serialized.
