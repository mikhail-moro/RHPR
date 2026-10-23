# Paper-priority GRID Beauty recipe and profile contract

Audit date: 2026-10-22. The user selected the article as the authority when it
conflicts with released example configurations. The immediate run is a runtime
profile, not a quality experiment. The registered runtime profile is in
[`amazon_beauty_grid_paper_profile.json`](../experiments/amazon_beauty_grid_paper_profile.json).
No GPU results are claimed in this document.

## Published target and source precedence

Target GRID's **Beauty, Flan-T5-XL → RK-Means 3×256 → encoder-decoder 4+4**
reference: Recall@10 **0.0639**, NDCG@10 **0.0347**. The paper reports averages
of five different seeds; one diagnostic run is not that five-seed result.

Use explicit article settings first, then the pinned author implementation for
unspecified details. Record those source defaults as fill-ins rather than proof
of the exact historical table configuration. The inspected author revision is
`2fe3475b2d369580234093f35d52b1a2f54d0472`; the paper is arXiv2507.22224v1.

| Setting | Resolved reference | Evidence / interpretation |
| --- | --- | --- |
| Data | Author P5-derived Amazon-2014 Beauty rows and leave-last-two split | Article §4; existing normalized rows preserve source text and IDs |
| Text encoder | Flan-T5-XL encoder, final-state masked mean, 2048D | Article §4; author embedding YAML specifies max128, FP32, batch8, special tokens including EOS |
| SID | RK-Means, three semantic levels, width256; normalized residuals; extra collision token | Article §§4–4.1; author RK and inference code |
| Generator layers | Four encoder + four decoder layers, d_model128, six heads, FF hidden1024 | Explicit article §4 |
| Other generator fields | d_kv64, dropout.15, mlp_layers2, tied weights, no user bins, separator enabled | Released generator YAML / constructor defaults; article does not specify these completely |
| Generator optimizer | Adam, LR5e-4, weight_decay1e-6 | Explicit article §4; not AdamW |
| Adam details | betas(.9,.999), eps1e-8, amsgradFalse | PyTorch Adam defaults; no overrides in released YAML |
| Effective model batch | 256 supervised examples per optimizer update | Article says “batch size256”, without a per-device qualifier for generation; this is the declared single-device interpretation |
| Loss | Sum of four hierarchy-wise mean cross-entropies | Actual author encoder-decoder model_step, including collision hierarchy |
| Scheduler / clipping | No scheduler; no gradient clipping | Article specifies neither; source scheduler:null and no Trainer gradient_clip_val |
| Reference generator dtype | FP32 | Article dtype unspecified; released generator precision32-true |
| Validation cadence | Every100 optimizer updates | Explicit article §4; do not inherit the hybrid's epoch cadence |
| Early stopping | Ten validation checks without validation NDCG@10 improvement | Explicit article §4 |
| Test checkpoint | Best validation Recall@10 | Explicit article §4; distinct from the early-stopping monitor |
| Beam | Free-form beam10, KV-cache, invalid paths remain misses at original rank | Released default; exact Table1 beam width is not specified in article |
| First diagnostic seed | 42 | Released seed; article's five seed values are unspecified |

Mixed FP16, larger physical batches and other throughput cases are profiling
candidates. They are not automatically adopted into the reference quality run.
A physical batch below256 may accumulate correctly weighted microbatches to
preserve effective256. A physical batch512 changes the effective batch and must
be labelled as an optimization change, not a pure execution speedup.

## Training stream and example accounting

Import the actual author `collate_with_sid_causal_duplicate`. It reads full
training histories, enumerates contiguous item subsequences of length≥2,
samples indices across a source-user batch, and then suffix-truncates to120
SID codes **including the target**. Four codes/item imply at most29 observed
items per training example. Author validation/test contexts remain capped at20
observed items.

The released source uses32 source users, at most512 selected augmented windows,
and accumulation16. Its sampled index vector is drawn with replacement but
membership-based enumeration emits a selected index only once, so actual model
batch size is variable and ≤512. Copying those three defaults does not implement
the article's stated batch256.

For the paper-priority adapter, keep the actual augmentation stream and buffer
its emitted examples in order into effective batches of256. Record source users,
augmented examples, optimizer updates and any retained buffered examples. This
is an explicit interpretation of the article's batch size, not a claim that the
example YAML has the same optimizer schedule. CPU checks must compare the
underlying collator output with the unchanged author function at fixed RNG,
including item0, long rows, padding, selected duplicates and target masking.

There is no natural finite epoch in this unbounded stream. Report runtime and
budgets in optimizer updates / target exposures. The user's earlier ten-epoch
validation preference belongs to the previous finite-prefix setup; the latest
paper-priority instruction selects the article's100-update cadence here.

## RK-Means: eight-device semantics

The article states **eight GPUs, 2,048 points/device, 1,000 steps/layer**, with
three semantic layers. Nominal exposure is16,384 point assignments per global
step,16,384,000 per layer and49,152,000 across three layers, before distinguishing
initialization from steady updates. These are repeated catalogue observations,
not distinct items. The released example's30 total steps and the hybrid's ten
catalogue passes are different budgets.

The author RK configuration uses KMeans++ initialization with a3,072-point
buffer and SGD(lr=.5), which is equivalent to its manual mini-batch centroid
update on one device. Initialization is based on rank0's buffer; a special loss
rescaling compensates DDP averaging for that initialization step. After
initialization, no world-size loss rescaling is applied.

**A single global batch of16,384 is not generally equivalent to the published
DDP update.** Each source MiniBatchKMeans object stores `cluster_counts` as a
plain rank-local tensor, not a registered synchronized buffer. For centroidj,
rankr's current count is n_rj, cumulative count after the batch is N_rj, and
batch mean is μ_rj. Source WeightedSquaredError and DDP's averaged gradients
with SGD(.5) give the initialized update:

```text
c_j(new) = c_j(old) - (1/R) * Σ_r [n_rj / N_rj] * (c_j(old) - μ_rj)
```

Ranks with n_rj=0 contribute zero. Merging all points and counts before one
running-mean update instead weights by global cumulative counts, which is a
different formula when rank histories differ.

A faithful one-device **steady-state operator benchmark** can run eight logical
rank contributions sequentially: shared pre-update centroids, eight persistent
local count tensors, actual source loss per rank, average the eight gradients,
then apply one common SGD(.5) update. Assignments for every rank must use the
same pre-update centroid version. This removes distributed communication and
does not reproduce the paper's wall-clock speedup or undocumented data order.
It does preserve the initialized update algebra for the supplied rank batches.

The profile should report local2,048-point and logical-eight-rank costs
separately. A single16,384-point proxy may also be timed, but must not be called
an exact DDP replacement. Do not spend3,000 fitting steps just to obtain a short
operator-time estimate; measure initialization and representative repeated
steady updates independently. A later full tokenizer fit must separately
record initialization, layer switching, sampling and rank-count semantics.

## Runtime profile outputs and limits

Use the aligned generator configuration and real source-length distributions.
Record the SIDs' provenance; random or proxy SIDs are valid only for explicitly
labelled runtime microbenchmarks. A fresh generator's ranking metrics are not
baseline quality. Set `quality_not_evaluated=true`, do not tune against test and
do not present this profile as a reproduction score.

Measure warmup-separated repeated cases with CUDA synchronization, elapsed
step time and peak allocated/reserved memory:

- Encoder FP32 batch8 reference and larger batches that fit; record tokens/items
  and observed finite outputs. Any FP16 comparison records numerical drift.
- Training physical batch32/64/128/256 at effective256, FP32 reference and FP16
  diagnostic; separate batch construction/H2D, model work and optimizer work.
- Autoregressive validation beam10 at several evaluation batches, with the
  real validation-context lengths. Subset timing is an estimate; a full
  validation timing is stronger evidence when within budget.
- RK initialization, local-step and logical-eight-rank steady-step timings.

Let T_step be measured time per effective256 update and T_val one complete
validation pass. For S optimizer updates, a paper-cadence cost estimate is:

```text
T_generator(S) ≈ S*T_step + ceil(S/100)*T_val + measured checkpoint overhead
T_total(S) ≈ measured encoder stage + measured RK stage + T_generator(S)
```

Report several exposure budgets (for example1k/10k updates), not a predicted
convergence epoch. Early stopping, data order and quality convergence are not
inferable from a short timing profile. Show startup/setup separately and state
when an entire-stage estimate is extrapolated from a sample.

## Primary sources

- [Article §4 and Tables1–3](https://arxiv.org/html/2507.22224v1#S4).
- [Embedding YAML](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/sem_embeds_inference_flat.yaml).
- [RK YAML](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/rkmeans_train_flat.yaml).
- [Generator YAML](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/tiger_train_flat.yaml).
- [Actual augmentation](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/data/loading/components/collate_functions.py).
- [MiniBatchKMeans](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/models/modules/clustering/mini_batch_kmeans.py).
- [WeightedSquaredError](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/components/loss_functions.py).
- [Initialization loop](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/src/components/training_loop_functions.py).

## Execution assets

The separate private input `carasik/rhpr-gr-flan-t5-xl-encoder` contains the
CPU-extracted, byte-preserved FP32 encoder and its original model card/license.
The [committed manifest](../runtime/gr/flan-t5-xl-encoder-manifest.json) pins every
asset. It restores only the original tied embedding alias; it changes no learned
tensor values. The existing Beauty/GRID/wheels input is reused independently.

Profile settings: 900-second cooperative budget and 25-minute Kaggle server
timeout, deterministic sample of catalogue items and validation contexts,
physical training batches 32/64/128/256 with effective 256, FP32 reference and
FP16 diagnostic. No full training run or test evaluation is part of this job.
