# Beauty baseline result and runtime profile

The first completed Sentence-T5 / RK-Means / GRID run is a functional reference,
with a reproducible quality gap and measured time accounting. **It does not
reproduce a published GRID result.** Following the user's clarification on
2026-10-22, faithful reproduction takes priority over profiling or another
hybrid quality run. The bounded profiler is implemented but has not been sent
to Kaggle. **The generator stays at 4 encoder + 4 decoder layers.**

## Frozen completed baseline

Source commit:
[`2975942b9a2baa92f9b46c20a0755ff870f80165`](https://github.com/mikhail-moro/RHPR/commit/2975942b9a2baa92f9b46c20a0755ff870f80165).
Private parent notebook:
[`carasik/rhpr-gr-beauty-sentence-t5-rkmeans`, version 2](https://www.kaggle.com/code/carasik/rhpr-gr-beauty-sentence-t5-rkmeans).
The version number is sequential; it is not a Kaggle internal session ID.
The preceding setup version 1 was cancelled, as confirmed by the user in Kaggle UI.

The run used Amazon-2014 Beauty: 22,363 users, 12,101 items, 153,776 training
events and 131,413 causal training examples per epoch. Its generator had
4,977,920 parameters, d_model=128, six attention heads with d_kv=64, ordinary
T5 FF1024, dropout .1, 2,000 user bins and no separator. Each epoch attempted
514 Adam updates at effective batch 256 / physical 128. Item representations were
complete Sentence-T5-base embeddings; tokenization was normalized-residual
GRID RK-Means, 3 semantic levels ×256 codes, followed by a collision suffix.
The original [baseline contract](gr-baseline.md) records the data and evaluation
conventions.

The run stopped by early stopping at epoch 80, rather than exhausting the
7,200-second training budget. The selected checkpoint was epoch 55:

| Metric | Validation at selected checkpoint | Held-out test |
| --- | ---: | ---: |
| Recall@5 | .04972499 | .03429772 |
| Recall@10 | .07624201 | .05348120 |
| NDCG@5 | .03288204 | .02251205 |
| NDCG@10 | .04141764 | .02862973 |

An independent item-level lookup and rank calculation reproduces all saved test
Recall/NDCG metrics to absolute tolerance1e-15. All complete four-token SIDs are
unique, all 22,363 test user IDs are unique, beam scores are finite and descending,
and there are no duplicate candidates. There are 450 invalid SIDs among 447,260
beam candidates (0.1006%); only 84 occur within top 10. They retain their original
miss positions. These checks provide no evidence that collision-prefix credit,
rank compaction or a large invalid-beam rate explains the quality gap.

The best validation NDCG@10 was .04141764 at epoch 55; the final value was
.04059404 at epoch 80 while training loss continued to decrease. This is
consistent with mild overfitting or selection noise. There were 11 AMP-skipped
updates among 41,120 attempts (0.0268%); this count alone does not explain the
runtime or the quality gap.

## Where the time went

| Component | Measured time |
| --- | ---: |
| Entire runner | 106.67min |
| Training loop, summed over 80 epochs | 91.47min, mean 68.60s/epoch |
| Seventeen validation passes | 12.87min, mean 45.43s/check |
| Sentence-T5 encoding, once | 56.49s |
| RK-Means, once | 22.35s |
| Selected-checkpoint test, once | 45.08s |
| Other overhead inside training block | 9.10s total |
| Offline runtime installation, outside runner | 37.64s |

Training accounts for 85.8% of runner time; validation accounts for 12.1%.
Checkpoint I/O was not instrumented separately: it is included in the small
9.10-second residual together with JSON writes, permutations and other overhead.
It cannot plausibly account for the observed minutes of runtime. Conversely,
aggregate training time cannot distinguish GPU kernels, Python launches,
per-microbatch synchronization, staging, backward and optimizer work. Those are
the profiler's targets.

The 25 epochs after the selected checkpoint cost approximately 32.06 min including
their validation checks. At an unchanged80-epoch trajectory, changing validation
from every 5 epochs to every 10 epochs would remove eight checks and save about
6.06 min. That is a counterfactual accounting estimate, not a measured speedup:
checkpoint selection and early stopping may change. Training already used
**FP16 autocast with FP32 parameters/optimizer state**; validation and test were
**FP32**. GRID's FP32 default therefore does not mean the completed training was
running without mixed precision.

## What GRID actually reports

In [GRID Table3 and §4.1](https://arxiv.org/html/2507.22224v1#S4),
**3×128 means three RK-Means levels with 128 centroids per level**, not a
three-layer Transformer. The generation backbone remains 4+4, d128 in this
comparison:

| Tokenizer | Beauty Recall@10 | Beauty NDCG@10 |
| --- | ---: | ---: |
| RK3×128 | .0617 | .0339 |
| RK3×256 | .0639 | .0347 |

The completed hybrid's gap from the RK3×256 reference is approximately−16.3%
Recall@10 and−17.5% NDCG@10. These are descriptive reference differences, not an
isolated tokenizer or representation effect: GRID uses Flan-T5 embeddings and
five-seed averages, while this is one Sentence-T5-base seed. Metadata,
augmentation, FF depth, user tokens and checkpoint selection also differ.

The [paper setup](https://arxiv.org/html/2507.22224v1#S4) specifies **Adam,
LR 5e-4, weight decay 1e-6, batch 256**, and validation every 100 optimizer steps.
Its text states NDCG@10 for early stopping but Recall@10 for checkpoint selection;
our declared selection criterion remains validation item NDCG@10. The released
[`tiger_train_flat.yaml`](https://github.com/snap-research/GRID/blob/2fe3475b2d369580234093f35d52b1a2f54d0472/configs/experiment/tiger_train_flat.yaml)
instead has LR 1e-3, weight decay 1e-4, dropout .15 and `precision: 32-true`.
Its validation interval 1600 counts training batches; with accumulation 16 this is
approximately 100 optimizer updates. Paper and YAML are distinct recipes.

The released generator also adds an extra 1024→1024 FF hidden layer, has no user
bins by default, and enables a separator. RHPR keeps the existing ordinary
T5 FF, dropout .1, user2000 and separator-off generator for the next experiment.
Changing the SID vocabulary to 128 reduces output-table/head work but leaves
the principal Transformer hidden computation and the four-token item length
unchanged. It is not a depth reduction. The
[framework audit](gr-framework-audit.md) records the remaining differences.

## Bounded profile, currently deferred

The profile loads the exact saved parent checkpoint, SIDs and manifest, with
artifact hashes checked before use. It measures short repeated optimizer steps
with restored parent weights and fresh optimizer/scaler state; these are runtime
microbenchmarks, not continued quality training. Validation precision agreement
uses a deterministic validation subset. Test labels and test predictions are not
used to choose runtime settings or the next model.

The prepared, unsubmitted profile can compare:

- Physical training batches 128, 256, 512, 1024 where memory allows. Batch 128 and 256
  preserve effective 256; larger effective batches are labelled as optimization
  changes, not pure implementation improvements.
- Evaluation batches 32, 64, 128, 256, with FP32 as the reference and FP16 evaluated
  for throughput, finite scores, candidate agreement and validation metric drift.
  BF16 is used only if natively supported; it is not assumed to accelerate T4.
- Full input pipeline versus GPU-resident batches, synchronized timing, peak
  memory and detailed CPU/CUDA events when the managed runtime supports them.
- RK-Means width 128 versus 256 using the same cached item embeddings; no repeated
  Sentence-T5 encoding or quality training is required for that tokenizer profile.

The previously proposed hybrid quality follow-up (Sentence-T5, RK3×128 and
paper LR/weight decay) is **cancelled before submission**. Changing those few
settings would still not reproduce GRID: the text encoder, augmentation and
generator configuration would remain different. Reproduction must first
identify and align the complete recipe. The user's preference for validation
every **10 epochs** remains recorded; GRID's step-based validation is a separate
protocol choice that must be explicit when establishing that recipe.

Matching GRID's reported quality is now the immediate goal. Faster kernels or a
combined recipe change would not independently demonstrate that Sentence-T5,
the quantizer, or a particular learning rate caused the previous gap.

## Evidence retained outside Git

Private artifacts include the original run outputs, checkpoint, user predictions
and full logs. The audit reads source SHA and input file hashes from those
artifacts; only aggregate findings are recorded here. Core parent identities:

| File | SHA256 |
| --- | --- |
| `run-manifest.json` | `d0d96892758eda471d2e8584158da9b9559615629e9cde7321d05144f6eed20d` |
| `generator-best.pt` | `f405506a71ceab9f3568e739a6b258f78ee338045c7ec15d13ab44bf73a499ea` |
| `semantic-ids.npy` | `b196711dceedce81a991d995aa5c86ccd35288d5cdfe3f7aedb174de86470dae` |
| `item-embeddings.npy` | `607895c3dbe7a6a5aa6272dde55dd190e8bc40fa97532830de161d118aca5314` |
| `metrics.json` | `1ce7d6026614777ed1e4401470cbdaf20ea7dde9735bec647f5cf38ae3ab841c` |
