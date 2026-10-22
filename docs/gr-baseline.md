# Amazon Beauty GR baseline

Phase 0 implements the user-selected **Sentence-T5 → RK-Means → encoder-decoder**
pipeline. It checks the infrastructure and establishes a functional reference.
It is a TIGER-style hybrid, **not an exact reproduction of a published score**:
TIGER uses RQ-VAE, GRID's reported baseline uses Flan-T5, and their released
example configurations also differ from the papers. The
[framework audit](gr-framework-audit.md) records those differences.

The research question and subsequent controlled comparisons are in the
[brief](gr-research-brief.md). No new representation model is implemented yet.

## Components

| Stage | Registered implementation |
| --- | --- |
| Data | Public GRID/P5 Amazon **2014** Beauty, 22,363 users, 12,101 items, 198,502 interactions |
| Text | Exact serialized title/brand/categories/price text from the supplied catalogue; no ID or interaction statistics added |
| Representation | Complete `sentence-transformers/sentence-t5-base`, revision `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`, max256 tokens; encoder → masked mean → learned Dense → normalization; 768D |
| Quantization | Actual GRID `MiniBatchKMeans`, 3 levels ×256 codes, normalize each incoming residual, k-means++ initialization, 10 complete catalogue passes per level |
| Disambiguation | Mandatory fourth token, stable ordinal 0…group_size−1, including0 for unique items; fail if suffix exceeds256-wide head |
| Generator | Actual GRID `SemanticIDEncoderDecoder` with HF T5 blocks, 4 encoder/4 decoder layers, d128, FF1024, 6 heads ×64 key/value dimensions, dropout.1 |
| Context | Latest20 observed items; 2,000 user bins using GRID remainder hashing; no separator or GRID additional FF hidden layer |
| Search | Upstream free-form beam20; four-token next item; exact SID→Item ID decoding; invalid candidates retain original miss positions |
| Training | Seed42, Adam lr.001/wd.0001, effective batch256/physical128, FP16 with standard dynamic loss scaling, clip1 |
| Selection | Validation item NDCG@10 every5epochs and epoch1, patience5 checks, minimum20epochs, maximum100epochs or7200training seconds |
| Test | Exactly once, after loading the validation-selected checkpoint; Recall/NDCG@5/10 over the full catalogue; no sampled negatives or seen-item filtering |

These dimensions follow TIGER's published description; the measured parameter
count is authoritative. In particular, 6 heads with d_kv64 do not imply
`d_model/num_heads`. GRID's extra `1024→1024` FF layer is intentionally disabled.
The optimizer and bounded training budget are a practical initial recipe; this
run does not match TIGER's 200k-update training schedule. Multiple seeds and
convergence checks are later requirements for research conclusions.

## Training and split contract

The prepared dataset preserves **all153,776 training events**. The last two
events per user are validation and test. GRID's supplied validation/test records
are clipped to20prior items, while training histories are full length. The
preparer checks their alignment for every user and retains full train histories.

Each epoch enumerates all **131,413 noninitial training positions exactly once**,
shuffled by seed+epoch. For target `train[t]`, input is
`train[max(0,t−20):t]`. Each example predicts the four SID tokens in one causal
teacher-forced decoder pass. Item targets are independent batch examples; this
is different from GRID's unbounded augmented-subsequence stream and the earlier
Yambda block training protocol.

Validation input is `train[-20:]`; test input is `(train+[validation])[-20:]`.
The respective target is absent from input. Static catalogue metadata and the
quantizer may include all catalogue items; no held-out interactions are used to
train them. This is a transductive catalogue protocol, not a cold-start result.
Chronological ordering is inherited from the GRID artifact; timestamps are not
provided in these records and are not independently reconstructed.

SID uniqueness is asserted. The independent evaluator maps every complete
predicted tuple to an Item ID, so predicting a different item with the same
three semantic codes is a miss. Invalid and duplicate candidates are recorded
and never repaired by inserting the target or dropping preceding rank slots.
The upstream constrained-search option is rejected because its first-step mask
bug was reproduced; no search algorithm is silently substituted.

## Reproduction and use

Configuration: [`amazon_beauty_gr_baseline.json`](../experiments/amazon_beauty_gr_baseline.json).
Pinned GRID revision: `2fe3475b2d369580234093f35d52b1a2f54d0472`.
The full Python source tree is hashed before import. Its only patch postpones a
type-only TFRecord iterator import, avoiding TensorFlow for array inputs. The
generator and centroid-update algorithms are upstream code. Source redistribution
retains GRID's noncommercial research license and third-party notices.

The private input dataset is `carasik/rhpr-gr-beauty-phase0`, with a configuration-
pinned resource manifest. It contains data, complete Sentence-T5 weights, an
opaque source archive, and45small dependency wheels (~380MB total). No data or
weights are committed to Git. The old multi-GB Mamba runtime is not mounted.
The GR bootstrap installs a hashed dependency lock offline and reuses the
platform PyTorch/CUDA. CPython3.13 dependency-wheel imports were checked locally;
actual CUDA imports and FP16 execution are checked by the Kaggle run itself.

To prepare from the public dataset on CPU:

```bash
python scripts/prepare_gr_beauty_data.py \
  --download /workspace/scratch/beauty-tfrecords \
  --output /workspace/scratch/beauty-normalized
```

Use `--help` for an existing-TFRecord input path. The preparation verifies
pinned archive-directory and normalized-data hashes. Unknown source embeddings
inside the downloaded archive are deliberately ignored.

For source/model/wheel inputs already downloaded at their pinned revisions:

```bash
python scripts/prepare_gr_resources.py \
  --grid /workspace/scratch/GRID \
  --data /workspace/scratch/gr-beauty-data/normalized-verified \
  --sentence-model /workspace/scratch/gr-resources/sentence-t5-base \
  --wheels /workspace/scratch/gr-resources/wheels \
  --output /workspace/scratch/new-gr-input
```

The builder performs no upload or GPU work. Rebuilt dependency wheels may change
bytes; register a new resource manifest rather than bypassing a hash mismatch.
The committed [dependency lock](../runtime/gr/requirements-cp313-lock.txt),
[wheel manifest](../runtime/gr/runtime-manifest.json), and
[Sentence-T5 inventory](../runtime/gr/sentence-t5-manifest.json) record the inputs.

Local development: `pip install -e '.[gr]'`. Core source integration tests require
`RHPR_GRID_SOURCE` pointing to the pinned public checkout:

```bash
RHPR_GRID_SOURCE=/workspace/scratch/GRID \
  python -m unittest discover -s tests -p 'test_gr_*.py' -v
```

The standard repository runner and private Kaggle workflow remain the entry
points. GPU execution must be explicitly requested; the present user task
authorizes this initial baseline, not subsequent searches.

## Extension points and outputs

Future encoders can provide a catalogue-ordered `.npy` matrix through
`representation.kind=external_embeddings`, with explicit artifact and catalogue
hashes and an objective recorded in the config. This seam does not transmit NIP
gradients through fixed RK-Means. A new encoder needs its own declared objective.
Do not change the generator, quantizer, data, or metadata availability while
attributing a gain to item representations.

Required outputs are `metrics.json`, `summary.md`, and `run-manifest.json`.
Additional artifacts include source/input provenance, representation metadata and
embeddings, centroids/SIDs, code usage/entropy and collisions, generator parameter
count, partially profiled forward FLOPs (explicit operation-count convention),
training/validation curves, selection reason, a best checkpoint and last-epoch
optimizer/scaler checkpoint, and test predictions for independent metric checks.
Full private checkpoints and predictions stay on Kaggle; do not commit them.

Encoding, quantization, training, and evaluation time are separated. Normalized-
residual MSE is not reconstruction error in the original embedding space.
The training budget exits at an epoch boundary, so one running epoch plus final
evaluation can extend beyond7200seconds. An epoch limit or time limit is not
evidence of convergence.
