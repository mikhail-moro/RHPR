# ML1M Last-8: fixed Pair128 merge-prefix pilot

Authorized2026-11-18(Moscow): four fresh generator fits on the existing
ML1M Last-8 data/index, seed42, merging64/96/112/120 old events inside a
128-item window. This is a quick40-epoch screen, VALevery5, with full resume
checkpoints. No new control fits, extra seeds, random tokenization,128-history
ablation,256-window fit, or ML10M experiment is included.

Registration: `experiments/ml1m_last8_pair128_2t4.json`.
Runner: `rhpr.gr.ml1m_last8_pairs`; intended private/offline two-T4 kernel
`carasik/rhpr-ml1m-pair128-screen`. Use the same three existing raw/runtime
datasets. No new datasets, parent mounts or cached artifact uploads.

## Compression and shared pair index

Within the latest128 valid events, protect the latest64/32/16/8 respectively.
From the oldest valid event, combine adjacent pairs before that protected
tail. Preserve an odd leftover as a single immediately before the tail. A
short history protects the same tail length and compresses only its older
excess; padding is never paired and no valid event is dropped. Train, VAL and
TEST use this fixed rule. Pair alignment follows the current valid window,
so moving to a later anchor can change pairs. This is the fixed Pair pilot,
not the previous Random-training augmentation.

Every single AND every pair has3semantic codes plus1collision suffix.
Separate four-level input namespaces distinguish pair tokens from singles;
all targets and decoder outputs remain single-item3+1SIDs.

| Maximum old events merged | Protected tail | Full-window pairs + singles | Encoder positions incl.USER | Supported B1 forward GFLOPs | Reduction vs ordinary128 |
|---|---:|---:|---:|---:|---:|
| 64 | 64 | 32+64 | 385 | 2.664203 | 32.49% |
| 96 | 32 | 48+32 | 321 | 2.098798 | 46.81% |
| 112 | 16 | 56+16 | 289 | 1.834982 | 53.50% |
| 120 | 8 | 60+8 | 273 | 1.707796 | 56.72% |

Ordinary128 has513positions/3.946108GFLOPs. These are supported CPU mathSDPA
teacher-forced4code forward operations, FMA=2, at fixed padded width; excludes
beam search, backward, text embedding and quantization. GPU throughput and
beam timings are separately saved; no serving-latency claim is implied.

The original all-rating ML1M catalogue/users/targets are unchanged:
6040users,3706items,48,320fixed TRAIN cases,6040VAL/TEST. Raw preparation keeps
the original256-slot arrays and hash pins; models read only their latest128.
The native title(year)+genres Sentence-T5 single embeddings use original
FP32/batch32/max256 settings. Recompute RQ-KMeans and require the original SID
file hash BEFORE computing the larger pair cache or training generators:
`80422aa36fbcabbd0c1133a0695547513c1a795f908c72b4ed412e9cc2c2b672`.

Pair template:
`First movie: {native title+genres text} Second movie: {native title+genres text}`.
Sentence-T5-base FP32/max2048/batch128; output cached once as768-dimensional
vectors for257184ordered pairs,790069248bytes. TRAIN union252253pairs,
VAL163691,TEST164627; the most compressed variant covers all other variants'
fixed pair keys. One shared normalized residual RQ-KMeans3x256, same seed42,
10passes/level, is fit on the union of TRAIN pair representations only.
VAL/TEST pairs get frozen-centroid assignments; suffix ordinals are allocated
TRAINfirst, then newVAL, then newTEST, each in ascending pair-key order.
No heldout target supervision enters fitting. Preserve raw assignments,
centroids and collision diagnostics even if suffix capacity256 is exceeded;
do not silently remove suffixes, merge identities or change vocabulary.

## Model and matched screening budget

Ordinary TIGER4+4/d128/FF1024/H6/dkv64/dropout.1/USER2000/noSEP. Preserve
every original parameter value, verified before and after the audited
`pair_unique4.attach4` adapter. The only added parameters are the separate
1024x128pair-input table:131072parameters, increasing total4,977,920to5,108,992
(2.63%). This is an explicit parameter-count difference from the historical
ordinary control; the Transformer and output heads do not change.

Common base initializer:
`d57fc2e783c0a04bfc567c0082541f757e241d1c53d7e79b3e8f281bf2eb9337`.
Four pair models share full initializer:
`34e7397ed277c78befd6254cbb736794df4907a011ed032bbd0735d9b7272286`.

Exactly40epochs,95attempted updates each, effective512/physical64 for every
variant. Keep FP16,SDPA,Inductor with CUDA Graphs disabled,Adam.003/wd1e-4/
clip1,two-epoch warmup and original60-epoch cosine floor.001. The scheduler
is truncated, not accelerated. Keep min40/EMA settings; all fits reach40
unless an operational failure/time cap intervenes. VALat5,10,...40, raw item
NDCG@10 checkpoint selection. A40-epoch endpoint is a screen, not a claim of
plateau. Save best and last with optimizer/scaler/scheduler/CPU+CUDA+Python+
NumPy RNG and curves so promising variants can be resumed later.

The existing ordinary128 control's best epoch35 is ALSO the best in its
first40epochs. Reuse that checkpoint's already saved TEST metrics and the
VALcurve limited to5..40; do not train or reevaluate the control. Its matched
reference is VAL.06705910548/TEST.05449578525. The exact snapshot/curve/hash
is embedded in `reference_control` and written to output. Restrict screening
comparisons to this common first40-epoch budget, despite the control later
having stopped at70. All four pair models choose their checkpoint using VAL;
TEST is reported once per selected checkpoint, FP32/beam20/evaluationbatch16,
full3706-item catalogue/no seen-item filtering. No TEST-based checkpoint or
hyperparameter selection.

Microbatch sizes differ from the old ordinary128physical32, and the encoder
layout changes USER-relative positions. Combined with one seed, these limit
causal claims. The pilot tests practical quality/compute tradeoffs, not a
unique explanation of any result. Frozen single/pair indices, pair alignments
and actual merged counts are retained for diagnosis.

## Preflight, measurements and limits

CPU preflight: `scripts/research/preflight_ml1m_last8_pairs.py`, report
`gr-ml1m-last8-pair128-preflight.json`. Real single SIDs and synthetic unique
pair SIDs test every full-window shape, finite forward/backward, nonzero
pair-table gradients, beam20, shared base parameters, actual pair-key counts,
and native AOTAutograd accumulation/profiling with state restoration. This
does not claim a locally fitted pair quantizer or local GPU verification.

Unit tests exhaust every valid length0..128 and all four merge caps, reversing
the tokenization back to exactly the original ordered events, and check odd
prefixes, protected tails, namespaces/suffixes, missing-key rejection, target
SIDs, registered budgets, and pair embedding batches.

Each T4 runs two independent fits, assigned round-robin. Before each long fit,
run one warmup and two measured effective512 updates of the actual compiled
path, restoring weights and RNG; no automatic eager/LR/precision fallback.
Save fixed-width FLOPs, first/warm epoch times, memory, all VAL prediction
paths and selected TEST paths. Limit each fit to7200s and the Kaggle session
to4h including preparation/compilation. Quota before dispatch19.22GPUhours.
Pair embeddings are a one-time preparation cost in this session and are
saved for future authorized use; new runs must not be silently dispatched.

## Mixed128 comparison

The separate [Mixed128 kernel](https://www.kaggle.com/code/carasik/rhpr-ml1m-mixed128-screen)
uses the same four merge prefixes, seed 42 and 40-epoch budget. For each
optimizer batch, it draws either the full SingleSID history or the fixed
PairSID view with probability 0.5 each. All microbatches in an update
receive the same view. The previous Mixed32 RNG stream is seeded by epoch
and shared across all four variants. Every training example contributes
its existing target once per epoch.

Full batches use 513 encoder positions, including USER. Pair batches
use the widths in the table. VAL every five epochs and TEST on the
selected checkpoint use the fixed PairSID view. The training curves
record batch and example counts for each view. A separate training
profile records the cost of full batches at the same physical batch
size as compressed batches.

The parallel session builds its own copy of the index from the same raw
data and pair fitting settings. Compare the saved single and pair SID
file hashes before attributing differences between sessions to mixing.
The config is `experiments/ml1m_last8_mixed128_2t4.json`.

## Mixed120 seed replication

Mixed120 led the seed 42 compressed variants on validation and test.
The follow-up assigns one T4 each to generator seeds 101 and 202 for
40 epochs with five-epoch validation checks. Each seed drives weight
initialization, dropout, shuffle and the 50/50 batch schedule. The original
seed 42 tokenizer and Last-8 cases are loaded from the completed Mixed128
kernel and verified against recorded hashes. The 60-pair, eight-single
layout still costs 1.708 GFLOPs at evaluation. The seed 42 Mixed120 and
Recent128 scores are retained as references.

The config is `experiments/ml1m_last8_mixed120_seeds_2t4.json`.
The private kernel uses a two-hour session cap, including startup.

The matched controls run Recent64 and Recent128 at seeds 101 and 202.
Recent64 is close to Mixed120 in forward cost (1.584 vs 1.708 GFLOPs),
and Recent128 sees the full 128-item context. The controls reproduce
the original window-specific microbatches (64 and 32 respectively),
with an effective batch of 512 for both. The base initialization,
Last-8 cases, single-item index, 40 epochs and selection rule match
the corresponding Mixed120 fit. The config is
`experiments/ml1m_last8_mixed120_controls_2t4.json`.
