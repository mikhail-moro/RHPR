# Full32 / Pair32 training mixture

Authorized 2026-11-15: test random training formats, with the existing Pair32
representation at inference. Random mixtures of individual pair/single SIDs
within a history remain a separate, unlaunched follow-up.

Exactly one new generator fit, seed42. Reuse the completed long-pair kernel's
frozen single and pair 3+1 indexes, original Pair32 and Recent32 predictions,
curves and selection records. No text embedding or quantizer fits. All44 parent
files and10 input files are byte-pinned. One worker on T4; no duplicate fits to
fill the second GPU supplied by Kaggle's two-T4 machine shape.

For each optimizer batch independently choose Full32 with probability0.5,
otherwise Pair32. Full32 retains all32 individual items; Pair32 compresses the
older16 into8 pair SIDs and retains the recent16 singles. Short training windows
use the unchanged existing pairing/padding policy. Keep native128/96 SID
positions plus USER, respectively: never pad the Pair view to the Full width.
Two compiled lengths (and last-batch shapes) can incur extra cold compilation.

Every original141137 TRAIN prefix target is used exactly once per epoch, in the
same seed/epoch permutation as the previous run. No doubled examples, dual
forward or consistency loss. A separate epoch-addressable NumPy RNG stream7301
chooses formats without consuming the original shuffle, dropout or evaluation
RNG streams. Actual per-format batch/example counts, schedule hashes and an
exactly-once example coverage audit are saved per epoch. This is augmentation
of input representations, not a proof that extra single-code supervision is
the mechanism. Pair-only training exposure is also reduced by the mixture.

All other generator/training settings match the completed Pair32: wide4+4,
d128/FF1024/dkv64/H6, FP16/SDPA/compile, train512, Adam LR0.003→0.001,
warmup2/cosine60, min40/max60epochs, validation every5, EMA score stopping
alpha0.5/delta0.0005/patience15. Raw VAL NDCG@10 selects the checkpoint.
VAL and final TEST use only Pair32, FP32/beam20/batch64, no seen-item filter.
TEST is evaluated once after selection; no settings or checkpoint choices use
TEST. This cohort's TEST has already been examined, so results are exploratory.

Keep original data, user/target/case hashes, initializer, indexes, all VAL
predictions, final TEST predictions, full best/last checkpoints, curves,
FLOPs, timing and peak memory. Compare selected and common fixed epochs with
the original Pair32 and Recent32. One examined pipeline seed does not establish
robust improvement; training time may increase while inference cost remains
the existing Pair32 cost (0.516500848 supported forward GFLOPs at B1).

Config: `experiments/kindle_mixed_views.json`; module `rhpr.gr.kindle_mixed_views`.
Private/offline kernel: `carasik/rhpr-kindle-mixed-views`. No additional seeds,
within-history mixtures, tuning, automatic retries or notifications authorized.

Preflight: all44 parent and10 input hashes verified; both native forward/
backward paths finite; all60 proposed epoch schedules cover original TRAIN
targets exactly once; initializer and Pair32 inference FLOPs match the completed
control. Full regression:1116tests,152optional skips,0failures. Embedded source
and exact configuration/private/offline T4 metadata verified before submission.
CPU evidence: `docs/gr-kindle-mixed-views-preflight.json`.
