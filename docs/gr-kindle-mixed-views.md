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

## Timeout and epoch30 recovery

The first submission was cancelled by its1800-second worker limit. The agent
incorrectly dispatched `timeout_minutes=30`, treating it as a CI wait limit;
`scripts/kaggle_job.py` passes it to `kaggle kernels push --timeout` even in
submit-only mode. The workflow input description now explicitly calls it the
worker hard limit including startup and compilation. This was an operational
mistake, not a numerical training failure or early stopping.

Thirty complete epochs/8280 successful updates/zero AMP skips are preserved.
The last logged epoch completed at1798.651seconds. Both last and best full
checkpoints survived; best is epoch30, VAL NDCG@10=0.01325165130971416.
Six saved validation predictions independently reproduce their metrics.
There is no final TEST, root metrics or completion manifest.

Recovery uses `experiments/kindle_mixed_views_resume.json`, a separate private
kernel `carasik/rhpr-kindle-mixed-resume`, and **timeout_minutes=180**. Continue
the same fit at epoch31 with unchanged min40/max60 and stopping/selection
protocol. No new generator seed, index, targets or optimizer policy. Restore
model, Adam, scaler, LR step, EMA and CPU/CUDA/Python/NumPy RNG; the view RNG is
epoch-addressable. Pin15 cancelled-parent files including best/last, six
predictions, original config and the three index arrays. The cancelled parent
lacks `run-manifest.json`; explicit byte pins and original config checks replace
the completed-parent locator only for this source. The completed long-pair
parent remains separately pinned and verified.

The first recovery submission (kernel `rhpr-kindle-mixed-resume/1`, source
38fc463) failed before launching any training worker: Kaggle silently omitted
the cancelled kernel from mounted sources. Its logs and metadata were preserved.
Checkpoint bytes are now additionally stored in the private ready dataset
`carasik/rhpr-kindle-mixed-epoch30-resume/1` (dataset13352852). The corrected
recovery mounts this dataset plus the completed long-pair parent, not the
cancelled kernel. The archive is73,708,688bytes, SHA256
`4e502b6c3063205a8c10bc9cbdc12e0527033835e98ceeeb034842a639d4ed47`.
Its15 members and each artifact hash are verified before restoration. No weights
were changed or retrained when copying this private archive.
Corrected archive-mount regression:1119tests,152optional skips,0failures.

Preflight verifies actual native GRID model/optimizer restoration and the
original30 view schedules. A dropout/Adam/scheduler CPU test verifies bitwise
equivalence of resumed and continuous training. CUDA compilation/fused kernels
can still prevent bitwise equivalence across sessions. Original runs are
preserved as failed history; do not call this a fresh independent repeat.
Recovery regression:1118tests,152optional skips,0failures.

Evidence: `gr-kindle-mixed-views-cancelled.json` and
`gr-kindle-mixed-resume-preflight.json`. Raw weights/predictions/logs are in
`/workspace/artifacts/kindle-pilot/mixed-views-results` and the original private
Kaggle version. The original submission receipt is historical, not live status.
