# Smaller mini-TIGER calibration

Authorized 2026-11-06 after the fixed seed202 transfer. This is a calibration
of ranking frozen SID indexes, not a claim of general tokenizer quality.

Two new architectures keep four encoder and four decoder layers, d_model64:

| Shape | d_ff | heads | d_kv | Parameters |
|---|---:|---:|---:|---:|
| ffn_small | 256 | 6 | 32 | 1,112,896 |
| attention_small | 256 | 4 | 16 | 719,552 |

Existing narrow baseline: d_ff512,6heads,d_kv32,1,375,040 parameters. Reuse its
saved results; no baseline retraining. Shape comparisons are exploratory and
use the same observed index family. This is not an independent holdout study.

Cross both shapes with twelve existing SUM/UW/GradNorm/PCGrad indexes at
pipeline seeds42/101/202 and independent generator seeds31415/27182:48freshfits.
All initializations are pinned with native Transformers5.22.1. One generator
seed per T4 worker;24fits each. Same raw clean ML1M cases,151166TRAINwindows,
2786VALusers,FP16training/FP32eval,nativeSDPA,512batch,Adam.001,clip1,
10epochs,2960updates,VAL5and10,noES/scheduler. Primary proxy score is fixed
**epoch10 VAL NDCG@10**, averaged over two generator seeds, not best-of10.

Target: saved full-TIGER fixedepoch30 VAL NDCG@10;best-of30 is secondary.
Report Spearman,top2overlap,selected-target regret,individual seed scores,
overall12indexes,within each pipeline seed,and separately SUM and each other
method across3indexes. Three-index correlations are weak evidence. Epoch5
is diagnostic; do not select a more flattering epoch after inspection.
Retain per-user VAL beam predictions for independent replay and conditional
shared-user bootstrap. Measure training epoch and total-fit wall time; distinguish
per-fit speed from cost of averaging two seeds. Compare only matched indexes
and epochs from saved narrow runs. Pipeline seed and generator seed in the
full references are coupled, a limitation of existing targets.

Only reuse existing private cleanMTL/replicates notebook outputs and three
existing datasets. No new data upload,quantizer/content/full-model training,
TEST inference,Telegram,retries,extra seeds,epoch extensions or hardware fallback.
Private kernel carasik/rhpr-ml1m-mini-width on2T4,offline. Internal deadline9900s
checked before each fit;server timeout180min,quota checked3.52h remaining.
Expected roughly1.5–2.5h,not a measured guarantee. Incomplete matrix must fail
and remain explicitly partial. No automatic resubmission.

Configuration: experiments/ml1m_mini_tiger_width_2t4.json.
