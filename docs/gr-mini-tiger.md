# Frozen-index mini-TIGER calibration

The experiment asks whether small generators preserve full TIGER's preferences
among plausible multimodal SID indexes. No tokenizer, content encoder, or full
generator is retrained. It is a pilot, not evidence of a universal proxy.

## Predeclared matrix

- Eight completed indexes: SUM/UW/GradNorm/PCGrad, tokenizer seeds42/101.
- Full references are the existing matching 4+4,d128 generators, trained30epochs.
  Each index has one full-generator seed, coupled to its original tokenizer seed.
- Three fresh proxies: shallow1+1 with original width (1,633,664parameters),
  narrow4+4,d64,d_ff512,6heads,d_kv32 (1,375,040), and compact2+2,d128,
  d_ff256,4heads,d_kv32 (1,175,424). Full TIGER has4,977,920parameters.
- Cross every index and architecture with generator seeds31415/27182, distinct
  from original full-model seeds.48fresh fits; indexes remain fixed byte-for-byte.
- Each fit20epochs, VAL every5, no early stopping/scheduler/LR or precision retry.
  Same151166TRAIN windows,2786VALusers,2874items,window20,FP16training,
  FP32evaluation,nativeHF5.22.1SDPA,512/512batches,AdamLR.001,wd.0001,
  dropout.1,clip1,2000userhashbins,3x256SID+originalsuffix,freebeam20.
- Two isolated T4workers,one generator seed per worker.3hserver timeout;
  internal deadline prevents starting new fits after10000seconds. Incomplete
  matrices fail visibly and are not reported as completed calibration.

## Inputs and egress

Existing private notebook outputs `carasik/rhpr-ml1m-clean-mtl` and
`carasik/rhpr-ml1m-clean-mtl-replicates` are mounted as kernel_sources.
Their manifests,plans,SID files,curves,selections and generator specs are pinned.
Only selected references are copied inside the Kaggle session. No new dataset
is uploaded. An attempted local dataset publication was rejected by automatic
approval review; that action was abandoned. Reuse reads artifacts already stored
at Kaggle and avoids uploading that research-data payload.

The three dataset_sources are existing Beauty runtime/GRID assets, existing clean
ML1M cases, and existing modern T5 runtime. No feature preparation is needed.
The server verifies resource integrity, then parses TRAIN/VAL cases only.
TEST cases may be hashed by generic manifest verification, but are never parsed,
scored, selected on, or exported as calibration references. No TEST metric is used.

## Analysis and limits

Primary target: fixed epoch30 full-generator VAL NDCG@10. Best-of30VAL is a
secondary, selection-dependent comparison. Proxy points are epochs5/10/15/20.
Compare Spearman,top2overlap,and target-quality regret for the proxy-selected
index, for each mini seed and the two-seed average. Also compute the same
statistics for early checkpoints of the original full model.

Report all eight indexes and each four-method tokenizer-seed group separately.
Epochs and generator seeds are repeated measurements, not additional index
samples. Only eight related indexes and one full-generator seed per index are
available. This cannot establish seed-robust target rankings or significance.
Any later optimization guided by this proxy needs separate transfer validation;
fixed-index correlation does not demonstrate reliable gradients or joint learning.

Keep raw per-user VAL predictions at all four checks,training/validation CE
curves,selected/last generator checkpoints,initialization hashes,actual timings,
frozen references,worker logs,full configuration and source revision. Quantizers
are not included in fit timing. Existing full-model validation users are reused;
this is exploratory calibration and cannot serve as a fresh final evaluation.

## Preflight

Native HF5.22.1 CPU probes with actual three mini architectures each completed
20epochs/4VALchecks on a tiny synthetic case fixture,checked initialization pins,
and produced no TEST predictions. CPU timings are not T4speed measurements.
Mounted-reference staging replayed all eight saved indexes and six full-VAL
checks per index exactly. Full repository regression and configuration/bundle
checks are recorded in `gr-mini-tiger-preflight.json` before deployment.
