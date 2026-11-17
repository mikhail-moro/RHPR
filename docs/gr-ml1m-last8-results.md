# ML1M Last-8: five ordinary TIGER context windows

All five registered windows8/16/32/64/128 have completed for seed42.
Windows8/32/128 are retained from the stopped original kernel/2; only the
missing16/64 were trained in the remaining kernel/1. Window256 is excluded
following the user's stop. No extra GPU training or inference was performed
to collect and analyze the results.

The128-window configuration improves both VAL and TEST. Compared with8,
VAL NDCG@10 rises28.96% and TEST rises19.02%; compared with64, TEST rises17.77%.
The relationship is not monotonic:16/32 are below8, and64 is close to8 on TEST.

| Window | VAL NDCG@10 | TEST NDCG@10 | TEST Recall@10 | B1 forward GFLOPs | Warm train epoch, s | Best / stop epoch |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.05199936 | 0.04578558 | 0.09453642 | 0.177736 | 6.75 | 35 / 60 |
| 16 | 0.05084036 | 0.04050114 | 0.08460265 | 0.340823 | 10.66 | 35 / 55 |
| 32 | 0.04988699 | 0.04138927 | 0.08774834 | 0.704770 | 21.47 | 40 / 60 |
| 64 | 0.05538826 | 0.04627123 | 0.09635762 | 1.583759 | 46.05 | 40 / 60 |
| 128 | 0.06705911 | 0.05449579 | 0.11158940 | 3.946108 | 124.94 | 35 / 70 |

FLOPs are supported CPU math-SDPA batch1 teacher-forced four-code forward
operations, FMA=2, at fixed padded width. They exclude backward, beam search,
the text encoder and quantizer. They are not serving latency. The128 window
costs2.492times64 and22.202times8 by this measure. Physical batch sizes and
evaluation batches differ by window, so evaluation wall times are not a
controlled serving benchmark. Encoder lengths are33/65/129/257/513positions.

New16/64 session duration was3643.05s (60.72minutes). Training plus validation
per fitted model was10.30/14.12/29.01/57.61/174.62minutes respectively, excluding
the initial index/profile setup and final TEST. Window128's warm train epoch
is about125s;256 was about367s and was stopped at32 completed epochs.

## Matched protocol and artifact audit

All6040users,3706rated films,1,000,209ratings,48,320fixed Last-8 TRAIN examples,
and6040VAL/TEST queries are shared. No user/positive-rating/external metadata
filters. Ordering is timestamp then movieID; ties retain that artificial order.
The latest TRAIN-eight targets, VAL, TEST, and their causal context anchors are
identical across models. The chosen maximum window alone crops the past.

Frozen Sentence-T5 title(year)+genres and residual KMeans3x256+collision suffix
are shared. Recomputed vectors, centroids and SID arrays from the two runs
are byte-identical. SID-file SHA256:
`80422aa36fbcabbd0c1133a0695547513c1a795f908c72b4ed412e9cc2c2b672`.
All five start with the same4,977,920parameter initialization. Architecture:
ordinary GRID TIGER4+4/d128/FF1024/H6/dkv64/dropout.1/USER2000/noSEP;
this is the current RQ-KMeans pipeline, not an exact original TIGER reproduction.

FP16,SDPA,Inductor with CUDA Graphs disabled,effective512,Adam.003to.001,
two-epoch warmup/60-epoch cosine,VALevery5,rawVALNDCGcheckpoint selection,
EMAalpha.5/min_delta.0005/patience20,minimum40/maximum120epochs are unchanged.
All five stop by EMA at55–70epochs; none is time/epoch capped, all have zero
AMP skipped updates. The best raw validation epochs are35–40. Validation CE
bottoms at20/25/25/30/35epochs and rises afterward; simply extending these same
trajectories is not supported by these curves. EMA stopping is not proof of
global convergence. TEST is evaluated once per selected checkpoint,
FP32/beam20/full catalogue/no seen-item filtering.

`scripts/research/analyze_ml1m_last8.py` checks shared cases/model/index,
optimizer-step/LR/EMA histories, checkpoint selection and resume states,
finite weights and optimizer tensors. It replays ALL61validation plus5TEST
prediction files:66files,7,972,800paths,4157invalid paths,0duplicates. Invalid
paths retain rank slots. All scalar ranking metrics reproduce from saved
paths. All10best/last checkpoints pass checks; no new forward passes.

Source kernels:

- `carasik/rhpr-ml1m-last8-windows/2`, stopped; source
  `5c1af46bddb1a848877acead858ec855e77201ed`, three finished models retained.
- `carasik/rhpr-ml1m-last8-remaining/1`, COMPLETE; source
  `eceafc4f9217e8a2f79c6fc2686936c80d542487`, two finished models.

## Conditional uncertainty and exploratory strata

Paired user bootstrap,10,000draws, conditional on the fixed fitted models and
selected checkpoints; it does not estimate seed variation, reselect checkpoints,
cluster items, or correct multiple comparisons:

| TEST comparison | NDCG absolute difference | Conditional95% CI |
|---|---:|---:|
| 128 minus8 | +0.008710 | [0.004473,0.012913] |
| 128 minus64 | +0.008225 | [0.003914,0.012496] |
| 64 minus8 | +0.000486 | [-0.003855,0.004799] |

On2333users with strict TRAIN-to-VAL and VAL-to-TEST timestamp boundaries,
TEST8/64/128 is0.023398/0.024287/0.031630. For128minus8 the difference is
+0.008232, CI[0.002935,0.013702]. Thus the observed gain is not confined to
users whose heldout boundaries share a timestamp. This post-hoc restriction
does not repair tied events inside TRAIN or establish true viewing order.

Strata use the number of past events before the TEST target, before window
cropping. All groups are exploratory:

| Available past events | Users | TEST8 | TEST64 | TEST128 |
|---|---:|---:|---:|---:|
| below32 | 915 | 0.060077 | 0.065275 | 0.083958 |
| 32–63 | 1354 | 0.059760 | 0.062282 | 0.066806 |
| 64–127 | 1393 | 0.046012 | 0.044414 | 0.051547 |
| at least128 | 2378 | 0.032197 | 0.030931 | 0.037877 |

For the2378longest histories,128vs64 is+22.46%, absolute+0.006947,
conditional CI[0.001122,0.012928]. Nevertheless128also wins on short histories
where64and128 contain the same observed items. The models were trained on
different windows across the whole population, and padding/USER relative
positions plus microbatch/dropout streams differ. These comparisons therefore
establish a better observed training configuration, not a causal demonstration
that tokens beyond64 produced the entire gain.

## Implication and saved artifacts

ML1M Last-8 with the128-window model is a useful candidate baseline for further
compression work: it has a visible quality advantage over shorter controls,
and its forward cost leaves room for savings. It is still one tokenizer and
generator seed. The16/32dips and short-history gains need not be assigned a
mechanistic explanation from this sweep.

A cheap next diagnostic, if authorized, is to evaluate the existing128checkpoint
with older events masked while keeping the128-slot layout and USER position.
This would test reliance on the old history without training new models.
Nothing further was launched during collection.

Machine-readable results: `gr-ml1m-last8-results.json`. Detailed artifacts in
`/tmp/ml1m-last8-analysis`: model-metrics.csv, learning-curves.csv, selected
user ranks, analysis.json and length-quality-compute.png/.pdf. Raw outputs and
all checkpoints remain in the two private Kaggle kernels and local download
directories recorded in `gr-ml1m-last8-completion.json`.
