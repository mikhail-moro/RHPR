# Random SID reassignment: completed fixed-budget screen

Private Kaggle session `carasik/rhpr-sid-corruption-screen`, executable commit
`1e87396aa65fd8b2dc9259a650e501cec25eec35`. Five fresh TIGER fits, identical
neural initialization and data/order seed42, one permutation seed2753,20epochs.
All catalogue prefix occupancies and full SID paths are fixed. Complete item
paths are reassigned within nested random item subsets; TRAIN-weighted token
frequencies and item associations are allowed to change.

## Primary endpoint: validation at20epochs

| Reassigned items | NDCG@10 | Change vs0% | Recall@10 | Hits /22363 |
|---|---:|---:|---:|---:|
|0%|0.03644569|—|0.07105487|1589|
|5%|0.03362916|−7.73%|0.06600188|1476|
|20%|0.02924565|−19.76%|0.05705853|1276|
|50%|0.02174110|−40.35%|0.04261503|953|
|100%|0.01513320|−58.48%|0.03027322|677|

The endpoint is monotonically worse as more assignments are randomized.
This supports a substantial benefit from the original item grouping at this
training budget, beyond merely preserving catalogue trie shape and occupancy.
It does not isolate pure textual semantics from all frequency/behavioral effects
of reassignment. The result also does not imply that a particular ALS swap
heuristic can improve an already useful original grouping.

| Reassigned | Absolute NDCG delta | Paired-user95% interval | Target-cluster95% interval |
|---|---:|---|---|
|5%|−0.00281653|[−0.00419536,−0.00148336]|[−0.00504566,−0.00071118]|
|20%|−0.00720003|[−0.00879524,−0.00570572]|[−0.01013758,−0.00434492]|
|50%|−0.01470458|[−0.01641953,−0.01295926]|[−0.01842320,−0.01112003]|
|100%|−0.02131249|[−0.02315242,−0.01953081]|[−0.02540270,−0.01702635]|

All intervals exclude zero conditional on these fitted models. They use2000
paired resamples,seed2754,are unadjusted across four comparisons,and exclude
training/permutation-seed uncertainty. The same users and nested random sets
appear across conditions; these are not independent replications.

## Generator likelihood and search

| Reassigned | Teacher-forced total NLL | First-token NLL | First-token top1 |
|---|---:|---:|---:|
|0%|8.77067|4.46815|11.61%|
|5%|8.88154|4.57712|10.30%|
|20%|9.00543|4.76901|7.98%|
|50%|9.34242|5.04149|4.53%|
|100%|9.43382|5.17095|1.78%|

These NLLs are in nats per example; the total sums the three semantic levels
and uniqueness suffix. Each head receives the true preceding target tokens
and retains its full256-token vocabulary. The first token has no preceding
target token, so its decline directly concerns predicting the first branch
from the history. Optimistic ranks handle exact ties as documented in the run.

The first-token NLL increase is0.70280nats, while total NLL increases0.66314.
The later levels therefore do not contribute a net worsening under supplied
true prefixes. At0%/100%, their NLLs are respectively3.32235/3.36980,
0.86526/0.80363,and0.11492/0.08944. Teacher-forced prediction also deteriorates,
so beam pruning alone cannot explain the retrieval loss. This does not rule out
an additional beam-width effect. Invalid final beams decrease from333of447260
at0% to0at100%; malformed output paths are not driving the observed degradation.

## Structure, frequencies, and training duration

Weighted T5-neighbor sharing of the first prefix declines from53.22% to48.06%,
33.98%,13.68%,0.62%. Actual moved counts are0,605,2420,6050,12101. Full prefix
keys and catalogue counts match exactly at all four depths. First-prefix TRAIN
distribution total variation versus original is0,.01690,.05815,.10519,.18072.
TRAINfirst-prefix entropy changes only from7.44586to7.47854bits at100%, but
this does not control the identity of frequent branches or conditional
relationships to histories. T5-neighbor similarity is a representation-based
diagnostic, not independently verified ground-truth semantics.

| Reassigned | Epoch5 | Epoch10 | Epoch15 | Epoch20 |
|---|---:|---:|---:|---:|
|0%|0.026672|0.029815|0.033597|0.036446|
|5%|0.023075|0.027715|0.030251|0.033629|
|20%|0.019742|0.023983|0.026551|0.029246|
|50%|0.014813|0.018575|0.021036|0.021741|
|100%|0.003530|0.008921|0.012471|0.015133|

The dose ordering holds at all four checks, which share trajectories and users.
All fits still improve at20epochs. In particular the100%condition rises from
0.012471to0.015133over the last five epochs. The experiment establishes a
finite-budget benefit, not that a random index can never catch up. Longer fits,
new neural seeds, or new permutation draws were not run in this analysis.

## Verification and artifacts

Each model completed20full epochs,2628260examples,10280updates,zero AMPskips,
4977920parameters,and four validation checks. Same native GRID4+4/d128,
FP16training,FP32validation batch32,free beam20,full catalogue12101,no seen mask.
Execution body5593.8seconds,93.2minutes. No test evaluations.

`scripts/analyze_sid_corruption.py` verifies exact source/config,common initial
parameter hashes,training case/order hashes,budgets,all pinned full-path/owner
hashes,regenerated derangements and nested item sets. It independently recomputes
all catalogue occupancy and semantic/frequency diagnostics. It rescored all20
raw beam files against each trial's item-to-path table,matching NDCG/Recall@5/10
and invalid/duplicate counts. It checks users/targets and the aggregates of all20
teacher-forced per-example NLL/rank sidecars. This is numerical verification of
saved diagnostics, not a new model forward pass.

Artifacts:`/workspace/artifacts/sid-corruption/analysis/` contains `README.md`,
`endpoint.json`,`epochs.json`,`training.json`,`diagnostics.json`,
`verification.json`,all20rank vectors in `user-metrics.npz`,and the PNG/SVG
figure `corruption-results`. Raw135numerical/report files,70.17MB,are cached
at`/tmp/rhpr-sid-corruption-completed`;the parent artifact directory retains
small manifest/suite/metric/download metadata. All ten best/last model-state
filenames were confirmed on private Kaggle,but checkpoint contents were not
downloaded or independently verified. Rank999means absent from saved20beams.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/rhpr-mpl-cache PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_sid_corruption.py --plot
```

Collection helper:`/workspace/artifacts/sid-corruption/collect.py`,using a fresh
official output listing and transient-read retries only. No new training,
inference,test evaluation,notification,or accelerator session was launched.
