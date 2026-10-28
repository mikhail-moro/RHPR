# Teacher age: completed shared-proposal screen

[Private Kaggle v1](https://www.kaggle.com/code/carasik/rhpr-generator-age-screen)
completed. Executable`316c1093a0a76ccf408562c5bce0a8640d72f322`.
All three ages evaluated the same115 pairs:95 crossing first-token branches,
20 sharing the first token, none sharing the first two. Each age consumed1987754
reserved scoring rows, including original-map static and sequential evaluation.

## Result

|Teacher epoch|Static positive pairs|Sequential accepted swaps|Changed items|Changed first token / second token|TRAIN NLL improvement|
|---|---:|---:|---:|---:|---:|
|1|2|2|4|2 / 2|0.002348%|
|5|2|2|4|4 / 0|0.000682%|
|20|0|0|0|0 / 0|0%|

Absolute mean TRAIN NLL:10.561140986->10.560892971 at1,
8.335156759->8.335099900 at5,7.452504137 unchanged at20.
Total likelihood gains32.5923655,7.4720830,0nats respectively.
Changes touch639 TRAIN cases /203 validation history-or-target cases at1,
138 /45 at5; exposure is descriptive, not a quality measurement.

The common source-rank-balanced pool has2000 pairs, but only its identical
budget-limited115-pair prefix is tested. This is not an exhaustive pair search.
The two swaps accepted in the preceding20epoch experiment are in the new2000
pool but outside its115-pair prefix. Therefore20epoch zero acceptance here does
not contradict the earlier two accepted swaps. Different pool/order/budget
accounting prevent direct comparison of acceptance counts between experiments.

## Same pair, different frozen teacher

Static original-map log-likelihood deltas, nats:

|Pair|Description|Original LCP|Epoch1|Epoch5|Epoch20|
|---|---|---:|---:|---:|---:|
|3023/3413|Two nail-art decorations|1|+3.080|-116.174|-167.732|
|8476/10004|DevaCurl gel / shampoo|0|-2.116|+5.650|-17.217|
|10655/11819|Two hyaluronic/vitamin-C serums|0|+29.512|+1.822|-23.017|

These three pairs constitute the union of accepted moves. The serum pair is
accepted by both early models; their other moves differ. TRAIN target counts:
97/43,11/10,5/14 respectively. The positive static deltas equal their sequential
deltas in this run. Other rejected sequential scores can change after earlier
acceptances:31 pairs at1,one at5,none at20 differ by>1e-7nats. Final indices remain
separate greedy outcomes; static positive moves were not combined into another
index.

The age effect is visible on identical original-map perturbations. However, it
is consistent both with learning a fixed old code assignment and with early
models making errors that later training corrects. Acceptance by an undertrained
teacher is not evidence that a reassigned tree is easier to learn from scratch.
Only2/115 pairs pass at either early age: moving earlier permits some upper-tree
changes, but does not remove the main sparsity of beneficial frozen-score moves.
No downstream quality benefit, stable semantic improvement or need for
alternating optimization has been established.

## Training, mining and validation of artifacts

Only5 new source epochs:657065 examples,2570 updates,zeroAMPskips,shared original
named-neural initialization hash`5f4a24fa...`,seed42,case order unchanged.
One native validation at5,not used for proposals/acceptance. No test,new index
training,semantic regularizer,extra seed or continuation.

All ages mined the same16384 TRAIN cases. Unique pair counts at1/5/20:
12501/13122/13261; invalid greedy paths2/0/0. Each retained2000 pairs before
balanced union/shuffle. Most proposals are rejected with substantially negative
static deltas; medians at1/5/20:-340.343/-358.180/-607.774nats.

All three complete TRAIN rescoring audits and identity probes have exactly zero
drift; frozen source weight/RNG guards pass. Owner permutations replay exactly;
all catalogue prefix occupancies remain identical. All three mining lists were
independently reconstructed from the saved probe arrays; shared pool/order/row
budget were reconstructed from raw histories. All cache aggregates and zero
changes outside affected TRAIN cases were verified.

The1/5 checkpoint files were downloaded and their weights-only contents checked;
the previously downloaded20 checkpoint was rechecked. Exact epochs1/5/20,
updates514/2570/10280,source configs,file hashes and model-state hashes match
the runtime reports. The three ages share initialization/stream, but20 comes
from a previous execution; CUDA numerical nondeterminism is not ruled out.

Actual notebook config and all180 embedded source files match the executable
commit. Private/offline/T4,two datasets and source-kernel mount were verified.
This resolves the runtime audit that was blocked by the earlier cloud outage.
Embedded packageSHA256:
`04448e74e87ff6bc41485d00d229c65fa121d55aacc089d73f42c9152c942508`.

Execution body30.24minutes. Downloaded60 numerical/report files11229807bytes
under`/tmp/rhpr-generator-age-completed`(plus two19.98MBearly checkpoints).
Durable analysis,figures,paired deltas,audits and copies of reports:
`/workspace/artifacts/sid-generator-age/`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/tmp/rhpr-mpl-cache \
  /workspace/rhpr-check/bin/python scripts/analyze_sid_generator_age.py --plot --verify-checkpoints
```

No new training or inference was used to analyze these outputs. No additional
GPU experiment has been submitted.
