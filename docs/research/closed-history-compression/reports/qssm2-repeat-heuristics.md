> Historical report from 2026-10-20; research branch closed on 2026-10-21. Any proposed runs or next steps below are historical and cancelled. See the [closure record](../README.md) for the final status.

# Fixed repeat heuristics on the saved validation anchors

**The learned baselines currently miss a strong elementary repeat signal.** On exactly the same 26,335 validation anchors / 4,384 users and exact 1,049,612-item catalogue, sorting the last 128 observed events by item frequency reaches **NDCG@10 0.097245**, compared with FuXi **0.024252** (4.01 times as high). Ranking unique recent items by last occurrence reaches 0.087914. No GPU inference or training was run for this diagnostic.

| Frozen method | Macro-user NDCG@10 | Macro-user NDCG@100 | HR@10 | HR@100 |
|---|---:|---:|---:|---:|
| FrequencyR128 | 0.097245 | 0.126156 | 0.161230 | 0.301267 |
| FrequencyVisible1152 | 0.064115 | 0.105953 | 0.116760 | 0.324644 |
| MostRecentUniqueR128 | 0.087914 | 0.121033 | 0.134764 | 0.300336 |
| FuXi, fresh rotating p60 | 0.024252 | 0.040383 | 0.040634 | 0.124210 |
| QSSM state16, fresh rotating p60 | 0.018300 | 0.033613 | 0.032666 | 0.112018 |
| QSSM2 state128, p60 | 0.017551 | 0.033032 | 0.031750 | 0.111961 |

FrequencyR128 minus FuXi paired user-bootstrap NDCG@10 difference is **+0.072993**, 95% interval **[+0.069228, +0.076796]** (5,000 draws, seed42). This interval measures user sampling of fixed predictions, not training-seed variation or checkpoint-selection uncertainty.

## Recipes fixed before evaluation

The exact three recipes and tie rules were saved in [qssm2-repeat-heuristic-recipes.json](../aggregates/qssm2-repeat-heuristic-recipes.json), SHA256 `5ed810de5aeeb15b5181d383c28d191589fb1476477e103f4042321bf7a7efb9`, before computing results. There was no tuning or search over recipes.

- FrequencyR128 sorts positive item IDs observed in the final 128 occupied context slots by frequency descending, last occurrence descending, then ID ascending.
- FrequencyVisible1152 applies the same rule to the complete visible H1024+R128 window.
- MostRecentUniqueR128 sorts positive IDs in R128 by last occurrence descending, then ID ascending.

Each priority list is followed by every remaining catalogue item in pre-cutoff cohort training-frequency order, ties by ID. This uses all 89,191,505 training events of the fixed cohort, whereas the neural models received about34m supervised target positions. It is not a matched-exposure or matched-capacity baseline. **However, this extra global prior makes no contribution at all to the reported NDCG@10 of the two R128 recipes:** an independent context-only `Counter` + sorting implementation reproduces 0.09724502983231559 / 0.08791441596695404 exactly. Only40 anchors have fewer than10 unique positive IDs in R128, and fallback adds no NDCG@10 on those anchors. The global prior therefore cannot explain the main NDCG@10 gap.

All catalogue IDs1..N are eligible, including observed items; ID0 is not a candidate. OOV targets receive zero credit. The target is only queried after constructing the ranking from past context. Exact full-catalogue target rank is computed analytically by accounting for priority IDs removed from the global ordering; the first10 known-target anchors were also checked against a literal full-catalogue permutation for all three recipes. Every actual context hash equals the saved neural input hash and every last observed timestamp precedes the target.

## Why this matters

In the previous paired slice analysis,86.1% of the FuXi–QSSM2 NDCG@10 gap came from targets already observed in R128. Within that target-informed repeat group, FrequencyR128 reaches NDCG@10 0.300091, versus FuXi0.059256 and QSSM2 0.042570. The broad architectural diagnosis must therefore account for weak exploitation of simple recent repetition. The fact that FuXi beats QSSM does not yet establish superior discovery of long-range temporal patterns.

The heuristic advantage does not imply older history is useless. On targets observed only in H1024, FrequencyVisible1152 reaches NDCG@10 0.028326 versus FuXi0.009301; FrequencyR128 receives zero there at10. These groups depend on the eventual target and are posthoc diagnostics, not permissible serving/routing features. They also overlap users and have different macro-user denominators, so their averages must not be combined using raw anchor shares.

## Scorer and alignment audit

The bounded source/artifact audit found no seen-item filtering, mapping offset, target alignment, or saved-family substitution problem:

- [benchmark.py](../../../../src/rhpr/sasrec/benchmark.py#L54) scores exact item rows1..N in FP32 and preserves seen items. Training positives and negatives use the same tied `item_embedding` as catalogue scoring.
- [daily_evaluation_data.py](../../../../src/rhpr/sasrec/daily_evaluation_data.py#L296) forms context strictly before the target, excluding that event. Both context and target use the fixed train-only 1-based item mapping.
- [query_rotating_protocol.py](../../../../src/rhpr/sasrec/query_rotating_protocol.py#L197) shifts training labels by one event; [losses.py](../../../../src/rhpr/sasrec/losses.py#L93) gathers those labels by the encoded source positions.
- Every saved neural rank/HR/NDCG/OOV record is internally consistent and each model's manually aggregated macro-user metrics exactly equal its own fresh selected p60 result. No model-name field in rank records is overridden by heuristic metadata. The audit did not reload weights or rerun neural logits.

Primary artifacts: [full aggregates and intervals](../aggregates/qssm2-repeat-heuristics.json), [independent context-only check](../aggregates/qssm2-repeat-heuristics-independent-check.json), [saved-rank audit](../aggregates/qssm2-saved-ranks-audit.json), [evaluation script](../analysis/evaluate-qssm2-repeat-heuristics.py).

All results use repeatedly inspected validation anchors, one neural training seed, and validation-selected checkpoints; there is no new test evaluation. All three neural curves are still increasing at the60-pass budget cap. This diagnostic identifies a missing repeat baseline and a weak exploitation of accessible signal; it does not isolate the cause among optimization, retrieval loss, model bias, cohort/anchor distribution, and architecture.
