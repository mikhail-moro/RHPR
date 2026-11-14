# Kindle generator capacity pilot: completed results

Two wide fits completed in59.637min on2T4;two narrow baselines reused. Fixed seed42,frozen item SID,data and optimizer/evaluation protocol. No tokenizer refit;TEST remains closed.

| Cell | Parameters | Selected VAL NDCG@10 | Top10 hits /958 | Best / last epoch | Warm epoch seconds | B1 forward GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| mini20 | 1440576 | 0.007464 | 13 | 30 / 45 | 21.42 | 0.117744 |
| mini32 | 1440576 | 0.006786 | 16 | 30 / 55 | 37.05 | 0.203128 |
| wide20 | 5108992 | 0.020492 | 38 | 55 / 60 | 31.25 | 0.427088 |
| wide32 | 5108992 | 0.016899 | 30 | 55 / 60 | 55.84 | 0.704770 |

Total parameter counts include the retained unused pair-input table; active counts excluding that table are1375040 and4977920. Forward counts use supported CPU math-SDPA operations,B1 full context,teacher-forced4-code forward with output heads,FMA=2;exclude backward/beam/text encoder. Times are per-fit median warmed train epoch excluding first-epoch compilation.

## Size and matched-budget comparisons

Selected wide vs mini:20 +174.54%(.007464->.020492),32 +149.05%(.006786->.016899). The wide fits ran60epochs and selected55; narrow controls stopped by EMA at45/55 and selected30. Thus selected maxima alone mix model shape with observed training duration.

At the predeclared fixed40 comparison,wide vs mini20 +150.94%,32 +143.09%. At a common selection budget through40, gains are+82.73%/+129.64%; through45,+87.71%/+129.64%. The direction therefore survives these checks and is not explained solely by extra epochs.

Conditional paired-user bootstrap95% intervals at fixed40:wide20−mini20 delta+.008204,[+.003552,+.013057];wide32−mini32+.009172,[+.003677,+.015154]. These condition on the four fitted models and fixed/index/seed;they exclude generator/tokenizer randomness,item clustering,selection effects and multiple comparisons. They do not establish seed-robust significance.

## Context gap remains unresolved

Among selected wide checkpoints,32 trails20 by17.53%;at fixed40 it leads14.25%;at fixed45 it trails29.11%. Selecting best within the same40/45budget gives wide32 leads14.25%/11.22%. All window comparisons and all difference-in-differences intervals include0. A general improvement from widening does not prove that the wider generator makes better use of older history.

Primary fixed40 interaction:(wide32−wide20)−(mini32−mini20)=+.000968,conditional user95%CI[−.005774,+.007798]. Selected interaction=−.002915,CI[−.010277,+.004785]. No confirmed capacity x context interaction.

## Learning and cost

Wide20/32 both selected55 and hit the60 cap;minimum observed teacher-forced VAL CE also at55,with a small rise at60. Last VAL CE9.1385/9.1047 vs narrow10.0674/9.8182;last train loss8.0422/7.9731 vs9.8699/9.5815. Do not claim converged optima or certain overfitting from one late check.

Widening4+4 from d64/FF512/dkv32 to d128/FF1024/dkv64 raises total parameters3.55x and supported forward operations3.63x/3.47x. Warm epoch time rises only1.46x/1.51x in the measured sessions. Narrow fits are historical,not a simultaneous controlled hardware benchmark. Wide peak CUDA allocated2.03/3.11GiB;actual serving latency was not benchmarked. Wide fit times34.11/58.96min;whole session59m38s.

## Verification and next inference boundary

Replayed all44 prediction files(24wide+20reused narrow):NDCG/Recall5/10,invalid/duplicate counts,scores finite,targets/users;verified all32parent file digests,source/config/input/index/case/initializer pins,all LR/EMA states and updates/AMP accounting. Wide fits have3AMP skips total,no failures or fallback. Across four trajectories:4927/843040 invalid paths,0duplicates. Original rank slots retained.

Two best and two last wide checkpoints confirmed in private Kaggle output. Full prediction/curve/index artifacts plus analysis JSON andPNG/PDF are local under /workspace/artifacts/kindle-pilot/capacity-results.

This is a useful positive screening result for a wider generator on this fixed cohort/index/seed. It does not establish a universal capacity bottleneck or improvement from longer context. Confirming the shape on remaining seeds and testing64would be separate runs requiring a new decision. No new GPU,extra epochs,TEST evaluation or notifications were started.

Replay:PYTHONPATH=src python scripts/research/analyze_kindle_capacity.py --results RESULTS --parent PARENT_RESULTS --data INPUT --output OUTPUT --plot.

Full local regression:1102tests,152optional skips,0failures.
