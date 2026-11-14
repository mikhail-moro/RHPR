# Kindle long-history window screen: completed results

All9 fits completed on2T4 in 3.816h; 103 VAL prediction files independently replayed. TEST remains closed.

Same fixed958 users and targets,141137 training prefixes,16177-book catalogue. Three coupled tokenizer+generator pipeline seeds42/101/202. Each window uses the same index and initializer within seed.

| Window | Selected VAL NDCG@10 mean ± sample SD | Recall@10 mean | Forward GFLOPs | Warm train epoch seconds | EMA / cap |
|---|---:|---:|---:|---:|---:|
| 20 | 0.009301 ± 0.002149 | 0.016701 | 0.117744 | 21.36 | 1 / 2 |
| 32 | 0.008465 ± 0.002327 | 0.019833 | 0.203128 | 37.00 | 1 / 2 |
| 64 | 0.008713 ± 0.000662 | 0.016006 | 0.500115 | 83.99 | 1 / 2 |

Forward: supported CPU math-SDPA operations,B1 full context,teacher-forced four-code forward with output heads,FMA=2. Excludes backward/beam/text encoding. Warm time is each-fit median excluding first epoch, then mean across three seeds; first epoch includes lazy compilation. Model parameters1440576, including the retained unused pair-input table.

## Paired results

| Pipeline seed | Recent20 | Recent32 | Recent64 |
|---|---:|---:|---:|
| 42 | 0.007464 | 0.006786 | 0.009070 |
| 101 | 0.011664 | 0.007489 | 0.007949 |
| 202 | 0.008775 | 0.011122 | 0.009119 |

Selected means: Recent32 vs20 −8.98%; Recent64 vs20 −6.32%. The winner changes across seeds:42->64,101->20,202->32. Absolute paired selected64−20 difference −0.000588, conditional shared-user bootstrap95%CI[−0.003705,+0.002599]. All nine window/criterion intervals cross0. These intervals condition on the fitted models and selected checkpoints and exclude seed/index/item-clustering and checkpoint-selection uncertainty.

Epoch-comparable checks also do not establish a robust gain: fixed40 means20=.006472,32=.006447,64=.007454; fixed45 means20=.005592,32=.006268,64=.005725. Thus64−20 changes from+15.17% at40 to+2.37% at45 and−6.32% for selected maxima. Fixed45 is the latest common observed epoch of all9; no filling stopped trajectories or treating missing later metrics as zero.

## Learning and stopping

| Trial | Best epoch | Last epoch | Stop | Top10 hits at selected epoch /958 |
|---|---:|---:|---|---:|
| recent20_42 | 30 | 45 | early_stopping | 13 |
| recent32_42 | 30 | 55 | early_stopping | 16 |
| recent64_42 | 40 | 55 | early_stopping | 13 |
| recent20_101 | 60 | 60 | epoch_limit | 20 |
| recent32_101 | 50 | 60 | epoch_limit | 17 |
| recent64_101 | 50 | 60 | epoch_limit | 17 |
| recent20_202 | 50 | 60 | epoch_limit | 15 |
| recent32_202 | 55 | 60 | epoch_limit | 24 |
| recent64_202 | 55 | 60 | epoch_limit | 16 |

Six fits hit60; three seed42 fits stopped by EMA at45/55/55. Every model has its lowest observed teacher-forced VAL CE at its final observed epoch. Continued CE improvement does not guarantee later ranking improvement, but the curves do not show convergence of that objective or a general overfitting explanation.

Only13–24 selected top10 hits per model among958 users. One rank1 success contributes1/958≈0.001044 to NDCG, versus overall means≈0.009. Absolute EMA min_delta=.0005 corresponds to≈5.4% of Recent20 mean on this harder cohort; its relative scale differs from the previous pilot. Do not present operational EMA exits as proof of a plateau.

## Verification and practical interpretation

Replayed all103 prediction files: NDCG/Recall5/10 and invalid/duplicate counts match1e-12; cases/targets/index/config/source/initialization pins, all LR and EMA states, epoch/update/AMP accounting verified. 515 total epochs,10 AMP skips, no failed fits or fallback. 20988/1973480 invalid paths across all checks;0 duplicates. Original rank slots retained.

Recent64 has4.25x the supported forward FLOPs and3.93x warmed epoch time of Recent20 without an established ranking improvement in this screen. This supports retaining Recent20 as the cheap control. It does not prove that older history lacks signal or that1.44M parameters is the bottleneck. No compression model was trained here.

Absolute metrics are not comparable to the previous4969-book/2625-user pilot: catalogue, users and interaction density changed. The small fraction of cold targets alone does not make the new task equally easy. Here413/958 VAL targets(43.1%) have<=5 original train next-item positions, versus181/2625(6.90%) in the previous dense pilot; their median target training frequency is6 versus21. This documents a changed support distribution, not a causal decomposition of the NDCG drop. Descriptive subgroup numbers are in the JSON; sparse hits and post-hoc inspection prevent strong subgroup claims.

Raw prediction files,full curves,indexes,embeddings,private user-level CSV andPNG/PDF curves are retained under /workspace/artifacts/kindle-pilot/long-window-results. Weights remain in the private Kaggle output. Replay: scripts/research/analyze_kindle_windows.py --results RESULTS --data INPUT --output OUTPUT --plot. No new GPU runs,TEST model evaluation,extra epochs or notifications.

Output inventory confirms9 best and9 last checkpoint files remain on private Kaggle. Full local regression after replay:1099tests,152optional skips,0failures.
