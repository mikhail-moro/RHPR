# Compiled mini-TIGER: two trials on one T4 give no throughput gain

Corrected private carasik/rhpr-kindle-concurrency-probe/version2 completed in6min18.76s. Sourceab5ac87e30d63739c97a5bccea0f29412ac045d9; workflow43530990541. One fixed Recent20/FP16/batch512/SDPA/compiled model shape, three100update repeats per case, no full fits or TEST.

| Transport / active trials | Aggregate updates/s, median | Range of three rounds |
|---|---:|---|
|Ordinary: 1, before|14.048|13.907–14.143|
|Ordinary: 2 together|12.367|11.878–12.584|
|Ordinary: 1, second idle|12.474|12.310–12.664|
|MPS: 1, before|12.721|12.414–12.959|
|MPS: 2 together|12.786|12.294–13.489|
|MPS: 1, second idle|12.737|12.408–12.871|

Ordinary two-client throughput is0.880x initial solo, but the later solo control with the other process idle is also0.888x initial solo. This temporal/context/clock drift prevents attributing the full12%gap causally to simultaneous execution. Paired throughput is0.991x the contemporaneous solo-post control: no measured positive gain.

MPS daemon started and both client PIDs were positively verified with its control interface. MPS paired throughput is1.005x its own solo-pre and1.004x solo-post, within short-run variability. Against initial ordinary solo it is0.910x, but transport/order effects are not isolated. MPS works technically here, not a useful throughput improvement for this workload.

Train NVML peak: ordinary solo1.50GiB, pair3.00GiB; MPS solo1.53GiB,pair3.03GiB. Concurrent FP32 VAL256 also fits: peak9.54/9.57GiB. All four full2625user beam predictions exactly match the saved parent checkpoint and NDCG replays1e-12. These peaks include the cached512shape training graph, not an additional485tail graph or longer windows.

Independent accounting verified all18timed rounds,2400applied updates,100updates/client/round, zero skipped measured updates, complete common start/finish windows, and both MPS clients. Warm-up/compilation are excluded from steady rates. Repeated fixed batches, one model/T4 and three repeats do not establish general results for other architectures or shapes.

Recommendation: retain one training trial per GPU; two separate T4s can run two independent trials. No bucketing, parallel HPO deployment or further fits were launched. The first version failed only in progress logging after its first solo measurements; it is preserved separately and contributes no final comparison numbers.

The previous LR0.003→0.001/EMA run completed successfully:60epochs, best VAL0.05133097 at50,17min session,13.88s median steady epoch. EMA did not trigger before the cap. Its source/initializer/SID/update/LR/EMA states and all12VAL prediction files were independently replayed. See docs/gr-kindle-lr-floor001-ema-results.md for its separate limitations.

Evidence:analysis.json, ordinary/mps-results.json, all individual commands/results/logs, gpu-utilization.csv, four validation ranking files, concurrency-throughput.png/.pdf. No TEST data or quality-trained checkpoints were created.
