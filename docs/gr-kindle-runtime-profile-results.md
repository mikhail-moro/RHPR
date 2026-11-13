# Kindle mini-TIGER hardware profile

Completed in154.29s on2T4. Source298137f4657c8cbcaaa88878a5ef5db73bcdff05; private carasik/rhpr-kindle-runtime-profile/version1; workflow43525147702. No full fit or TEST access. Pinned cosine_003 seed42best weights. All8train cases,4validation batch sizes and compile probe completed.

| Train physical batch | Effective batch | Pipeline step ms | Examples/s | NVML peak GiB | GPU busy mean |
|---|---:|---:|---:|---:|---:|
|256|512|160.66|3187|1.06|54.1%|
|512|512|125.93|4066|1.83|87.3%|
|1024|1024|232.08|4412|3.33|92.9%|
|2048|2048|457.20|4479|6.36|95.7%|

At512, GPU-resident fixed-batch step126.48ms versus CPU-assembly/transfer125.93ms: no measured gain. Synchronized host assembly0.33ms, H2D2.00ms; instrumented forward57.62ms/backward81.83ms/Adam6.53ms. Stage synchronization/profiling overhead means these must not be added to reproduce uninstrumented126ms. GPU copy_ operators also include AMP/device casts, not just host input copies.

Compiled warm512 step72.20ms, 1.74x versus eager pipeline. Cold case including compilation/warm-up45.34s; one Dynamo graph, zero graph breaks, all20timed updates applied. Full-epoch speedup remains to be measured; the485example tail may compile a second graph. This is standard training-forward compilation, not beam-search compilation.

| VAL batch | Full2625user pass mean s | IDs/ranking vs old batch32 checkpoint |
|---|---:|---|
|32|5.515|exact match|
|64|4.703|exact match|
|128|4.387|exact match|
|256|4.301|exact match|

All four complete VAL predictions were independently checked against the old cosine_003epoch50predictions; user/targets/SIDs match exactly and NDCG replays1e-12. Two timed passes per batch. TEST remains closed. VAL acceleration alone affects only~5.6%of prior fit time.

Current train512 fits in~1.83GiB NVML used of15GiB reported capacity;2048uses6.36GiB. Capacity is not limiting these batches. Larger batches improve throughput only8.5–10.2%and change updates/epoch. Do not confuse GPU busy87%with achieved compute efficiency; the trace shows many small operators/casts and attention work, and compile fusion provides a larger measured benefit. Native SDPA is already using efficient CUTLASS attention on T4.

Training arrays are RAM-resident; disk is not accessed per update.27.07%of original train positions are padding; every sampled random512batch has a20item example, so dynamic batch trimming alone removes none. Bucketing changes update order, and naive left-padding removal shifts USER/item T5 relative bias. A native CPU check on a short context changed hidden outputs by max_abs1.6168; it is not a numerically equivalent optimization with the current layout.

Adopt compile and VAL256 for one corrected LR0.003→0.001/EMA intermediate run while preserving train512 and original data order. Keep unwrapped checkpoints. Its quality cannot isolate LR floor versus compilation/EMA; record first-epoch compilation cost, steady epoch times, loss and actual stop. No repeated nine-model matrix.

Artifacts:analysis.json,training-profile-trace.json,operator-profile.json,gpu-utilization.csv,train/evaluation/compile-profile.json and all four VAL ranking files.
