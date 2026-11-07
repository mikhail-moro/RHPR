# T4 batch pilot: completed

Private Kaggle `carasik/rhpr-t5-batch-pilot/1`, executable
`b777b516014355e967db6ffcb869c307d5717e0f`, workflow41998201870.
Completed2026-11-05. Experiment section570.12seconds (~9.50minutes), including
558.23seconds of child execution; offline bootstrap precedes this timer.

All16predeclared attempts executed.15completed with zero AMPskips and finite
losses/gradient norms;1expected OOM,0numerical_failed. No fallback, reduced batch,
retry, new quantizer/embeddings or full quality training was launched.

## Resident-input throughput and peak allocated CUDA memory

TIGER4+4,d128,6heads,d_kv64,FF1024, FP16/dropout.1, AdamLR.001/wd.0001,clip1,
GradScaler1024. Same trained sum42checkpoint, fresh optimizer/scaler/process for
every attempt on physical TeslaT4 GPU0. PyTorch2.11.0+cu130 unchanged; historical
Eager4.47.0 vs nativeSDPA5.22.1.50timed steps after10warmup;10separate end-to-end
steps include production SID batch construction and H2D. Counts below are
training examples/second, not optimizer updates/second or full-epoch measurements.

| Window | Effective/physical batch | Eager examples/s | SDPA examples/s | Eager allocated GiB | SDPA allocated GiB |
|---|---:|---:|---:|---:|---:|
|20|256/128|1969|1938|0.84|0.75|
|20|256/256|2283|2452|1.56|1.37|
|20|512/512|2354|2651|3.03|2.67|
|20|1024/1024|2371|2771|5.93|5.22|
|50|256/128|708|859|2.35|1.77|
|50|256/256|722|910|4.59|3.43|
|50|512/512|736|934|9.10|6.76|
|50|1024/1024|OOM|938|OOM|13.42|

Eager50/1024failed requesting474MiB with326.81MiBfree,14.56GiBtotal GPU
capacity. Allocated at failure13.75GiB, reserved14.11GiB. SDPA50/1024fits with
13.42GiBpeak allocated; that is a narrow capacity margin, not a recommended
production default or proof all workloads will fit. No allocator tuning tried.

## Practical comparisons

Relative to the current historical eager256/128 baseline, native SDPA:

- At20,256/256gives1.245×throughput (+24.5%),512/512gives1.346×(+34.6%),
  1024/1024gives1.407×(+40.7%). At256/128it remains slightly slower here.
- At50,256/256gives1.284×(+28.4%),512/512gives1.319×(+31.9%),
  1024/1024gives1.324×(+32.4%).
- Moving fromSDPA512to1024adds only4.54%throughput at20 and0.388%at50,
  while allocated memory nearly doubles.512achieves95.7%and99.6%of1024's
  throughput respectively.

At the SAME physical/effective256/256, SDPA outperforms eager at both widths;
the early256/128pilot did not exhaust the useful batch configurations.

End-to-end throughput (same order256/128,256/256,512,1024):

| Window | Backend | End-to-end examples/s |
|---|---|---|
|20|Eager|1945,2262,2321,2378|
|20|SDPA|1916,2431,2616,2727|
|50|Eager|707,718,748,OOM|
|50|SDPA|846,900,914,949|

Thus gains survive including batch preparation/H2D. The pilot did not measure
a full epoch, checkpoint I/O, validation or catalogue quantizer time. Avoid
multiplying whole-run duration by these short-step throughput ratios as a
measured promise.

## Training implications

256/128→256/256preserves the effective batch, mean objective and optimizer
update count per epoch. It changes microbatch execution, dropout RNG and
floating point reductions; identical trajectories/NDCG are not promised.
SDPA256/256is the strongest candidate when preserving the existing optimization
schedule matters. At20its measured step is104.38ms vs129.99ms historical baseline.

512/512is a reasonable throughput-oriented candidate for NEW training: close to
the saturation point, much less memory than1024. It halves updates per epoch
relative to effective256. Quality/convergence must be assessed under that
choice. User accepted this as an allowable change; the pilot did not establish
quality equivalence. LR remained.001 for ALL configurations, no automatic
scaling or tuning. No production defaults changed from collecting this run.

Fixed batch reused for70updates per attempt:17920/35840/71680examples consumed
at effective256/512/1024, respectively. Those are repeated examples, not distinct
training-data exposure. Loss decreases are diagnostic minibatch overfitting and
cannot rank recommendation quality. The frozen trained checkpoint permits
capacity/speed comparisons; it is not fresh-model convergence evidence.

Single sweep, fresh processes, alternating backends per window/batch pair, no
independent hardware repetitions/randomized order or clock controls. Timing SD
within50steps is not a confidence interval on general performance. CUDA
allocated/reserved metrics exclude some driver/context/library allocations.
No claim about capacity beyond1024 or performance on other hardware.

## Verification and artifacts

CPU-only replay `scripts/research/analyze_t5_batch_pilot.py`:

- Checks executable/config, all16individual result files, runtime/device and
  exact predeclared schedule against combined results/metrics.
- Recomputes throughput from750timed and150end-to-end samples.
- Independently reconstructs80268strict-past training cases from retained raw
  histories, without importing training helpers. Reproduces all6window/batch
  sample hashes, proving paired backends and both256cases used the same data.
- Checks70updates, examples consumed, zero AMPskips and finite curves for all
 15successful configurations; verifies the single OOM placement.

Aggregate JSON `docs/gr-t5-batch-results.json`; raw numerical outputs/logs in
`/tmp/batch-completed`; plots/replay analysis in `/tmp/batch-analysis`.
Portable `/tmp/t5-batch-pilot-evidence.zip` includes raw downloaded metadata,
curves/logs, figures, replay, source audit and this report; no credentials or
model checkpoint weights. Outputs and source checkpoint stay on private Kaggle.
Code/report published in `codex/movielens-rich-semantics`.

26targeted tests and2real GRID CPU accumulation probes passed before submission.
This collection changes reports/replay only, with replay assertions checked;
no runtime changes, additional tests of unchanged training, or GPU reruns.
