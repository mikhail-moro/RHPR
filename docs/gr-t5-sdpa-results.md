# T4 native SDPA pilot: complete

Private `carasik/rhpr-t5-sdpa-pilot/1`, executable
`d6edbf7815a3814867f915a1cb733a805131b2be`, workflow41993141529.
Completed2026-11-05; experiment section194.20seconds, including173.30seconds
of sequential child execution. Offline bootstrap precedes that experiment timer.
No quality training, ML1M run, or extra GPU experiment was started.

## Full optimizer-step measurements

One physical TeslaT4, PyTorch2.11.0+cu130. Same trained sum42 singleton checkpoint,
4+4T5/d128,6heads,d_kv64,FF1024, frozen SIDs, real strict-past contexts.
FP16/dropout.1, batch256/physical128, Adam, clip1, GradScaler1024.
Each row has10warmup and50timed optimizer steps on the same fixed minibatch.

| History items | Eager4.47 step | Eager5.22.1 step | SDPA5.22.1 step | SDPA time vs historical eager |
|---|---:|---:|---:|---:|
|20|124.67ms|144.46ms|128.88ms|**+3.38%**|
|50|348.85ms|371.25ms|308.27ms|**−11.63%**|

A separate10step measurement includes production SID batch construction and
H2D. Historical→SDPA:125.18→129.52ms at20(+3.47%);349.82→309.36ms
at50(−11.57%). Thus the sign of the observed result survives batch-preparation
inclusion. These are short fixed-minibatch steps, not measured full epochs.

| History items | Eager4.47 allocated | Eager5.22.1 allocated | SDPA allocated | SDPA vs historical eager |
|---|---:|---:|---:|---:|
|20|0.841GiB|0.837GiB|0.747GiB|**−11.25%**|
|50|2.353GiB|2.353GiB|1.773GiB|**−24.64%**|

Peak reserved CUDA memory: historical→SDPA0.924→0.832GiB at20 and
2.643→1.924GiB at50. Allocated memory excludes some CUDA/library/context memory;
these values do not describe total nvidia-smi usage or maximum feasible batches.

## Actual dispatch and correctness

Profiler records `aten::_scaled_dot_product_efficient_attention`,
`aten::_efficient_attention_forward`, and their backward operators at BOTH
widths. This is actual memory-efficient CUDA execution, not simply setting a
config flag. Do not sum inclusive profiler operator times: parent/child events
overlap. The short50step timing samples were collected outside the profiler.

CUDA FP32/eval-with-grad parity on the trained checkpoint:

- Old→new eager hidden/loss exact; max gradient difference7.45e-9.
- Old→SDPA max hidden/loss difference9.54e-7; max gradient difference1.79e-7.
- All95 gradient tensors compared, including learned relative-position bias.
- All8×20cached beam paths exact, max SDPA beam-score difference3.73e-8.
- SDPA vs modern eager passed the same output/gradient checks.
- All6FP16/dropout.1 short training benchmarks completed with zero AMP skips.

FP32 numerical parity and FP16 finite training are different checks. This does
not prove identical dropout masks, FP16 trajectories, converged NDCG or long-run
numerical equivalence. Repeated fixed-batch loss reductions are overfitting of
a diagnostic batch, not evidence of better recommendation quality.

## Interpretation

For our CURRENT20item training window, this pilot does not support migrating
for speed: SDPA5.22.1 is3.4%slower than the deployed4.47eager path, while using
11.2%less allocated memory. At50items, SDPA is a useful candidate:11.6%less step
time and24.6%less allocated memory in this measurement.

Comparing only against MODERN eager would overstate the benefit for our current
system: SDPA saves10.78%step time at20 and16.96%at50 versus5.22.1eager, but that
eager implementation itself is slower here than our original4.47implementation.
Native SDPA is working; the library-migration overhead and short contexts limit
the practical gain. No claim is made about why the newer eager is slower.

One sequential runtime order, no independent repeats/randomized orders or GPU
clock controls. Within-run step SDs are2–4ms; these are NOT confidence intervals
on general performance. Treat the measured3%small-window difference cautiously.
The memory benefit and50item result merit preserving, but neither resolves the
roughly10.5×increase in examples when moving fromML100K toML1M.

Historical defaults remain unchanged. No full-model retraining or another
pilot was authorized by collecting these outputs.

## Reproducible evidence

`docs/gr-t5-sdpa-results.json` holds all aggregated measurements/parity checks.
`scripts/research/analyze_t5_sdpa_pilot.py` independently recomputes all6means
from300timings, verifies the three result files against combined outputs,
checks config/executable/device/budgets, and generates PNG/PDF figures.

Local collection `/tmp/sdpa-completed`; analysis `/tmp/sdpa-analysis`; portable
bundle `/tmp/t5-sdpa-pilot-evidence.zip`. The bundle has downloaded numerical
outputs/logs/profile tables, replay script, source audit, and this report.
It excludes the large parity tensor files and model checkpoints, which remain
on the original private Kaggle sources. No credentials are included.

Before submission975repository tests passed(13skipped),13additional GRID tests
passed with modern Transformers, and full CPU checkpoint parity passed.
Collection changes only reports/replay; no runtime source change or test rerun.
