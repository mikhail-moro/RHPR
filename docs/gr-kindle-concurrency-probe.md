# One versus two compiled trials on one T4

User-authorized cheap concurrency test; no full training, bucketing, new index,
new features or TEST access. The ongoing/completed LR experiment is not modified.
Use the frozen Recent20 seed42cosine_003best checkpoint and exact Kindle cases.

All clients use the same physical T4, selected by GPU UUID. FP16 batch512,
train-forward torch.compile(reduce-overhead,fullgraph=False,dynamic=False),
AdamLR0.003/wd0.0001/clip1, one CPU thread per client. Model/optimizer states
reset outside each measured round;8warm-up updates populate the optimizer and
CUDA graph. Every timed round contains100real optimizer updates per active
client. Require all updates finite/applied; do not count skipped batches as work.

Per transport:3solo rounds before the second client exists,3simultaneous rounds,
3solo controls with the second client resident/idle. File barriers arm clients
before a common future start; aggregate throughput=sum(applied updates)/(last
finish−first actual start), not a sum of independently measured rates. Save start
skew and individual step times. Warm-up and compilation excluded from steady
throughput but their wall cost remains logged. One combined GPU NVML sampler
records total memory/utilization. Two full FP32 VAL256 passes run concurrently
at the end to check memory feasibility with cached training graphs. No TEST.

Ordinary CUDA processes are required. MPS is optional only if its binary exists
and the remaining10min compute budget permits. Launch a separate user-owned
MPS daemon restricted to this session's GPU and unique pipe/log paths; never
change GPU compute mode or another daemon. Verify both worker PIDs are registered
with the server before labelling results MPS. If unavailable/unverified/failed,
record that status explicitly; no silent fallback or automatic retry. Quit only
our own daemon and terminate only our own client process groups in cleanup.

Pin all input/model files and parent sourceac7fc6a3aa587aa0366de504182d9c831f0c00ff.
No new learned parameters/checkpoints are retained. These fixed-batch disposable
updates measure throughput, not quality or production-trial scheduling. Identical
shapes/weights and reset state improve comparability; models remain independent
processes/optimizers/RNG. CUDA graph compilation caches can be reused within the
probe to avoid repeated cold cost. Report solo-pre/solo-post variability, repeated
round ranges, memory and validation feasibility. Projecting to four trials on
2T4 or different sequence lengths requires another measurement.

Bound600s after setup; per-command/startup deadlines and process-group cleanup.
Server15min including environment startup. Expected3–7min actual GPU session,
not a guarantee. Preserve partial results on failure. No long training or
follow-up deployment is automatic from this probe.

Initial version1(c083bc3)failed after its first solo measurement because the
new progress event used phase both positionally and as a keyword. GPU execution
completed100updates, but no paired measurements were reached; there is no
concurrency conclusion from v1. All failure artifacts are saved, including a
local evidenceZIP. Repair changes the logging payload key to round_kind and
adds a complete parent-mode test exercising the actual logger. One corrective
repeat of the same fixed benchmark is required to complete the authorized test.
