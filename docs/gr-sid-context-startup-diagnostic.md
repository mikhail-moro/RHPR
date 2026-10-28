# Silent context-robustness startup: diagnosis and prepared correction

The user reported more than30minutes in Running with empty stdout for
`carasik/rhpr-context-robustness-replay`, original code
`d4305b9c563a77d967ff1e2b2dcefaec33f07c1e`.

Confirmed observations:

- GitHub submission40511232348 succeeded; only one version was submitted.
- Kaggle reports RUNNING, with no persisted log/output files available.
- The user confirms Running with empty stdout in the UI.
- Global GPU quota increased during diagnosis. This is not per-session billing
  telemetry, but continuing silently without verified progress risks quota.
- A bounded CLI live-log attempt yielded no events. The SDK log endpoint returned
  HTTP400 with no explanatory body. Neither revealed a session ID or startup phase.
- The cancellation API requires a kernel-session ID, which the available status
  and metadata APIs do not return. The user was asked to stop via the UI.

The precise root cause is **not established**. The old runner prints its first
message only after dependency preparation, so empty stdout alone cannot separate
platform startup from dependency/import/I/O delay. Do not claim a confirmed pip
hang, model failure or infrastructure failure.

Prepared corrections preserve all models, inputs, inference settings and cohorts:

1. Emit `kernel_python_started` before any output-directory writes or dependency
   bootstrap. The marker contains only source SHA and experiment name.
2. Emit GR runtime prepare/pip/import stages, and limit the small offline pip
   install to300seconds by default. Imports keep their original order.
3. Emit validation resource/parent verification stages before any long preparation.
4. Check known parent mounts first. Recursive scanning of remote `/kaggle/input`
   becomes a lazy fallback, preserving hash/source guards and versioned mounts.
   The previous eager scan was unnecessary; it is not proven to be this incident.

Validation:15resource/runtime tests,13parent/precision contracts and1early-marker
test pass. The marker test executes the runner prefix before filesystem setup;
the parent test forbids recursive scanning on an existing known mount; the pip
test confirms TimeoutExpired propagates without importing model libraries.

Update: the original session completed all ten models and forty passes without
a replacement or API cancellation. Its first runtime-ready log is at34.65seconds,
installation/imports take33.86seconds, and logs end at1117.78seconds (18min38sec).
Thus this run does not show a long pip or parent-verification hang. The external
wait/live-log silence remains unexplained by the persisted log. The prepared
changes improve future observability; they did not produce the completed result.
Do not restart the completed run.

Local evidence: `/workspace/artifacts/sid-context-robustness/startup-diagnostic/`,
`live-diagnostic.log`, `session-log-probe-detail.log` and existing launch metadata.
