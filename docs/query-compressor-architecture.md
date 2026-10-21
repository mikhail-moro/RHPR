# Fixed-LR compressor architecture search

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

The completed H1024 learning-rate run selected backbone/compressor LR 0.001 for
both families. Its best checkpoints were at pass 50; the last pass was60.
Full-history validation NDCG@10/@100 was 0.06512496/0.07505025 for Cross Attention
and 0.06468601/0.07437090 for QSSM. These are validation selection results from one
training seed, not a new test comparison. QSSM's higher compressor LR 0.003 failed
in pass 4: same-batch diagnostic replay found FP16 overflow while FP32 stayed
finite. The successful LR 0.001 models do not show that failure.

## Search and stopping

Keep H1024, Q=H[-128:], R128, K4, the original catalogue, target/negative schedule,
seed42, full32768-position initializer and SASRec predictor. Both optimizer
groups use LR 0.001 and the original weight decay, AMP and batch sizes. Only the
compressor architecture changes:

| Family | Trial | History readout | Inner width | State size |
|---|---|---|---:|---:|
| Cross Attention | ca_base | One read, four heads; reused control | — | — |
| Cross Attention | ca_read2 | Two sequential reads, four heads each | — | — |
| Cross Attention | ca_heads8 | One read, eight heads | — | — |
| QSSM | q_base | Reused control |128|16|
| QSSM | q_inner64 | Narrower scan |64|16|
| QSSM | q_state32 | Larger recurrent state |128|32|

The query generator always has four heads. The two-read CA updates the queries
after its first global read and reads the same memory again, then applies one
shared final FFN. It is not two complete Q-Former layers. CA heads8 retains the
same parameter count; QSSM width/state variants have different compute and
parameter counts, which are reported rather than treated as equal budgets.

Validate every ten passes, allow20–60 passes, and stop after two consecutive
checks without strict improvement in macro-user NDCG@10. There is no gap-based
pruning: a lower-scoring configuration that continues improving can run to 60.
The good controls improved through pass 50, so an unconditional20/30-pass cap
would bias this search toward faster starters. These passes supervise one final
128-target block per user; they are not exhaustive epochs over every event.

Within each new trial retain the earliest strictly best H1024-validation
checkpoint, then evaluate it once using the full observed validation history.
Select the family winner using full-history validation NDCG@10, with registered
order breaking exact ties. This is inference-length transfer after short-history
training. It does not replace later training/replication at long lengths. No test
ranks are loaded or new test predictions computed.

## Reusing controls and recording cost

The two baseline controls come from the completed private kernel
`carasik/rhpr-query-prefix-1k-search`, source
`66a0322b46b17cf9b480952512d49810c1efcd8e`. Config, manifests, report, weights
and validation ranks are pinned. Replaying the new stopping policy over each
saved curve must retain the same pass 50 checkpoint through the same 60-pass
budget. Only then can a control enter selection. Reused checkpoints retain
their original source and mounted path; the new output does not claim to have
trained them again. Their validation ranks are copied into the new output for
reproducible comparisons.

Four architectures receive new training. Each receives its own disposable
real-update and short/full inference preflight before the budget decision;
baseline timing is not substituted for a more expensive shape. The budget
prices all four at 60 passes, validation, full-history transfer and checkpoint
I/O. Early stopping can save time but cannot silently fund a larger grid.

Numerical failures and OOM exclude the affected trial and retain diagnostics;
they do not change learning rate or precision. Best weights and rolling complete
Adam/scaler/RNG checkpoints remain available. Progress messages mark evaluation
and paired-statistic phases, including work after the last training trial.
Serving cache experiments remain a separate protocol in
[cached-compressor-serving.md](cached-compressor-serving.md).
