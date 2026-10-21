# QSSM summary-count and memory-initialization search

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This validation-only first stage tests eight QSSM compression recipes on
H1024+R128. It varies summary count K4/K8/K16/K32 and the log-uniform initial
step-size range: default dt `[0.0001, 0.1]` or slower dt `[0.0001, 0.01]`.
The smaller upper bound initializes slower decay; it does not constrain learned
step sizes or establish the trained model's effective memory horizon.

All recipes use inner width64, state size16, gate rank32, compressor dropout0,
and backbone/compressor LR0.001. The SASRec predictor remains d128, two layers,
four heads, FFN multiplier4, dropout0.2 and its original32768-position table.
Parameters bearing Mamba no-weight-decay flags retain the earlier all-parameter
AdamW decay convention. CUDA selective scan is mandatory. No automatic LR,
precision, physical-batch or architecture fallback is permitted.

| Summary tokens | Default dt | Slow dt | SASRec input tokens |
|---:|---|---|---:|
|4|Reuse verified completed inner64 control|Fresh training|132|
|8|Fresh training|Fresh training|136|
|16|Fresh training|Fresh training|144|
|32|Fresh training|Fresh training|160|

The reused control comes from `carasik/rhpr-query-compressor-architecture`, source
`a8f6f0dc760624e83d1f8261bd19333cfbd4dc02`, trial `q_inner64`. Its exact selected
weights, source configuration, report/manifest hashes, validation ranks, original
contexts, Q/R preservation and stopping-policy replay are verified before reuse.
Only QSSM artifacts and weights are opened; Cross Attention weights and test rank
files are not read. The registered alias `dt_default` preserves the original
trial definition in provenance. The verified weights are copied into this run's
output so checkpoint paths remain portable.

## Training and primary selection

Keep the parent catalogue, global daily splits, seed42, original full-prefix
user endpoints, final128 shifted targets and512 sampled negatives. A pass
visits one final window for each user, reshuffling users and drawing new
identity-bound negatives. It is not an epoch over all chronological targets.
Crop H on the CPU to1024 events before embedding, retain R128, and compute the
same128 causal losses in parallel. Queries read H's final128 events. Occupied
OOV events keep their chronological positions.

Physical training batch32 and effective batch128 stay fixed. Train at least20
and at most60 passes, validating every10, with patience2 and strict improvement
in validation macro-user NDCG@10. Select the earliest best checkpoint. NDCG@100
and full-catalogue daily-anchor ranks are also saved. No test ranks are
deserialized and no test predictions are computed.

Resolve every short-context training trial before choosing full-history
diagnostics. Short-context NDCG@10 selects recipes independently within each K;
registered order breaks exact ties. All eight primary results are retained, and
an optional full-history failure cannot invalidate a successful primary result.
The search uses a single seed and validation-set selection, so it does not yet
support a claim of tuned family superiority or a noise-free optimal K.

## Budget and full-history diagnostics

Profile every fresh shape using two real effective Adam updates, short-context
catalogue inference, parameter counts and explicitly scoped arithmetic estimates.
The prospective six-hour ceiling charges all seven fresh recipes at60 passes,
all scheduled validation checks, checkpoint/statistics overhead and safety1.25.
Reserve2400 seconds for optional frozen full-H evaluation. Early stopping may
save compute; it does not fund extra unregistered trials.

Only the two best successful fresh recipes by primary short-context NDCG@10
receive frozen-checkpoint full-history transfer, at physical evaluation batch4.
The reused K4 control already has verified full-history ranks. Other recipes
have no fabricated full-history metrics. The two selected transfers are a
length-transfer diagnostic after1K training, not full-history training or an
additional selection criterion. Numerical/OOM failures and insufficient budget
are recorded with primary results retained.

The run retains best FP32 weights and complete optimizer/scaler/RNG state for
the two best fresh recipes, allowing deliberate later continuation without
training every candidate past60 passes. Report uncached rates, memory and
arithmetic scope separately; these measurements do not establish cached-serving
latency. More seeds, deeper architecture/regularization tuning and full-history
training belong to subsequent experiments.
