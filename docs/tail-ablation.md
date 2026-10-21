# Frozen MixedDepth checkpoint: tail128 ablation

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

This evaluation asks whether the old prefix influences next-item predictions
of the jointly trained MixedDepth checkpoint. The new kernel performs no
training, creates no optimizer, and leaves test untouched.

Configuration: `experiments/yambda_mixed_depth_tail128_ablation.json`.
Attach `carasik/rhpr-mixed-depth-15ep` as a private kernel output source alongside
the original Yambda dataset. Before loading weights, verify the configured
checkpoint SHA256 and the training revision against the source manifest.
All source configuration, predictions, history features, and frozen feature
bins must come from that same attachment. A newer source version with different
weights fails verification rather than silently changing the experiment.

Evaluate three modes on the same frozen checkpoint:

- `depth_0`: complete context, no compression.
- `depth_3`: complete context, three applications of the shared PairMLP.
- `tail_128`: keep only the final `min(n, 128)` original item embeddings.

Strictly load the original recursive model before selecting a mode. For
`tail_128`, temporarily replace the compressor with a parameter-free selector;
restore the original compressor afterwards. Keep `max_seq_len=1028`, all
backbone/item/position weights, the original validation context, and its target.
The selector returns original source positions, so the surviving tail receives
the same recency PE as in the complete input. Its positional values are
`min(n,128)-1` through zero. The existing `recent` compressor has a different
budget and is not this ablation.

The raw DataLoader input remains the full original context. This preserves
the existing evaluator's `context_hash` and length diagnostics. Predictions
also record `reference_context_hash`, `reference_window_length`,
`input_context_hash`, and `input_length`; the latter pair describes retained
events rather than pretending the ablated input equals the reference.
The encoder sees at most 128 events in the tail-only mode.

Use the original validation users, targets, seed, catalogue mapping, full-catalog
FP32 top-10, and seen-item policy. Re-evaluate `depth_0` and `depth_3` in the same
run; report differences from prior predictions separately. Identity mismatches
fail, while GPU floating-point differences in rankings are measured rather than
required to be bitwise equal. Frozen history bins describe the original full
histories and are not refitted to truncated inputs. Compare tail-only against
both reference modes with paired user bootstrap intervals, including long
histories, actual compression, and the original long/high-uniqueness slice.

Benchmark all three modes on the same real histories and fixed-length cases.
Tail-only timing includes the full raw embedding lookup and GPU selection, then
the shorter encoder and full-catalog scoring. It does not measure CPU input
cropping, an optimized tail-only serving pipeline, a cache, or service queues.
Use the same physical B32 recipe and report B1 separately. Benchmark work is
outside evaluation and checkpoint loading; no new training budget is consumed.

Outputs include `metrics.json`, `summary.md`, `run-manifest.json`,
`predictions.parquet`, `tail-cohorts.json`, `reproduction-check.json`, and
per-mode benchmark artifacts. Flat root metrics contain
`training_performed=false`, `training_epochs=0`, `test_evaluated=false`, and
`tail_ablation_complete=true` only after all required evaluation and diagnostics
complete. Do not copy the large source checkpoint into the new kernel output.

Interpretation: a tail-only quality drop establishes that removing the old prefix
affects this checkpoint. It does not isolate learned summaries from a shift in
input distribution. A small drop means this checkpoint works well on the tail;
it does not establish that old history is useless for every model. Independently
trained Recent128 and Mean controls would be subsequent experiments. User
bootstrap conditions on this single selected checkpoint and seed, and overlapping
cohort intervals are exploratory rather than adjusted for multiple comparisons.
