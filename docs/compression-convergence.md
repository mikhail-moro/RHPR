# Compression convergence experiment

> Historical research, closed 2026-10-21. See the [closure archive](research/closed-history-compression/README.md) for final results and status.

Train Full/OFF and PairMLP/OFF from scratch with maximum 30 epochs and patience
5, using `experiments/yambda_pairwise_compression_convergence_30ep.json`.
This checks whether their similar validation quality after ten epochs persists
after longer training. It performs no hyperparameter search and reserves test.

Both variants retain the previous pilot's seed42, raw context1028, tail128 loss,
512 negatives, architecture, optimizer, batch sizes and full-catalog validation.
Only the epoch limit and patience change. The optional operator profiler is
disabled; synchronized B1/B32 timings, analytic matmul FLOPs and memory
measurements remain enabled. The suite budget is four hours.

Start from fresh initialization: the previous checkpoints contain model weights
without optimizer state, so restarting their optimizer would change the training
trajectory. Compare the first ten epochs with the previous pilot as a
reproducibility check, then inspect the remaining validation curves, best
checkpoints, training times and paired-user quality differences.

Use the private kernel `carasik/rhpr-compression-convergence-30ep` to preserve
the original pilot's outputs. Dispatch the existing Kaggle workflow against the
exact branch revision with GPU enabled and Telegram notifications disabled.

The runner supports either the original Full/PairMLP x Time OFF/ON grid or the
complete Full/OFF versus PairMLP/OFF comparison. A missing matched control is
rejected; cohort analysis only reports time settings present in the experiment.

Best checkpoints still at epoch30 indicate that convergence remains unresolved.
If early stopping occurs, inspect the full curves rather than assuming the
architecture has reached its maximum possible quality. Any quality limitation
is conditional on this common recipe; later HPO and multiple seeds are separate
experiments. Context2048 and caching are also separate follow-ups.
