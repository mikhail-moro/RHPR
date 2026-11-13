# Unique-pair study: plateau continuation

Continue all nine Pair20 3+1, Recent15 and Full20 models from their exact epoch-50 checkpoint. Preserve Adam and gradient scaler, CPU/CUDA/Python/NumPy random states, original item and pair indexes, old curves and the previously selected best checkpoint. Inputs and source files are content-pinned.

Keep the original constant learning rate, training data, batch size, FP16 and validation every five epochs. Stop after 30 epochs without exceeding the existing best validation NDCG@10, including the history before epoch 50. Run at least through epoch 55; cap total training at 120 epochs and report any `epoch_limit` or `time_budget` separately from `early_stopping`. Plateau is operational, not proof of convergence. Evaluate the validation-selected best and final checkpoints on the previously viewed test set; this remains exploratory and does not change stopping decisions.
