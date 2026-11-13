# Recent15 LR pilot

One seed-42 model, trained from scratch with the same clean ML1M cases, 3+1 item index, 4+4 d64 architecture, initialization, Adam settings and batches as the unique-pair controls. Baseline is the saved seed-42 constant-LR 0.001 curve and model, not a new training run.

Pilot starts at LR 0.002 without warmup. ReduceLROnPlateau observes validation NDCG@10 every five epochs: after three consecutive validation checks without improvement, halve LR down to 0.0001. Early stopping uses 30 epochs since the best val NDCG, minimum 30 epochs, maximum 100. This is a bounded screening run; hitting the cap is not evidence of convergence. Teacher-forced validation CE is saved at every check; its sum over SID levels is a diagnostic, not item NDCG. Record AMP skips and wall-clock time. Evaluate both fixed last and val-selected best on the already inspected test set; test is exploratory, not used for LR or checkpoint selection.

Scheduler and early stopping are two changes relative to the saved constant-LR baseline, so the result cannot isolate the causal effect of a higher initial LR. Validation-loss computation is additional overhead; report optimizer and end-to-end times separately.
