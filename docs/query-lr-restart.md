# Learning-rate diagnostic from the Query SSM pilot checkpoints

Configuration: `experiments/yambda_query_lr_restart.json`, seed 42.
This follow-up checks whether the weak cross-attention result was limited by
training progress or the common learning rate. It changes neither compressor
architecture nor predictor capacity. It is a small diagnostic, not a complete
hyperparameter search or a new test of Query SSM superiority.

| Parent model | Restart learning rates | Maximum additional passes |
| --- | --- | --- |
| Recent128 | 0.001, 0.003 | 48 per arm |
| Recent-conditioned cross-attention, four summaries | 0.001, 0.003 | 48 per arm |

The private parent output `carasik/rhpr-query-ssm-block-pilot` supplies the
pass-48 weights. Source revision, file lengths and SHA256 digests are pinned in
the configuration and verified before loading. Kaggle's kernel-source reference
does not itself pin an output version; checkpoint content hashes do. Incompatible
data, model, query protocol or checkpoint metadata stops execution.

The parent saved FP32 model weights only. Each arm therefore starts a **fresh
AdamW**, AMP scaler and random stream. This is an optimizer restart from pass 48,
not an exact continuation of its optimizer trajectory. Both rates within a model
family load the same checkpoint and train every parameter, including the item
embeddings, HSTU and compressor. Weight decay, clipping, loss and runtime remain
the same as in the parent experiment.

Training uses global schedule indices 49 through 96. Every pass supplies one
random-ended full training window per eligible user, not every possible prefix
of every user's history. All arms share endpoints, window order, targets and
negative samples at a given global index. Recent128 and cross-attention retain
the parent's causal 32-target protocol: training raw tail 159, query support the
first 128 tail events, and serving tail 128. See
[the parent protocol](query-ssm-block-pilot.md) for the exact alignment.

Validation runs at restart pass 0 and every eight additional passes. The parent
weights remain eligible if further training fails to improve validation.
Checkpoint selection uses the earliest strictly best macro-user NDCG@10.
Early stopping becomes eligible after 24 additional passes, with patience of
three scheduled validation checks and zero minimum improvement. Ties count as
non-improvements. NDCG@100 is recorded as a secondary metric. The maximum of 48
additional passes is fixed; the cost preflight may reject an unaffordable run,
but must not silently change that cap.

Learning-rate selection happens independently within each model family using
its best validation score; a tie favors the first registered rate, 0.001.
All four arms finish their selection before either selected model is tested.
Only the selected checkpoint for each family receives a new test evaluation.
Test scores cannot change learning rate, stopping or checkpoint selection.
Because the held-out period has already been inspected in earlier experiments,
this remains an exploratory follow-up. A paired user bootstrap describes
uncertainty conditional on this training seed.

Physical training batch 32, effective batch 128 and evaluation batch 32 match
the successful parent run. Disposable parent-loaded models profile successful
updates at both rates. Timing warmups never change the weights used for quality
training. The maximum planned work is checked against a 7,200-second execution
budget with safety factor 1.25. Expected T4 runtime is roughly one hour, with
additional startup and checkpoint I/O. The workflow has a separate 180-minute
cancellation timeout. The same hash-locked offline runtime is retained.

Rich outputs record parent verification, baseline validation, all training and
validation passes, global schedule fingerprints, stopping reasons, validation
selection and selected-model per-anchor predictions. Best FP32 model weights
and final optimizer/scaler/random state remain private Kaggle outputs. Standard
`metrics.json`, `summary.md` and `run-manifest.json` retain the experiment
contract. Checkpoint files are excluded from automatic GitHub artifact download.
