# ML1M Last-8 depth pilot

Fit Recent128 and Thinned64 from scratch with both four and eight encoder
layers. The original configuration fits both history variants at each depth in
one session. To use one Colab session per trial, append `#<trial_id>` to the
experiment path in the workflow input. Supply the paths as a JSON array in
`serial_experiments` when they should run one at a time. `after_run_id`
can delay the batch until an existing Colab workflow releases its session.
Each trial has up to 16,800 seconds
for training and validation within a 300-minute remote session. All sessions
emit the same cohort metrics for their view. Compute the cross-run differences
after all four fits complete. Keep four decoder layers, d_model 128, FFN width 1024, six heads,
FP16 training and FP32 beam-20 evaluation. All four fits use seed 101,
60 epochs, 95 updates per epoch, effective and physical batches of 512 and
32, and validation every five epochs. Select each checkpoint by raw
validation NDCG@10.

The single-trial launcher runs training independently of the SSH connection and fetches
the latest checkpoint after validation. If the VM disappears, GitHub Actions
exports the latest retrieved copy as `checkpoint.tar.age`, including last and
best weights, training curves and optimizer state in the last checkpoint.

The eight-layer model copies the initial weights of all common layers from
the four-layer model. Its four added encoder blocks keep their own initial
weights. Both history variants use the same model and initial tensors at
each depth. Recent128 always reads the full window. Thinned64 trains with
the registered 50/50 full and thinned optimizer-batch schedule, and uses
the compact view at evaluation. Thinning keeps every second event of the
old prefix and all 64 recent events.

The frozen index, training examples, validation examples and test examples
are pinned to the same MovieLens1M Last-8 parent. We stage the GRID source,
wheel bundles and pinned notebook files. The frozen index is reused directly.

| Encoder layers | History | Encoder positions | Active parameters | B1 forward GFLOPs |
| --- | --- | ---: | ---: | ---: |
| 4 | Recent128 | 513 | 4,977,920 | 3.946 |
| 4 | Thinned64 | 385 | 4,977,920 | 2.664 |
| 8 | Recent128 | 513 | 6,813,952 | 7.448 |
| 8 | Thinned64 | 385 | 6,813,952 | 4.990 |

FLOPs count supported operations in an FP32 teacher-forced CPU forward with
math SDPA and a multiply-add counted as two operations. The pair embedding
table is present in both histories and receives only singleton-token inputs.

Subtract the four-layer NDCG@10 gap between Recent128 and Thinned64 from
the eight-layer gap in the same direction. Use all test users and the
predefined group with at least 128 past events. The latter compares examples
that fill the nominal window. The encrypted results retain ranked
predictions, validation curves, checkpoints and paired cohort means. This
single-seed experiment tests depth under a fixed training budget.
