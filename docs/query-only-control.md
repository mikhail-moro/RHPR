# QueryOnly training control

Configuration: `experiments/yambda_query_only_control.json`, seed 42.
This is one fresh training arm: four learned queries attend to the final 128
events of H, followed by the shared summary FFN and the same SASRec predictor.
It removes both the full-H readout and the SSM scan. It is distinct from
truncating the inputs of an already trained QSSM checkpoint.

For the complete train-only sequence T, the original fixed schedule retains
`end=len(T)-1` and its identity-bound negative seed. Input collation takes
`T[end-256:end]` before embedding lookup; its first 128 events supply Q, and
its last 128 events supply R. The labels are the same final 128 next items as
the full-prefix pilot. Earlier events are not transferred to the GPU or read
by the compressor. No shortened-cohort or changed endpoint is substituted.

The architecture emits `q + FFN(LayerNorm(q))` for each query, then packs four
summaries before R128. Query generation and FFN have the same initial weights
as the parent query compressors. The item table, SASRec and random stream
share their initial state with the parent. In particular, the positional
table remains sized for 32768: changing it to 256 would change initialization
and parameter count. Summary recency remains128 and raw positions127..0.

Training is fixed at 30 passes, physical batch 32/effective 128, with the same
AdamW, learning rate 0.001, weight decay, clipping, AMP, order, negatives and
128 supervised positions per user. A disposable model profiles real successful
updates before training starts; profiling weights are discarded. Validation
at 10/20/30 selects the earliest strictly best macro-user NDCG@10 checkpoint.
No hyperparameter or architecture grid is included in this control.

The globally split cohort, catalogue, anchors and OOV policy remain pinned.
Validation and test use only the observed query128+raw128 suffix and one final
prediction per anchor, ranking the full catalogue at 10/100. Test is evaluated
once after the checkpoint is fixed. Comparisons reuse the parent saved ranks,
whose content hashes and manifest are verified. Since that test period has
already informed exploratory research, these are exploratory comparisons;
additional seeds and an independent temporal replication are still needed.

The run records actual 256-event context hashes separately from the common
reference hashes, training/validation costs, per-anchor ranks and paired
user intervals. Best FP32 weights and the final Adam/scaler/RNG state stay
private Kaggle outputs. The execution budget is 1200 seconds, with a separate
60-minute Actions timeout. A shortened scan is not used; QueryOnly contains
no selective-scan operator and needs no native SSM gradient audit.
