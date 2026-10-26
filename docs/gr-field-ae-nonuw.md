# Non-UW continuous-AE controls

User requested alternative loss aggregation after the first suite. Four fresh
5000-update trials on two isolated T4 workers,config
`experiments/amazon_beauty_field_ae_nonuw_2t4.json`. Same catalogue,frozen title
targets,all inputs,native neural initialization,latent64,batch2048,FP32 and
neural Adagrad0.001. No RQ,SID,generator or recommendation evaluation.

The completed first suite already supplies the UW and fixed-normalized controls.
Original UW:brand accuracy69.75%,category F1 49.52%,price MAE9.68,title
MSE0.00917425. Fixed normalization:11.02%,70.50%,0.647,0.000240686.
Title-only MSE0.000121211;constant mean vector MSE0.000287248. The strong
tradeoff motivates checking both scale calibration and adaptive weights.

Let L be raw mean CE/BCE/Huber/MSE. Constants c are catalogue errors of
fixed predictors:brand empirical prior6.5500196,categories marginal
probabilities0.05163986,median standardized price0.25537452,mean frozen
title vector0.0002872481. No next-item targets are used to fit them.

|Trial|Objective|
|---|---|
|equal_sum|Sum L_i;no learned weights|
|sqrt_normalized|Sum L_i/sqrt(max(c_i,1e-6));softer calibration than previous L_i/c_i|
|dwa|Sum w_i L_i/c_i;DWA with temperature2|
|gradnorm|Sum stop_gradient(w_i) L_i/c_i;GradNorm alpha1.5|

Equal sum and square-root normalization interpolate exponent0 and0.5 against
the already completed exponent1 control. These are separate numerical
objectives;do not rank their total loss. All per-field error functions and
reductions remain unchanged;this suite studies aggregation,not cosine/focal
loss or class-rebalancing. No unused UW parameters receive gradients.

DWA and GradNorm are adapted from LibMTL revision
`4336804847eaa5e0b924b743d76beec7ac3fdc97`,MIT license already bundled.
References:LibMTL/weighting/DWA.py and GradNorm.py;implementation in
`src/rhpr/gr/vendor/libmtl_balancing.py`,with attribution in vendor NOTICE.
No additional package installation is required. This is an adaptation to the
existing loop,not an unmodified invocation of the full LibMTL trainer.

DWA uses observed-item-weighted task mean errors from two complete shuffled
catalogue passes. Here a pass has six batches,including the shorter last batch.
First two passes use w_i=1;then r_i=mean_i(t-1)/mean_i(t-2) and
w=K softmax(r/T),using the two most recently completed passes.
Weights sum to K=4;fixed loss normalization precedes DWA.
Magnitude calibration and relative-rate adaptation thus have separate roles.

GradNorm uses K softmax(logits),logits initially1,as in LibMTL. First catalogue
pass uses equal multipliers and establishes the reference task means. Later
updates compute each normalized task's gradient norm on the last shared
decoder Linear weight tensor. This is an actual shared parameter gradient,
not a representation-gradient proxy. For G_i=w_i*norm_i and relative training
rate r_i=(loss_i/reference_i)/mean(loss/reference),minimize sum|G_i-target_i|
with detached target_i=mean(G)*r_i^1.5. Gradients from this auxiliary objective
update only the logits;the neural objective treats weights as constants.
No second-order neural gradients. A separate native Adagrad group uses LR0.025
for logits;neural LR remains0.001. This optimizer choice is an explicit
adaptation,not a claim about LibMTL's complete trainer configuration.

An absent task is excluded from the neural objective as before. GradNorm skips
only its weight update when a batch lacks a task;all four tasks are observed
in actual large batches. Complete passes must observe every task. Pass sums,
counts,recent means,initial references,weights/logits and optimizer state are
checkpointed;partial final passes remain recorded without fabricating a full
pass. Log effective weights and auxiliary gradient norms/targets every500steps.

Worker0 runs GradNorm then equal sum;worker1 runs DWA then square-root
normalization. Require two CUDA devices with no hardware fallback. The
already-tested subprocess orchestration retains successful trials on sibling
failure and fails the overall suite if any trial fails or initial hashes differ.
Fresh same-seed trials are diagnostic controls,not independent replications.

Compare per-field accuracy/F1/MAE/MSE and title error relative to mean-vector
baseline;inspect task-weight trajectories. Equalizing scalar losses,gradient
norms or relative learning rates are different goals. None guarantees equal
task quality or better downstream recommendation. A reconstruction tradeoff
does not by itself identify a gradient-conflict cause or fix SID collisions.
