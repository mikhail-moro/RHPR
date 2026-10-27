# User-level validation replay

Requested2026-10-27 after the first graph-placement TIGER comparison. The training
run kept only aggregate validation metrics; its user predictions were saved only
for test. This job restores all four selected epoch20checkpoints and evaluates
validation once. No training,optimizer,quantizer,graph changes,model selection,
hyperparameter search or test evaluation. CPUbenchmark:128users take15.85seconds
with two threads,so full four-model replay would take about3hours. Use a short
private two-T4 inference session instead,approximately5–8minutes including setup.

Config:`experiments/amazon_beauty_sid_validation_replay_2t4.json`. Parent outputs
are source0c1c44cd graph suite and source6e37388a native RQ-VAE baseline. Byte-pin
parent manifests,each selected checkpoint,complete SID matrix and original
training-selection report. Verify catalogue,data,generator,seed,training budget
and epoch20. Preserve suffix ownership. Restore weights strictly and verify
the full model state is unchanged after evaluation.

Use original GRID source/runtime,FP32,batch32,beam20,one held-out validation target
per user with last20TRAINitems as context. Save raw beam paths/scores,user IDs,
target IDs,per-user ranks,NDCG/Recall@5/10,and deviations from the original
GPU aggregate. Replay numerical drift is recorded explicitly,not hidden.
Two independent workers evaluate two checkpoints each;no hardware fallback.

Analyse paired users and predeclared descriptive cohorts:changed target paths,
changed context items,TRAINtarget frequency,history length,and membership in the
75%CF graph-fit users versus25%held-out users. These are exploratory validation
slices,not new confirmatory holdouts. Do not treat thousands of identical zero
NDCG scores as identical recommendation lists or user-level generalization proof.

No Telegram,extra training seeds,automatic retries or other new experiments.
