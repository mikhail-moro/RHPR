# Plateau completion recovery after an AMP accounting assertion

Private carasik/rhpr-ml1m-proxy-plateau/1 ERROR,executable64500ca852e7bc478df3127e3421d661f36bfa12.
Four models reached the specified early-stopping criterion. The post-fit guard
incorrectly required successful Adam updates=all scheduled batches. The engine
already counts successful updates and AMP skips separately and enforces finite
losses and a maximum16gradient-overflow skips. Correct invariant is
successful_updates+amp_skipped_updates=ceil(examples/batch)*completed_epochs.
The fix does not change training,precision,LR,model,examples or stopping policy.

| Original trial | End epoch | Successful updates | AMP skips | State |
|---|---:|---:|---:|---|
| SUM42 |60|17760|0|Completed |
| GradNorm42 |60|17760|0|Completed |
| SUM101 |100|29593|7|ES criterion at100,best80 |
| GradNorm101 |65|19238|2|ES criterion,best45 |
| SUM202 |30|8880|0|Not started in continuation |
| GradNorm202 |30|8880|0|Not started in continuation |

Workers stopped after the erroneous assertion on101,so202 never started. The
four completed results are recoverable without training. Replayed all57saved
validation predictions against users/targets/frozenSID,NDCG/Recall5/10 and invalid/
duplicatecounts match1e-12. Every epoch satisfies the corrected update accounting.
Last/best weights of all4were downloaded/hashed/inspected:epochs/updates/skips,
best checkpoint,95Adam state entries and1CUDA RNG agree with metadata. No full
experiment metrics or run-manifest exists because the suite aborted.

New private kernel carasik/rhpr-ml1m-proxy-plateau-finish preserves the failed
notebook output. The first dispatch42376103455 at3ec6719d was rejected by SaveKernel
HTTP400 before a session/version or GPU work was created. It attempted to attach
this failed output as a fourth kernel source;the response did not expose a detailed
cause. Revised deployment retains only the original three successful kernel sources.
The failed source plan,experiment,4last/best checkpoints,sidecars and57predictions
(83files total) remain size/SHA256 pinned as provenance. Their already verified
selection/curve/generator records are embedded in the private config as external
completed results. Weights and predictions stay in the original private failed
kernel and local evidenceZIP;they are not silently claimed as newly generated or
copied into the finish kernel. Final CPU analysis joins these existing57prediction
files with the new202outputs. Recovery writes explicit external-checkpoint-reference
sidecars. Continue ONLY SUM202 andGradNorm202 from the ORIGINAL full30 source with
exactoptimizer/RNG.
Same six-model final matrix,min60/max100,VAL5,patience20,LR.001,FP16SDPA512.
No new seeds,tokenizer/proxy fits,TEST,HPO,hardware fallback or repeated completed
training. This completes the originally authorized six models,not a new sweep.

Suite5500s,perfit5000s,server100min,twoT4;expectedroughly35–85min includingsetup,
depending on ES. No automatic further retry or extension. Config
experiments/ml1m_proxy_plateau_finish_2t4.json.

Regression includes the exact65epoch/2skip and100epoch/7skip cases and still
rejects missing batches/time exits. Native CPU recovery copied the real65epoch
model and all pinned sidecars/predictions bit-for-bit while a patched training
function would raise if called. Original FP32 continuation/RNGprobe remains valid;
no model-training code changed in this fix. Source and submission receipt identify
the new deployment;original failed evidence is retained separately.

External-reference regression also forbids model construction or training for all four reused results. This fallback uploads no new dataset and repeats no completed training.
