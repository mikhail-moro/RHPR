# Six full TIGERs: exact continuation to an operational validation plateau

User authorized2026-11-07 after the independent full30 proxy diagnostic.
Continue ONLY the six full models from carasik/rhpr-ml1m-proxy-target/1,epoch30,
seed161803. Same frozen SUM/GradNorm indexes x42/101/202,same data,initialization,
AdamLR.001/wd.0001,FP16/scaler,SDPA512,dropout,beam and validation protocol.
No proxy/tokenizer fits,old-seed retraining,TEST,HPO,new data,Colab,Telegram or retries.

Validation every5,total minimum60 and maximum100 epochs. Stop when there has been
no strictly better validation item NDCG@10 for20 epochs(4checks),after minimum60.
No LR scheduler is added. At100 without the stopping criterion,report censored,
not converged. Time exits/incomplete fits fail the suite. Selected best checkpoint
over inherited+new validation history is the new target;fixed30 remains reported
as a separate historical primary. Early stopping is an operational plateau rule,
not proof of global convergence. Same six related indexes and validation users;
no claim of held-out tokenizer-family/dataset generalization.

Reuse full epoch30 last checkpoints with Adam state,scaler,torchCPU/CUDA,Python,
NumPy RNG,and the original selected checkpoint and curves. Every original file
is SHA256/size pinned in config;parent manifest/plan/experiment also pinned.
Six actual last checkpoints inspected:30epochs/8880updates/95Adam states/1CUDA RNG;
best checkpoint epoch/score agrees with saved selection. Restore enforces same
SID map,cases,optimizer-affecting hyperparameters and original data permutation
schedule. RNG restoration checked immediately before the first continued epoch.
This does not promise bitwise deterministic CUDA kernels.

Private kernel carasik/rhpr-ml1m-proxy-plateau;twoT4 workers,three resumes each.
Three existing datasets and three completed private parent sources;no uploads.
Per-fit additional wall cap5000s,suite15500s,server270min. Expected1.5–4h session
depending on stopping. Observed43.43h GPU remaining before submission.
Save all inherited/new predictions,curves,last/best checkpoints,resume verification
and censoring flags. Do not extend automatically if models reach100.

Preflight: native real epoch30 model/Adam restore,small real30->60+ continuation,
RNG verification,existing resume tests and repository regression. Config
experiments/ml1m_proxy_plateau_2t4.json;preflight JSON records actual details.
