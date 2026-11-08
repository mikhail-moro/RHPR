# Independent full-generator-seed proxy diagnostic

Authorized 2026-11-07: exactly six fresh full-TIGER fits, one new generator seed
161803, 30 epochs, validation every5, two T4 workers. No proxy/tokenizer/content
training, TEST evaluation, tuning, extra seeds, retries, Colab deployment or
notifications. Config experiments/ml1m_proxy_target_replication_2t4.json.
Private kernel carasik/rhpr-ml1m-proxy-target, offline with three existing datasets
and two existing private completed parent kernels. Server cap135min; internal
suite7500s and each fit2300s. Expected1.5–2h session, not a measured new runtime.
Quota observed45.00h GPU remaining2026-11-07 before submission.

## Candidate and evaluation rule fixed before new outcomes

SUM and GradNorm indexes crossed with tokenizer seeds42/101/202. This balanced
six-index subset is selected by method/seed labels, not optimized on the forthcoming
target scores. Old indexes and holdouts have already been examined; this is an
exploratory replication of generator randomness, not a held-out tokenizer benchmark.
MTL is not newly trained or tuned. Broader tokenizer-family or dataset experiments
are not authorized by this pilot.

The original frozen SID arrays, catalogue, TRAIN/VAL cases and exact source hashes
are verified against the completed cleanMTL parents. Native GRID T5 4+4,d128,
FF1024,H6,kv64,4977920parameters,FP16train/FP32eval,nativeSDPA,512batch,Adam.001/
wd.0001,dropout.1,unmaskedbeam20,userhash2000. Same151166training windows per epoch,
2786validation users,history20. Every new model uses the same fresh initialization
and permutation seed161803, independent of historical full seeds42/101/202 and
mini seeds31415/27182. All30epochs/8880updates/sixVALchecks are required.

Primary target: fixed epoch30 VAL item NDCG@10, preserving the historical proxy
calibration target. Best-of30 is not substituted for fixed30. This cap does not
establish convergence. New early full5/10 checkpoints are within-trajectory
comparators, not independent-seed transfer. Historical full30 versus new full30
checks reference stability. Existing narrow mini epoch10 scores, separately per
seed and their mean, are copied from already verified original artifacts, including
kernel/code/SID/evidence hashes. Their complete reference object has its own SHA256
pin. No new notebook-parent mount or private-data upload is needed for these scalars.

Report Spearman, selected-index target regret, relative regret, top2 overlap and
top3 shortlist retention/regret, pooled and per method/index seed. Two-candidate
seed subgroups only describe ordering; six correlated candidates do not support
strong significance/universal-proxy claims. Preserve every validation prediction,
loss curve, initialization/index/case pin, checkpoint and measured training/fit time.
No automatic advancement or early stopping based on a handpicked rho threshold.

## Decision after collection

If mini retains good candidates against the independent full seed, a small new
tokenizer-family test becomes reasonable. If full-reference ordering is unstable,
quantify that noise before broader architecture search. If early full gives equally
useful choices for similar cost, retain it as the practical screening baseline.
None of these outcomes automatically authorizes another GPU run.

## Preflight

The preflight JSON records six real full-model initializer hashes (must agree),
original SID/case verification, balanced3+3workers and two tiny CPU training probes
through the actual30epoch/sixVAL native loop. Full repository regression runs before
publication. New runtime paths reuse the existing frozen-index mini worker; older
mini phases retain their contracts. Missing fits or source/index mismatches fail
the experiment; no shortened matrix is reported as complete.
