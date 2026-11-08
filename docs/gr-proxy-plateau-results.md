# Proxy calibration after full-TIGER plateau continuation

COMPLETE: all six full TIGERs satisfy the predeclared operational ES rule:
minimum60/maximum100totalepochs,validation5,20epochs without a strictly better
VALitem NDCG10. No scheduler,LR/model/data/index change. Same generator seed161803.
These are six existing frozen SUM/GradNorm indexes x42/101/202;no new MTL,
tokenizer or mini training. No TEST. No additional compute authorized.

Four completed fits were recovered from private carasik/rhpr-ml1m-proxy-plateau/1
(executable64500ca852e7bc478df3127e3421d661f36bfa12). Only the two202models continued
in carasik/rhpr-ml1m-proxy-plateau-finish/1(executableac6b5de04dc492fce22af3e885bdc190c1d7bd7b).
The finish job COMPLETE in56.119min at2026-11-07T14:28:51UTC,not the runtime of all
six from scratch. All models resumed original full30Adam/scaler/RNG checkpoints.
The four recovered models were not retrained.

## Quality and stopping

All values below are full-generator validation scores. Best first30 follows the
ordinary checkpoint-selection policy;fixed epoch30 is a distinct earlier target.

| Index | Best VAL NDCG10 within first30 | Selected after continuation | Best epoch | Stop epoch | Total AMP skips |
|---|---:|---:|---:|---:|---:|
| sum_seed42 | 0.065339 | 0.069266 | 40 | 60 | 0 |
| gradnorm_seed42 | 0.065429 | 0.065429 | 30 | 60 | 0 |
| sum_seed101 | 0.067229 | 0.070717 | 80 | 100 | 7 |
| gradnorm_seed101 | 0.066328 | 0.066399 | 45 | 65 | 2 |
| sum_seed202 | 0.069079 | 0.069456 | 65 | 85 | 5 |
| gradnorm_seed202 | 0.070583 | 0.072976 | 45 | 65 | 2 |

No model is marked budget-censored. SUM101 stops at100withbest80:the20epoch
criterion is met there,not merely the budget cap. GN42 best30stops60because of
minimum60. Other models stop20epochs after their best. Operational ES does not
prove global convergence. Mean per-index relative gain versus best first30is
2.54%;five of six find a later strict improvement,three gain about3.4–6.0%.
These are selected VAL maxima,not independent TEST gains;more checks can raise a
maximum through noise. Do not report SUM101's fixed30->bestlong rise as the whole
benefit of longer training:best first30 was already .067229 versus .062603 at30.

## Proxy result and reference definition

| Comparison, SAME six indexes | Spearman |
|---|---:|
| Mini10(two-seed mean) -> full fixed30 |-.028571|
| Mini10(two-seed mean) -> full best within30 |.428571|
| Mini10(two-seed mean) -> full plateau-selected |.428571|
| Full fixed30 -> full plateau-selected |-.085714|
| Full best within30 -> full plateau-selected |.771429|
| Full best within10 -> full plateau-selected |.371429|
| Full best within15 -> full plateau-selected |.657143|
| Full best within20 -> full plateau-selected |.828571|

The earlier fixed30slice was a fragile reference. The proxy's positive .43also
already appears when evaluating against best-within30;therefore the change from
-.03to+.43cannot be attributed solely to extra training epochs. The identical
.43values do not imply identical target rankings. Longer training improves some
scores and changes order,while early checkpoint selection explains part of the
previous apparent failure. Fixed full25happens to correlate .942857with final,
but selecting epoch25after inspecting this is post-hoc and not a validated recipe.
Full-best-within30and the selected-after-ES target also share the same trajectory;
this is not independent-seed replication. Historical different-seed full30 versus
new ES target is budget-mismatched,so not a clean estimate of training-seed noise.

Mini individual seeds31415/27182 give .142857/.428571 against final. Within the
three SUM indexes mini rho=-.5,within three GN rho=.5;groups are too small and
related to infer a general method-specific rule. Pooled .43is not established
portable calibration or proof of useful optimization gradients.

Mini chooses SUM202:finalfull .06945629 versus bestGN202 .07297643,regret
.00352014(relative4.8237%). The chosen index now ranks3/6. Mini top2contains the
winner;one six-candidate result is weak evidence of shortlist reliability.
Conditional shared-user bootstrap95%gap interval[-.004163,+.010996]includes0;
mini correlation interval[-.542857,.885714]is very wide. These intervals hold
selected fitted checkpoints fixed and exclude training/tokenizer randomness,
checkpoint/index selection correction and item dependence. No equivalence or
robust quality-loss claim. Six pre-examined related indexes,one long full seed,
same2786VALusers;no new tokenizer-family/dataset transfer was tested.

## Cost and verification

Mean cumulative full fit73.34min/index including inherited first30;mini two-seed
10epoch cost13.36min/index(training plus recorded ranking validation),full15about
15.14min on that narrower timing basis. These estimates exclude tokenizer/content
fitting and are not a measured end-to-end search speedup. Early full with checkpoint
selection must remain a comparator for proxy-based selection.

Independently replayed105predictionfiles:87fulltrajectory files(57recovered+30from
new202models,including inherited early predictions),12mini10files and6historical
full30files. Same users/targets/SIDs,NDCG/Recall5/10,invalid/duplicatecounts match
1e-12. All435total model-epochs satisfy successfulupdates+AMPskips=296*epoch;
16total skips across six models. Selected checkpoint,case/index/source/proxy
pins,all75nonweight recovered artifact hashes,worker aggregation and zero-training
recovery flags verified. All8recovered last/best weight files were inspected/hashed
before the recovery launch;new202weights remain in their private finish output.
Source-state RNG/optimizer audits match. Full metrics/summary/manifest valid and
saved correlations recomputed exactly. No GPU,inference or TEST in collection.
Repository checks1032tests/91optional skips/0fail;CPU replay supplies numeric
verification. JSON docs/gr-proxy-plateau-results.json;replay
scripts/research/analyze_proxy_plateau.py. Local raw/tmp/proxy-plateau-finish-completed,
recovered/tmp/proxy-plateau-failed,analysis/tmp/proxy-plateau-analysis;
plot proxy-plateau.png/.pdf. Evidence ZIP retains both result sources and references,
TRAIN/VALcases,verification and reproduction instructions;weights stay on Kaggle.

The earlier assertion that the proxy failed to transfer needs this qualification:
it failed the fixed30reference check;under normal selected-checkpoint references
it has a positive but weakly established signal. Do not treat either result as a
universal proxy verdict. Subsequent work should fix a reference/selection protocol,
retain early-full baselines,and validate on fresh tokenizer candidates before
broad architecture or joint-learning claims. No new training is implied.
