# Independent full-TIGER seed check: completed exploratory results

Private carasik/rhpr-ml1m-proxy-target/1 COMPLETE,93.3434min,finished2026-11-07
09:54:04UTC. Executable5ed498804545ea41686eee9eaa8d9f67747be0db. All six fits
30epochs/8880updates/sixVALchecks/151166TRAINwindows per epoch/0AMPskips.
New common generator seed161803;frozen SUM/GradNorm indexes x42/101/202.
No TEST evaluation or new tokenizer/proxy fitting. No additional compute authorized.

## Primary result

On these SAME six candidates, narrow mini epoch10(two-seed mean) versus old full30
rho=.885714; versus new independent full30 rho=-.028571. Historical full30 versus
new full30 rho=.257143. Single mini seeds31415/27182 give .257143/-.028571 against
new full. New full10 versus new full30 rho=.028571, so short full is not a successful
reference-selection baseline on this run either. Early full shares the target
trajectory;mini and historical full do not share the new generator seed.

| Index | Historical full30 VAL NDCG10 | New full30 VAL NDCG10 | Mini10 two-seed mean | New best epoch |
|---|---:|---:|---:|---:|
| sum_seed42 | 0.064870 | 0.065339 | 0.033294 | 30 |
| gradnorm_seed42 | 0.065226 | 0.065429 | 0.032892 | 30 |
| sum_seed101 | 0.058728 | 0.062603 | 0.031356 | 25 |
| gradnorm_seed101 | 0.064202 | 0.066328 | 0.030949 | 30 |
| sum_seed202 | 0.071987 | 0.065194 | 0.038320 | 20 |
| gradnorm_seed202 | 0.071192 | 0.070583 | 0.034692 | 30 |

Mini chooses SUM202, now rank5/6: .06519357 against best GN202 .07058343,
absolute regret .00538986,relative7.636%. Historical full30 and new earlyfull10
choose the same SUM202. Mini shortlist3 retains GN202;with six candidates, merely
retaining one winner in three is weak evidence(random uniform shortlist3 retains
it with probability.5). All estimates are based on this finite observed candidate
pool;no uniform superiority or inferiority claim follows.

Several rankings concern near ties: new SUM42=.06533901,GN42=.06542940 and
SUM202=.06519357. A low rank correlation need not imply large practical differences
between every swapped pair. Even the observed winner-versus-mini-choice gap has
conditional shared-user bootstrap95% interval[-.001107,+.012239]. This is for fixed
fitted models and a pair identified after viewing outcomes;it excludes training
seed uncertainty and selection correction,so it does not establish equivalence
or a robust7.6%loss. Conditional rho intervals are similarly wide:mini/new
[-.542857,.885714],old/new[-.257143,.942857]. Only six related candidates and one
new generator seed were evaluated.

New GN beats paired SUM in all three tokenizer seeds, but deltas are .00009038,
.00372471,.00538986;all three conditional user intervals crosszero. Mini ranks SUM
above GN for each pair. This is descriptive;one new training seed does not establish
an architecture-by-tokenizer interaction or a GradNorm advantage.

## Post-hoc sensitivity (not replacement primary targets)

Mini versus new best-of30 rho=.428571;versus mean of new epochs20/25/30 rho=.485714.
Old/new full late-epoch mean rho=.485714. Mini versus the mean of both full-seed
results rho=.657143. These diagnostics indicate sensitivity to the selected
reference/training trajectory;they do not restore a prospectively validated proxy.
The two-full-seed mean partly includes the old target used to choose mini settings.
Four of six new best checkpoints are at30,so neither fixed30 nor best-of30 is
established converged performance. Historical full seeds42/101/202 were coupled
to tokenizer-seed groups;the new seed is common across all candidates. This is a
replication of full training per fixed index,not a balanced multi-seed variance study.

## Cost and verification

Mean full epoch training58.608s,mean complete fit1826.437s(30.44min). Existing
mini two-seed epoch10 estimates cost about13.4min/index versus about10.1min for new
full10(training plus recorded ranking validation only;setup/CEvalidation omitted).
Thus no robust budget-constrained selection advantage has been established.

Independently replayed36newVAL+12historicalminiVAL+6historicalfullVALpredictionfiles
against identical users,targets and SID arrays. NDCG/Recall5/10,invalid and duplicate
counts match to1e-12. Config/source/manifests/casehashes,all six identical initializers,
parent artifact/proxy reference hashes,training budgets and worker aggregation
match. Original saved correlations recomputed exactly. Top metrics/summary/manifest
all valid. CPU bootstrap and plots only;no new inference or GPU compute.

Full data docs/gr-proxy-target-results.json;replay
scripts/research/analyze_proxy_target.py;local raw/tmp/proxy-target-completed and
analysis/tmp/proxy-target-analysis,plot proxy-target.png/.pdf. EvidenceZIP retains
numerical outputs,reference predictions,TRAIN/VAL cases and replay instructions;
full weights stay on private Kaggle.

## Interpretation

The candidate mini proxy has not passed this independent target-seed check. This
also exposes weak repeatability of the full-model reference on closely performing
indexes;it does not isolate small-model architecture as the cause or disprove
proxy selection in other regimes. Do not expand a broad proxy-architecture or
joint-learning sweep based on the original pooled correlation. A next protocol
needs to address reference uncertainty,practically meaningful score differences
and/or larger candidate diversity. No new GPU run is implied by this report.
