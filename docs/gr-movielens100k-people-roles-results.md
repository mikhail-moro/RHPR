# MovieLens100K: actor/director ablation results

Collected2026-11-02 MSK. Private carasik/rhpr-ml100k-people-roles v1
completed all9fits without failures in227.51min (3h47m31s) on2T4.
All stopped by early stopping, none hit the100epoch/time guard. This exceeded
our earlier2–3h estimate. No new training was launched during collection.

## Selected recommendation quality

Each test uses its own best validation checkpoint. NDCG@10:

| Input people role | Seed42 | Seed101 | Seed202 | Mean ± sample SD | Relative to fresh control |
|---|---:|---:|---:|---:|---:|
| None | .04561 | .04536 | .05013 | .04703 ± .00269 | — |
| Cast only | .04775 | .04311 | .04490 | .04525 ± .00234 | -3.8% |
| Directors only | .03568 | .04695 | .04282 | .04182 ± .00570 | -11.1% |

Mean Recall10 is .09579/.09544/.09155. Both role variants beat the control
on only1of3test seeds. These point estimates do not establish stable harm:
conditional shared-user bootstrap95% intervals for absolute mean testNDCG
changes are cast[-.00809,+.00431], directors[-.01159,+.00104]. The intervals
condition on the fitted models, exclude training/runtime variance and are
not adjusted for multiple comparisons. The3pipeline seeds share943holdout
users/targets and are not independent holdout replications.

Validation ranking differs: mean selected valNDCG10 none .06411,
cast .06845 (+6.8%,positive3/3seeds), directors .06284 (-2.0%,positive2/3).
This repeats the validation/test disagreement seen previously; it is not
proof of incorrect splitting or permission to select on test.

## The proposed director subgroup did not transfer to test

Same director means the target shares a director with one of the visible
last20films. These pre-existing, target-dependent groups are error diagnostics,
not usable serving-time routing rules.

| Group | No people | Cast only | Directors only |
|---|---:|---:|---:|
| Validation, same director, n54 | .05860 | .06757 | .08275 |
| Test, same director, n57 | .09735 | .08127 | .08705 |

Directors-only gains41.2% on this validation group, positive3/3seeds, but
loses10.6% on test, positive1/3. Its conditional subgroup delta95%CI is
[-.00695,+.05660] on val and[-.04670,+.02459] on test. Across171test user-seed
cases, top10hits are31control versus30directors, also30cast. Thus the old
full-graph subgroup observation did not establish a robust benefit that a
standalone director channel could reproduce. It does not prove that directors
or graph features are intrinsically useless. Complete subgroup tables also
retain cast-overlap and complement groups, not only this selected example.

Use the fresh no-people arm as primary control. Against historical no-people
models, seed42 reproduces the AE neural state and full semantic codes exactly,
but has a different generator result (old .05315,new .04561). Historical101/202
used adaptive AE durations4000/5000, while this matched experiment uses3500
for every arm; their indices therefore differ. Do not mix historical controls
into this comparison or attribute all repeat differences to numerical noise.

## Training and intervention verification

Best/stopped epochs by42/101/202:
- None30/50,22/42,29/49.
- Cast38/58,44/64,23/43.
- Directors10/30,27/47,23/43.

All inputs use the exact same frozen Sentence-T5/poster/MPNet vectors and
original shared-table DeepWalk features. Cast/director64D halves are separately
catalogue-centered/RMS-normalized, inserted into original128D coordinates with
inactive coordinates zero. Decoder loss only covers selected64coordinates.
No-people excludes the block/loss. All other active weights remain1. This tests
role readout under the current shared DeepWalk, not typed graph retraining,
DeepWalk against person identity, or graph benefit under another objective.

Independently verified:
- Official Kaggle COMPLETE, exact executable/config, all9trial results and
  complete worker reports. Output listing was paginated: both pages fetched.
 494numerical/metadata/log files downloaded (44,538,935bytes), hashes recorded;
 large .pt checkpoints remain on the private notebook.
- Independently decoded/rescored all291saved prediction files. Full943users,
 targets, NDCG/Recall5/10, invalid and duplicate beam slots match outputs.
 Catalogue order, semantic-code-to-SID conversion and frozen resume hashes match.
- Quantizer and generator neural initializations match declared seed-specific
 pins. Role normalization/loss columns, active modalities, raw/weighted AE/RQ
 losses,3500AE/3000RQ updates, native initialization and fresh RQ optimizer
 transfer/latent checks are consistent.
- Prefix20curves/selection, saved checkpoint provenance, Adam/scaler/allRNG
 resume at20/6680, same temporal case hashes, and all later LR/ES decisions
 independently replayed. Every epoch has85470examples/334attempted updates.
- Three AMP-skipped updates total: one each in cast42,cast101,directors101;
 all others0. No automatic precision/LR fallback. Test invalid slots:12in
 none202,1in directors202,0others; no test duplicates.
- All944repository tests pass (88skipped). No production code changed for
 collection; numerical analysis script and outputs are preserved separately.

The experiment provides no positive downstream result for isolating people
roles in this quantizer. Further coefficient/depth tuning is not supported
by these data alone. The graph representation, reconstruction objective and
rating-prediction protocol remain possible explanations, not established causes.

## Evidence

Executable682a23dabed6803d3340607f6ecd7924483ce3ba;
config experiments/movielens100k_people_roles_2t4.json.
Full analysis/replay/CSV/plots/ranks: /tmp/movie-roles-analysis.
Compact durable report/tables/script: artifacts/movielens100k-people-roles/analysis.
Raw numerical output: /tmp/movie-roles-completed. Standalone evidence ZIP:
/tmp/movielens100k-people-roles-numerical-evidence.zip. Temporary paths may
not survive environment replacement; original outputs/checkpoints remain Kaggle.
