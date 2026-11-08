# Full actor-graph comparison: three-seed results

Collected2026-11-03 MSK. Private carasik/rhpr-ml100k-actor-full-training v1
completed all6fresh pipelines without failures on2T4 in159.34min (2h39m).
All stopped by early stopping, none reached the100epoch or time guard.
No new compute was submitted during collection.

## Selected recommendation quality

The actor coefficient0.1 was frozen from prior validation screening. Each
model uses its own best validation checkpoint; test is evaluated once.

| Seed | No-people test NDCG10 | Actor graph .1 | Relative change |
|---|---:|---:|---:|
|42|.04530|.04593|+1.4%|
|101|.05071|.04298|-15.2%|
|202|.04833|.05273|+9.1%|
|Mean|**.04811**|.04722|**-1.9%**|
|Sample SD|.00271|.00500|—|

Graph wins2/3test seeds, but the loss on101outweighs those gains. Mean
Recall10 is .10392 without people versus .10145 with graph (-2.4%).
The conditional shared-user bootstrap95%CI for absolute mean testNDCG delta
is[-.00714,+.00539], crossing zero. It conditions on the3fitted pairs,
excludes training/runtime/graph-seed uncertainty and multiplicity, and does
not establish significance, equivalence or noninferiority.

On validation the graph loses on ALL3seeds. Mean selected valNDCG10 is
.07077 for control versus .06521 for graph (-7.9%). Per-seed relative
changes are -3.0%/-15.0%/-5.2%; conditional shared-user deltaCI also
includes zero[-.01287,+.00177]. There is no basis here for claiming that
the selected graph variant gives a robust downstream quality improvement.
Seed202was not used for coefficient selection and favors graph on test,
but one new seed does not provide independent confirmation on a new holdout.

Best/stopped epochs(42/101/202): control26/46,31/51,34/54;
graph31/51,48/68,29/49. The graph101fit already runs much longer than its
control and selects after LR reduction; no epoch-cap failure explains it.

## How this relates to the thirty-epoch screen

The preceding screen reported mean test+7.4% on42/101. The new full run is
an end-to-end fresh comparison under the declared full schedule, not a
continuation of those exact screened indices. The AE/RQfits were also fresh.
The screened constant-LR-through30 checkpoints could not reproduce the
full schedule with LRadaptation from21.

For both candidates on42/101, independently compared with the screen:
- Original feature hashes, catalogue order and normalization match.
- Quantizer training configuration matches except the added202initialization
  pin. Resolved native model, initial AE neural-state hash and generator
  initial parameter hash match.
- Final AE neural-state hashes and semantic-code matrices differ in all4
  comparisons. This is not just token relabeling: first-level adjusted Rand
  agreement is .263/.235 for control42/101 and .310/.268 for graph42/101.
- Generator training CE differs already at epoch1, before the LR schedules
  diverge. The precise numerical/runtime cause has not been identified.

Thus the earlier gain did not survive this fresh full-pipeline comparison,
but its disappearance cannot be causally attributed solely to additional
generator epochs, convergence or LR scheduling. There is additional
representation-training variation despite matching seeds/initial states.
A strictly isolated duration experiment would require frozen source indices
and a common generator training history. No such extra run is authorized
or launched here. Current fresh paired controls remain the primary comparison;
do not substitute the earlier controls or select checkpoints using test.

This limits the positive article claim. The present data document sensitivity
and an inconclusive/negative full-budget comparison; they do not establish
that graph features or DeepWalk consistently improve GR. They also do not
establish that graph features are universally harmful. No actor-identity or
random-feature control was included, so any individual gain cannot yet be
credited specifically to learned graph structure.

## Protocol and verification

Same1273film/943user timestamp-group split and85470strict-past next-rating
cases per epoch, visible history20, same Sentence-T5 text/poster/MPNetplot.
Frozen actor-only DeepWalk:15998nodes,graph seed42, no interactions. Actor
64D input is normalized separately, zero-padded into the128D slot and only
its64decoder coordinates contribute people MSE. Weight.1 applies in both
3500fixedAE and3000nativeRQ updates; other active weights1. Control skips
people block/loss. RQ3x256/latent64 and TIGER4+4/d128 unchanged.

Fresh20epoch prefix,val5/10/15/20; verified last/Adam/scaler/allRNG resume
at20/6680. Afterwards validation each epoch,min30/max100,ES20, prior
ReduceLRfrom21. One final validation-selected test, none at phase boundary.

- Official Kaggle COMPLETE, exact executable/config and6successful trial and
  worker results.372numerical/metadata/log files downloaded (34,345,527bytes)
  from one listing page; all downloaded hashes verified. Large .pt files
  remain on the private notebook.
- All229saved val/test prediction files independently decoded/rescored;
  matching943users/targets, NDCG/Recall5/10, invalid/duplicate beam slots.
  Independently reconstructed all train/val/test cases and verified digests.
- Actor graph/sourceID/provenance hashes and each arm's normalization verified
  from the original raw feature arrays. Seed-specific neural initialization
  pins, AEtransfer/freshRQoptimizer/initial latent checks and raw/weighted
  losses match the frozen protocol.
- Verified prefix histories, source index hashes and optimizer/scaler/RNG
  resume; replayed every LR and early-stopping decision. Every completed
  epoch has85470cases/334attempted updates, accounting for AMP skips.
- AMP-skipped updates: graph42=1,graph101=3,control202=1; others0. All prefix
  phases have0skips. No automatic precision/LR fallback. Test invalid slots:
  control42=2,control202=1,others0; no duplicates. They do not explain the
  main seed101quality gap.
- All949repository tests pass (90skipped). No production code changed during
  collection; new numerical replay/report are preserved separately.

The same holdout has already been inspected during development, and42/101
were used for earlier tuning. Three pipeline seeds are not three independent
holdout replications. This remains next-rating prediction, not observed VOD
viewing chronology, a cold-start benchmark or a definitive publication claim.

## Saved evidence

Executable70a0c16d65fab69df3d64742fee527975e12df70.
Configuration experiments/movielens100k_actor_full_2t4.json.
Raw outputs: /tmp/movie-actor-full-completed.
Analysis/CSV/plots/per-user ranks/replay: /tmp/movie-actor-full-analysis;
workspace link artifacts/movielens100k-actor-full/analysis.
Bundle /tmp/movielens100k-actor-full-numerical-evidence.zip includes numerical
outputs plus raw frozen features and the four screened reference fits needed
for replay. No large checkpoints or signed URLs. Temporary local data may
not survive environment replacement; source runs and weights remain Kaggle,
actor features are in their private dataset and report/replay are in the repo.
