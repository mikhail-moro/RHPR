# Full training of the validation-selected actor graph, three seeds

User-authorized2026-11-03. Six fresh pipelines: no-people control versus
actor-only DeepWalk with people reconstruction coefficient0.1, crossed with
fixed pipeline seeds42/101/202. This coefficient was selected by mean
validation in the two-seed30epoch screen and is frozen here. No new HPO.

All model/data/features/quantizer settings match the screen: same1273films,
943users, timestamp-group leave-two holdout,85470strict-past cases per epoch,
context20, Sentence-T5 text without names, poster/MPNetplot, native RQ3x256/
latent64,3500fixed AE updates then3000RQ updates. Actor loss coefficient0.1
in BOTH AE/RQ; text/poster/plot weights1. Actor graph is the same frozen
15998-node CPU SGNS seed42 graph. Only cast64D enters the128D people slot;
inactive half is zero, normalization and MSE cover only selected64dimensions.
No-people omits the block and its loss. Neural initialization is pinned within
seed, including the restored seed202 declarations. Graph seed is not varied.

Full generator policy deliberately matches the earlier full-training work:
TIGER4+4/d128, FP16train/FP32eval, Adam.001, same256effective/128physical
batches; fresh20epoch prefix with validation5/10/15/20 and constantLR;
verified last/Adam/scaler/allRNG continuation in place from20. Subsequently
validate every epoch, min30/max100, ES20epochs since best raw validation
NDCG10. ReduceLR starts21: factor.5 after5failed checks, floor1e-5,
absolute threshold1e-4. One final test from validation-best weights, none at
phase boundary. Save per-level validation CE and all predictions/curves.

The screened30epoch checkpoints have intact optimizer/RNG states, but their
constant-LR history through30 differs from this schedule starting21.
Therefore all6pipelines are freshly fitted, including AE/RQ, for one common
full protocol. No weights-only continuation or silent LR-history substitution.
Historical screening and older full-training models remain references;
the fresh paired no-people fits are the primary controls.

Two T4 workers, three trials each. Pairs42 and101 share their respective
worker, pair202 is split across two same-type T4s to balance the6fits. Seeds
and initial weights do not ensure bitwise deterministic CUDA trajectories.
Expected roughly2–4hours plus queue/startup at recent speed; near-cap fits
may take longer. Private kernel carasik/rhpr-ml100k-actor-full-training.

Save raw/weighted per-head AE/RQ losses, books/codes/maps, source hashes,
initialization hashes, transfer/resume audits, prefix and full curves,
checkpoints, per-user validation/test beams and invalid/duplicate SID counts.
Report each seed, mean/sampleSD and paired user differences. The old holdout
was examined during screening; these3pipeline seeds include2used for tuning,
so this is an exploratory full-budget replication, not an untouched independent
confirmation. There is no actor-identity/random-feature baseline here and no
claim that DeepWalk itself is responsible for any gain.

No coefficient changes, extra seeds, automatic extensions/retries, TPU or
notifications. No earlier studies or paused TPU changes are overwritten.
Configuration: experiments/movielens100k_actor_full_2t4.json.
