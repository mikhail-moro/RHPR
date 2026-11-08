# Actor versus director features in MovieLens100K TIGER

User-authorized run, 2026-11-02. Nine fresh pipelines: no people, cast only,
and directors only, crossed with fixed seeds42/101/202. Two T4 workers;
private carasik/rhpr-ml100k-people-roles. Existing shared DeepWalk vectors
are frozen: this is a role/readout ablation, not retraining typed graphs.

All arms retain Sentence-T5 text without names, ViT poster and MPNet plot,
the exact1273-film/943-user timestamp-group split, 20-item strictly-past
contexts, all85470 training cases per epoch, native RQ3x256/latent64,
and TIGER4+4/d128. This compares next-rated-item prediction, including all
rating values, not observed viewing order or a cold-start benchmark.

Architecture and initialization: preserve all4 modality heads, including
128D people input/output. No-people omits that entire block and its loss.
Role arms normalize the selected64 dimensions with their catalogue mean
and scalar RMS, then insert them into their original coordinates of an
otherwise zero128D block. Decoder MSE averages ONLY those64 selected
coordinates. Inactive-role inputs, biases in omitted decoder outputs and
inactive reconstruction targets contribute no loss. All active modality
reconstruction weights remain1. Quantization/commitment losses unchanged.
Equal schemas preserve the pinned neural initialization within each seed;
normalization and per-head metrics record selected coordinates explicitly.

Every arm uses3500 fixed continuous-AE updates, weight transfer, native
residual KMeans initialization and3000 RQ reconstruction updates. Fresh
Adagrad accumulators and AE transfer/initial latent drift are verified.
Fresh generator prefix20 epochs with validation5/10/15/20, then verified
resume of last weights/Adam/scaler/allRNG. Continue under prior policy:
min30/cap100, validation each new epoch, ES20 since best raw valNDCG10,
ReduceLR from21 (.5 after5 failures, threshold1e-4, floor1e-5). FP16 train,
FP32 evaluation; one final test from validation-selected weights, none at
prefix boundary. Existing models remain historical references; fresh no-
people models are the matched primary control. CUDA numerical variation
is not eliminated by matching seeds and initialization.

Preserve quantizer/head losses, AE/RQ curves, codes/books/maps, generator
curves/checkpoints, per-epoch validation beams/user IDs/targets and final
test predictions. Report whole-split quality and the exploratory same-
director/cast groups defined in the preceding audit; target-dependent
groups are diagnostics, not routing rules. Three fits share the holdout;
positive group results require caution about small groups and multiplicity.
No HPO, random-feature control, additional seeds, retries, changed graph
training, TPU or notifications are included. Any later identity baseline
would test whether DeepWalk adds value beyond shared person identity.

Configuration: experiments/movielens100k_people_roles_2t4.json.
