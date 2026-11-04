# Context50 replication: seeds 101 and 202

Authorized 2026-11-04 (Europe/Moscow): repeat the context-50 ordinary-T5 experiment
on two more pipeline seeds. Four fresh generators: no-people control and actor
graph 0.1 for each seed 101/202. Reuse the completed seed42 results unchanged.

Config: `experiments/movielens100k_context50_seeds101_202.json`.
New private Kaggle kernel: `carasik/rhpr-ml100k-context50-replication`.
Two T4 workers each train one model at a time, seed101 then seed202. This is one
Kaggle session, without parallel duplicate notebooks.

Exactly retain the completed seed42 context50 protocol and the prior context40
101/202 replication: TIGER 4+4/d128/FF1024, 6 heads with d_kv64, ReLU, ordinary
eager T5 attention and learned relative bias. Context50 = 201 encoder SID tokens
including the prepended user token. FP16 AMP with FP32 weights/GradScaler,
physical128/effective256, LR0.001, weight decay0.0001. FP32 unfiltered beam20
evaluation. All30 epochs, validation5/10/15/20, exact optimizer/scaler/RNG resume,
then validation every epoch21-30 and the unchanged plateau LR schedule. Select
by validation NDCG@10, test once at that checkpoint. This cap is not convergence.

Use the original corresponding frozen SID index for each candidate/seed from
`carasik/rhpr-ml100k-actor-full-training v1`. No graph, encoder or quantizer fits.
Fresh predictor construction must match the original101/202 initial parameter
hashes. All parent files are SHA256 pinned. These are pipeline seeds: frozen
indices differ between seeds as well as generator initialization. They do not
isolate generator randomness on one common index.

Keep 85,470 training targets, 943 heldout users, 1,273 items and timestamp-group
splits. Verify original20 and40 suffixes, identical users/targets and split hashes.
Preserve all validation predictions/losses, epoch curves, test predictions,
repeat diagnostics, timing/memory and index/initialization/resume provenance.
No seen masking, wider beam, FlashAttention, TPU, batch/precision fallback, HPO,
extra windows/seeds, automatic resubmission or runtime extensions.

Quota before submission:3.57 GPU hours. The measured context50seed42 run took
66.87 minutes for two models; four models are provisionally estimated at2-2.5h
elapsed with2T4. Keep the per-model7200s guard and set the session timeout180min
to remain within the available quota; incomplete fits must not count as results.

After completion, aggregate all3seeds and report validation/test separately.
Compare validation against context20/40 using the same first30epoch selection
budget. Historical20test checkpoints are not universally budget matched.
The same holdouts have already been examined and window50 was chosen after
exploration. This replication measures robustness, not independent confirmation.
With fixed left padding, changing context width also changes relative distances
from the prepended user token; it does not isolate the value of older events.

Before submission:958 repository tests passed (91 skipped). Four full-size
constructor/backward probes at context50 matched original per-seed initializers
and had finite gradients. All pinned parent files, user/target orders and recent
20/40 suffixes were verified. Runtime changes only admit the registered new
seed/window matrix; no predictor/training/attention implementation changes.
