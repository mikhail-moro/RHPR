# ML1M clean multimodal MTL: two additional seeds

Authorized 2026-11-05: repeat **all four** methods (sum, UW, GradNorm, PCGrad) with two additional pipeline seeds, 101 and 202. Combine these with the completed seed42 screen for three matched seeds per method. This is seed replication, not hyperparameter search.

Config: `experiments/ml1m_clean_mtl_replicates_2t4.json`. Private target: `carasik/rhpr-ml1m-clean-mtl-replicates`, on two T4. The original `carasik/rhpr-ml1m-clean-mtl/1` output remains intact.

All data/splits/fields, model shapes, 3,500 AE + 3,000 RQ steps, normalization, optimizers, learning rates, FP16 TIGER/native SDPA, batch512/512, fixed30 generator epochs, validation every5, free beam20 and one test per validation-selected checkpoint remain unchanged. Quantizer training stays FP32. Frozen text/poster/description-only MPNet features are computed once and shared; no people graph or training-data modification. No new pretrained assets or runtime installation changes. Existing inputs are declared explicitly, internet disabled. Estimated session duration is about 2.5 hours, extrapolating the previous 73.34-minute four-fit screen; queue/startup and concurrent-worker variation are additional.

The same trial seed controls tokenizer initialization/warm-up/data order, fresh generator initialization and generator training permutations. Each trial receives its own config copy with `seed` set to the trial's seed, preventing accidental use of the parent seed101 for seed202 training. Initial weights are pinned before GPU use and must match across all methods within each seed. Seed42 pins were reproduced as a backward compatibility check. No continuation from seed42 checkpoints.

Round-robin assignments:

| Worker | Ordered trials |
|---|---|
| T4 0 | sum101, GradNorm101, sum202, GradNorm202 |
| T4 1 | UW101, PCGrad101, UW202, PCGrad202 |

Preflight: all eight tiny native end-to-end pipelines completed30epochs with6validation checks; final checkpoint configs and reported seeds agree with their trial seed and the parent suite is not mutated. Contract tests verify complete method×seed coverage, required initialization pins, balanced assignments and unchanged non-seed protocol settings. Full regression results and pinned hashes are recorded in `gr-ml1m-clean-mtl-replicates-preflight.json`.

Artifacts retain the same structure as seed42: all training/MTL probes, normalization, code usage/collisions, semantic codes and suffix SIDs, per-epoch losses, six validation prediction files, validation-selected and last generator checkpoints, final test predictions and trial/suite summaries. Final comparisons should report paired seed-level deltas, each seed's val/test outcome and conditional paired-user uncertainty. With three seeds, do not treat user bootstrap alone as training-seed evidence. Do not pick epochs or loss methods by test results.

No extra methods, HPO, automatic retries, hardware fallbacks, Colab deployment or Telegram messages are included. Colab is an available future option, not needed for this bounded Kaggle run. Once submitted, monitor this exact version and do not duplicate it.
