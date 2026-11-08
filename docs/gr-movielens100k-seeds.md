# MovieLens100K text ladder: two extra pipeline seeds

The first seven-fit run completed. Title-only had the largest observed test
NDCG@10 (0.0471667), while adding TMDB keywords had the largest validation
NDCG@10 (0.0690003). Full text scored 0.0394417 on test. Those observations do
not establish that rich semantics help or harm: one fitted realization and
943 users leave considerable uncertainty. All 35 saved validation/test files
passed independent item-level rescoring.

The user authorized two additional seeds for each of the seven variants on
2026-11-01. Config: `movielens100k_text_ladder_seeds_2t4.json`.

- Fourteen fresh fits: seeds 101 and 202 × the same seven cumulative text stages.
  Reuse the completed seed 42 run for three pipeline replicates per stage.
- One private 2×T4 session, new slug `carasik/rhpr-ml100k-text-ladder-seeds`.
  Worker 0 runs all seven stages at 101; worker 1 runs them at 202.
- Pipeline seed controls RQ-VAE initialization/shuffling, generator initialization,
  dropout and per-epoch training-case permutation. Within a seed these are matched
  across text stages. This estimates joint pipeline variability; it cannot
  separately attribute variance to the quantizer and generator.
- Frozen Sentence-T5, its revision, batch size and 2048-token cap stay fixed.
  Recompute embeddings cheaply and preserve their hashes for equality checks.
- Input catalogue, all 943 users, train/validation/test timestamp-group split,
  embargo, source IDs, case digests and strict-past context builder stay fixed.
- Preserve the entire previous training/quantizer/generator/evaluation recipe:
  RQ-VAE 3×256 / 3000 updates, TIGER 4+4 / d128, FP16 train / FP32 eval,
  twenty complete epochs, validation at 5/10/15/20, one test from the best
  validation NDCG@10 checkpoint, free beam 20 and no seen-item mask.
- Runtime checks pin generator initialization separately for each pipeline seed.
  Trial-specific seed is also passed in the trainer config and saved checkpoints;
  setting only torch.manual_seed would leave case shuffling on the wrong seed.
- Keep per-epoch curves, predictions, embeddings, SID maps, quantizer reports,
  best/last checkpoints and manifests as in the first run.

Seed 101 generator initial parameter SHA256:
`3579338214a2f9d276f655150e0769ac74085188cb68ec0da24ee15cda8ffce9`.
Seed 202:
`eceff6a83a00e4e757c528381dfa1686db548869660faf77e2f99fb7affc172c`.
These were computed using pinned native GRID components; seed 42 was recomputed
and matched the previous `5f4a24fa…` initialization.

Observed first-run epochs were 50.7–54.6 seconds, RQ fits 72–74 seconds. Budget
seven fits per worker at about 2h15–2h40 wall time plus queue/setup; nominal
accelerator consumption is roughly 4.5–5.3 GPU-hours if both cards are counted.
Quota inspection before preparation: 27.99 GPU-hours unreserved.

After completion, report all three scores and mean/spread for each stage at its
validation-selected checkpoint. Compare paired stage-minus-title differences per
seed and show all validation trajectories and selected epochs. Three pipeline
seeds remain exploratory; they do not establish independent quantizer/generator
effects. Test results do not select a new configuration.

No repeat of seed 42, tuning, extra seeds, automatic retries, TPU, Telegram
notifications or revival of earlier studies is authorized by this run.
