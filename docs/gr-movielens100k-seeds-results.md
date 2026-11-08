# MovieLens100K text ladder: three pipeline seeds

All fourteen additional fits finished twenty epochs and saved test predictions. Kaggle marked the session ERROR because the final aggregate compared initial weights across seeds instead of within each seed. The submitted source was `60612f5968131745c23b88b8443bca028a93fdf9`; private kernel `carasik/rhpr-ml100k-text-ladder-seeds`, version 1.

Original root metrics.json, summary.md and run-manifest.json are absent. Raw failed-run evidence is preserved. Local recovered-metrics.json reconstructs the fourteen saved trial results; it does not pretend the original run passed its contract. No training was retried. The config now enables compare_initialization_by_seed, and validation requires it for replicated ladders. A regression test reproduces the original false failure, accepts different weights across seeds and still rejects inconsistent weights within a seed.

## Test NDCG@10 at validation-selected checkpoints

Each row adds a field to the previous row. Original titles retain the year; explicit year repeats that information. Keywords are from TMDB. Means and sample standard deviations describe three coupled pipeline seeds, not three independent datasets.

| Cumulative text | Seed 42 | Seed 101 | Seed 202 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| Title (year retained) | 0.04717 | 0.04462 | 0.04388 | 0.04522 ± 0.00173 |
| + Genres | 0.03563 | 0.04373 | 0.04511 | 0.04149 ± 0.00512 |
| + Explicit year | 0.04426 | 0.03382 | 0.04470 | 0.04093 ± 0.00616 |
| + Runtime | 0.04027 | 0.04653 | 0.05076 | 0.04585 ± 0.00528 |
| + TMDB keywords | 0.04341 | 0.03756 | 0.04331 | 0.04143 ± 0.00335 |
| + Cast | 0.03793 | 0.04074 | 0.03652 | 0.03839 ± 0.00215 |
| + Directors | 0.03944 | 0.03822 | 0.04243 | 0.04003 ± 0.00216 |

## Findings

- Full text loses to title-only on test in all three seeds: mean 0.04003 versus 0.04522 (−11.5%). This does not identify the causal role of people names: all preceding fields also differ.
- Adding cast to the keywords variant hurts in two seeds and helps in one: mean test difference −7.3%. Adding directors to the cast variant helps in two seeds and hurts in one: mean +4.3%. Both people fields together versus keywords average −3.4%, with mixed signs across seeds.
- Runtime has the largest observed mean test score, 0.04585, only +1.4% over title. Its paired effect changes sign across seeds. This is descriptive test inspection, not a validated optimum.
- Mean selected validation NDCG@10 for full text is 0.05989 versus 0.05613 for title, while mean test has the opposite ordering. The split and evaluation cases were unchanged.
- Frozen Sentence-T5 embeddings are bitwise identical across pipeline seeds for every stage. The coupled RQ/generator/dropout/case-order seed explains the observed between-seed randomness; these runs cannot separate its individual sources or establish that rare names are encoded poorly.

## Verification

All 105 saved prediction files (21 fits × four validation checkpoints plus test) were independently rescored against their own full SID maps. Users/targets match across files and the exact split report is shared. Every fit consumed 1,709,400 cases and 6,680 optimizer updates with zero AMP skips. Initial weights match their pinned pipeline seed. No text was truncated. Only 17 of 396,060 test beams were invalid; invalid/duplicate slots retain their original ranks.

Fixed recipe: 1,273 films, 943 users, 85,470 strictly earlier-timestamp training cases per epoch, at most 20 context items. Distinct final timestamp groups supply one validation and one test target; other events in those groups are embargoed. All rating values are retained. Frozen Sentence-T5, RQ-VAE 3×256 / 3,000 updates, TIGER 4+4 / d128 / 4,977,920 parameters, FP16 training and FP32 evaluation. Twenty complete epochs, validation at 5/10/15/20, test once at the best validation NDCG@10 checkpoint. Free beam 20, full catalogue, no seen-item mask.

Three pipeline seeds provide descriptive replication. Conditional user-bootstrap intervals saved with the analysis omit training-seed uncertainty and multiple-comparison correction. The 943 users are shared across fits.

## Evidence

Local analysis: `/workspace/artifacts/movielens100k-text-ladder-seeds/analysis/`. It contains the report, all seed/stage metrics, all 420 training epochs, compressed per-user ranks, plots, verification, explicit reconstructed metrics and a numerical-evidence ZIP. Raw fourteen-fit outputs are in `/tmp/rhpr-ml100k-seeds-completed` and retained on private Kaggle. Original seed-42 evidence remains in `/workspace/artifacts/movielens100k-text-ladder/analysis/`. Large model checkpoints and embeddings remain on Kaggle; the compact ZIP preserves all prediction files, SID maps, reports, source hashes and stdout.

Validation of the aggregation fix: 920 repository tests OK (81 skipped), including completed-worker comparison across both pipeline seeds. No new GPU jobs were submitted.
