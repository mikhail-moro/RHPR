# MovieLens100K multimodal MPNET ladder: results

Private Kaggle notebook `carasik/rhpr-ml100k-multimodal` v2 completed all twelve fits. Executable `addd3c326dbc55ddf18d2dad2c832bf43000a8e0`. Experiment body: 117.50 minutes. The cancelled Sentence-T5-plot v1 is excluded.

Base text is title including year, genres, explicit year, runtime and TMDB keywords. The four cumulative stages use a new shared RQ-VAE with modality MLP heads and fixed-normalized summed reconstruction losses. The historical plain reference uses exactly the same frozen text vectors and the previous RQ-VAE recipe.

## Validation-selected test NDCG@10

| Variant | Seed 42 | Seed 101 | Seed 202 | Mean ± sample SD | Mean Recall@10 |
|---|---:|---:|---:|---:|---:|
| Plain RQ-VAE, same text | 0.04341 | 0.03756 | 0.04331 | 0.04143 ± 0.00335 | 0.08766 |
| Text + heads | 0.03438 | 0.03623 | 0.03103 | 0.03388 ± 0.00264 | 0.07070 |
| + Poster | 0.03921 | 0.03859 | 0.04167 | 0.03982 ± 0.00163 | 0.08059 |
| + MPNET description | 0.04143 | 0.03070 | 0.04067 | 0.03760 ± 0.00599 | 0.07777 |
| + DeepWalk people | 0.03893 | 0.03577 | 0.03849 | 0.03773 ± 0.00171 | 0.08236 |

## Findings

- Posters improve the headed text control in all three test replicates: mean +17.5%. They remain 3.9% below the historical plain reference on average, with mixed signs versus that reference.
- Adding the MPNET description helps in one seed and hurts in two: mean −5.6%. Adding DeepWalk people helps in one seed and hurts in two, with a near-zero mean effect (+0.3%).
- Full multimodal input improves the headed control on average (+11.4%, positive in two seeds), but loses to the historical plain reference in all three seeds (mean −8.9%).
- The headed text-only control itself loses 18.2% to the previous plain RQ-VAE on the identical text vectors. Head architecture, fusion, target normalization and optimization changes are coupled; this experiment does not isolate which causes that loss.
- Validation does not show the same ranking: its selected mean NDCG@10 is 0.05704 / 0.05602 / 0.05598 / 0.05412 for the four stages. These are three exploratory pipeline seeds on the same 943 evaluation users, not a claim of population significance.

## Quantizer diagnostics

| Variant | Unique semantic triples before suffix | Used first-level codes / 256 |
|---|---:|---:|
| Plain reference | 1271–1273 | 98–140 |
| Text with heads | 1025–1088 | 14–15 |
| + Poster | 888–953 | 11–16 |
| + Description | 825–889 | 12–17 |
| + People | 652–756 | 10–14 |

All full SIDs remain unique after appending collision suffixes: no film was dropped. Semantic diversity continues to increase across the saved 500–3000 update probes. Normalized text reconstruction MSE ends around 0.81–0.84 for text-only and 0.87–0.88 for all channels, against a fixed constant-predictor MSE of 1. Other full-model head MSEs are roughly 0.87–0.92. These losses are not directly comparable to the old unit-normalized RQ loss. Summing more reconstruction heads also increases total reconstruction weight relative to the fixed quantization coefficient.

The weak same-text control and these diagnostics motivate checking quantizer training/convergence before judging the usefulness of richer modalities. Code occupancy alone is not a causal explanation, and these results do not prove that longer training or any particular remedy will improve recommendations.

## Protocol and verification

All 60 saved validation/test prediction files were independently rescored against their own full SID maps, plus three historical plain-reference test files. Users, targets and split report align exactly. Every fresh fit completed 20 epochs, 1,709,400 training cases and 6,680 optimizer updates with zero AMP skips. Per-seed initial generator hashes match the reference; quantizer initial parameters are equal across all four stages within a seed. Frozen inputs and normalization are shared. The base text vectors are identical to the plain control. Saved final codebook histograms match the exported semantic codes.

Unchanged data: 1273 films, 943 users, strict earlier-timestamp training contexts capped at 20 events. Distinct final timestamp groups supply one validation and one test target; other events in those groups are embargoed. All ratings retained. RQ-VAE 3×256 / latent 64 / 3000 updates, TIGER 4+4 / d128 / 4977920 parameters, FP16 training and FP32 evaluation. Validation at epochs 5/10/15/20; test once from the best validation NDCG@10 checkpoint. Free beam 20, full catalogue, no seen-item mask.

Plot preparation used pinned all-mpnet-base-v2 on description only: 1272 observed descriptions, 1285 chunks, maximum 657 document tokens, no truncation. It took 21.3 seconds. Descriptions are mostly short plot summaries, not long synopses; one missing plot stays masked. Poster and people inputs are unchanged. Mean generator epoch duration ranges 50.95–52.83 seconds; fresh quantizer fits take about 94–117 seconds.

## Artifacts

`/workspace/artifacts/movielens100k-multimodal/mpnet-replacement/analysis/` contains seed metrics, all 240 generator epochs and 72 quantizer probes, compressed per-user ranks, verification, conditional paired-user comparisons, PNG/PDF figures and numerical-evidence.zip. The ZIP includes generated frozen features, raw reports, predictions, SID maps, final codebooks, download hashes and stdout. Large checkpoints remain on private Kaggle v2. The analysis script reproduces these checks. No additional GPU work was submitted.
