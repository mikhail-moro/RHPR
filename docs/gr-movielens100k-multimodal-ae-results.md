# MovieLens100K staged multimodal ladder: completed results

Private notebook v1 COMPLETE; executable `ba160aac3155831e06286b6e6924eb720b82dfc2`. All12fits/20epochs; body 121.16min. No new GPU work.

| Representation | Direct test mean ± SD | AE test mean ± SD | Change | Positive seeds | AE selected val mean |
|---|---:|---:|---:|---:|---:|
| Text | 0.03388 ± 0.00264 | 0.04016 ± 0.00261 | +18.5% | 3/3 | 0.05684 |
| + Poster | 0.03982 ± 0.00163 | 0.04127 ± 0.00174 | +3.6% | 2/3 | 0.06171 |
| + Description | 0.03760 ± 0.00599 | 0.04292 ± 0.00305 | +14.1% | 2/3 | 0.05925 |
| + People | 0.03773 ± 0.00171 | 0.04326 ± 0.00636 | +14.7% | 2/3 | 0.05676 |

Historical plain RQ on exactly the same text: test 0.04143 ± 0.00335, selected val 0.05866. All values are validation-selected checkpoints; test was not used to choose an epoch.

## AE suite by seed

| Stage | Seed42 | Seed101 | Seed202 |
|---|---:|---:|---:|
| Text | 0.03762 | 0.04004 | 0.04283 |
| + Poster | 0.03962 | 0.04308 | 0.04111 |
| + Description | 0.03940 | 0.04468 | 0.04467 |
| + People | 0.03605 | 0.04563 | 0.04808 |

## Paired comparisons

| Comparison | Mean relative test change | Positive seeds | Conditional user bootstrap 95% interval for absolute NDCG delta |
|---|---:|---:|---:|
| AE vs direct, stage1 | +18.5% | 3/3 | [+0.00039, +0.01248] |
| AE vs direct, stage2 | +3.6% | 2/3 | [-0.00498, +0.00782] |
| AE vs direct, stage3 | +14.1% | 2/3 | [-0.00081, +0.01137] |
| AE vs direct, stage4 | +14.7% | 2/3 | [-0.00060, +0.01170] |
| Poster increment | +2.8% | 2/3 | [-0.00481, +0.00731] |
| Description increment | +4.0% | 2/3 | [-0.00481, +0.00804] |
| People increment | +0.8% | 2/3 | [-0.00545, +0.00623] |
| All vs text | +7.7% | 2/3 | [-0.00261, +0.00908] |
| AE text vs plain | -3.1% | 1/3 | [-0.00716, +0.00470] |
| AE all vs plain | +4.4% | 2/3 | [-0.00421, +0.00797] |

AE improves mean test versus direct RQ in all four input combinations, but these are three seeds and the improvements are not uniform across seeds. Adding modalities in the AE suite yields small positive mean test increments with mixed signs; validation prefers the poster stage. Full multimodal input does not establish a stable superiority over the same-text plain reference. Its mean Recall@10 is0.08766, equal to the plain reference; description-stage recall is0.09084, the highest in the AE ladder. Conditional intervals omit seed/runtime variation and multiple comparisons.

## Quantizer and timing

| Stage | Semantic triples range | Used level-1 codes range | AE steps | AE seconds | Quantized text MSE |
|---|---:|---:|---:|---:|---:|
| Text | 1270–1273 | 228–252 | 3500–4500 | 56.1–74.1 | 0.7196–0.7551 |
| + Poster | 1273–1273 | 235–250 | 3500–5000 | 63.8–91.3 | 0.7852–0.7906 |
| + Description | 1273–1273 | 247–256 | 3500–5000 | 75.9–104.0 | 0.7563–0.7933 |
| + People | 1270–1273 | 251–255 | 3500–4500 | 80.4–104.5 | 0.8049–0.8136 |

RQ fit seconds 92.4–115.1; mean TIGER epoch seconds 49.87–50.71. AE stopping reasons: {'spectral_plateau': 12}. Spectral plateau is not reconstruction convergence or a proven quality optimum.

## Seed42 replay limitation

The repeated text42 fit has exactly the same AE neural state, exported latents, RQ loss/occupancy curves, final codebooks/SIDs and generator initial parameters as the pilot. Nevertheless generator training losses diverge from epoch1: 9.355196830 versus 9.355121774. Selected epoch changes 15 → 20, test 0.04210 → 0.03762. This localizes the divergence after quantization, during generator training. The exact numerical/runtime cause is not established; do not claim fixed seeds ensure bitwise reproducibility or treat the pilot as a fourth seed.

## Verification and evidence

Independently rescored 140 prediction files:60new+60direct+15plain+5pilot, each with943matched users/targets and their own full SID map. Same split, frozen vectors and masks, normalization, initial headed quantizer and generator weights; every new fit20epochs/1709400examples/6680updates/zeroAMPskips. All12untouched-book/neural+BNtransfer/fresh optimizer checks pass, initial latent max difference0 on all1273items. Final counts match saved codes; full SIDs identify every film.

AE extra training, initialized latent geometry and first-batch evaluation mode change together; this does not isolate the cause or prove occupancy determines NDCG. No modalities-only causal claim. The task is next rating in MovieLens with all ratings retained, not observed VOD viewing chronology.

Reports/CSV/plots/per-user ranks are under `/workspace/artifacts/movielens100k-multimodal-ae/analysis`. Full numerical bundle (without large weights) is `/tmp/movielens100k-multimodal-ae-numerical-evidence.zip`; large checkpoints stay on private Kaggle. Source/config submission audit remains in the parent artifact directory.
